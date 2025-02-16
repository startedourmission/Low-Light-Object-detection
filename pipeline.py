import os
import shutil
import yaml
from pathlib import Path
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from models.brightness_cnn import BrightnessCNN
from models.zerodce import enhance_net_nopool
from ultralytics import YOLO

class ObjectDetectionPipeline:
    def __init__(self, cnn_weights, zerodce_weights, yolo_weights, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # 모델 초기화
        self.brightness_classifier = BrightnessCNN().to(self.device)
        self.brightness_classifier.load_state_dict(torch.load(cnn_weights))
        self.brightness_classifier.eval()
        
        self.zerodce = enhance_net_nopool().to(self.device)
        self.zerodce.load_state_dict(torch.load(zerodce_weights))
        self.zerodce.eval()
        
        self.detector = YOLO(yolo_weights)
        
        self.brightness_transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.Grayscale(1),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def classify_brightness(self, image):
        with torch.no_grad():
            img = self.brightness_transform(image).unsqueeze(0).to(self.device)
            output = self.brightness_classifier(img)
            return output.item()
            
    def enhance_image(self, image):
        with torch.no_grad():
            transform = transforms.ToTensor()
            img_tensor = transform(image).unsqueeze(0).to(self.device)
            _,enhanced,_ = self.zerodce(img_tensor)
            return transforms.ToPILImage()(enhanced.squeeze().cpu())
            
    def prepare_eval_directory(self, test_dir, temp_dir, threshold=0.5):
        """평가를 위한 임시 디렉토리 준비"""
        # 임시 디렉토리 생성
        temp_images_dir = Path(temp_dir) / 'images'
        temp_labels_dir = Path(temp_dir) / 'labels'
        temp_images_dir.mkdir(parents=True, exist_ok=True)
        temp_labels_dir.mkdir(parents=True, exist_ok=True)
        
        # 통계 초기화
        stats = {
            'total_images': 0,
            'dark_images': 0,
            'bright_images': 0,
            'brightness_scores': []
        }
        
        # 이미지 처리
        for img_file in os.listdir(test_dir):
            if not img_file.endswith(('.jpg', '.jpeg', '.png')):
                continue
                
            stats['total_images'] += 1
            image_path = os.path.join(test_dir, img_file)
            
            # 이미지 로드 및 밝기 분류
            image = Image.open(image_path).convert('RGB')
            brightness_score = self.classify_brightness(image)
            stats['brightness_scores'].append(brightness_score)
            
            # 라벨 파일 복사
            label_name = os.path.splitext(img_file)[0] + '.txt'
            src_label = Path(str(Path(test_dir)).replace('images', 'labels') + '/' + label_name)
            if src_label.exists():
                shutil.copy2(src_label, temp_labels_dir / label_name)
            
            # 어두운 이미지 향상 또는 밝은 이미지 복사
            if brightness_score < threshold:#len(img_file) < 17:#
                stats['dark_images'] += 1
                enhanced_image = self.enhance_image(image)
                enhanced_image.save(temp_images_dir / img_file)
            else:
                stats['bright_images'] += 1
                image.save(temp_images_dir / img_file)
        
        return stats
        
    def create_temp_yaml(self, temp_dir, original_yaml):
        """임시 YAML 파일 생성"""
        # 원본 YAML 읽기
        with open(original_yaml, 'r') as f:
            yaml_data = yaml.safe_load(f)
        
        # 경로 수정
        yaml_data['path'] = str(Path(temp_dir).absolute())
        yaml_data['val'] = 'images'  # 이미지 디렉토리 지정
        
        # 임시 YAML 파일 저장
        temp_yaml = Path(temp_dir) / 'temp.yaml'
        with open(temp_yaml, 'w') as f:
            yaml.dump(yaml_data, f)
            
        return temp_yaml

    def evaluate_directory(self, test_dir, data_yaml): #test_dir
        """디렉토리 평가 수행"""
        # 임시 디렉토리 설정
        temp_dir = 'temp_eval'
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        
        try:
            # 평가용 임시 디렉토리 준비
            print("\nPreparing evaluation directory...")
            stats = self.prepare_eval_directory(test_dir, temp_dir)
            
            print(f"\nProcessed {stats['total_images']} images:")
            print(f"Dark images: {stats['dark_images']}")
            print(f"Bright images: {stats['bright_images']}")
            print(f"Average brightness score: {np.mean(stats['brightness_scores']):.4f}")
            
            # 임시 YAML 생성
            temp_yaml = self.create_temp_yaml(temp_dir, data_yaml)
            
            # YOLOv5 validation 수행
            print("\nRunning YOLOv5 validation...")
            results = self.detector.val(
                data=str(temp_yaml),
                verbose=False
            )
            
            return stats, results
            
        finally:
            # 임시 디렉토리 정리
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

if __name__ == "__main__":
    pipeline = ObjectDetectionPipeline(
        cnn_weights='weights/best.pth',
        zerodce_weights='weights/zero_dce.pth',
        yolo_weights='weights/yolov5s.pt'
    )
    
    test_dir = '../datasets/generic_sample/images/all'
    # test_dir = "../datasets/reorganized_dataset91/images/L09/"
    data_yaml = 'generic.yaml'
    stats, results = pipeline.evaluate_directory(test_dir, data_yaml)