# Lowlight Object Detection

저조도 환경에서의 객체 탐지 성능을 향상시키기 위한 파이프라인을 제안합니다.

## 주요 특징

- CNN 기반 저조도 이미지 분류기
- ZeroDCE 저조도 이미지 향상 기법
- YOLOv5 객체 탐지 모델

## 작동 방식

1. 입력 이미지의 저조도 여부를 CNN 모델로 판단
2. 저조도로 판단된 이미지는 ZeroDCE를 통해 향상
3. 향상된 이미지 또는 원본 이미지를 YOLOv5로 객체 탐지

## 시작하기

### 요구사항
- PyTorch
- CUDA 지원 GPU
- Python 3.8+

### 설치
```bash
git clone [repository URL]
cd ODLE
pip install -r requirements.txt
```

## 프로젝트 구조
```
ODLE/
├── models/
│   ├── brightness_cnn.py     # CNN 모델
│   ├── zerodce.py           # ZeroDCE 모델
│   └── yolov5/              # YOLOv5 모델
├── weights/                  # 사전 학습된 가중치
└── pipeline.py              # 메인 파이프라인
```

## 개발 환경
- CentOS 7.9
- CUDA 12.4
- Tesla V100 GPU
