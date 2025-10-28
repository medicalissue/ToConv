# Vision Token Compression System

CLIP 비전 트랜스포머의 토큰을 효율적으로 압축하는 딥러닝 시스템입니다.

## 주요 기능

- **CLIP ViT-L/14 통합**: Frozen CLIP 비전 인코더로 고품질 비전 토큰 추출
- **2D 컨볼루션 기반 압축**: 토큰의 공간적 지역성을 활용한 효율적인 압축
- **WGAN-GP 손실**: 압축 전후 토큰 분포를 일치시켜 정보 보존
- **AutoEncoder 손실**: Receptive field 기반 토큰 재구성으로 정보 손실 방지
- **GPU 최적화**: 모든 연산을 GPU에서 병렬 처리
- **Hydra 설정 관리**: 유연한 실험 설정 관리
- **W&B 로깅**: 실시간 학습 모니터링

## 아키텍처

```
ImageNet 이미지 (336×336)
    ↓
CLIP ViT-L/14 (Frozen)
    ↓
토큰 (24×24 grid, 576 tokens)
    ↓
2D Conv Compressor
    ↓
압축된 토큰 (k×k grid, 예: 6×6 = 36 tokens)
    ↓
├─→ Discriminator (WGAN-GP) → 분포 정렬 손실
└─→ AutoEncoder Decoder → 재구성 손실
```

## 설치

```bash
# 저장소 클론
git clone <repository-url>
cd ToConv

# 의존성 설치
pip install -r requirements.txt
```

## 필요 사항

- Python 3.8+
- PyTorch 2.0+
- CUDA 지원 GPU (권장)
- ImageNet 데이터셋

## 사용법

### 1. 설정 파일 수정

[vision_token_compression/configs/config.yaml](vision_token_compression/configs/config.yaml)에서 설정을 수정합니다:

```yaml
# ImageNet 경로 설정
data:
  imagenet_root: "/path/to/imagenet"

# CUDA 디바이스 선택
hardware:
  cuda_device: 0

# 압축 그리드 크기 설정
model:
  compressor:
    output_grid_size: 6  # 6x6 = 36 tokens
```

### 2. 학습 시작

```bash
# 기본 설정으로 학습
python train.py

# 특정 설정 오버라이드
python train.py model.compressor.output_grid_size=8 training.batch_size=16

# CUDA 디바이스 선택
python train.py hardware.cuda_device=1
```

### 3. W&B 로깅 (선택사항)

```bash
# W&B 로그인
wandb login

# 설정 파일에서 W&B 활성화
# experiment.use_wandb: true
# experiment.wandb_project: "your-project-name"
```

## 프로젝트 구조

```
ToConv/
├── vision_token_compression/
│   ├── models/
│   │   ├── clip_encoder.py       # CLIP 비전 인코더 래퍼
│   │   ├── token_compressor.py   # 2D Conv 토큰 압축기
│   │   ├── discriminator.py      # WGAN-GP Discriminator
│   │   └── autoencoder.py        # AutoEncoder Decoder
│   ├── losses/
│   │   ├── wgan_gp.py           # WGAN-GP 손실 함수
│   │   └── autoencoder.py       # AutoEncoder 손실 함수
│   ├── data/
│   │   └── imagenet_dataset.py  # ImageNet 데이터로더
│   ├── configs/
│   │   └── config.yaml          # Hydra 설정 파일
│   └── trainer.py               # 학습 루프
├── train.py                     # 메인 학습 스크립트
├── requirements.txt             # 의존성
└── README.md                    # 이 파일
```

## 주요 컴포넌트 설명

### 1. CLIP Vision Encoder
- CLIP ViT-L/14 모델을 사용하여 이미지를 토큰으로 변환
- 완전히 freeze되어 학습 중 업데이트되지 않음
- 336px 해상도에서 24×24 grid (576 tokens) 생성

### 2. Token Compressor
- 2D 컨볼루션을 사용한 점진적 다운샘플링
- 토큰의 공간적 관계 보존
- 설정 가능한 압축률 (예: 576 → 36 tokens)

### 3. WGAN-GP Discriminator
- 압축 전후 토큰 분포를 구분
- Gradient Penalty로 안정적인 학습
- 압축된 토큰이 원본 토큰과 구분 불가능하도록 학습

### 4. AutoEncoder Decoder
- 압축된 토큰으로부터 원본 토큰 재구성
- Convolutional 또는 Attention 기반 디코더 선택 가능
- 정보 손실 최소화

## 손실 함수

### WGAN-GP Loss
```python
L_disc = -E[D(real)] + E[D(fake)] + λ_gp * GP
L_gen = -E[D(fake)]
```

### AutoEncoder Loss
```python
L_ae = MSE(reconstructed, original) + Cosine(reconstructed, original)
```

### Total Loss
```python
L_total = α * L_gen + β * L_ae
```

## 하이퍼파라미터 튜닝

주요 하이퍼파라미터:

- `output_grid_size`: 압축 후 그리드 크기 (6, 8, 12 등)
- `lambda_gp`: Gradient penalty 가중치 (기본값: 10.0)
- `n_critic`: Discriminator 업데이트 횟수 (기본값: 5)
- `ae_weight`, `wgan_weight`: 손실 함수 가중치

## 성능 최적화

- 모든 연산은 GPU에서 벡터화되어 실행
- Python 루프 제거
- 효율적인 텐서 연산
- 배치 처리 및 데이터로더 최적화
- Mixed precision 학습 지원 (설정 가능)

## 체크포인트

체크포인트는 `./checkpoints` 디렉토리에 저장됩니다:

- `checkpoint_epoch_N.pt`: 주기적 체크포인트
- `best_model.pt`: 검증 손실이 가장 낮은 모델
- `latest.pt`: 가장 최근 체크포인트

## 테스트

각 모듈을 개별적으로 테스트할 수 있습니다:

```bash
# CLIP 인코더 테스트
python -m vision_token_compression.models.clip_encoder

# 토큰 압축기 테스트
python -m vision_token_compression.models.token_compressor

# Discriminator 테스트
python -m vision_token_compression.models.discriminator

# AutoEncoder 테스트
python -m vision_token_compression.models.autoencoder

# 손실 함수 테스트
python -m vision_token_compression.losses.wgan_gp
python -m vision_token_compression.losses.autoencoder
```

## 라이선스

MIT License

## 참고 자료

- [CLIP Paper](https://arxiv.org/abs/2103.00020)
- [WGAN-GP Paper](https://arxiv.org/abs/1704.00028)
- [Vision Transformers](https://arxiv.org/abs/2010.11929)

## 문의

이슈나 질문이 있으시면 GitHub Issues를 이용해주세요.
