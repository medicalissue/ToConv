# Getting Started - Vision Token Compression

빠르게 시작하기 위한 가이드입니다.

## 1. 빠른 설치

```bash
# 프로젝트 디렉토리로 이동
cd ToConv

# 의존성 설치
pip install -r requirements.txt
```

## 2. 시스템 테스트

설치가 제대로 되었는지 확인:

```bash
python scripts/quick_test.py
```

모든 테스트가 통과하면 다음과 같은 메시지가 표시됩니다:
```
ALL TESTS PASSED! System is ready for training.
```

## 3. 설정 파일 수정

[vision_token_compression/configs/config.yaml](vision_token_compression/configs/config.yaml) 파일을 열어 다음 항목들을 수정하세요:

### 필수 설정

```yaml
# ImageNet 데이터셋 경로 (필수!)
data:
  imagenet_root: "/path/to/imagenet"  # 실제 ImageNet 경로로 변경

# CUDA 디바이스 선택
hardware:
  cuda_device: 0  # 사용할 GPU 번호
```

### 선택적 설정

```yaml
# 압축률 조정
model:
  compressor:
    output_grid_size: 6  # 6x6=36 tokens (기본값)
                         # 더 높은 압축: 4 (16 tokens)
                         # 더 낮은 압축: 8 (64 tokens)

# W&B 로깅 (선택사항)
experiment:
  use_wandb: true  # W&B 사용 시 true
  wandb_project: "vision-token-compression"
  wandb_entity: "your-username"  # W&B 사용자명
```

## 4. 학습 시작

### 방법 1: 빠른 테스트 (데이터 서브셋)

설정이 제대로 작동하는지 확인:

```bash
python train.py \
    data.use_subset=true \
    data.subset_size=1000 \
    training.epochs=5 \
    training.batch_size=16
```

### 방법 2: 전체 학습

```bash
python train.py
```

### 방법 3: 특정 설정 오버라이드

```bash
# 8x8 압축으로 학습
python train.py model.compressor.output_grid_size=8

# 다른 GPU에서 학습
python train.py hardware.cuda_device=1

# 배치 크기 변경
python train.py training.batch_size=64
```

## 5. 학습 모니터링

### 콘솔 출력

학습 중 실시간으로 다음 정보가 표시됩니다:

```
Epoch 1/100
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100%
D_loss: 0.1234 | G_loss: 0.5678 | AE_loss: 0.0123 | W_dist: 0.4567
```

### W&B 대시보드 (선택사항)

W&B를 활성화한 경우:

1. 터미널에 표시된 W&B 링크 클릭
2. 실시간 그래프와 메트릭 확인
3. 다른 실험과 비교

## 6. 체크포인트

모델은 자동으로 저장됩니다:

```
./checkpoints/
├── best_model.pt          # 검증 손실이 가장 낮은 모델
├── latest.pt              # 가장 최근 체크포인트
└── checkpoint_epoch_N.pt  # 주기적 체크포인트
```

## 7. 학습된 모델로 추론

```bash
python inference.py \
    --checkpoint ./checkpoints/best_model.pt \
    --image /path/to/your/image.jpg \
    --output result.png
```

결과:
- 원본 토큰 시각화
- 압축된 토큰 시각화
- 재구성 품질 히트맵

## 8. 주요 설정 조정

### 압축률 변경

더 높은 압축 (정보 손실 증가):
```bash
python train.py model.compressor.output_grid_size=4  # 24x24 → 4x4 (36배 압축)
```

더 낮은 압축 (정보 보존 증가):
```bash
python train.py model.compressor.output_grid_size=12  # 24x24 → 12x12 (4배 압축)
```

### 손실 함수 가중치

AutoEncoder 손실을 더 중요하게:
```bash
python train.py loss.weights.ae=2.0 loss.weights.wgan=1.0
```

WGAN 손실을 더 중요하게:
```bash
python train.py loss.weights.wgan=2.0 loss.weights.ae=1.0
```

### Discriminator 업데이트 빈도

더 강한 Discriminator:
```bash
python train.py training.n_critic=10  # Generator 1회당 Discriminator 10회 업데이트
```

더 약한 Discriminator:
```bash
python train.py training.n_critic=1  # 동일한 빈도로 업데이트
```

## 9. 문제 해결

### CUDA Out of Memory

배치 크기 줄이기:
```bash
python train.py training.batch_size=16  # 또는 8
```

### ImageNet 경로 오류

설정 파일에서 경로 확인:
```yaml
data:
  imagenet_root: "/path/to/imagenet"
```

ImageNet 구조:
```
/path/to/imagenet/
├── train/
│   ├── n01440764/
│   ├── n01443537/
│   └── ...
└── val/
    ├── n01440764/
    ├── n01443537/
    └── ...
```

### 데이터셋 없이 테스트

작은 데이터 서브셋 사용:
```bash
python train.py data.use_subset=true data.subset_size=100
```

## 10. 다음 단계

- 다양한 압축률 실험
- 손실 함수 가중치 튜닝
- Attention 기반 디코더 시도
- 커스텀 데이터셋 사용
- 압축된 토큰을 downstream task에 적용

## 도움말

더 자세한 정보는 다음 문서를 참고하세요:

- [README.md](README.md): 전체 프로젝트 개요
- [vision_token_compression/configs/config.yaml](vision_token_compression/configs/config.yaml): 모든 설정 옵션
- 각 모듈의 docstring: 구현 세부사항

질문이나 이슈가 있으면 GitHub Issues를 활용해주세요.

## 빠른 참조

### 기본 명령어

```bash
# 시스템 테스트
python scripts/quick_test.py

# 빠른 학습 테스트
python train.py data.use_subset=true data.subset_size=1000 training.epochs=5

# 전체 학습
python train.py

# 추론
python inference.py --checkpoint checkpoints/best_model.pt --image image.jpg
```

### 주요 하이퍼파라미터

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `output_grid_size` | 6 | 압축 후 그리드 크기 (k×k) |
| `batch_size` | 32 | 배치 크기 |
| `learning_rate` | 1e-4 | Generator 학습률 |
| `lambda_gp` | 10.0 | Gradient penalty 가중치 |
| `n_critic` | 5 | Discriminator 업데이트 빈도 |
| `ae_weight` | 1.0 | AutoEncoder 손실 가중치 |
| `wgan_weight` | 1.0 | WGAN 손실 가중치 |

성공적인 학습 되시길 바랍니다!
