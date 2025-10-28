# Vision Token Compression - 프로젝트 요약

## 프로젝트 개요

CLIP 비전 트랜스포머의 토큰을 2D 컨볼루션을 이용하여 효율적으로 압축하는 딥러닝 시스템입니다.

## 핵심 특징

### 1. 아키텍처
- **CLIP ViT-L/14 (Frozen)**: 336px 이미지 → 24×24 토큰 그리드 (576 tokens)
- **2D Conv Compressor**: 토큰의 공간적 지역성을 활용한 점진적 압축
- **WGAN-GP Discriminator**: 압축 전후 토큰 분포 정렬
- **AutoEncoder Decoder**: 압축된 토큰으로부터 원본 토큰 재구성

### 2. 손실 함수
- **WGAN-GP Loss**: 압축된 토큰이 원본 토큰과 구분 불가능하도록 학습
- **AutoEncoder Loss**: Receptive field 내 토큰들의 정보 보존
- **Gradient Penalty**: 안정적인 WGAN 학습을 위한 정규화

### 3. 최적화
- 모든 연산 GPU 벡터화
- Python 루프 제거
- 효율적인 배치 처리
- Mixed precision 지원

## 프로젝트 구조

```
ToConv/
├── vision_token_compression/          # 메인 패키지
│   ├── models/                       # 모델 정의
│   │   ├── clip_encoder.py          # CLIP 비전 인코더 래퍼
│   │   ├── token_compressor.py      # 2D Conv 토큰 압축기
│   │   ├── discriminator.py         # WGAN-GP Discriminator
│   │   └── autoencoder.py           # AutoEncoder Decoder
│   ├── losses/                       # 손실 함수
│   │   ├── wgan_gp.py               # WGAN-GP 손실
│   │   └── autoencoder.py           # AutoEncoder 손실
│   ├── data/                         # 데이터 로딩
│   │   └── imagenet_dataset.py      # ImageNet 데이터셋
│   ├── configs/                      # Hydra 설정
│   │   └── config.yaml              # 메인 설정 파일
│   └── trainer.py                    # 학습 루프
├── scripts/                          # 유틸리티 스크립트
│   ├── quick_test.py                # 시스템 테스트
│   └── train_example.sh             # 학습 예제
├── train.py                          # 메인 학습 스크립트
├── inference.py                      # 추론 스크립트
├── requirements.txt                  # Python 의존성
├── README.md                         # 프로젝트 문서
├── GETTING_STARTED.md               # 빠른 시작 가이드
└── PROJECT_SUMMARY.md               # 이 파일
```

## 구현된 기능

### ✅ 완료된 기능

1. **모델 아키텍처**
   - [x] CLIP Vision Encoder 래퍼 (freeze 지원)
   - [x] 2D Convolutional Token Compressor
   - [x] WGAN-GP Discriminator
   - [x] Convolutional AutoEncoder Decoder
   - [x] Attention-based AutoEncoder Decoder (선택적)
   - [x] Multi-scale Discriminator (선택적)

2. **손실 함수**
   - [x] WGAN-GP Loss with Gradient Penalty
   - [x] MSE Reconstruction Loss
   - [x] Cosine Similarity Loss
   - [x] Hybrid Loss (MSE + Cosine)
   - [x] Perceptual Reconstruction Loss (공간적 일관성)

3. **데이터 처리**
   - [x] ImageNet 데이터로더
   - [x] CLIP 전처리 파이프라인
   - [x] 효율적인 배치 처리
   - [x] 멀티워커 지원

4. **학습 인프라**
   - [x] 전체 학습 루프
   - [x] Discriminator/Generator 교대 학습
   - [x] 체크포인트 저장/로딩
   - [x] 검증 루프
   - [x] W&B 통합
   - [x] Hydra 설정 관리

5. **유틸리티**
   - [x] 추론 스크립트
   - [x] 시각화 도구
   - [x] 시스템 테스트
   - [x] 예제 스크립트

## 기술 스택

- **PyTorch 2.0+**: 딥러닝 프레임워크
- **Transformers**: CLIP 모델 로딩
- **Hydra**: 설정 관리
- **W&B**: 실험 추적
- **torchvision**: 데이터 로딩 및 전처리

## 핵심 파라미터

### 모델 설정
```yaml
model:
  clip:
    model_name: "openai/clip-vit-large-patch14-336"
    freeze: true

  compressor:
    output_grid_size: 6  # 압축 후 그리드 크기
    num_layers: 3
    use_residual: true
```

### 손실 가중치
```yaml
loss:
  wgan:
    lambda_gp: 10.0

  weights:
    wgan: 1.0
    ae: 1.0
```

### 학습 설정
```yaml
training:
  epochs: 100
  batch_size: 32
  learning_rate: 1e-4
  n_critic: 5  # Discriminator 업데이트 빈도
```

## 압축 비율 예시

| 출력 그리드 | 압축 토큰 수 | 압축률 | 권장 용도 |
|------------|------------|--------|----------|
| 4×4 | 16 | 36x | 극도의 압축 |
| 6×6 | 36 | 16x | 균형잡힌 압축 (기본값) |
| 8×8 | 64 | 9x | 정보 보존 우선 |
| 12×12 | 144 | 4x | 최소 압축 |

## 사용 예시

### 1. 시스템 테스트
```bash
python scripts/quick_test.py
```

### 2. 빠른 학습 테스트
```bash
python train.py \
    data.use_subset=true \
    data.subset_size=1000 \
    training.epochs=5
```

### 3. 전체 학습
```bash
python train.py
```

### 4. 커스텀 압축률
```bash
python train.py model.compressor.output_grid_size=8
```

### 5. 추론
```bash
python inference.py \
    --checkpoint checkpoints/best_model.pt \
    --image image.jpg
```

## 성능 지표

학습 중 추적되는 주요 메트릭:

- **Wasserstein Distance**: 원본과 압축 토큰 분포 간 거리
- **Reconstruction Loss**: 재구성 품질
- **Cosine Similarity**: 토큰 간 유사도 (0~1, 높을수록 좋음)
- **Gradient Penalty**: 안정성 지표

## 예상 결과

잘 학습된 모델의 경우:
- Cosine Similarity: > 0.85
- Wasserstein Distance: < 0.1
- 시각적으로 원본과 유사한 재구성

## 최적화 팁

1. **메모리 부족 시**:
   - 배치 크기 감소: `training.batch_size=16`
   - 압축률 증가: `model.compressor.output_grid_size=4`

2. **학습 불안정 시**:
   - n_critic 증가: `training.n_critic=10`
   - Gradient penalty 증가: `loss.wgan.lambda_gp=20.0`

3. **재구성 품질 향상**:
   - AE 가중치 증가: `loss.weights.ae=2.0`
   - Attention decoder 사용: `model.autoencoder.use_attention=true`

## 확장 가능성

이 시스템은 다음과 같이 확장 가능합니다:

1. **다른 CLIP 모델**: ViT-B/16, ViT-H/14 등
2. **커스텀 데이터셋**: 도메인 특화 데이터
3. **다운스트림 태스크**: 분류, 검색, 생성 등에 압축 토큰 활용
4. **동적 압축**: 이미지 복잡도에 따라 압축률 조정

## 라이선스

MIT License

## 참고 문헌

- [CLIP: Learning Transferable Visual Models](https://arxiv.org/abs/2103.00020)
- [Improved Training of Wasserstein GANs](https://arxiv.org/abs/1704.00028)
- [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929)

## 작성자

AI Assistant를 통해 구현된 연구 프로젝트

---

**마지막 업데이트**: 2025년 10월 28일
