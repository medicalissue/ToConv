# RF-Based Vision Token Compression System

**Receptive Field 단위 1:1 비교 기반 비전 토큰 압축 시스템**

## 프로젝트 개요

CLIP 비전 트랜스포머의 토큰을 **Receptive Field (RF) 단위로** 효율적으로 압축하는 딥러닝 시스템입니다.

### 핵심 개선사항

기존 시스템과의 차이점:

| 구분 | 기존 (Global) | 신규 (RF-based) |
|------|--------------|----------------|
| **Discriminator 비교** | 576개 vs 36개 (불공정) | 36개 vs 36개 (공정, 1:1) |
| **샘플링 방식** | 전역 비교 | RF 내 랜덤 샘플링 |
| **AutoEncoder** | 36개 → 576개 전체 재구성 | 36개 → 각각 16개 RF 재구성 |
| **Spatial awareness** | 없음 | RF 단위로 공간 정보 보존 |
| **해석 가능성** | 낮음 | RF별 품질 분석 가능 |

## RF 기반 아키텍처

```
ImageNet 이미지 (336×336)
    ↓
CLIP ViT-L/14 (Frozen)
    ↓
토큰 (24×24 grid, 576 tokens)
    ↓
2D Conv Compressor
    ↓
압축된 토큰 (6×6 grid, 36 tokens)
    ↓
├─→ RF Discriminator
│   └─ 각 압축 토큰 vs RF 내 랜덤 샘플 1개 (36개 비교)
│
└─→ RF AutoEncoder Decoder
    └─ 각 압축 토큰 → 자신의 4×4 RF (16개) 재구성
```

### Receptive Field 구조

```
24×24 Original Grid               6×6 Compressed Grid
┌────────────────────────┐       ┌──────┐
│ RF0  │ RF1  │...│ RF5 │       │0│1│..│5│
│ 4×4  │ 4×4  │   │ 4×4 │       │6│7│..│ │
├──────┼──────┼───┼─────┤       │ │ │  │ │
│ RF6  │ RF7  │...│ RF11│  →    │ │ │  │ │
│ 4×4  │ 4×4  │   │ 4×4 │       │ │ │  │ │
├──────┼──────┼───┼─────┤       │ │ │  │35│
│  ...   ...     ...    │       └──────┘
│ RF30 │ RF31 │...│ RF35│
└────────────────────────┘

각 압축 토큰은 자신의 4×4 RF (16개 원본 토큰)에 대응
```

## 설치

```bash
cd ToConv
pip install -r requirements.txt
```

## 빠른 시작

### 1. 시스템 테스트

```bash
python scripts/test_rf_system.py
```

모든 컴포넌트가 정상 작동하는지 확인합니다.

### 2. 설정 파일 수정

[vision_token_compression/configs/rf_config.yaml](vision_token_compression/configs/rf_config.yaml)에서:

```yaml
# ImageNet 경로 설정 (필수!)
data:
  imagenet_root: "/path/to/imagenet"

# CUDA 디바이스 선택
hardware:
  cuda_device: 0

# 압축 그리드 크기
model:
  compressor:
    output_grid_size: 6  # 6x6 = 36 tokens
```

### 3. RF 기반 학습 시작

```bash
# 전체 학습
python train_rf.py

# 빠른 테스트 (데이터 서브셋)
python train_rf.py data.use_subset=true data.subset_size=1000 training.epochs=5

# 다른 압축률
python train_rf.py model.compressor.output_grid_size=8  # 8x8 = 64 tokens

# CUDA 디바이스 선택
python train_rf.py hardware.cuda_device=1
```

## RF 기반 컴포넌트

### 1. RF Discriminator

**위치**: `models/rf_discriminator.py`

**특징**:
- 단일 토큰 쌍 비교 (batch_size, 1024)
- 경량 MLP 아키텍처
- 36개 토큰 쌍에 병렬 적용

**작동 방식**:
```python
for each compressed token at position (i,j):
    # 1. RF에서 랜덤으로 1개 샘플
    rf_tokens = original_tokens[RF_region]  # 16개
    sampled = random_choice(rf_tokens)       # 1개

    # 2. 비교 (1:1)
    score = discriminator(compressed_token[i,j])
    vs
    score = discriminator(sampled)
```

### 2. RF AutoEncoder Decoder

**위치**: `models/rf_autoencoder.py`

**특징**:
- 각 압축 토큰이 자신의 RF만 재구성
- 2D Conv upsampling: 1×1 → 4×4
- 또는 Attention 기반 디코더 (선택)

**입출력**:
- 입력: (B, 36, 1024) - 압축 토큰
- 출력: (B, 36, 16, 1024) - 36개 RF, 각각 16개 토큰

### 3. RF WGAN-GP Loss

**위치**: `losses/rf_wgan_gp.py`

**핵심 함수**:
- `sample_rf_tokens()`: 각 RF에서 랜덤 샘플링
- `compute_rf_gradient_penalty()`: 단일 토큰용 GP

**비교 방식**:
```python
# 36개 압축 토큰
compressed_tokens: (B, 36, 1024)

# 각 RF에서 1개씩 샘플 → 36개
sampled_tokens: (B, 36, 1024)

# 1:1 비교
discriminator(compressed_tokens)  # (B*36, 1)
vs
discriminator(sampled_tokens)     # (B*36, 1)
```

### 4. RF AutoEncoder Loss

**위치**: `losses/rf_autoencoder_loss.py`

**핵심 함수**:
- `extract_rf_targets()`: RF 매핑으로 타겟 추출
- Per-RF 재구성 손실 계산
- RF별 통계 제공

**손실 계산**:
```python
for each RF (36개):
    target_rf = original_tokens[RF_indices]  # (B, 16, 1024)
    recon_rf = reconstructed_rfs[:, i, :, :]  # (B, 16, 1024)
    loss += compute_loss(recon_rf, target_rf)
```

### 5. RF 유틸리티

**위치**: `utils/rf_utils.py`

**제공 기능**:
- `get_rf_indices()`: RF 인덱스 계산
- `visualize_rf_reconstruction()`: RF 재구성 시각화
- `compute_rf_statistics()`: RF별 상세 통계
- `create_rf_heatmap()`: RF 품질 히트맵

## 학습 모니터링

### W&B 대시보드

학습 중 추적되는 RF 특화 메트릭:

- **RF Similarity Distribution**: RF별 코사인 유사도 분포
- **RF Quality Heatmap**: 6×6 그리드의 RF 품질 히트맵
- **Per-RF Statistics**:
  - Excellent RFs (>0.9)
  - Good RFs (0.8-0.9)
  - Fair RFs (0.6-0.8)
  - Poor RFs (<0.6)
- **Spatial Patterns**: Row/Column별 평균 유사도

### 시각화 저장

자동으로 저장되는 시각화:

- `visualizations/rf_heatmap_epoch_N.png`: RF 품질 히트맵
- `visualizations/rf_best_epoch_N_idx_X.png`: 최고 품질 RF
- `visualizations/rf_worst_epoch_N_idx_X.png`: 최저 품질 RF
- `visualizations/rf_median_epoch_N_idx_X.png`: 중간 품질 RF

## 주요 하이퍼파라미터

### 압축률 조정

```bash
# 더 높은 압축 (4x4 = 16 tokens, RF size 6x6)
python train_rf.py model.compressor.output_grid_size=4

# 중간 압축 (기본값, 6x6 = 36 tokens, RF size 4x4)
python train_rf.py model.compressor.output_grid_size=6

# 낮은 압축 (8x8 = 64 tokens, RF size 3x3)
python train_rf.py model.compressor.output_grid_size=8
```

### 손실 함수 조정

```bash
# AutoEncoder 손실 강화
python train_rf.py loss.weights.ae=2.0 loss.weights.wgan=1.0

# WGAN 손실 강화
python train_rf.py loss.weights.wgan=2.0 loss.weights.ae=1.0

# Gradient penalty 조정
python train_rf.py loss.rf_wgan.lambda_gp=20.0
```

### Discriminator 업데이트 빈도

```bash
# 더 강한 discriminator
python train_rf.py training.n_critic=10

# 더 약한 discriminator
python train_rf.py training.n_critic=1
```

## 기존 시스템과 비교

### 파일 구조

```
ToConv/
├── train.py                     # 기존: Global 방식
├── train_rf.py                  # 신규: RF 기반 방식
│
├── vision_token_compression/
│   ├── models/
│   │   ├── discriminator.py           # 기존: Global discriminator
│   │   ├── autoencoder.py             # 기존: Global decoder
│   │   ├── rf_discriminator.py        # 신규: RF discriminator
│   │   └── rf_autoencoder.py          # 신규: RF decoder
│   │
│   ├── losses/
│   │   ├── wgan_gp.py                 # 기존: Global WGAN-GP
│   │   ├── autoencoder.py             # 기존: Global AE loss
│   │   ├── rf_wgan_gp.py              # 신규: RF WGAN-GP
│   │   └── rf_autoencoder_loss.py     # 신규: RF AE loss
│   │
│   ├── trainer.py               # 기존: Global trainer
│   ├── rf_trainer.py            # 신규: RF trainer
│   │
│   ├── configs/
│   │   ├── config.yaml          # 기존 설정
│   │   └── rf_config.yaml       # RF 설정
│   │
│   └── utils/
│       └── rf_utils.py          # RF 유틸리티
│
└── scripts/
    ├── quick_test.py            # 기존 시스템 테스트
    └── test_rf_system.py        # RF 시스템 테스트
```

### 병행 사용

두 시스템을 모두 유지하여 비교 가능:

```bash
# 기존 Global 방식
python train.py

# 신규 RF 방식
python train_rf.py
```

## 예상 개선 효과

### 1. 공정한 비교
- **기존**: 576개 vs 36개 → 개수만 세도 구분 가능
- **RF**: 36개 vs 36개 → 동일 개수, 공정한 비교

### 2. Spatial Awareness
- **기존**: 전역 비교, 공간 정보 손실
- **RF**: RF 단위 비교, 공간 구조 보존

### 3. 해석 가능성
- **기존**: 전체 손실만 확인 가능
- **RF**: RF별 품질 분석, 히트맵 시각화

### 4. 학습 안정성
- **기존**: Shape mismatch, broadcasting 이슈
- **RF**: 명확한 1:1 대응, 안정적인 gradient

### 5. 메모리 효율
- **기존**: 대형 global discriminator
- **RF**: 경량 token-level discriminator

## 문제 해결

### CUDA Out of Memory

```bash
# 배치 크기 감소
python train_rf.py training.batch_size=16

# 또는 더 높은 압축률
python train_rf.py model.compressor.output_grid_size=4
```

### RF 품질이 낮을 때

```bash
# AutoEncoder 가중치 증가
python train_rf.py loss.weights.ae=3.0

# 더 많은 디코더 레이어
python train_rf.py model.rf_autoencoder.num_layers=5
```

### Discriminator 과적합

```bash
# Critic 업데이트 줄이기
python train_rf.py training.n_critic=3

# Dropout 증가
python train_rf.py model.rf_discriminator.dropout=0.2
```

## 성능 벤치마크

잘 학습된 RF 기반 모델의 예상 메트릭:

- **Overall Cosine Similarity**: > 0.85
- **Excellent RFs (>0.9)**: 20-25 / 36
- **Good RFs (0.8-0.9)**: 10-15 / 36
- **Fair RFs (0.6-0.8)**: 0-5 / 36
- **Poor RFs (<0.6)**: 0-1 / 36
- **Wasserstein Distance**: < 0.1

## 참고 자료

- [CLIP Paper](https://arxiv.org/abs/2103.00020)
- [WGAN-GP Paper](https://arxiv.org/abs/1704.00028)
- [Vision Transformers](https://arxiv.org/abs/2010.11929)
- [Original README](README.md) - 기존 Global 방식 문서

## 라이선스

MIT License

---

**마지막 업데이트**: 2025년 10월 28일
**작성자**: Claude AI Assistant
