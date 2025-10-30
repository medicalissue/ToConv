# CLIPVisionTower + ToConv Token Compression Integration

LLaVA의 CLIPVisionTower에 ToConv 토큰 압축 기능이 통합되었습니다. 이를 통해 CLIP Vision Encoder에서 생성된 토큰을 효율적으로 압축하여 메모리 사용량과 연산량을 줄일 수 있습니다.

## 주요 기능

- **압축 on/off 설정**: `use_token_compression` 플래그로 압축 기능을 켜고 끌 수 있습니다
- **유연한 압축 비율**: 입력/출력 그리드 크기를 자유롭게 설정할 수 있습니다
- **지원하는 압축 설정**:
  - 24×24 → 12×12 (75% 토큰 감소)
  - 24×24 → 8×8 (88.9% 토큰 감소)
  - 16×16 → 12×12 (43.75% 토큰 감소)
  - 16×16 → 8×8 (75% 토큰 감소)

## 사용 방법

### 1. 기본 사용 (압축 없음)

```python
from llava.model.multimodal_encoder.clip_encoder import CLIPVisionTower

class Args:
    mm_vision_select_layer = -2
    mm_vision_select_feature = 'patch'
    use_token_compression = False  # 압축 비활성화

args = Args()
vision_tower = CLIPVisionTower('openai/clip-vit-large-patch14-336', args)

# Forward pass
images = torch.randn(batch_size, 3, 336, 336)
features = vision_tower(images)
# Output: (batch_size, 576, 1024) - 24x24 = 576 tokens
```

### 2. 압축 활성화 (24×24 → 12×12)

```python
class Args:
    mm_vision_select_layer = -2
    mm_vision_select_feature = 'patch'
    use_token_compression = True  # 압축 활성화
    compression_input_size = 24
    compression_output_size = 12

args = Args()
vision_tower = CLIPVisionTower('openai/clip-vit-large-patch14-336', args)

# Forward pass
images = torch.randn(batch_size, 3, 336, 336)
features = vision_tower(images)
# Output: (batch_size, 144, 1024) - 12x12 = 144 tokens (75% 감소!)
```

### 3. 고압축 설정 (24×24 → 8×8)

```python
class Args:
    mm_vision_select_layer = -2
    mm_vision_select_feature = 'patch'
    use_token_compression = True
    compression_input_size = 24
    compression_output_size = 8  # 더 높은 압축률

args = Args()
vision_tower = CLIPVisionTower('openai/clip-vit-large-patch14-336', args)

# Forward pass
images = torch.randn(batch_size, 3, 336, 336)
features = vision_tower(images)
# Output: (batch_size, 64, 1024) - 8x8 = 64 tokens (88.9% 감소!)
```

## 압축 효과

| 설정 | 입력 토큰 | 출력 토큰 | 토큰 감소율 | 압축률 |
|------|----------|----------|-----------|--------|
| 24→12 | 576 | 144 | 75% | 4:1 |
| 24→8 | 576 | 64 | 88.9% | 9:1 |
| 16→12 | 256 | 144 | 43.75% | 1.78:1 |
| 16→8 | 256 | 64 | 75% | 4:1 |

## 구현 세부사항

### 통합된 파일
- `LLaVA/llava/model/multimodal_encoder/clip_encoder.py`
  - `CLIPVisionTower` 클래스에 압축 기능 추가
  - `CLIPVisionTowerS2` 클래스에도 동일하게 적용

### 주요 변경사항

1. **초기화 시 설정 추가**:
   ```python
   self.use_token_compression = getattr(args, 'use_token_compression', False)
   self.compression_input_size = getattr(args, 'compression_input_size', 24)
   self.compression_output_size = getattr(args, 'compression_output_size', 12)
   ```

2. **모델 로드 시 TokenCompressor 초기화**:
   ```python
   if self.use_token_compression:
       from vision_token_compression.models.token_compressor import TokenCompressor
       self.token_compressor = TokenCompressor(
           input_grid_size=self.compression_input_size,
           output_grid_size=self.compression_output_size,
           hidden_dim=hidden_dim
       )
   ```

3. **Forward pass에서 압축 적용**:
   ```python
   if self.use_token_compression and self.token_compressor is not None:
       image_features = self.token_compressor(image_features)
   ```

4. **동적 패치 수 계산**:
   ```python
   @property
   def num_patches(self):
       if self.use_token_compression and self.token_compressor is not None:
           return self.compression_output_size ** 2
       return (self.config.image_size // self.config.patch_size) ** 2
   ```

## 테스트

테스트 스크립트를 실행하여 통합을 검증할 수 있습니다:

```bash
python test_clip_compression.py
```

모든 테스트가 성공적으로 통과하면 다음과 같은 출력을 볼 수 있습니다:

```
Testing CLIPVisionTower with ToConv compression integration

============================================================
Test 1: CLIPVisionTower WITHOUT compression
============================================================
Input shape: torch.Size([2, 3, 224, 224])
Output shape: torch.Size([2, 256, 1024])
Expected patches: 256 (16x16)
Hidden size: 1024
✓ Test passed!

============================================================
Test 2: CLIPVisionTower WITH compression (24x24 → 12x12)
============================================================
TokenCompressor [24to12]: 24×24 → 12×12
  Architecture: PartialConv(k=3, s=2, p=1)
  Theoretical RF: 3×3
ToConv Token Compression enabled: 24x24 → 12x12
Input shape: torch.Size([2, 3, 336, 336])
Output shape: torch.Size([2, 144, 1024])
Expected patches: 144 (12x12)
Hidden size: 1024
✓ Test passed!

============================================================
Test 3: CLIPVisionTower WITH compression (24x24 → 8x8)
============================================================
TokenCompressor [24to8]: 24×24 → 8×8
  Architecture: PartialConv(k=3, s=3, p=0)
  Theoretical RF: 3×3
ToConv Token Compression enabled: 24x24 → 8x8
Input shape: torch.Size([2, 3, 336, 336])
Output shape: torch.Size([2, 64, 1024])
Expected patches: 64 (8x8)
Hidden size: 1024
✓ Test passed!

============================================================
All tests passed successfully!
============================================================
```

## 사용 시 고려사항

1. **CLIP 모델 선택**:
   - 224×224 입력: `openai/clip-vit-large-patch14` (16×16 토큰)
   - 336×336 입력: `openai/clip-vit-large-patch14-336` (24×24 토큰)

2. **압축률 선택**:
   - 높은 압축률(24→8)은 메모리와 연산량을 크게 줄이지만 정보 손실이 더 클 수 있음
   - 중간 압축률(24→12)은 성능과 효율성의 균형을 제공

3. **학습 vs 추론**:
   - 추론 시에만 압축을 사용하려면 학습 시 `use_token_compression=False`로 설정
   - 압축을 사용하여 학습하려면 처음부터 `use_token_compression=True`로 설정

## 향후 개선 방향

- [ ] 사전 학습된 압축 모델 체크포인트 제공
- [ ] 다양한 압축 비율에 대한 벤치마크 결과
- [ ] 학습 스크립트에 압축 설정 통합
- [ ] 추론 속도 및 메모리 사용량 프로파일링

## 관련 파일

- `LLaVA/llava/model/multimodal_encoder/clip_encoder.py` - 통합 코드
- `vision_token_compression/models/token_compressor.py` - ToConv 압축 모델
- `test_clip_compression.py` - 테스트 스크립트
