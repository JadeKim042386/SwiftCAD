# Drawing2CAD 모델 개선 실험 계획 보고서

**작성일**: 2026-04-13
**대상 논문**: Drawing2CAD (ACM Multimedia 2025)
**목적**: 추론 성능 및 생성 품질 동시 개선

---

## 1. 연구 배경 및 동기

### 1.1 기존 모델 개요

Drawing2CAD는 손으로 그린 공학 도면(SVG)을 3D CAD 명령 시퀀스로 변환하는 Seq2Seq Transformer 모델이다.

```
SVG Input (3 views × 100 tokens) → Encoder → Bottleneck → Decoder → CAD Output (60 commands)
```

| 구성요소 | 사양 |
|---------|------|
| Encoder | 4 Transformer layers, d_model=144, 8 heads |
| Bottleneck | Mean pooling → z(144-dim) → Residual MLP |
| Decoder | 4 Transformer layers, Non-autoregressive (ConstEmbedding) |
| Output | CommandFCN (6 classes) + ArgsFCN (16×257 args) |
| 파라미터 | 3,550,406 (13.54 MB) |

### 1.2 선행 최적화 이력

| 최적화 | 변경 내용 | 결과 |
|--------|---------|------|
| Embedding MLP 제거 | SVGEmbedding → SVGEmbeddingNoMlp | 파라미터 감소, 성능 변화 미미 |
| Decoder 통합 | CommandDecoder + ArgsDecoder → 단일 Decoder | 파라미터 ~30% 감소 |
| d_model 축소 | 256 → 144 | 모델 크기 ~40% 감소 |

**문제**: 모델 크기는 줄었으나 **추론 시간**과 **생성 품질** 모두 거의 개선되지 않음.

### 1.3 근본 원인 분석

**원인 1: Overhead-bound 추론**
- 모델이 compute-bound가 아니라 CUDA 커널 런칭 오버헤드에 의해 지배됨
- ~75+ 커널 × 5-15μs = ~750μs 오버헤드 vs 실제 연산 ~100μs
- d_model 축소는 행렬 크기만 줄일 뿐 커널 수는 동일 → 추론 시간 불변

**원인 2: 정보 병목 (Information Bottleneck)**
- Encoder 출력 (300 tokens × 144 dims = 43,200 values)이 mean pooling으로 단일 벡터 z (144 dims)로 압축 (300:1 압축비)
- z를 `linear_global(z)`로 projection 후 모든 60개 decoder 위치에 **동일하게 broadcast** (cross-attention이 아님)
- 각 decoder 위치가 입력 SVG의 특정 부분에 선택적으로 접근 불가
- 출력 최소 정보량 (~7,835 bits) > z 벡터 정보 용량 (~4,608 bits)

---

## 2. 제안 아키텍처

### 2.1 아키텍처 개요

```
[기존]
SVG(300 tok) → Encoder(4 layers) → mean pool → z(1×144) → broadcast → Decoder(4 layers) → CAD(60 tok)

[제안]
SVG(300 tok) → AlternatingEncoder(4F+4G layers) → memory(300×144) → cross-attn → Decoder(4 layers) → CAD(60 tok)
```

3가지 핵심 변경을 통합하여 단일 모델로 구현 완료:

| 변경 | 내용 | 근거 |
|------|------|------|
| Alternating Attention Encoder | Frame-wise + Global attention 교대 | VGGT (CVPR 2025 Best Paper) |
| Cross-Attention Decoder | Bottleneck 제거, full memory cross-attention | 정보 병목 해소 |
| nn.MultiheadAttention + SDPA | Custom MHA 교체, need_weights=False | torch.compile 호환, 커널 오버헤드 해소 기반 |

### 2.2 Alternating Attention Encoder (VGGT-style)

**핵심 아이디어**: Encoder의 self-attention을 두 가지 패턴으로 교대 적용

```
Block 1: Frame-wise Attention → Global Attention
Block 2: Frame-wise Attention → Global Attention
Block 3: Frame-wise Attention → Global Attention
Block 4: Frame-wise Attention → Global Attention
```

- **Frame-wise attention**: 각 view (Front/Top/Side) 내부에서만 self-attention
  - 개별 뷰의 기하학적 특징(stroke 형상, 곡선 관계) 정제
  - Reshape: `(300, N, D)` → `(100, 3*N, D)` (view와 batch를 merge)
- **Global attention**: 모든 view의 토큰이 함께 self-attention
  - Cross-view 대응관계 학습 (Front의 edge ↔ Top의 edge 매칭)
  - 원본 shape `(300, N, D)` 그대로 사용

**Reshape 정합성 검증 완료** (sanity_check.py):
```python
# Feature: (300, N, D) → (3, 100, N, D) → permute → (100, 3*N, D)
x = x.view(3, 100, N, D).permute(1, 0, 2, 3).reshape(100, 3*N, D)
# Mask: (N, 300) → (N, 3, 100) → permute → (3*N, 100)
mask = mask.view(N, 3, 100).permute(1, 0, 2).reshape(3*N, 100)
```
- Roundtrip 무손실 (atol=1e-7)
- Padding 위치 attention weight = 0.00e+00 확인

**Decomposed Positional Encoding**:
- 기존: Global PE (0-299) — Front 10번째 토큰(PE=10) ≠ Top 10번째 토큰(PE=110)
- 제안: **Intra-view PE (0-99)** + **View Embedding (0-2)**
  - 동일 위치의 뷰 간 토큰이 같은 PE를 공유 → Frame-wise layer에서 뷰 간 공통 패턴 학습 용이
  - View Embedding은 기존 `nn.Embedding(4, 4)` (learned) 활용

### 2.3 Cross-Attention Decoder

**핵심 변경**: Global broadcast → 진정한 Cross-Attention

| 항목 | 기존 (GlobalImproved) | 제안 (Improved) |
|------|----------------------|----------------|
| Encoder→Decoder 연결 | `linear_global(z)` broadcast | `multihead_attn(tgt, memory, memory)` |
| Memory shape | `(1, N, 144)` — 단일 벡터 | `(300, N, 144)` — 전체 시퀀스 |
| 위치별 conditioning | 모든 위치 동일 | 위치별 선택적 attend |
| Mask 지원 | 불필요 (broadcast) | `memory_key_padding_mask` 전파 |

**구현 변경 사항**:
1. `Encoder.forward()`: `return z` → `return memory, key_padding_mask`
2. `Decoder.__init__()`: `TransformerDecoderLayerGlobalImproved` → `TransformerDecoderLayerImproved`
3. `Decoder.forward(z)` → `Decoder.forward(memory, memory_key_padding_mask)`
4. `SVG2CADTransformer`: Bottleneck 제거, mask 전파 파이프라인 구축

**Cross-attention mask 검증 완료**:
- Encoder padding 위치로의 attention weight = 0.00e+00
- 서로 다른 mask를 가진 배치 간 output 차이 = 1.17 (충분히 구분됨)

### 2.4 nn.MultiheadAttention 전환

기존 custom `MultiheadAttention` (attention.py + functional.py ~400줄)을 PyTorch native `nn.MultiheadAttention`으로 교체.

| 항목 | 기존 | 제안 |
|------|------|------|
| Attention 구현 | `torch.bmm` + manual softmax | SDPA (Flash/Memory-efficient auto-select) |
| Self/Cross 판별 | `torch.equal(query, key)` — 전체 텐서 비교 | 불필요 (SDPA가 내부 처리) |
| torch.compile | Graph break 발생 | 호환 가능 |
| Attention weights | 항상 계산 | `need_weights=False` — SDPA 최적 경로 |

---

## 3. 파라미터 분석

### 3.1 Component별 파라미터 비교

| Component | 기존 (Baseline) | 제안 (Ours) | 변화 |
|-----------|---------------:|------------:|-----:|
| **Encoder** | | | |
| - Embedding (SVGEmbeddingNoMlp) | 98,900 | 98,900 | ±0 |
| - Encoder Layers | 464,416 (4 layers) | 1,857,664 (4F+4G) | +300% |
| - LayerNorm | 288 | 288 | ±0 |
| **Bottleneck** | 20,952 | 0 (제거) | -100% |
| **Decoder** | | | |
| - ConstEmbedding | 8,640 | 8,640 | ±0 |
| - Decoder Layers | 1,021,280 | 1,264,352 | +24% |
| - CommandFCN | 13,290 | 13,290 | ±0 |
| - ArgsFCN | 1,438,064 | 1,438,064 | ±0 |
| **Total** | **3,550,406** | **4,681,198** | **+31.8%** |
| **Size (float32)** | **13.54 MB** | **17.86 MB** | +31.9% |

### 3.2 파라미터 증가 원인

- **Encoder 레이어 2배**: Frame-wise 4개 + Global 4개 = 8개 (기존 4개)
  - 각 레이어: self-attention(144²×3 = 62,208) + FFN(144×512×2 = 147,456) + norms ≈ 232,104
  - 추가 4개 레이어 × 232,104 ≈ +928,416
- **Decoder cross-attention 추가**: 기존 `linear_global(144→144)` 대신 `MultiheadAttention(144, 8)` 4개
  - 각 cross-attn: Q/K/V projections(144²×3) + output(144²) + norm ≈ +83,232 per layer
- **Bottleneck 제거**: -20,952

---

## 4. 실험 비교 매트릭스

### 4.1 Variant 정의

총 5개 variant를 200 epochs, scratch부터 학습하여 비교:

| Variant | Encoder | Decoder | Bottleneck | 예상 Params | 예상 추론 시간 |
|---------|---------|---------|------------|------------|-------------|
| **(a) Baseline** | Standard (4 layers) | Global broadcast | Mean pool | 3.55M (기준) | 기준 |
| **(b) Cross-attn only** | Standard (4 layers) | Cross-attention | 제거 | ~3.78M (+6%) | ↑ 소폭 |
| **(c) Cross-attn + bottleneck** | Standard (4 layers) | Cross-attention | Element-wise 유지 | ~3.80M (+7%) | ↑ 소폭 |
| **(d) Alt-attn only (대조군)** | Alternating (4F+4G) | Global broadcast | Mean pool | ~4.46M (+26%) | → 동일 |
| **(e) Alt-attn + Cross-attn** | Alternating (4F+4G) | Cross-attention | 제거 | ~4.68M (+32%) | ↑ 소폭 |

### 4.2 Variant별 가설

- **(a) Baseline**: Phase 1.1 (nn.MHA 전환) 적용 후 기준선. 기존 custom MHA 모델 대비 동일 성능 예상.
- **(b) Cross-attn only**: 정보 병목 해소가 성능의 주 요인인지 검증. **가장 강력한 성능 향상 예상.**
- **(c) Cross-attn + bottleneck**: Element-wise bottleneck이 추가 정규화로서 일반화에 기여하는지 검증. Bottleneck이 per-token FFN과 중복될 가능성.
- **(d) Alt-attn only (Negative Control)**: **"인코더만 좋게 만들어도 decoder 병목이 남아있으면 성능이 안 오른다"를 증명하는 대조군.** 성능 향상 미미할 것으로 예상.
- **(e) Alt-attn + Cross-attn**: 인코더 + 디코더 동시 개선. 최고 성능 기대, 추론 시간 trade-off 분석 대상.

### 4.3 Baseline 정의

> Baseline = Phase 1.1에서 `nn.MultiheadAttention`으로 전환 + d_model 조정(필요 시) 완료 후, scratch부터 200 epochs 학습한 모델.
> 기존 custom MHA 모델이 아님. 모든 variant가 동일 코드 기반에서 출발하여 공정 비교.

---

## 5. 학습 전략

### 5.1 기본 학습 설정

| 항목 | 값 |
|------|---|
| Optimizer | Adam, lr=1e-3 |
| LR Schedule | Linear warmup (2000 steps) → One-cycle cosine decay |
| Epochs | 200 |
| Batch size | 256 |
| Gradient clipping | norm=1.0 |
| Loss | Command CE (weight=1.0) + Gumbel loss (weight=2.0, tolerance=3) |

### 5.2 Cross-Attention 도입에 따른 추가 전략

**과적합 대응**:
- 기존 mean pooling이 의도치 않은 강력한 정규화 역할을 하고 있었음
- Cross-attention 도입 시 모델 표현력 급증 → 과적합 위험
- **Attention Dropout 상향**: 0.1 → 0.2~0.3 (변인 실험)
- **Stochastic Depth (DropPath)**: `timm` 라이브러리 활용, Encoder/Decoder 블록에 적용

**Weight Initialization**:
- Cross-attention 레이어의 K/V projection: Xavier uniform 명시 초기화
- `nn.MultiheadAttention` 기본 초기화는 수렴이 느릴 수 있음

**Layer-wise Learning Rate**:
- 새로 추가된 cross-attention params: `base_lr × 1.5~2.0`
- 기존 encoder params: `base_lr`

**Phased Training**:
- Phase A: Encoder freeze, Decoder(cross-attention)만 학습
  - 전환 시점: decoder validation loss plateau 또는 Command Acc 80% 달성 시
  - 완료 시점에 체크포인트 별도 저장 (반복 실험 시간 절약)
- Phase B: 전체 해동, end-to-end fine-tuning

### 5.3 Data Augmentation 강화

Cross-attention 도입으로 decoder가 개별 SVG 토큰에 직접 attend → 스케치 순서/방향 민감도 증가 예상.

| Augmentation | 설명 |
|-------------|------|
| Stroke permutation | 동일 뷰 내 획 순서 랜덤 셔플 |
| Stroke reversal | 획 방향 (시작↔끝) 랜덤 반전 |
| Coordinate jittering | 좌표에 가우시안 노이즈 (σ=1~2, args_dim=256 기준) |
| Global scaling/rotation | 전체 스케치 소폭 스케일 (±5%) / 회전 (±3°) |

### 5.4 Bucket Batching

Cross-attention 연산량이 시퀀스 길이에 비례 → 길이 편차가 큰 배치에서 패딩 연산 낭비.
- `BucketBatchSampler`: SVG 유효 길이 기준 정렬 우선 적용
- 메모리 스파이크 시 CAD 길이까지 고려한 2차원 버킷팅으로 확장

---

## 6. 평가 지표

| 지표 | 설명 | 적용 Phase |
|------|------|-----------|
| **Command Accuracy** | 명령 타입 정확도 (Line/Arc/Circle/EOS/SOL/Ext) | 학습 중 |
| **Argument Accuracy** (tol=3) | Line/Arc/Circle/Plane/Transform/Extrusion 별 | 학습 중 |
| **Argument MAE** | 연속형 지표. Command별 MAE 분리 시각화 (WandB). Tolerance 밖 수렴 추적 | 학습 중 |
| **Syntax Validity Rate** | 유효 CAD 시퀀스 비율. Validation: 경량 텐서 checker, Test: full E2E 파서 | 평가 |
| **Chamfer Distance** | 3D 형상 품질 (2000 point sampling). 렌더링 실패 시 Max Penalty (99th pctl) | 최종 평가 |
| **Forward Pass Time** | batch_size=1,16,256 순수 forward 시간 (warmup 100 iters 후 평균) | 전 단계 |
| **Inference Time (E2E)** | 전처리~후처리 포함 end-to-end | torch.compile 후 |
| **Parameter Count** | 모델 크기 | 전 단계 |
| **FLOPs** | component별 연산량 | Profiling |

### 6.1 CD 평가 실패 샘플 정책

- Syntax Invalid → 3D mesh 변환 불가 시 `try-except`으로 루프 중단 방지
- 실패 샘플에 Max Penalty CD 값 부여 (전체 test set CD의 99th percentile)
- 렌더링 성공/실패 비율을 SVR 지표로 별도 기록

---

## 7. 추론 최적화 계획 (Phase 3)

구현은 완료되어 있으며, 학습 완료 후 적용 예정:

### 7.1 torch.compile

```python
model = torch.compile(model, mode="reduce-overhead")
```

- CUDA Graph로 커널 런칭 오버헤드 제거 → 30-60% 추론 가속 예상
- 사전 조건: `nn.MultiheadAttention` 전환 완료 (graph break 제거)
- `TORCH_LOGS=graph_breaks`로 잔여 graph break 점검

**주의사항**:
- CPU-GPU Sync 금지: `.item()`, `.nonzero()`, `.cpu()`, tensor 기반 `if`, `torch.unique()`
- 고정 배치 크기 권장 (CUDA Graph 메모리 풀)
- `torch._dynamo.mark_static()` 적용

### 7.2 AMP (Automatic Mixed Precision)

```python
with torch.autocast(device_type='cuda', dtype=torch.float16):
    outputs = model(...)
```

- 5-15% 추가 속도 향상
- `GradScaler` 통합 (학습 시)

### 7.3 head_dim 호환성

현재 d_model=144, nhead=8 → head_dim=18. FlashAttention 최적 성능은 head_dim이 8의 배수일 때.
- 추론 속도 미달 시: d_model=128 (head_dim=16) 또는 d_model=160 (head_dim=20) 조정 검토
- Phase 1.2 profiling 결과에 따라 결정

### 7.4 시스템 요구사항

- **PyTorch 2.2 이상** (가급적 2.3+ 권장)
- 현재 환경: PyTorch 2.11.0+cu128 (요구사항 충족)

---

## 8. 구현 현황

### 8.1 완료된 작업

| Phase | 내용 | Commit | 상태 |
|-------|------|--------|------|
| 1.1 | Custom MHA → nn.MultiheadAttention + SDPA | `08e2823` | 완료 |
| 1.5 | Reshape/Mask Sanity Check (6개 테스트) | sanity_check.py | 통과 |
| 2.1 | Cross-Attention Decoder (bottleneck 제거, mask 전파) | `67701ea` | 완료 |
| 2.2 | Alternating Attention Encoder (Decomposed PE) | `67701ea` | 완료 |

### 8.2 수정된 파일

| 파일 | 변경 내용 |
|------|----------|
| `model/layers/transformer.py` | `nn.MultiheadAttention` import, `AlternatingTransformerEncoder` 추가, `need_weights=False` |
| `model/layers/improved_transformer.py` | `nn.MultiheadAttention` import, `need_weights=False`, `**kwargs` 호환 |
| `model/model.py` | Encoder 반환값 변경, Decoder cross-attention, Bottleneck 제거, Decomposed PE |
| `sanity_check.py` | 6개 검증 테스트 (reshape, mask, attention weight) |

### 8.3 남은 작업

| Phase | 내용 | 우선순위 |
|-------|------|---------|
| 학습 | 5개 variant × 200 epochs scratch 학습 | **필수** |
| 3.1 | torch.compile + AMP 적용 | 학습 후 |
| 4.1 | Mask-Predict (오프라인 high-fidelity 모드) | 선택적 |
| 4.2 | Output head factorization | 선택적 (fine-tuning 단계) |

---

## 9. 참고 논문

| 논문 | 연도 | 핵심 기여 | 적용 |
|------|------|----------|------|
| VGGT | CVPR 2025 | Alternating attention (frame-wise + global) | Encoder 설계 |
| AVGGT | 2025 | Early global→local conversion (8-10x speedup) | 향후 최적화 |
| CAD-SIGNet | CVPR 2024 | Sketch Instance Guided cross-attention | Decoder 설계 참고 |
| CMLM | EMNLP 2019 | Mask-Predict iterative refinement | Phase 4 (오프라인) |
| Return of the Encoder | 2025 | 2/3 encoder - 1/3 decoder 최적 비율 | 향후 layer 배분 |
| Gated Linear Attention | ICML 2024 | Short-sequence efficient attention | 추가 최적화 후보 |

---

## 10. 예상 결과 요약

| Variant | Command Acc | Arg Acc | 추론 시간 | 비고 |
|---------|-----------|---------|----------|------|
| (a) Baseline | ~82% | ~79% | 기준 | nn.MHA 전환만 |
| (b) Cross-attn | **~88-92%** | **~85-90%** | +10-20% | **핵심 개선** |
| (c) Cross-attn+BN | ~87-91% | ~84-89% | +10-20% | BN 효과 확인 |
| (d) Alt-attn (대조군) | ~83% | ~80% | ±0% | 병목 미해소 증명 |
| (e) Alt+Cross | **~89-93%** | **~86-91%** | +15-25% | 최고 성능 기대 |
| (e) + torch.compile | ~89-93% | ~86-91% | **-20~40%** | 속도 회복 |

> 예상치는 정보 병목 해소 효과(20-35% 향상)에 기반한 추정이며, 실제 결과는 학습 dynamics에 따라 달라질 수 있음.

---

## 11. 실험 결과

### 11.1 Variant (a) Baseline — 완료 (2026-04-14)

**학습 설정**: encoder_type=standard, decoder_type=broadcast, input_option=3x, 200 epochs, batch_size=256, lr=1e-3
**파라미터**: 7,654,330 (29.20 MB, float32)
**wandb run**: [olive-oath-90](https://wandb.ai/jujoo/Drawing2CAD/runs/8sbd2gms)
**학습 시간**: 약 11시간 30분 (02:00 ~ 13:30, A100 80GB)
**체크포인트**: `proj_log/variant_a_baseline/model/` (epoch 100, 200, latest)

#### Training Loss (에폭별 평균)

| Epoch | loss_cmd | loss_args |
|-------|----------|-----------|
| 50    | 0.5673   | 3.8272    |
| 100   | 0.4564   | 3.5545    |
| 150   | 0.3813   | 3.3838    |
| 200   | 0.3396   | 3.3034    |

#### Final Metrics (wandb summary, epoch 200)

| Metric | Value |
|--------|-------|
| **train/loss_cmd** | 0.3544 |
| **train/loss_args** | 3.4597 |
| **validation/loss_cmd** | 1.0828 |
| **validation/loss_args** | 4.4707 |
| **args_acc/plane** | 87.99% |
| **args_acc/circle** | 71.77% |
| **args_acc/line** | 58.64% |
| **args_acc/arc** | 51.97% |
| **args_acc/trans** | 49.99% |
| **args_acc/extent** | 36.59% |
| **learning_rate** | 8.854e-4 |

#### 분석
- Train vs Validation loss 차이가 큼 (cmd: 0.35 vs 1.08, args: 3.46 vs 4.47) → **과적합 경향**
- Plane accuracy가 가장 높고 (88%), Extent가 가장 낮음 (37%)
- 이 baseline 결과를 기준으로 variant (b)~(e)의 cross-attention/alternating attention 효과를 비교 예정
