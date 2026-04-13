# Drawing2CAD 모델 개선 연구 계획

## Context

Drawing2CAD는 손으로 그린 SVG 스케치를 CAD 명령 시퀀스로 변환하는 seq2seq Transformer 모델이다. 기존에 Encoding MLP 제거, Decoder 통합(weight sharing), d_model 축소(256→144)를 적용하여 모델 크기를 줄였으나, 추론 시간과 성능이 거의 개선되지 않았다.

**연구를 통해 밝혀진 근본 원인:**

1. **추론 시간이 개선되지 않은 이유**: 모델이 compute-bound가 아니라 **overhead-bound**이다. CUDA 커널 런칭 오버헤드(~75+ 커널 × 5-15μs = ~750μs)가 실제 연산 시간(~100μs)을 압도한다. d_model을 줄여도 커널 수는 동일하므로 추론 시간이 줄지 않는다.

2. **성능이 개선되지 않은 이유**: Decoder의 **정보 병목**이 핵심 제한 요인이다.
   - Encoder 출력(300×144 = 43,200 값)을 mean pooling으로 단일 벡터 z(144차원)로 압축 (300:1 압축비)
   - 이 z를 linear projection 후 모든 60개 decoder 위치에 동일하게 broadcast
   - 각 decoder 위치가 입력 SVG의 어떤 부분과도 선택적으로 상호작용할 수 없음
   - 출력에 필요한 최소 정보량(~7,835 bits) > 144 float32 벡터의 정보 용량(~4,608 bits)

---

## 제안 방법론

### 방법론 1: VGGT-style Alternating Attention (Encoder)

**출처**: VGGT (CVPR 2025 Best Paper), AVGGT (2025)

**핵심 아이디어**: Encoder의 self-attention을 두 가지 패턴으로 교대 적용
- **Frame-wise attention**: 각 view(front/top/side) 내부에서만 self-attention → 개별 뷰의 기하학적 특징 추출
- **Global attention**: 모든 view의 토큰이 함께 self-attention → cross-view 대응관계 학습

**Drawing2CAD 적용**:
- 3-view 입력(front 100 + top 100 + side 100 = 300 토큰)이 VGGT의 frame/global 구조와 자연스럽게 매핑
- Dataset에서 view가 연속 100토큰 블록으로 저장됨 확인 (rows 0-99=view0, 100-199=view1, 200-299=view2)
- 기존 4개 encoder layer를 4개 alternating block으로 교체 (frame→global→frame→global)

**구현 세부사항**:
- `_get_clones`로 동일 레이어를 복제하는 현재 방식 대신, `nn.ModuleList`로 두 종류의 레이어를 교대 배치
- Frame-wise attention reshape (**순서 주의 - naive reshape 금지**):
  ```python
  # Feature tensor: (300, N, D) → (100, 3*N, D)
  x = x.view(3, 100, N, D)           # (3, 100, N, D) - view별 분리
  x = x.permute(1, 0, 2, 3)          # (100, 3, N, D) - sequence dim을 앞으로
  x = x.reshape(100, 3*N, D)         # (100, 3*N, D) - view와 batch를 merge
  
  # Padding mask: (N, 300) → (3*N, 100)
  mask = mask.view(N, 3, 100)        # (N, 3, 100) - view별 분리
  mask = mask.permute(1, 0, 2)       # (3, N, 100) - view dim을 앞으로
  mask = mask.reshape(3*N, 100)      # (3*N, 100) - view와 batch를 merge
  ```
- Global attention: 원본 shape `(300, N, D)` 그대로 사용
- **Positional encoding 전략 (Decomposed PE)**:
  - 기존 global PE (0-299)를 **Intra-view PE (0-99) + View Embedding (0-2)**으로 완전 분리
  - **이유**: Global PE 사용 시 Front의 10번째 토큰(PE=10)과 Top의 10번째 토큰(PE=110)이 서로 다른 PE 대역을 갖게 되어, Frame-wise layer에서 뷰 간 공통 패턴(같은 위치의 유사한 geometric feature) 학습이 어려움
  - Intra-view PE: 각 뷰 내 토큰 위치 (0-99). `PositionalEncodingLUT(d_model, max_len=100)` → Frame-wise attention이 뷰 내부 형상에 집중
  - View Embedding: 뷰 ID (0-2). 기존 `view_embed: nn.Embedding(4, 4)` 활용 (Learned embedding — 3개 고정 뷰의 고유 기하학적 관계를 학습하는 데 Sinusoidal보다 유리) → Global attention이 뷰 간 관계를 엮는 데 활용
  - 합산: `token_embedding + intra_view_PE + view_embedding`
- **4x mode**: `(400, N, D)` → `(4, 100, N, D)` 동일 패턴 적용

**예상 효과**: 인코더 품질 향상 (더 나은 multi-view feature 추출)
**구현 복잡도**: MEDIUM (~120줄 신규 코드)

### 방법론 2: Cross-Attention Decoder (Bottleneck 제거)

**핵심 아이디어**: 현재의 aggressive bottleneck(mean pooling → single z vector → linear broadcast)을 제거하고, Decoder가 Encoder의 full memory sequence에 cross-attention하도록 변경

**현재 구조** (`TransformerDecoderLayerGlobalImproved` 사용):
```
Encoder(300 tokens) → mean pool → z(1×144) → linear_global(z) broadcast → Decoder(60 positions)
```
- Decoder는 cross-attention이 아닌 단순 linear projection + broadcast를 사용
- 모든 60개 decoder 위치가 동일한 conditioning을 받음

**제안 구조** (`TransformerDecoderLayerImproved` 사용):
```
Encoder(300 tokens) → memory(300×144) + key_padding_mask → cross-attention → Decoder(60 positions)
```

**구현 세부사항**:

1. **Encoder 반환값 변경** (`model.py:114-123`):
   - 현재: `return z` (mean-pooled, shape `(1, N, 144)`)
   - 변경: `return memory, key_padding_mask` (shape `(S, N, 144)`, `(N, S)`)
   - mean pooling 제거, full encoder memory 반환

2. **Decoder layer 교체** (`model.py:216, improved_transformer.py`):
   - `TransformerDecoderLayerGlobalImproved(cfg.d_model, cfg.dim_z, ...)` → `TransformerDecoderLayerImproved(cfg.d_model, ...)`
   - `linear_global` broadcast → `multihead_attn` cross-attention
   - `cfg.dim_z` 파라미터 불필요해짐

3. **Decoder.forward() 시그니처 변경** (`model.py:222`):
   - 현재: `def forward(self, z)` 
   - 변경: `def forward(self, memory, memory_key_padding_mask=None)`
   - 내부 호출: `self.decoder(src, memory, memory_key_padding_mask=memory_key_padding_mask)`

4. **SVG2CADTransformer.forward() 업데이트** (`model.py:257-263`):
   - `memory, enc_padding_mask = self.encoder(...)` 
   - `command_logits, args_logits = self.decoder(memory, enc_padding_mask)`

5. **memory_key_padding_mask 전파** (Critical):
   - Encoder의 `key_padding_mask` (`_get_key_padding_mask_svg`로 생성)를 Decoder까지 전파
   - **주의**: mask convention 확인 필요 - 현재 코드는 `True=무시`이고 `TransformerDecoderLayerImproved`의 `multihead_attn`도 같은 convention 사용

6. **Bottleneck 처리**:
   - 기본: 제거 (가장 단순, 권장)
   - 대안 variant (c): Bottleneck을 full memory에 element-wise 적용
     - 단, 원래 bottleneck은 300개 토큰을 1개로 압축하는 "정보 병목"이었으나, element-wise 적용 시 per-token nonlinear transform이 되어 의미가 달라짐
     - 이는 사실상 추가 FFN layer와 유사 → Transformer layer의 FFN과 중복될 수 있음
     - Variant (c)는 "bottleneck 유무" 비교가 아닌 "추가 per-token FFN의 효과" 확인용으로 해석
   - 추가 고려: Bottleneck을 완전 제거 대신 **소규모 Bottleneck 유지** (예: 72-dim 또는 36-dim projection) → 일반화 성능에 긍정적일 수 있음. Parameter-efficient 관점에서 관찰 필요

7. **ConstEmbedding**: 변경 불필요. `z` 대신 `memory`가 전달되어도 `memory.new_zeros()`로 동일하게 동작 (device/dtype 추론용). 단, call site에서 `self.embedding(z)` → `self.embedding(memory)`로 인자 변경 필요

**⚠️ 과적합(Overfitting) 리스크**:
- 기존 mean pooling은 의도치 않게 강력한 정규화 역할을 수행. Cross-attention으로 전환 시 모델 표현력 급증 → 과적합 위험
- **대응**: 
  - Attention Dropout 상향 (0.1 → 0.2~0.3)
  - **Stochastic Depth (DropPath)**: `timm` 라이브러리의 DropPath를 Encoder/Decoder 블록에 적용 — 단순 Dropout보다 정규화 효과 우수
  - Variant (b), (c), (e)에서 dropout 0.1 vs 0.2 비교 포함
- **Cross-attention weight initialization**: `nn.MultiheadAttention` 기본 초기화는 수렴이 느릴 수 있음. Cross-attention 레이어의 K/V projection을 Xavier uniform으로 명시 초기화 + Warmup 2000 steps 이상 충분히 확보
- **Layer-wise Learning Rate Decay**: 새로 추가된 cross-attention 레이어는 기존 encoder 레이어보다 높은 LR 필요. 구현: `param_groups`에서 cross-attention params와 나머지를 분리, cross-attn LR = base_lr × 1.5~2.0

**⚠️ 입력 분산(Variance) 노출 리스크**:
- 기존 mean pooling은 SVG 획 순서(stroke order)와 미세 노이즈를 smooth out하는 효과가 있었음
- Cross-attention 도입 시 디코더가 개별 SVG 토큰에 직접 attend → 스케치 순서/방향의 무작위성에 민감해질 수 있음
- **대응**: Sequence-level Data Augmentation 강화
  - **Stroke permutation**: 학습 시 DataLoader에서 동일 뷰 내 획 순서를 무작위로 셔플
  - **Stroke reversal**: 획의 방향(시작↔끝)을 랜덤하게 뒤집기
  - **Coordinate jittering**: 좌표에 미세 가우시안 노이즈 추가 (σ=1~2, args_dim=256 기준)
  - **Global scaling/rotation**: 전체 스케치에 소폭 스케일(±5%)/회전(±3°) 적용
  - 모델이 순서/특정 좌표 값이 아닌 '기하학적 형태' 자체에 집중하도록 유도

**⚠️ Latency Trade-off 주의**:
- 기존 broadcast: 1개 linear projection (144→144), ~83K FLOPs
- Cross-attention: QKV projections + attention over 300 tokens, ~60M FLOPs (약 700x 증가)
- 절대량은 여전히 작지만 (GPU에서 trivial), kernel launch overhead 추가 (~28 kernels)
- Phase 3의 torch.compile이 이를 상쇄할 수 있으나, 실험 2.1에서 **순수 Forward Pass 시간 비교를 필수 지표로 포함**하여 trade-off 정량화 필요

**예상 효과**: 성능 +20-35% 향상 (각 decoder 위치가 관련 SVG stroke에 선택적 attend)
**구현 복잡도**: LOW-MEDIUM (기존 `TransformerDecoderLayerImproved` 재활용)

### 방법론 3: Custom Attention → nn.MultiheadAttention + torch.compile (추론 최적화)

**핵심 아이디어**: 추론 시간의 근본 원인인 kernel launch overhead를 해결

**Step 1**: Custom `MultiheadAttention` + `multi_head_attention_forward` → PyTorch native `nn.MultiheadAttention` 교체
- 현재 `functional.py`가 explicit `torch.bmm` (line 228, 246) + `torch.equal()` 분기 (line 90, 94) 사용
- `torch.equal(query, key)`: 매번 전체 텐서 비교, torch.compile graph break 유발
- `nn.MultiheadAttention`(PyTorch 2.0+)은 내부적으로 SDPA 사용, Flash/memory-efficient backend 자동 선택
- **FlashAttention-2 백엔드 확인**: PyTorch 2.2+에서 `torch.backends.cuda.sdp_kernel(enable_flash=True)`로 FA-2 사용 강제 가능. 300 시퀀스 길이에서도 소폭 추가 이득 기대
- **⚠️ head_dim 호환성**: 현재 d_model=144, nhead=8 → head_dim=18. 일부 HW 가속 커널(FlashAttention 등)은 head_dim이 8의 배수일 때 최적 성능. 추론 속도 미달 시 조정 검토:
  - **d_model=128 (head_dim=16)**: 파라미터 ~20% 감소 → 표현력 부족 여부 Variant (b) 학습 시 면밀히 관찰
  - **d_model=160 (head_dim=20)**: 성능 향상 폭 커질 수 있으나 추론 오버헤드 증가
  - d_model 조정 결정은 Phase 1.2 profiling 결과에 따라, Phase 2 시작 전에 확정
- `functional.py` ~250줄 + `attention.py` custom class 제거

**주의사항**:
- **Mask convention**: 현재 `key_padding_mask`의 boolean convention (`True=무시`) 확인. `nn.MultiheadAttention`도 동일 convention 사용 (PyTorch 2.0+에서)
- **need_weights**: `visualize_attention.py`가 attention weight를 사용함. Visualization 모드에서만 `need_weights=True` 전달, SDPA fallback으로 동작하도록 처리
- **Projection logic**: `nn.MultiheadAttention`이 자체 Q/K/V projection을 관리하므로, custom projection weights 마이그레이션 필요 (또는 scratch부터 학습)

**Step 2**: `torch.compile(mode="reduce-overhead")` 적용
- CUDA Graph를 사용하여 kernel launch overhead 제거
- 30-60% 추론 속도 향상 예상
- 사전 조건: Step 1 완료 (torch.equal graph break 제거)
- `TORCH_LOGS=graph_breaks`로 남은 graph break 점검 (특히 `SVGEmbeddingNoMlp`의 `if S == 100` 분기 - input_option 고정이므로 static branch)

**⚠️ CUDA Graph CPU-GPU Sync 주의**:
- `mode="reduce-overhead"`는 CUDA Graph를 사용하므로, forward pass 내부에 CPU-GPU 동기화를 유발하는 코드가 **단 하나라도** 있으면 graph가 깨지고 성능 저하
- **금지 패턴**: `.item()`, `.nonzero()`, `.cpu()`, 텐서 값에 의존하는 `if` 제어문, `print(tensor)`
- **대응**: 마스킹/패딩 처리의 조건문을 `torch.where()` 등 벡터화 연산으로 대체
- Phase 3 진행 시 `TORCH_LOGS=graph_breaks` 외에도 코드 내 data-dependent control flow 전수 검토 필요

**⚠️ Dynamic Shapes 대응**:
- SVG 시퀀스: input_option에 따라 100/300/400 중 하나로 고정. 학습/추론 시 max length로 패딩되므로 동적 크기 문제 없음
- CAD 시퀀스: 항상 60으로 패딩. 고정 크기
- **권장**: 추론 시 `torch._dynamo.mark_static()`으로 고정 차원을 명시적으로 선언하여 불필요한 재컴파일 방지
- 만약 가변 배치 크기가 필요한 경우: `torch._dynamo.mark_dynamic(tensor, dim=1)`로 batch 차원만 동적으로 선언
- **CUDA Graph 메모리 풀 주의**: `mode="reduce-overhead"`는 고정 메모리 풀 사용. 가변 배치 크기 시 여러 Graph 생성 → 메모리 점유율 증가. **추론용 배치 크기 고정 권장** (예: batch=1 전용 또는 batch=256 전용)
- **추가 동기화 패턴 점검**: `.item()`, `.nonzero()` 외에도 `torch.unique()`, 텐서 값 기반 `assert`문 등을 `TORCH_LOGS=graph_breaks`로 철저히 확인

**Step 3**: FP16 inference (`torch.autocast`)
- `torch.autocast` / `GradScaler` 사용 (전체 모델 FP16 캐스팅 대신)
- Training loop의 `update_network`에 AMP 통합

**예상 효과**: 추론 시간 30-60% 감소
**구현 복잡도**: MEDIUM

**⚠️ 시스템 요구사항**: **PyTorch 2.2 이상 (가급적 2.3+ 권장)**
- PyTorch 2.0~2.1에서는 SDPA + torch.compile CUDA Graph 조합 시 메모리 누수 및 커널 에러가 보고된 알려진 버그 존재
- 2.2+에서 대부분 수정됨. 프레임워크 단 버그 디버깅에 불필요한 시간 소비 방지

### 방법론 4: Mask-Predict Iterative Refinement (선택적 — High-fidelity 오프라인 모드)

**출처**: CMLM (Ghazvininejad et al., EMNLP 2019)

**⚠️ 포지셔닝**: 본 방법론은 추론 시간 K배 증가를 수반하므로, 본 연구의 1차 목표(속도 개선)와 상충됨. **실시간성이 요구되지 않는 High-fidelity 오프라인 렌더링 시나리오를 위한 별도의 파생 모델**로 포지셔닝. 메인 모델의 single-pass 추론과 병행하여, 품질 최우선 시나리오에서 선택적으로 활용.

**핵심 아이디어**: NAT 디코딩 후 confidence가 낮은 토큰을 mask하고 재예측하는 반복 정제

**구현 세부사항**:
- `ConstEmbedding`을 `PartialPredictionEmbedding`으로 교체 필요
  - 1차 예측: 기존 ConstEmbedding 사용 (zeros + positional encoding)
  - 2-3차 재예측: 이전 예측의 command/args embedding + learned mask token 사용
  - Unmasked 위치: 이전 예측 결과 embedding, Masked 위치: learnable `[MASK]` token
- Cross-attention과 호환 가능 (memory는 변하지 않음)
- Masking 전략: softmax confidence 기반 하위 K% 토큰 mask

**예상 효과**: 품질 추가 향상 (특히 argument 정확도)
**구현 복잡도**: MEDIUM-HIGH
**추론 시간**: 반복 횟수에 비례하여 증가 (2-3x) — 오프라인 전용

### 방법론 5: Output Head 최적화 (선택적)

ArgsFCN이 전체 파라미터의 40.5%(1.44M)를 차지하면서 16×257 = 4,112 logit을 flat하게 출력

**제안**:
- **Factored heads**: sketch-args head (args 0-4, 5개)와 extrusion-args head (args 5-15, 11개)로 분리
  - CAD_CMD_ARGS_MASK에 따라 Line/Arc/Circle은 sketch args만, Ext는 extrusion args만 사용
  - **⚠️ Loss masking 필수**: GT 명령어 타입에 따라 비활성 헤드의 loss를 0으로 마스킹해야 함
    - 예: 현재 명령이 Line(sketch)이면 extrusion-args head의 loss는 0
    - 구현: `loss_sketch = loss * sketch_cmd_mask`, `loss_ext = loss * ext_cmd_mask`
    - 마스킹 없이 학습 시 비활성 헤드가 노이즈로 간섭하여 수렴 불안정
  - **Dynamic Loss Weighting**: 특정 명령어(예: Line)가 데이터셋에 압도적으로 많으면 한쪽 헤드만 과적합 위험. Uncertainty-based weighting(Kendall et al.) 또는 inverse frequency weighting으로 학습 균형 유지
- **Low-rank factorization**: `Linear(288, 4112)` → `Linear(288, 64) + Linear(64, 4112)`

**예상 효과**: 파라미터 ~75% 감소, 품질 중립~소폭 향상
**구현 복잡도**: LOW-MEDIUM (loss masking 로직 추가 필요)

---

## 비추천 방법론 (조사 후 배제)

| 방법론 | 배제 이유 |
|--------|---------|
| Mamba/SSM | 시퀀스가 너무 짧음(60/300), NAT decoder와 구조적 비호환 (Mamba는 본질적으로 causal/autoregressive) |
| Linear Attention | N=300, d=144에서 linear attention이 오히려 더 비쌈 (N²=90K < Nd²=6.2M) |
| GQA/MQA | KV cache 없는 NAT decoder에서 이점 없음, 모델이 너무 작음 |
| Flash Attention 단독 | 시퀀스 길이 300/60에서 kernel launch overhead가 지배적, head_dim=18은 최적화 대상 외 |

---

## 실험 계획

### Phase 1: 기반 구축 (Prerequisites)

**실험 1.1**: Custom Attention → `nn.MultiheadAttention` 전환
- `model/layers/functional.py` 제거 또는 최소화
- `model/layers/attention.py`의 custom `MultiheadAttention` → `nn.MultiheadAttention`
- 모든 encoder/decoder layer에서 PyTorch native MHA 사용
- 검증: 새로운 모델로 short training (10 epochs) 후 loss 수렴 확인
- Note: 기존 checkpoint와 호환 불가 → scratch부터 학습

**실험 1.2**: Baseline 추론 시간 측정
- batch_size=1, 16, 64, 256에서 추론 시간 프로파일링
- `torch.profiler`로 component별 (encoder / bottleneck / decoder / heads) 시간 분석
- FLOPS/latency 비율로 overhead vs compute 비율 정량화

### Phase 1.5: Reshape / Mask Sanity Check (신규)

**필수 검증 단계** — Phase 2 본 학습 전에 수행:

1. **Alternating Attention Reshape 검증**:
   - 임의 위치에 padding을 넣은 더미 데이터(3 views × 100 tokens) 생성
   - `(300, N, D)` → `(100, 3*N, D)` reshape 후 frame-wise attention 수행
   - Attention weight 가시화: padding 위치의 attention weight가 0인지 확인
   - Reshape 역연산 `(100, 3*N, D)` → `(300, N, D)`이 원본과 일치하는지 수치 검증 (atol=1e-7)

2. **Cross-Attention Mask 검증**:
   - Encoder memory에 known padding을 배치
   - Decoder cross-attention 수행 후, padding 위치로의 attention weight가 0인지 확인
   - `memory_key_padding_mask` 전파 파이프라인의 end-to-end 무결성 검증

### Phase 2: 핵심 아키텍처 개선

**실험 2.1**: Cross-Attention Decoder (방법론 2)
- Encoder.forward() 반환값 변경: `z` → `(memory, key_padding_mask)`
- Decoder layer 교체: `TransformerDecoderLayerGlobalImproved` → `TransformerDecoderLayerImproved`
- `memory_key_padding_mask` 전파 파이프라인 구현
- Bottleneck: (a) 제거, (b) element-wise 적용 두 가지 비교
- 학습 조건: 200 epochs, Adam lr=1e-3
- Warmup: 기존 `GradualWarmupScheduler`(config에 `warmup_step=2000` 존재하나 비활성) 재활성화 후 one-cycle LR과 조합
  - Linear warmup 2000 steps → one-cycle cosine decay
  - Cross-attention의 random 초기화된 K/V projection이 안정적으로 수렴하기 위해 필요
  - **Phased Training 고려**: 초기에 encoder를 freeze하고 decoder(cross-attention) 레이어만 학습 → 이후 전체 해동. Cross-attention 파라미터가 먼저 안정화된 후 encoder와 함께 fine-tune하여 수렴 안정성 확보
  - **전환 시점**: 고정 1000 steps 대신 동적 전략 권장 — decoder validation loss plateau 감지 또는 Command Acc 80% 달성 시 해동
  - **Checkpointing 전략**: Phase 1 (encoder freeze) 완료 시점에 체크포인트 별도 저장 → Phase 2 (전체 학습) 하이퍼파라미터 튜닝 시 Phase 1부터 재개 가능. 반복 실험 시간 대폭 절약
- **Bucket Batching (학습 속도 최적화)**:
  - Cross-attention 도입 시 메모리/연산이 시퀀스 길이에 비례 → 길이 편차가 큰 배치에서 패딩 낭비 심각
  - 유효 토큰 길이가 비슷한 샘플끼리 배치를 구성하는 Bucket Batching 적용
  - `BiSequenceDataset`에 `BucketBatchSampler` 추가
  - **우선**: SVG 유효 길이 기준으로만 정렬 (Encoder 연산 효율화에 충분)
  - **확장**: Cross-attention 연산량은 SVG길이 × CAD길이에 비례. 학습 중 메모리 스파이크 발생 시 CAD 길이까지 고려한 2차원 버킷팅으로 확장

**실험 2.2**: Alternating Attention Encoder (방법론 1)
- `AlternatingEncoder` 구현: `nn.ModuleList`에 frame-wise / global layer 교대 배치
- Frame-wise: `(300, N, D)` → `(100, 3*N, D)` reshape + view mask
- 실험 2.1의 cross-attention decoder와 결합

**실험 비교 매트릭스** (5개 variants):

| Variant | Encoder | Decoder | Bottleneck | 예상 Param 변화 | 예상 Inference Time 방향 |
|---------|---------|---------|------------|----------------|------------------------|
| (a) Baseline* | Standard | Global broadcast | Mean pool | 기준 (3.55M) | 기준 |
| (b) Cross-attn only | Standard | Cross-attention | 제거 | +5~10% (cross-attn KV proj 추가, bottleneck 제거) | ↑ 소폭 증가 (추가 attention) |
| (c) Cross-attn + bottleneck | Standard | Cross-attention | Element-wise 유지 | +5~15% | ↑ 소폭 증가 |
| (d) Alt-attn only (**대조군**) | Alternating | Global broadcast | Mean pool | ±0% (layer 수 동일) | → 거의 동일 |
| (e) Alt-attn + Cross-attn | Alternating | Cross-attention | 제거 | +5~10% | ↑ 소폭 증가 |

**\* Baseline 정의**: Phase 1.1에서 `nn.MultiheadAttention`으로 전환 + d_model 조정(필요 시) 완료 후, scratch부터 200 epochs 학습한 모델. 기존 custom MHA 모델이 아님. 모든 variant가 동일한 코드 기반 및 하드웨어 설정(Phase 1 완료 후)에서 출발하여 공정 비교. **d_model 변경 결정은 반드시 Phase 1.2 profiling 후, Baseline 학습 전에 확정.**

**\*\* Variant (d) 가설**: 인코더를 VGGT-style로 개선하되 디코더 병목(mean pool → broadcast)은 유지. 인코더가 기하학적 특징을 잘 추출해도 최종 단계에서 1개 벡터로 압축되므로 성능 향상이 미미할 것으로 예상. **"인코더 개선만으로는 디코더의 근본적 정보 병목을 극복할 수 없음을 증명하는 Negative Control"**로 설정. Variant (e)와의 비교를 통해 cross-attention의 기여를 분리 확인.

- 각 variant의 실제 parameter count 기록
- 모든 variant를 scratch부터 200 epochs 학습
- **Forward Pass 시간 비교**: compile 적용 전 순수 forward 시간을 variant 간 비교하여 cross-attention latency 증가량 정량화

### Phase 3: 추론 최적화

**실험 3.1**: torch.compile 적용
- Phase 2 최적 모델에 `torch.compile(mode="reduce-overhead")` 적용
- `TORCH_LOGS=graph_breaks`로 graph break 점검
- 고정 차원에 `torch._dynamo.mark_static()` 적용
- FP16: `torch.autocast` 적용
- 추론 시간 비교: compile 전 vs 후, FP16 vs FP32
- **핵심 검증**: Phase 1의 overhead 절감이 Phase 2의 연산량 증가를 상쇄하는지 확인

### Phase 4: 추가 개선 (선택적, Phase 2-3 결과에 따라)

**실험 4.1**: Mask-Predict refinement (High-fidelity 오프라인 모드)
- `PartialPredictionEmbedding` 구현
- 2-3 iterations으로 quality/latency trade-off 분석
- 메인 모델과 별도로 포지셔닝

**실험 4.2**: Output head factorization / low-rank (Fine-tuning 단계)
- ArgsFCN 분리 (sketch/extrusion) 또는 low-rank factorization
- **우선순위 조정**: Phase 2 핵심 아키텍처 개선의 성능 안정화 이후 도입. 경량화 목적이므로 성능 향상보다는 모델 크기 최적화에 초점

---

## 평가 지표

| 지표 | 설명 | 적용 Phase |
|------|------|-----------|
| **Command Accuracy** | 명령 타입 정확도 | Phase 2, 4 |
| **Argument Accuracy** (tolerance=3) | Line/Arc/Circle/Plane/Transform/Extrusion 별 | Phase 2, 4 |
| **Argument MAE (L1 Error)** | tolerance 경계에서의 지표 불연속성 보완. 정답에 "얼마나 가까워지고 있는지" 미세 수렴 추적. Tolerance 밖이어도 오차 감소 확인 가능. **Command별 MAE를 분리 시각화** (WandB/TensorBoard): 특정 명령어(예: Arc 반지름)의 이상 오차 → 해당 명령어 전용 data augmentation/head tuning 근거 | Phase 2, 4 |
| **Syntax Validity Rate** | 유효한 CAD 시퀀스 생성 비율 (SOL/EOS 매칭, 명령 순서, arg 범위). **이원화 전략**: Validation 시 경량 텐서 기반 마스크 매칭 checker 사용, Test 시에만 full E2E CAD 파서 적용 | Phase 2, 4 |
| **Chamfer Distance** | 3D 형상 품질 (2000 point sampling). **붕괴 방지**: try-except으로 렌더링 실패 샘플 처리 (아래 정책 참조) | Phase 2, 4 |
| **Forward Pass Time** | batch_size=1, 16, 256에서 순수 forward 시간 (warmup 100 iters 후 평균) | Phase 1.2, 2, 3 |
| **Inference Time (E2E)** | 전처리~후처리 포함 end-to-end 추론 시간 | Phase 3 |
| **Parameter Count** | 모델 크기 비교 | 전 Phase |
| **FLOPs** | component별 연산량 | Phase 1.2, 2 |

---

## 핵심 파일 및 수정 내용

| 파일 | 수정 내용 |
|------|----------|
| `model/layers/functional.py` | 제거 또는 minimal utility로 축소 |
| `model/layers/attention.py` | Custom MHA 제거, `nn.MultiheadAttention` 사용 |
| `model/layers/improved_transformer.py` | `TransformerDecoderLayerImproved` 활용, 새 `AlternatingEncoderLayer` 추가 |
| `model/layers/transformer.py` | `AlternatingTransformerEncoder` 추가 (nn.ModuleList 기반) |
| `model/model.py` | Encoder 반환값 변경, Decoder 시그니처 변경, Bottleneck 처리, ConstEmbedding call site 변경 |
| `config/config.py` | `dim_z` 제거 또는 deprecated, alternating attention 관련 설정 추가 |
| `trainer/trainer.py` | `torch.autocast`, warmup scheduler 추가 |
| `trainer/base.py` | AMP `GradScaler` 통합 |

---

## 참고 논문

| 논문 | 핵심 기여 | 적용 방향 |
|------|----------|----------|
| VGGT (CVPR 2025) | Alternating attention (frame-wise + global) | Encoder multi-view attention |
| AVGGT (2025) | Early global→local conversion으로 8-10x speedup | 향후 encoder 최적화 |
| CMLM (EMNLP 2019) | Mask-Predict iterative refinement | Phase 4 오프라인 품질 개선 |
| CAD-SIGNet (CVPR 2024) | Sketch Instance Guided cross-attention | Cross-attention decoder 설계 참고 |
| Return of the Encoder (2025) | 2/3 encoder - 1/3 decoder 최적 비율 | 향후 layer 배분 최적화 |
| Gated Linear Attention (ICML 2024) | FlashAttention보다 빠른 short-sequence attention | Phase 3 추가 최적화 후보 |

---

## 검증 방법

0. **CD 평가 실패 샘플 처리 정책**:
   - CAD 시퀀스 렌더링 실패(Syntax Invalid → 3D mesh 변환 불가) 시 try-except으로 루프 중단 방지
   - **정책**: 실패 샘플에 Max Penalty CD 값 부여 (전체 test set CD의 99th percentile). SVR(Syntax Validity Rate)이 간접적으로 CD에 반영됨
   - 별도로 렌더링 성공/실패 비율을 SVR 지표로 기록하여 분리 분석 가능
1. **Sanity Check**: Phase 1.5에서 reshape/mask 정합성 가시화 기반 unit test
2. **Loss 수렴**: `nn.MultiheadAttention` 전환 후 short training loss 수렴 확인
3. **Profiling**: `torch.profiler`로 component별 추론 시간 분석
4. **Training**: 200 epochs 학습 후 validation accuracy + syntax validity 비교 (5개 variants)
5. **Forward Pass**: variant 간 순수 forward 시간 비교 (cross-attention latency trade-off 정량화)
6. **Inference**: torch.compile 전후 E2E 추론 시간 비교
7. **3D evaluation**: Chamfer Distance로 최종 형상 품질 비교
8. **Ablation**: Bottleneck 유무, alternating attention 유무 등 개별 기여 분석
