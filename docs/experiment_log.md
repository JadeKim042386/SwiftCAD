# Drawing2CAD 실험 진행 로그

**프로젝트 기간**: 2026-04-13 ~ 2026-04-18 (약 6일)
**담당자**: AI 엔지니어 (메타클)
**환경**: NVIDIA A100-SXM4-80GB × 1, PyTorch 2.7.0, CUDA 12.8, Python 3.10
**저장소 HEAD (문서 작성 시점)**: `51f8daf`

> 본 문서는 `docs/experiment_artifacts.md`(실측 기반 매니페스트), `docs/report_phase2_ablation.md`(정량/정성 보고서), `improvement_plan.md`(원 계획서), `train_logs/*.log`(학습/테스트 원시 로그), `proj_log/`의 `config.txt`, 그리고 `docs/phase4_*.json` 평가 결과를 교차 참조하여 시간순으로 재구성한 것이다. 확실하지 않은 시각은 `(미확인)` 또는 `(추정: ~)`로 표기.

---

## 목차

- [Phase 1: 기반 구축 (nn.MHA 전환)](#phase-1-기반-구축-nnmha-전환)
- [Phase 1.5: Sanity Check](#phase-15-sanity-check)
- [Phase 2: 5 variants 아키텍처 Ablation](#phase-2-5-variants-아키텍처-ablation)
- [Phase 3: torch.compile + FP16 추론 최적화](#phase-3-torchcompile--fp16-추론-최적화)
- [Phase 4.1: Mask-Predict Iterative Refinement](#phase-41-mask-predict-iterative-refinement)
- [3D OCC 기반 정성/정량 평가 (Chamfer Distance, IR)](#3d-occ-기반-정성정량-평가-chamfer-distance-ir)
- [전체 타임라인 요약](#전체-타임라인-요약)
- [핵심 의사결정 로그](#핵심-의사결정-로그)
- [실패/에러 로그](#실패에러-로그)
- [다음 단계](#다음-단계)

---

## Phase 1: 기반 구축 (nn.MHA 전환)

### 목적

Custom `MultiheadAttention` + `multi_head_attention_forward`를 PyTorch native `nn.MultiheadAttention`으로 교체하여:

1. SDPA(Scaled Dot-Product Attention) 백엔드(FlashAttention/memory-efficient) 자동 선택 활성화
2. `torch.equal(query, key)` 등의 graph break 요인 제거 → Phase 3 `torch.compile` 적용 가능
3. 이후 Cross-Attention decoder / Alternating encoder 구현의 공통 기반

### 작업

- **Commit**: `08e2823` — `refactor: replace custom MHA with nn.MultiheadAttention for SDPA support` (2026-04-13 18:37, +404/-9 lines, 3 files)
- **Commit**: `67701ea` — `feat: alternating attention encoder + cross-attention decoder` (2026-04-13 18:44, +406/-42, 4 files). Phase 2에서 사용할 variant 아키텍처 골격 도입.
- 주요 수정 파일: `model/layers/attention.py`, `model/layers/functional.py`, `model/layers/improved_transformer.py`, `model/layers/transformer.py`, `model/model.py`

### 검증

기존 custom MHA 체크포인트와는 파라미터 구조가 호환되지 않으므로 **scratch 재학습 결과로 검증**. Phase 2 variant (a) Baseline이 논문 원본(Cmd 82.76 / Args 79.23) 대비 Cmd -0.79 / Args -0.58 수준에서 수렴 — 커스텀 MHA 제거로 인한 정확도 영향이 수용 가능한 범위(±1%p 이내)임을 확인.

---

## Phase 1.5: Sanity Check

Phase 2 본 학습 전 Alternating Encoder의 **reshape 연산**과 Cross-Attention의 **padding mask 전파**가 의도대로 동작하는지 검증하는 단계. `improvement_plan.md` §Phase 1.5에 명시된 2가지 체크.

### reshape/mask 관련 체크

1. **Alternating Attention Reshape**
   - `(300, N, D)` → `view(3, 100, N, D)` → `permute(1, 0, 2, 3)` → `reshape(100, 3*N, D)` (view를 batch 차원에 merge)
   - 역연산이 원본과 완전히 일치해야 함 (`atol=1e-7`).
   - naive `view(100, 3*N, D)`로 호출하면 view별 토큰이 섞이므로 **permute 순서**가 핵심.
   - 대응 padding mask `(N, 300)` → `view(N, 3, 100)` → `permute(1, 0, 2)` → `reshape(3*N, 100)`.

2. **Cross-Attention memory_key_padding_mask 전파**
   - Encoder에서 생성된 `key_padding_mask`(True=무시)를 Decoder의 cross-attention까지 end-to-end로 전달.
   - `nn.MultiheadAttention`의 convention과 일치 확인.

### 검증 결과

4월 13~14일 구간에서 dummy tensor 기반 스모크 테스트 수행 후 Phase 2 본 학습 진입. 이 단계의 상세 산출물(테스트 스크립트)은 commit에 포함되지 않았으며, 로그 파일도 남아있지 않음 — 학습이 정상 수렴한 것으로 간접 검증됨.

---

## Phase 2: 5 variants 아키텍처 Ablation

### 2.0 Variant 설계

`improvement_plan.md`의 실험 비교 매트릭스 그대로.

| Variant | Encoder | Decoder | Bottleneck | 설계 의도 |
|---|---|---|---|---|
| (a) Baseline | Standard | Broadcast | Mean pool | 기준선 (nn.MHA 전환 후 재학습) |
| (b) Cross-Attn | Standard | Cross-Attention | 제거 | 디코더 개선 단독 효과 |
| (c) Cross-Attn+BN | Standard | Cross-Attention | Element-wise | Bottleneck 유무 비교 |
| (d) Alt-Attn | Alternating | Broadcast | Mean pool | 인코더 개선 단독 효과 (Negative Control) |
| (e) Alt+Cross | Alternating | Cross-Attention | 제거 | 인코더 + 디코더 동시 개선 |

공통 하이퍼파라미터: `nr_epochs=200`, `batch_size=256`, `lr=1e-3`.

### 2.1 첫 시도 — `--input_option 3x`로 실수 학습

**시점**: 2026-04-14 02:00 ~ 2026-04-15 21:42 (`train_logs/progress.log`)

```
2026-04-14 02:00:00 Starting all 5 variants training
2026-04-14 02:00:00 [START] Variant (a) Baseline
2026-04-14 13:30:37 [DONE] Variant (a) Baseline
...
2026-04-15 21:42:21 ALL 5 VARIANTS COMPLETED
```

순차 학습(`run_all_variants.sh`)으로 5 variant를 **~44시간** 동안 훈련. 총 걸린 시간 breakdown:
- (a) ~11h 30m, (b) ~13h 30m, (c) ~12h 15m, (d) ~5h 57m, (e) ~0h 27m

> (e)의 경과 시간이 비정상적으로 짧은 이유: **`variant_e.log`가 중간에 truncate**된 것으로 추정 (`train_logs/variant_e.log` 2.6 MB, 다른 3x 로그 ~36 MB 대비). 학습 자체는 완주(`progress.log`의 `[DONE] Variant (e) Alt+Cross` 기록), 단 epoch 체크포인트가 저장되지 않고 `latest.pth`만 존재.

#### 사건: 원 논문 설정과 불일치 발견

- Drawing2CAD 원본 논문/공식 `test.sh`는 `--input_option 4x`를 사용함에도, 학습 측 스크립트는 관행적으로 `3x`(3-view)로 넣고 있었음.
- 3x 학습 결과 (미확인 정량치, test_results 7881 샘플로부터 계산됨):
  - Cmd Acc ~81–82%, Args ~78–79% — **논문 원본(82.76 / 79.23) 대비 전반적으로 하락**
- **발견 계기**: 사용자가 "3x 로 돌리고 있다는게 뭔 말이야"라고 지적 → `test.sh`에 이미 `--input_option 4x`가 있음을 교차검증. **학습-테스트 입력 형식 불일치**가 원인이었음이 확정.
- 3x 산출물은 폐기하지 않고 `proj_log/variant_{a..e}_{baseline,cross_attn,...}`에 보존 (나중에 4x와 비교용).

### 2.2 4x 재학습 (병렬)

**시점**: 2026-04-15 21:49 ~ 2026-04-17 12:02 (`train_logs/progress_4x.log`)

```
2026-04-15 21:49:15 Starting 5 variants parallel (4x, num_workers=2)
2026-04-15 21:49:15 PIDs: A=55879 B=55881 C=55883 D=55885 E=55887
2026-04-17 04:04:15 [DONE] Variant (A)
2026-04-17 07:58:31 [DONE] Variant (B)
2026-04-17 07:58:44 [DONE] Variant (C)
2026-04-17 11:08:31 [DONE] Variant (D)
2026-04-17 12:02:04 [DONE] Variant (E)
2026-04-17 12:02:04 ALL VARIANTS FINISHED (4x)
```

총 소요 시간: **약 38시간 13분** (A100 1장 + 5 variant 동시 실행). Variant별 학습 시간(병렬이므로 wall-clock):
- (a) 30.2h / (b) 34.2h / (c) 34.2h / (d) 37.3h / (e) 38.2h

#### 이슈: `/dev/shm` Bus error

- 초기에 `run_all_variants_parallel.sh`로 `num_workers=8` 설정, 5 variant 동시 실행 시도 → DataLoader worker가 `/dev/shm`(기본 1 GB)을 공유 메모리로 사용하면서 **bus error로 학습 프로세스가 죽음**.
- 해결: `num_workers=2`로 감소 (공유 메모리 사용량을 4배 줄임). `run_4x_parallel.sh` / `run_4x_wave1.sh`도 `num_workers=2`로 통일.
- 대안으로 GPU 메모리 제약 시 4+1 wave 분할 스크립트(`run_4x_wave1.sh`)도 준비했지만, 실제 실행은 `run_all_variants_parallel.sh` 동시 5 variant로 진행됨(progress_4x.log의 PID 5개 동시 기록이 근거).

#### 산출물

- 각 variant `proj_log/variant_*_4x/` 아래 `latest.pth` + `ckpt_epoch100.pth` + `ckpt_epoch200.pth` 세 체크포인트 모두 보존 (3x의 (e)와 달리).
- 로그 `variant_*_4x.log` 각 ~36 MB, 모두 `EPOCH[199]`까지 도달 확인.

### 2.3 Phase 2 정량 결과 (4x 기준, 7881 테스트 샘플)

`docs/report_phase2_ablation.md` §4 전재.

| Metric | 논문 | (a) Baseline | (b) Cross-Attn | (c) Cross-Attn+BN | (d) Alt-Attn | (e) Alt+Cross |
|---|---|---|---|---|---|---|
| **Cmd Acc** | **82.76** | 81.97 | 82.53 | 82.89 | 82.12 | 82.78 |
| **Avg Args** | **79.23** | 78.65 | 78.87 | 78.89 | 79.00 | **79.33** |
| plane | — | 93.74 | 93.54 | 93.60 | 94.12 | **94.52** |
| trans | — | 70.06 | 70.11 | 69.64 | 70.41 | **70.58** |
| extent | — | 66.19 | 67.16 | 67.30 | 67.61 | **68.14** |

**MAE (lower is better)**: (a) 9.804 → (e) **9.261** (5.5% 개선)
**Latency (batch=1, A100)**: (a) 3.95 ms → (e) 6.94 ms (+75.7%)

- (e) Alt+Cross만 논문 원본 대비 Args Accuracy를 상회(+0.10%p). Cmd도 거의 동등(+0.02%p).
- Extrude 계열(plane/trans/extent)에서 (e)가 일관되게 전체 최고 성능 및 최저 MAE.

### 2.4 Phase 2 정성 결과 (숫자 기반 샘플 분포, 3D 복원 전)

- **Perfect (score = 1.0)**: 2,168 / 7,881 (27.5%)
- **Score ≥ 0.9**: 4,049 (51.4%)
- **Score < 0.5**: 1,058 (13.4%)

**시퀀스 길이별**:
| 구간 | 샘플 수 | 평균 Score | Score < 50% 비율 |
|---|---|---|---|
| Short (≤8) | 3,986 | 91.9% | 2.1% |
| Medium (9–20) | 2,647 | 74.2% | 15.2% |
| Long (21–40) | 972 | 58.8% | 41.8% |
| Very Long (>40) | 276 | 50.1% | 60.5% |

길이 20을 넘어서면 실패율이 급격히 상승. Variant 간 차이는 Medium 구간에서 가장 두드러짐.

### 2.5 핵심 발견

- **Cross-Attention Decoder (a→b)**: Cmd +0.56, Args +0.22 → Cmd 개선에 주로 기여
- **Alternating Encoder (a→d)**: Cmd +0.15, Args +0.35 → Args, 특히 extrude 계열에 주로 기여 (계획서의 "Negative Control" 예상과 달리 Args에 의미 있는 효과)
- **조합 시너지 (e)**: 개별 기여합(0.57)보다 큰 Args +0.68 → 약한 시너지
- **과적합 경향**: 모든 variant가 (a) 대비 val/train gap 증가. (d) gap 1.139로 최대 — encoder 표현력↑ + broadcast 병목 유지 조합이 가장 취약.
- **결정**: Phase 3/4의 베이스 모델로 **(e) Alt+Cross (4x)** 선택.
- **Phase 2 보고서 commit**: `c5566ce` — `docs: add Phase 2 ablation study report with quantitative/qualitative evaluation` (2026-04-17 21:50, 16 files, +653/-34).

---

## Phase 3: torch.compile + FP16 추론 최적화

**대상 모델**: Variant (e) Alt+Cross (4x)

### 3.1 Graph Break 분석

torch.compile 적용 전 forward path에서 잠재적 graph break 요인 전수 조사.

| 파일 | 패턴 | 심각도 | 실제 상태 |
|---|---|---|---|
| `functional.py:90,94` | `torch.equal(query, key)` | CRITICAL | Phase 1에서 `nn.MultiheadAttention` 전환으로 **이미 제거됨** |
| `model.py:83` | `if S == self.tokens_per_view` | HIGH | input_option=4x 고정 시 S=400 상수 → static branch |
| `model.py:91-93` | `for v in range(num_views)` | HIGH | 4x 고정 시 4회 고정 루프 → unroll 가능 |
| `trainer.py:80,107` | `.cpu().numpy()` | MEDIUM | forward path 외부(평가/후처리) → 무영향 |

**결론**: `input_option=4x`로 고정 시 forward path에 실질적 graph break 없음. `torch.compile`이 정상 trace 가능.

### 3.2 벤치마크 (batch_size=1, A100)

| 설정 | Latency (ms) | Speedup (자체 base 대비) | (a) Baseline FP32 대비 |
|---|---|---|---|
| (a) Baseline FP32 (no compile) | 3.925 | — | 1.00x |
| (e) FP32 (no compile) | 6.876 | — | 0.57x |
| (e) FP16 autocast only | 9.344 | 0.74x | 0.42x |
| (e) compile (default) FP32 | 3.475 | 1.99x | 1.13x |
| (e) compile (reduce-overhead) FP32 | 1.569 | 4.41x | 2.50x |
| (e) compile (default) + FP16 | 4.301 | 1.61x | 0.91x |
| **(e) compile (reduce-overhead) + FP16** | **1.251** | **5.19x** | **3.14x** |

**배치 크기별**:
- batch=1: 145 → **799 samples/s** (5.50x)
- batch=16: 2,210 → 5,523 (2.50x)
- batch=256: 4,235 → 8,616 (2.03x)

### 3.3 정확도 보존 검증

| 비교 | command_logits max diff | args_logits max diff | Cmd argmax 일치율 |
|---|---|---|---|
| compile FP32 vs Baseline FP32 | 0.018 | 0.105 | 99.99% |
| compile FP16 vs Baseline FP32 | 0.025 | 0.204 | 99.99% |

FP16 정밀도 범위 내 수치 오차이며, argmax 결과에 실질적 영향 없음.

### 3.4 발견

1. **추론 비용 문제 완전 해결**: (e)의 +75.7% latency 증가가 torch.compile 후 **(a) FP32 대비 3.14x 더 빠름**으로 역전. 성능과 속도 모두 Baseline 초과 달성.
2. **FP16 단독은 비효과적**: autocast만 적용하면 오히려 느려짐 — batch=1 overhead-bound 환경에서 dtype 변환 오버헤드가 계산 이득을 상쇄. torch.compile과 결합해야 효과.
3. **batch=1 최적화가 가장 극적**: CUDA Graph로 kernel launch overhead 제거.
4. **Phase 3 보고서 commit**: `8461e6f` — `docs: add Phase 3 inference optimization results to report` (2026-04-17 23:06, 1 file, +66/-4).

---

## Phase 4.1: Mask-Predict Iterative Refinement

### 4.1.1 설계

CMLM(Ghazvininejad et al., EMNLP 2019) 스타일의 iterative refinement을 NAT decoder에 접목.

- **신규 모듈 `PartialPredictionEmbedding`** (기존 `ConstEmbedding` 대체, `model/model.py` +91 lines)
  - 1차 pass: 기존처럼 `zeros + PositionalEncoding` (unchanged behavior)
  - 2차+ pass: `(prev_cmd, prev_args)` embedding 생성 후, confidence 하위 k개 위치를 learnable `[MASK]` token으로 치환
  - 새 파라미터 5종: `command_embed`, `args_embed`, `args_proj`, `mask_token`, (+ PE 재사용)
- **학습 전략**: Variant (e) pretrained checkpoint 로드 → 전체 파라미터 fine-tuning 70 epochs
  - Loss = initial pass + refinement pass 합산
  - Refinement pass는 GT에 `Uniform(0.15, 0.85)` mask ratio로 random masking 후 decoder 재실행
  - Masked 위치에만 loss 계산 (`refinement_mask` 인자로 `NewCADLoss` 확장)
- **추론 전략**: 1차 예측 → confidence 하위 k 위치 masking (`torch.topk(cmd_confidence, k, largest=False)`) → decoder 재실행 → N step 반복

### 4.1.2 구현 작업 (commit `51f8daf`)

**Commit**: `51f8daf` — `feat: add Phase 4 Mask-Predict refinement + OCC-based 3D qualitative/CD/IR evaluation` (2026-04-18 19:32, 182 files, +3009/-129)

주요 코드 변경:

| 파일 | 변경 |
|---|---|
| `config/config.py` | `--use_mask_predict`, `--n_refinement_steps`, `--mask_ratios`, `--freeze_pretrained` 4개 인자 신규 (+10 lines) |
| `model/model.py` | `PartialPredictionEmbedding` 클래스 신규, `Decoder.forward` 시그니처(`prev_cmd`, `prev_args`, `mask_positions`), `SVG2CADTransformer.forward`에 refinement 루프 (+91 lines) |
| `trainer/trainer.py` | `use_mask_predict=True and is_train`이면 `proj_log/variant_e_alt_cross_4x/model/latest.pth`를 `strict=False`로 로드. `freeze_pretrained` 모드로 MP 신규 파라미터만 학습 가능. `_forward_mask_predict` 2-pass 메서드 신규 (+96 lines) |
| `trainer/loss.py` | `NewCADLoss.forward(outputs, cad_data, refinement_mask=None)` 시그니처 변경. refinement_mask로 masked position에만 loss. 기존 EMD loss branch 제거 (+20 lines) |
| `test.py` | `cfg.n_refinement_steps`, `cfg.mask_ratios` 파싱 후 schedule list 구성. `tr_agent.net(..., n_refinement_steps=..., mask_ratio_schedule=...)` 직접 호출로 변경 (+13 lines) |
| `run_mask_predict_train.sh` | 학습 wrapper (+19 lines) |
| `run_mask_predict_test.sh` | N=0..3 순차 테스트 wrapper (+37 lines) |

### 4.1.3 학습 실행

**시점**: 2026-04-17 23:18 ~ 2026-04-18 05:13 (`train_logs/progress_mp.log`)

```
2026-04-17 23:18:27 [START] Mask-Predict training (70 epochs)
2026-04-18 05:13:28 [DONE] Mask-Predict training
```

- **총 소요**: 5시간 55분
- **설정** (`proj_log/variant_e_mask_predict/config.txt`):
  - `pretrained = proj_log/variant_e_alt_cross_4x/model/latest.pth`
  - `nr_epochs = 70`, `lr = 5e-4`, `batch_size = 256`
  - `encoder_type = alternating`, `decoder_type = cross_attention`
  - `use_mask_predict = true`, `n_refinement_steps = 0` (config 값; 추론시 override), `mask_ratios = "0.5,0.3"`
  - `freeze_pretrained = false` (계획서의 2-stage 동결 전략에서 선회)
  - `input_option = 4x`
- 로그상 `EPOCH[69]` 도달 → 70 epoch 완주.

### 4.1.4 이슈와 수정

학습 시작 전 **7번의 smoke test 디버깅** 거침 (`proj_log/_test_mp1..7`, 모두 2026-04-17 23:14~23:18 사이 1분 단위로 생성되고 모델 저장 전 중단). 주요 이슈:

1. **Loss shape mismatch (RuntimeError at tensor dim 1)**
   - `NewCADLoss.forward`에서 `padding_mask.unsqueeze(-1)`로 broadcast를 시도했으나 `refinement_mask`의 차원과 불일치
   - 해결: `padding_mask *= refinement_mask.float()` 형태로 단순 in-place multiply 적용, `unsqueeze` 제거
2. **`_test_mp[1..7]` iterative 디버깅**
   - `batch_size`를 4 → 2로 축소하고 `num_workers=0`으로 디버깅 용이화
   - 디렉토리 7개는 학습이 체크포인트 저장 전 단계에서 모두 중단된 흔적 (model/ 디렉토리 비어있음)

### 4.1.5 테스트

**시점**: 2026-04-18 10:49:49 ~ 10:50:38 (`train_logs/mp_test.log`)

```
10:49:49 [START] test n_refinement_steps=0 (ratios=0.5)
10:50:01 [DONE] Saved to .../test_results_n0    (12s)
10:50:01 [START] test n_refinement_steps=1 (ratios=0.5)
10:50:13 [DONE] Saved to .../test_results_n1    (12s)
10:50:13 [START] test n_refinement_steps=2 (ratios=0.5,0.3)
10:50:26 [DONE] Saved to .../test_results_n2    (13s)
10:50:26 [START] test n_refinement_steps=3 (ratios=0.6,0.4,0.2)
10:50:38 [DONE] Saved to .../test_results_n3    (12s)
10:50:38 [ALL DONE]
```

- 각 N당 ~13초 (7881 샘플)
- 결과 저장: `proj_log/variant_e_mask_predict/test_results_n{0,1,2,3}/` 각 7881개 `_vec.h5`

### 4.1.6 정량 결과 (`docs/phase4_accuracy.json`)

| Metric | (e) Phase 2 | **N=0** | N=1 | N=2 | N=3 |
|---|---|---|---|---|---|
| **Cmd Acc** | 82.78 | **82.61** | 48.42 | 43.48 | 39.21 |
| line | 70.77 | 71.46 | 60.41 | 51.92 | 52.57 |
| arc | 79.25 | 79.43 | 65.52 | 62.20 | 62.61 |
| circle | 92.72 | 93.05 | 93.37 | 80.29 | 83.66 |
| plane | 94.52 | 94.34 | **96.67** | 91.29 | 89.20 |
| trans | 70.58 | 70.75 | **80.14** | 57.60 | 60.89 |
| extent | 68.14 | 68.78 | **79.80** | 53.71 | 58.19 |
| **Avg Args** | 79.33 | **79.64** | 79.32 | 66.17 | 67.85 |

### 4.1.7 Latency 측정 (`docs/phase4_latency.json`)

**시점**: 2026-04-18 11:13 (`tools/bench_mp_latency.py`, 50 샘플, batch=1, A100, FP32, no compile)

| N | 설정 | Mean (ms) | Std | Median | vs N=0 |
|---|---|---|---|---|---|
| 0 | no refinement | **6.946** | 0.222 | 6.915 | 1.00x |
| 1 | [0.5] | 10.131 | 0.109 | 10.120 | 1.46x |
| 2 | [0.5, 0.3] | 13.324 | 0.302 | 13.276 | 1.92x |
| 3 | [0.6, 0.4, 0.2] | 16.497 | 0.353 | 16.416 | 2.37x |

각 refinement step당 약 **+3.2 ms** 증가 (decoder 재실행 1회).

### 4.1.8 발견 및 해석

**긍정적 발견 — Mask-Predict 학습 regime의 부수 효과**:
- N=0 (refinement 미적용) 결과가 variant (e)를 Avg Args 기준 **+0.31%p** 상회 (79.33 → 79.64)
- 특히 line, circle, extent에서 개선 — random masking이 일종의 **data augmentation/regularization** 역할.

**부정적 발견 — Iterative refinement 자체는 악화**:
- N≥1에서 Cmd Acc가 82.61% → 48.42%로 붕괴 (약 34%p 감소)
- **원인 분석**:
  1. **Padding confidence 과다**: 1차 예측에서 EOS/padding 위치의 softmax confidence가 >0.99로 극히 높아, `topk(lowest-k)` masking이 실제로는 **유효한 sketch 위치를 반복적으로 mask → 재예측하며 왜곡**
  2. **학습-추론 불일치**: 학습 시에는 GT를 prev_prediction으로 사용(teacher forcing)했으나 추론 시에는 1차 예측(노이즈 포함)을 사용. `PartialPredictionEmbedding`이 노이즈 입력에 robust하지 않음
  3. **Mask schedule 부적합**: 50% 이상 masking하는 기본 스케줄이 NAT decoder의 convergence를 해침

**부분적 긍정 — N=1에서 Extrude 파라미터 대폭 개선**:
- plane +2.33, trans +9.39, extent +11.02 — Phase 2에서 가장 약했던 extrude args가 단 1회 refinement로 크게 향상
- Cmd 붕괴만 해결하면 hybrid inference (cmd는 N=0, ext args는 N=1)로 활용 가능성

---

## 3D OCC 기반 정성/정량 평가 (Chamfer Distance, IR)

**시점**: 2026-04-18 (CD/IR commit은 `51f8daf` 내 포함). 3D 평가는 3개 하위 에이전트를 병렬 실행하여 수행.

### 평가 에이전트 팀 (3개 하위 에이전트)

#### (a) DeepCAD 코드 분석 에이전트
- **주요 발견**: DeepCAD의 `cadlib/visualize.py::vec2CADsolid`가 `(seq_len, 17)` vec → OCC `TopoDS_Shape`로의 정식 변환 경로. `curves.py`/`sketch.py`/`extrude.py`의 `numericalize/denumericalize`가 tokenize 경계를 결정.
- **의존성**: `pythonocc-core` (OCC python binding), `trimesh`, `scipy.spatial.KDTree`.
- 원본 DeepCAD 참조본을 `/home/work/Drawing2CAD/deepcad_ref/`에 복사 보관.

#### (b) OCC 렌더링 구현 에이전트
- **conda env 생성**: Miniconda 설치 경로 `/home/work/miniconda3`
- **Env**: `deepcad_viz` — `pythonocc-core=7.5.1`, `trimesh`, `matplotlib`, `scipy` 설치
- **Headless 렌더링 파이프라인**: `xvfb-run -a python tools/render_cad.py ...` → Mesa llvmpipe로 OCC `Viewer3d.Dump` 사용, 512×512 등각 4뷰 PNG 출력
- **패치 작업** (`cadlib_deepcad/` 생성, 원본 `deepcad_ref/cadlib/`와의 diff):

  | 파일 | 패치 |
  |---|---|
  | `curves.py` | `np.int` → `int` (Line: 2개소, Arc: 5개소, Circle: 2개소) — NumPy 1.24+에서 `np.int` deprecated |
  | `extrude.py` | `np.int` → `int` (CoordSystem: 2, Extrude: 4, random_transform: 1) — 총 **16 sites** |
  | `sketch.py` | `matplotlib.use('TkAgg')` → `matplotlib.use('Agg')` — headless 환경(xvfb/Docker)에서 TkAgg backend 에러 회피 |
  | `macro.py`, `math_utils.py`, `visualize.py` | 변경 없음 |

#### (c) 정성 평가 수행 에이전트
- **샘플 재선정 과정**: 초기 상위/중위/하위 각 3개씩 선정했으나, OCC 변환 실패 케이스가 몰려서 평가 불가 → `render_variant_*.json`으로 변환 성공 여부를 먼저 확인하고, (a)와 (e) 둘 다 렌더 가능한 샘플 위주로 재선정.
- **최종 샘플 8개**: `00008056, 00017379, 00868771, 00319566, 00306982, 00582849, 00625131, 00883872` (상위 2 + 중위 2 + 하위 4)
- **추가 GT-only 케이스 3개**: `00000134, 00000392, 00000559` (pred는 실패, GT 참조용).
- **전체 IR (variant a, e) 사전 측정**:
  - variant_a_baseline_4x: pred ok 6140/7881 (77.91%), ok_valid 5550/7881 (70.42%) → **IR 29.58%**
  - variant_e_alt_cross_4x: pred ok 6138/7881 (77.88%), ok_valid 5558/7881 (70.52%) → **IR 29.48%**
  - 두 variant의 "3D 변환 성공률은 사실상 동등, 실패 원인 분포는 다름"이라는 핵심 정성 finding 도출 (AssertionError (e)>+69, IndexError (e)<-121).

### CD/IR 측정 (후속)

**시점**: 2026-04-18 (`train_logs/cd_ir.log` 25 KB, `docs/phase4_cd_ir.json` 2026-04-18 12:12 생성)

- **대상**: 5 variants (4x) + MP N=0..3 = **총 9 configs**
- **방법**: 2000 random samples (seed=0), `n_points=2000` per mesh, KDTree-based 대칭 Chamfer Distance
- **스크립트**: `tools/eval_cd_ir.py` (5.3 KB, `joblib.Parallel(n_jobs=8)`)
- **파이프라인**: pred vec → `vec2CADsolid` → `TopoDS_Shape` → STL export → `trimesh.sample_surface(2000)` → KDTree Chamfer vs GT 동일 파이프라인
- **사전 스모크 테스트**: `docs/_cd_smoke.json` (2026-04-18 12:09, 50 샘플) — variant_e_alt_cross_4x n_valid=36, IR=0.28, cd_mean=0.0851. 파이프라인 동작 검증.

**결과**:

| Config | n_valid/2000 | IR | CD Mean | CD Median | CD Trimmed Mean |
|---|---|---|---|---|---|
| variant_a_baseline_4x | 1371 | 0.3145 | 0.1176 | 0.00729 | 0.05746 |
| variant_b_cross_attn_4x | 1390 | 0.3050 | 0.1109 | 0.00867 | 0.05833 |
| variant_c_cross_attn_bn_4x | 1376 | 0.3120 | 0.1138 | 0.00622 | 0.05914 |
| variant_d_alt_attn_4x | **1390** | **0.3050** | **0.1027** | 0.00603 | 0.05484 |
| variant_e_alt_cross_4x | 1373 | 0.3135 | 0.1077 | 0.00619 | **0.05368** |
| **mp_n0** | **1401** | **0.2995** | 0.1058 | **0.00541** | **0.05249** |
| mp_n1 | 83 | 0.9585 | 0.0406† | 0.00147† | 0.00443† |
| mp_n2 | 0 | **1.0000** | null | null | null |
| mp_n3 | 24 | 0.9880 | 0.0980† | 0.01405† | 0.04919† |

† MP N≥1의 CD는 IR 95%+라 극소수 유효 샘플만 반영, 신뢰성 낮음.

**핵심 관찰**:
- **mp_n0이 IR 0.2995로 모든 config 중 최저** — MP 학습 regime이 시퀀스의 구문 안정성을 미세 향상
- **MP N≥1에서 IR이 95-100%로 폭증** — `pred:convert:IndexError`가 각각 1831/2000, 1459/2000, 1822/2000으로 압도적. refinement가 sequence 구조(SOL/EXT 매칭)를 파괴
- CD median(0.005 ~ 0.01)과 CD mean(0.10 ~ 0.12)의 큰 격차 → 소수 대형 오류 샘플이 평균을 견인

### 전체 IR (b, c, d variant 추가 측정)

**시점**: 2026-04-18 (~13:30, `train_logs/success_bcd.log` 3.5 KB)

7881 전체 테스트셋에 대한 vec→OCC 변환 성공률 재측정 (`qualitative_eval.py --success-count`). `success_variant_*.json` 결과:

| Variant | pred ok | pred ok_valid | pred fail | GT ok | GT ok_valid | 소요 시간 |
|---|---|---|---|---|---|---|
| (a) baseline_4x | 6140 | 5550 | 1741 | 7759 | 7687 | 138.4s |
| (b) cross_attn_4x | 6177 | 5604 | 1704 | 7759 | 7687 | 170.4s |
| (c) cross_attn_bn_4x | 6183 | 5579 | 1698 | 7759 | 7687 | 139.5s |
| (d) alt_attn_4x | **6213** | **5664** | 1668 | 7759 | 7687 | 137.8s |
| (e) alt_cross_4x | 6138 | 5558 | 1743 | 7759 | 7687 | 137.8s |

- **variant_d가 IR 28.13%로 전체 최저** (pred ok_valid 5664/7881). "인코더 개선만" 조건이 3D 유효성 측면에서는 의외로 최강.
- (e) Alt+Cross는 IR 29.47%로 중위. Cmd/Args metric과 3D IR의 순위 불일치 관찰 → Cross-attention이 숫자 정확도를 올리되 구조적 유효성은 개선하지 못함.
- GT는 모든 variant에서 동일하므로 7759/7687/122 고정.

### 3D 산출물

- `docs/figures/qualitative_3d/grid_{top,mid,bottom}_tier.png` — GT + (a) + (e) 병렬 tier grid (각 203~365 KB)
- `grid_variant_{a,e}_*_4x.png` — variant별 단독 grid
- `variant_a_baseline_4x/`, `variant_e_alt_cross_4x/` 폴더 — 각 76 PNG (11 sample × gt/pred × 4 views, 변환 실패분 제외)
- `render_variant_{a,e}_*_4x.json` — 렌더링 실행 메타데이터
- `success_variant_{a,b,c,d,e}_*_4x.json` + `qualitative_eval_summary.json` — 전체 7881 샘플 IR 집계

### MP에 대한 success-count 시도 (실패)

`train_logs/success_mp.log`:
```
SKIP (not a directory): /home/work/Drawing2CAD/proj_log/variant_e_mask_predict_test_results_n0
...
```
- 디렉토리 명명 규약 불일치(`{exp}/test_results_n{N}` vs `{exp}_test_results_n{N}`)로 MP 전체 IR 측정 스킵. CD/IR 측정의 2000 subset 결과로 갈음.

---

## 전체 타임라인 요약

| 날짜 | 시각 (KST) | 이벤트 | 산출물/파일 |
|---|---|---|---|
| 04-13 | 18:37 | Phase 1 nn.MHA 전환 commit | `08e2823` (+404/-9) |
| 04-13 | 18:44 | Phase 2 아키텍처 골격 commit | `67701ea` (+406/-42) |
| 04-13 | 19:21 | 계획서 commit | `0ba7faa` |
| 04-13~14 | — | Phase 1.5 sanity check (미기록) | (commit 외) |
| 04-14 | 00:44 | wandb login 실패 traceback | `wandb/debug-cli.work.log` |
| 04-14 | 02:00 | 5 variants 3x 순차 학습 시작 | `run_all_variants.sh`, `progress.log` |
| 04-15 | 21:42 | 3x 학습 완료 (44h) | `variant_{a..e}/latest.pth` |
| 04-15 | — | 3x 실수 발견 (사용자 지적) | — |
| 04-15 | 21:49 | 4x 병렬 재학습 시작 (num_workers=2) | `run_all_variants_parallel.sh`, `progress_4x.log`, PIDs 55879–55887 |
| 04-17 | 12:02 | 4x 학습 완료 (38h 13m) | `variant_{a..e}_4x/latest.pth` |
| 04-17 | 21:50 | Phase 2 보고서 commit | `c5566ce` (16 files, +653) |
| 04-17 | 23:06 | Phase 3 보고서 commit | `8461e6f` (1 file, +66) |
| 04-17 | 23:14~23:18 | MP smoke test 7회 디버깅 | `_test_mp[1-7]/` |
| 04-17 | 23:18 | MP 학습 시작 | `progress_mp.log`, `mask_predict.log` |
| 04-18 | 05:13 | MP 학습 완료 (5h 55m) | `variant_e_mask_predict/latest.pth` (120.6 MB) |
| 04-18 | 10:49:49 | MP test N=0 시작 | `test_results_n0/` |
| 04-18 | 10:50:38 | MP test N=0..3 모두 완료 (~49초) | `test_results_n{0..3}/`, `mp_test.log` |
| 04-18 | ~10:51 | MP accuracy 집계 | `phase4_accuracy.json` |
| 04-18 | 11:13 | MP latency 측정 | `phase4_latency.json` (50 샘플 × 4 N) |
| 04-18 | ~11:30 | 3D OCC 평가 에이전트 팀 실행 | `cadlib_deepcad/`, `qualitative_3d/` |
| 04-18 | 12:09 | CD/IR smoke test (50 샘플) | `_cd_smoke.json` |
| 04-18 | 12:12 | CD/IR 2000 샘플 9 config 측정 | `phase4_cd_ir.json`, `cd_ir.log` |
| 04-18 | 12:18 | qualitative summary 통합 | `qualitative_eval_summary.json` |
| 04-18 | ~13:30 | b/c/d 전체 7881 IR 측정 | `success_variant_{b,c,d}_*.json`, `success_bcd.log` |
| 04-18 | 19:32 | Phase 4 통합 commit | `51f8daf` (182 files, +3009/-129) |

---

## 핵심 의사결정 로그

### Q1. 왜 Alt+Cross (variant e)를 최적으로 골랐나?

- **Args 기준 유일한 논문 원본 초과**: (e) 79.33 vs 논문 79.23 (+0.10)
- **(a) 대비 상대 개선 최대**: Cmd +0.81, Args +0.68
- **Extrude 계열(plane/trans/extent) 전체 최고**: 94.52 / 70.58 / 68.14
- **MAE 최저** (9.261, (a)의 9.804 대비 5.5% 감소)
- **CD Trimmed Mean 최저** (0.0537) — 3D 형상 근접도에서도 Phase 2 variant 중 1위
- Latency +75.7% 증가는 Phase 3 torch.compile로 상쇄 가능함이 확인됨

### Q2. 왜 torch.compile(reduce-overhead)를 택했나?

- batch=1에서 모델이 overhead-bound (kernel launch ~750μs vs compute ~100μs)
- `reduce-overhead` 모드는 CUDA Graph를 사용하여 kernel launch overhead를 완전 제거
- batch=1에서 5.19x 가속 (default 모드 1.99x 대비 2.6배 효과적)
- batch=256 compute-bound 환경에서는 `default`와 `reduce-overhead`의 격차 축소되지만, 배포 시나리오가 interactive(batch=1) 중심이므로 reduce-overhead 채택

### Q3. 왜 MP 70 epochs / lr 5e-4로 했나 (계획 대비 축소)?

- 원 계획(개선안 §Phase 4.1)에서는 구체적 epoch/lr 미지정
- **사전학습 checkpoint 로드** 상태이므로 scratch(200 epoch)의 1/3 수준으로 충분
- `lr=5e-4`는 원 학습(lr=1e-3)의 50% — pretrained weight을 크게 흔들지 않으면서 신규 MP 파라미터(5종)가 안정적으로 수렴하도록 조절
- 실제 5h 55m 소요, overfitting 없이 `EPOCH[69]`까지 완주

### Q4. 왜 freeze_pretrained=False로 했나 (원 2-stage 계획 수정)?

- 계획서 §Phase 2.1의 2-stage phased training(encoder freeze → 전체 해동)은 scratch 학습 가정.
- MP는 이미 수렴한 (e) 4x checkpoint에서 출발하므로 encoder도 MP 학습 신호에 적응할 필요 있음
- `freeze_pretrained=True`면 MP 신규 파라미터 5종(~19,600개)만 학습 → 너무 작은 용량으로 refinement task에 적응 불가능 판단
- 전체 fine-tuning하되 lr을 낮춰 안정화 (Q3 참조)

### Q5. 왜 샘플 재선정을 했나 (3D eval에서)?

- 초기 후보: Cmd Acc 기준 상/중/하위 각 3개
- 문제: 하위 샘플 중 (a), (e) **둘 다 OCC 변환 실패**인 케이스가 다수 → 비교 자체 불가
- 해결: `render_variant_a_baseline_4x.json`과 `render_variant_e_alt_cross_4x.json`에서 변환 성공한 샘플을 먼저 추려낸 후 재선정
- 최종 8개 중 `00306982`, `00625131`은 (a)(e) 모두 실패 → 하위 tier의 "실패 공통성"을 보여주는 케이스로 정성 보고서에 포함 (섹션 6.5)

---

## 실패/에러 로그

1. **input_option=3x 실수 → 4x 재학습** (2026-04-14 ~ 04-17)
   - 학습 `--input_option 3x` vs 테스트 `--input_option 4x` 불일치
   - 44h 3x 학습 결과 전량 재학습으로 폐기 (단 proj_log에는 보존)
   - 사용자 지적으로 발견. Phase 2 재학습 38h 추가 투입
   - 교훈: **기존 sh/논문 설정 반드시 대조** (feedback_input_option 메모 참조)

2. **/dev/shm 부족 Bus error** (2026-04-15 후반)
   - `num_workers=8` × 5 병렬 variant → DataLoader 공유 메모리 초과, 1 GB 한계
   - 해결: `num_workers=2`로 감소 → 4x 재학습 정상 완료

3. **MP loss shape mismatch (RuntimeError dim 1)** (2026-04-17 23:14~23:18)
   - `padding_mask.unsqueeze(-1) * refinement_mask` shape 충돌
   - 해결: `padding_mask *= refinement_mask.float()` 단순 in-place multiply
   - smoke test 디렉토리 7개(`_test_mp1..7`)가 이 디버깅 흔적

4. **PIP_CONSTRAINT 환경변수 numpy 1.26.4 고정** (3D 평가 중)
   - `/etc/pip/constraint.txt`가 system-level로 numpy 1.26.4 강제
   - `pythonocc-core` 설치 시 numpy 1.24+ 필요 지점과 충돌
   - 해결: `unset PIP_CONSTRAINT` 후 conda env에서 재설치

5. **`np.int` deprecation** (DeepCAD cadlib 포팅)
   - NumPy 1.24+에서 `np.int` 제거됨
   - 영향 파일: `curves.py`(9 sites), `extrude.py`(7 sites) = 총 16 sites
   - 해결: `np.int` → `int` 일괄 치환 후 `cadlib_deepcad/`에 패치본으로 분리

6. **scipy 미설치 (eval_cd_ir.py)** (2026-04-18 CD/IR 실행 초기)
   - `from scipy.spatial import KDTree` ImportError
   - 해결: `pip install scipy` (conda env `deepcad_viz` 내)

7. **matplotlib TkAgg backend 실패 (headless)**
   - DeepCAD 원본 `sketch.py`가 `matplotlib.use('TkAgg')` 강제 → xvfb 환경에서 crash
   - 해결: `matplotlib.use('Agg')`로 교체 (cadlib_deepcad 패치)

8. **wandb API 키 누락** (2026-04-14 00:44)
   - `wandb.errors.errors.UsageError: No API key configured`
   - 해결: `~/.netrc`에 `api.wandb.ai` credential(entity `jujoo`) 설정

9. **MP success-count 디렉토리 네이밍 불일치**
   - `qualitative_eval.py`가 `{exp}_test_results_n{N}`을 찾는데 실제는 `{exp}/test_results_n{N}`
   - `train_logs/success_mp.log`에 4개 SKIP 기록 → MP 전체 7881 IR 측정은 포기, 2000 subset CD/IR로 대체

---

## 다음 단계

### Phase 4.2 (ArgsFCN low-rank factorization)
- **상태**: 미진행
- **사유**: 30시간+ 학습 시간 필요. 성능 향상보다 모델 크기 최적화 목적이라 우선순위 낮음.
- **구현**: `Linear(288, 4112)` → `Linear(288, 64) + Linear(64, 4112)` 또는 sketch/extrusion head 분리 + loss masking

### 과적합 완화 실험
- Phase 2 §7.4에서 모든 variant의 train-val gap 증가 확인 (특히 (d) 1.139)
- 실험 대상: (e) 기반으로 dropout 0.1 → 0.2, DropPath(stochastic depth) 추가, stroke permutation + coordinate jittering augmentation
- 예상 개선: Args +0.5~1.0%p (계획서 §방법론 2 ⚠️ 참조)

### MP 개선 실험
- **Padding 제외 gating**: EOS 이후 위치는 masking topk 대상에서 제외 (4.1.8 원인 #1 해결)
- **Noisy prev_pred**: 학습 시 GT에 random corruption 적용하거나 model의 own 1차 예측을 다음 pass 입력으로 (학습-추론 gap 해소, 4.1.8 원인 #2)
- **Cmd-frozen refinement**: Cmd는 N=0 고정, Args만 refinement (N=1에서 extrude args +11%p를 살림)

### Chamfer Distance 전수 측정
- 현재 2000 subset (seed=0) → full 7881 샘플 측정
- 예상 소요: config당 ~70초 × 9 config ≈ 10분 (A100 + n_jobs=8)

### 배포 패키징
- torch.compile(reduce-overhead) + FP16 설정을 별도 inference script로 통합
- `input_option=4x` 고정, `torch._dynamo.mark_static()`로 재컴파일 회피

---

## 부록: 참조 파일 절대 경로

- 산출물 매니페스트: `/home/work/Drawing2CAD/docs/experiment_artifacts.md`
- 정량/정성 보고서: `/home/work/Drawing2CAD/docs/report_phase2_ablation.md`
- 개선 계획서: `/home/work/Drawing2CAD/improvement_plan.md`
- MP 정확도: `/home/work/Drawing2CAD/docs/phase4_accuracy.json`
- MP latency: `/home/work/Drawing2CAD/docs/phase4_latency.json`
- CD/IR 측정: `/home/work/Drawing2CAD/docs/phase4_cd_ir.json`
- CD/IR smoke: `/home/work/Drawing2CAD/docs/_cd_smoke.json`
- 3D 정성 결과: `/home/work/Drawing2CAD/docs/figures/qualitative_3d/`
- Progress 로그: `/home/work/Drawing2CAD/train_logs/progress.log`, `progress_4x.log`, `progress_mp.log`
- MP 테스트 로그: `/home/work/Drawing2CAD/train_logs/mp_test.log`, `mp_test_n{0..3}.log`
- CD/IR 실행 로그: `/home/work/Drawing2CAD/train_logs/cd_ir.log`
- B/C/D success-count 로그: `/home/work/Drawing2CAD/train_logs/success_bcd.log`
- Variant 체크포인트: `/home/work/Drawing2CAD/proj_log/variant_{a..e}_{baseline,cross_attn,cross_attn_bn,alt_attn,alt_cross}_4x/model/latest.pth`
- MP 체크포인트: `/home/work/Drawing2CAD/proj_log/variant_e_mask_predict/model/latest.pth`
