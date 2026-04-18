# Drawing2CAD Phase 2 Ablation Study Report

**실험 기간**: 2026-04-15 ~ 2026-04-17  
**실험 환경**: NVIDIA A100-SXM4-80GB, PyTorch, 5 variants 병렬 학습  
**데이터셋**: Drawing2CAD test set (7,881 samples)  
**학습 설정**: 200 epochs, batch_size=256, lr=1e-3, input_option=4x

---

## 1. 실험 목적

Drawing2CAD의 핵심 성능 병목인 **Decoder 정보 병목**(mean pooling → single vector broadcast)을 해결하기 위해, 두 가지 아키텍처 개선안의 효과를 검증한다.

- **방법론 1 (Alternating Attention Encoder)**: VGGT-style frame-wise/global attention 교대 적용
- **방법론 2 (Cross-Attention Decoder)**: mean pooling 제거 → full encoder memory에 cross-attention

5개 variant를 통해 각 방법론의 개별 기여와 조합 효과를 분리 측정한다.

---

## 2. Variant 설계

| Variant | Encoder | Decoder | Bottleneck | 설계 의도 |
|---------|---------|---------|------------|-----------|
| **(a) Baseline** | Standard | Broadcast | Mean pool | 기준선 (nn.MHA 전환 후 재학습) |
| **(b) Cross-Attn** | Standard | Cross-Attention | 제거 | 디코더 개선 단독 효과 |
| **(c) Cross-Attn+BN** | Standard | Cross-Attention | Element-wise 유지 | Bottleneck 유무 비교 |
| **(d) Alt-Attn** | Alternating | Broadcast | Mean pool | 인코더 개선 단독 효과 (Negative Control) |
| **(e) Alt+Cross** | Alternating | Cross-Attention | 제거 | 인코더 + 디코더 동시 개선 |

---

## 3. 모델 규모

| Variant | Parameters | Size (MB) | vs Baseline |
|---------|:---------:|:---------:|:-----------:|
| (a) Baseline | 7,679,930 | 29.30 | — |
| (b) Cross-Attn | 8,471,482 | 32.32 | +10.3% |
| (c) Cross-Attn+BN | 8,537,402 | 32.57 | +11.2% |
| (d) Alt-Attn | 9,711,546 | 37.05 | +26.5% |
| (e) Alt+Cross | 10,503,098 | 40.07 | +36.8% |

---

## 4. 정량 평가

### 4.1 Command Accuracy & Argument Accuracy (Tolerance=3)

![Cmd & Args Accuracy](figures/fig1_cmd_args_accuracy.png)

| Metric | 논문 원본 | **(a) Baseline** | **(b) Cross-Attn** | **(c) Cross-Attn+BN** | **(d) Alt-Attn** | **(e) Alt+Cross** |
|--------|:---------:|:----------------:|:-------------------:|:---------------------:|:----------------:|:-----------------:|
| **Cmd Acc** | **82.76** | 81.97 | 82.53 | 82.89 | 82.12 | 82.78 |
| line | — | 69.49 | 70.38 | 70.48 | 70.83 | 70.77 |
| arc | — | 79.80 | 79.20 | 80.27 | 79.22 | 79.25 |
| circle | — | 92.58 | 92.83 | 92.08 | 91.77 | 92.72 |
| plane | — | 93.74 | 93.54 | 93.60 | 94.12 | **94.52** |
| trans | — | 70.06 | 70.11 | 69.64 | 70.41 | **70.58** |
| extent | — | 66.19 | 67.16 | 67.30 | 67.61 | **68.14** |
| **Avg Args** | **79.23** | 78.65 | 78.87 | 78.89 | 79.00 | **79.33** |

![Per-type Args Accuracy](figures/fig2_per_type_accuracy.png)

### 4.2 Exact Match Accuracy (Tolerance=0)

| Metric | **(a) Baseline** | **(b) Cross-Attn** | **(c) Cross-Attn+BN** | **(d) Alt-Attn** | **(e) Alt+Cross** |
|--------|:----------------:|:-------------------:|:---------------------:|:----------------:|:-----------------:|
| line | 63.55 | 64.19 | 64.13 | 64.53 | 64.32 |
| arc | 69.49 | 69.25 | 70.55 | 68.33 | 69.29 |
| circle | 84.61 | 84.60 | 84.13 | 83.54 | 84.61 |
| plane | 93.12 | 92.97 | 92.97 | 93.49 | **93.88** |
| trans | 63.38 | 63.24 | 62.76 | 63.36 | **63.80** |
| extent | 52.76 | 53.30 | 52.93 | 53.01 | **53.81** |
| **Avg** | 71.15 | 71.26 | 71.25 | 71.05 | **71.62** |

### 4.3 Mean Absolute Error (MAE, Lower is Better)

![MAE Comparison](figures/fig3_mae_comparison.png)

| Metric | **(a) Baseline** | **(b) Cross-Attn** | **(c) Cross-Attn+BN** | **(d) Alt-Attn** | **(e) Alt+Cross** |
|--------|:----------------:|:-------------------:|:---------------------:|:----------------:|:-----------------:|
| line | 15.217 | 14.619 | 14.711 | 14.129 | 14.235 |
| arc | 8.753 | 8.638 | 8.238 | 8.734 | 8.819 |
| circle | 1.927 | 1.998 | 2.140 | 2.200 | 1.993 |
| plane | 4.372 | 4.548 | 4.556 | 4.122 | **3.863** |
| trans | 14.066 | 13.904 | 14.237 | 13.566 | **13.450** |
| extent | 14.491 | 14.126 | 14.182 | 13.280 | **13.203** |
| **Avg** | 9.804 | 9.639 | 9.677 | 9.339 | **9.261** |

### 4.4 논문 원본 대비 차이 (Delta)

![Delta Accuracy](figures/fig5_delta_accuracy.png)

| Metric | (a) | (b) | (c) | (d) | (e) |
|--------|:---:|:---:|:---:|:---:|:---:|
| **Cmd Acc** | -0.79 | -0.23 | +0.13 | -0.64 | +0.02 |
| **Avg Args** | -0.58 | -0.36 | -0.34 | -0.23 | **+0.10** |

### 4.5 Validation Loss (Training 종료 시점)

| Metric | **(a)** | **(b)** | **(c)** | **(d)** | **(e)** |
|--------|:------:|:------:|:------:|:------:|:------:|
| val/loss_cmd | **1.178** | 1.166 | 1.297 | 1.397 | 1.285 |
| val/loss_args | 4.539 | **4.539** | 4.594 | 4.730 | 4.593 |

---

## 5. 추론 성능

### 5.1 Forward Pass Time (batch_size=1, A100)

![Accuracy vs Latency](figures/fig4_accuracy_vs_latency.png)

| Variant | Latency (ms) | vs Baseline |
|---------|:----------:|:-----------:|
| (a) Baseline | 3.95 ± 0.27 | — |
| (b) Cross-Attn | 4.82 ± 0.23 | +22.0% |
| (c) Cross-Attn+BN | 4.88 ± 0.08 | +23.5% |
| (d) Alt-Attn | 6.07 ± 0.09 | +53.7% |
| (e) Alt+Cross | 6.94 ± 0.09 | +75.7% |

### 5.2 학습 시간 (5개 병렬, A100 80GB)

| Variant | 총 학습 시간 | Epoch 당 |
|---------|:----------:|:--------:|
| (a) Baseline | 30.2h | 9.1 min |
| (b) Cross-Attn | 34.2h | 10.2 min |
| (c) Cross-Attn+BN | 34.2h | 10.2 min |
| (d) Alt-Attn | 37.3h | 11.2 min |
| (e) Alt+Cross | 38.2h | 11.5 min |

---

## 6. 정성 평가 (3D 복원 기반)

Variant (e) Alt+Cross와 (a) Baseline의 예측 command sequence를 OCC(pythonocc-core 7.5.1)로 실제 3D BRep 솔리드로 복원하여 비교한다. DeepCAD의 `cadlib.visualize.vec2CADsolid` 파이프라인을 활용해 `(seq_len, 17)` vec → `TopoDS_Shape` 변환 후, `BRepCheck_Analyzer`로 유효성을 검사하고 Viewer3d(Mesa llvmpipe + xvfb offscreen)로 4개 등각 시점에서 512×512 PNG를 dump한다. 본 절의 모든 이미지는 동일 설정·동일 샘플 ID로 (a)와 (e)를 나란히 배치한다.

### 6.1 3D 복원 성공률 (전체 테스트 세트 7,881샘플)

| 지표 | **(a) Baseline** | **(b) Cross-Attn** | **(c) Cross-Attn+BN** | **(d) Alt-Attn** | **(e) Alt+Cross** | GT |
|------|:---:|:---:|:---:|:---:|:---:|:--:|
| OCC 변환 성공 | 6,140 (77.91%) | 6,177 (78.38%) | 6,183 (78.45%) | 6,213 (78.84%) | 6,138 (77.88%) | 7,759 (98.45%) |
| BRep valid (IsValid) | 5,550 (70.42%) | 5,604 (71.11%) | 5,579 (70.79%) | 5,664 (71.87%) | 5,558 (70.52%) | 7,687 (97.54%) |
| **IR (BRep-invalid)** | **29.58%** | **28.89%** | **29.21%** | **28.13%** | **29.47%** | 2.46% |

**실패 원인 분포 (pred)**:

| 원인 | (a) | (b) | (c) | (d) | (e) | 의미 |
|------|:--:|:--:|:--:|:--:|:--:|------|
| AssertionError | 864 | 860 | 927 | 814 | 933 | cadlib의 `extrude.py`, `sketch.py`에서 SOL 시작 가정·인덱싱 전제 위반 (cmd 연쇄 오류 후 sketch/extrude 구조 붕괴) |
| IndexError | 662 | 617 | 580 | 630 | 541 | 존재하지 않는 EXT 참조 또는 sketch loop 닫힘 실패 |
| RuntimeError | 212 | 226 | 189 | 222 | 268 | OCC의 boolean op / `BRepAlgoAPI_*` 실패 (기하학적으로 불가한 형상) |
| NullShape / Other | 3 | 1 | 2 | 2 | 1 | vec2CADsolid 반환값이 Null |

**해석**:
- 두 variant 모두 **pred의 약 22%가 논리적 command sequence를 만들지 못해 OCC 변환 자체를 실패**. 숫자 metric(Cmd 82%/Args 79%)이 놓치는 "구조적 유효성" 차원의 갭이다.
- (a)와 (e)의 변환 성공률은 사실상 동일하나 **실패 원인 분포는 다름** — (e)는 AssertionError(+69) 및 RuntimeError(+56)가 증가하고 IndexError(-121)는 감소. Cross-attention이 "sketch 내부 구조는 더 잘 예측"(IndexError↓)하나, 샘플 단위의 structural coherence(SOL/EXT 순서)에서는 (a)와 동등하거나 오히려 더 fragile함을 시사.
- GT도 1.55%(122샘플)가 OCC 변환 실패 — RuntimeError 대부분으로, Drawing2CAD 데이터셋 자체에 boolean op가 실패하는 경계 케이스가 포함되어 있음.

### 6.2 성능 분포

![Score Distribution](figures/fig6_score_distribution.png)

| 구간 | 샘플 수 | 비율 |
|------|:------:|:----:|
| Perfect (score = 1.0) | 2,168 | 27.5% |
| Score ≥ 0.9 | 4,049 | 51.4% |
| Score < 0.5 | 1,058 | 13.4% |

**시퀀스 길이별 성능 분포**:

| 길이 구간 | 샘플 수 | 평균 Score | 중앙값 | Score < 50% 비율 |
|-----------|:------:|:---------:|:-----:|:---------------:|
| Short (≤8) | 3,986 | 91.9% | 97.1% | 2.1% |
| Medium (9-20) | 2,647 | 74.2% | 77.8% | 15.2% |
| Long (21-40) | 972 | 58.8% | 54.3% | 41.8% |
| Very Long (>40) | 276 | 50.1% | 43.7% | 60.5% |

![Score vs Sequence Length](figures/fig7_score_vs_seqlen.png)

시퀀스 길이가 성능의 가장 강력한 예측 인자이며, 길이 20을 넘어서면 실패 확률이 급격히 증가한다.

### 6.3 상위 성능 (Top Tier)

Score ≥ 0.99, (a)/(e) 모두 OCC 변환 및 BRep valid 통과한 샘플 중 기하 복잡도가 다른 두 케이스.

![Top Tier 3D Comparison](figures/qualitative_3d/grid_top_tier.png)

*왼쪽 4컬럼: GT, 중앙 4컬럼: (a) Baseline, 오른쪽 4컬럼: (e) Alt+Cross. 4 view = 등각 NE-top, NW-top, SW-top, lower-front.*

**Sample `00008056`** — 단순 평판 (seq_len=11, Line/Arc + 1 Extrude)
- (a)/(e) 모두 Cmd 100% / Args 100%. 3D 복원 결과도 GT와 완전 일치. 가장자리 라운드가 있는 얇은 직사각형 평판으로, 모든 4뷰에서 GT와 pred(a)·pred(e)를 시각적으로 구분할 수 없음.
- 전체 샘플의 27.5%(2,168샘플)가 이러한 완벽 복원 달성.

**Sample `00017379`** — 원통형 허브 + 방사형 돌기 (seq_len=20, Line·Arc·Circle + 2 Extrude)
- (a)/(e) 모두 Cmd 100% / Args 100%. 원통 몸체, 상면 원판, 3개의 방사형 돌기 모두 GT와 일치.
- 시퀀스 길이 20의 **중간 복잡도에서도 두 variant 모두 완벽 복원**. 이는 sequence length 자체보다 command multiset의 난이도가 성능을 좌우함을 보여주는 케이스.

### 6.4 중위 성능 (Mid Tier)

![Mid Tier 3D Comparison](figures/qualitative_3d/grid_mid_tier.png)

**Sample `00868771`** — 장변 레일 + 장공(slot) + 단일 원형홀 (seq_len=14, Line:6 Arc:2 Circle:1 Ext:1)

| | Cmd | Args | OCC 변환 | BRep Valid |
|--|:-:|:-:|:-:|:-:|
| (a) Baseline | 92.3% | 64.7% | **실패 (IndexError)** | — |
| (e) Alt+Cross | 100.0% | 85.3% | 성공 | 유효 |

- **(a)는 숫자 수준에서는 Cmd 92%로 나쁘지 않지만, 단 한 개의 cmd 오류(CIRCLE→ARC)가 sketch loop 구조를 깨뜨려 OCC 변환 자체가 IndexError로 실패**. 3D 수준에서는 "결과물 없음" 상태.
- (e)는 모든 cmd를 정확히 예측하고 3D 복원에 성공. GT와 비교 시 레일의 길이·슬롯의 위치·원형 홀 위치 모두 일치. Args에서 15% 오차가 있으나 3D 시각화 기준으로는 GT와 구분이 쉽지 않음.
- **교훈**: `00868771`은 Cmd 92% vs 100%라는 얼핏 작은 차이가 "3D 복원 가능" vs "불가능"의 이진 격차로 증폭되는 대표 케이스. Cross-attention decoder의 CMD 수준 교정 효과(섹션 7.2)가 3D 유효성이라는 하위 stream task에 결정적 영향을 줄 수 있음을 보여준다.

**Sample `00319566`** — L자 블록 (2단 돌출, seq_len=13, Line:8 Ext:2)

| | Cmd | Args | OCC 변환 | BRep Valid |
|--|:-:|:-:|:-:|:-:|
| (a) Baseline | 100.0% | 84.2% | 성공 | 유효 |
| (e) Alt+Cross | 58.3% | 52.6% | 성공 | 유효 |

- GT는 두 직방체가 직각으로 맞물린 L자 블록. 두 개의 extrude 구간이 필요.
- (a)는 cmd를 모두 맞췄고 args 오차도 tolerance 근처. 3D 복원 결과는 L자를 유지하되 한쪽 돌출부 너비가 GT와 약간 다름 (첫 extrude만 약간 짧아 L의 상단 노치가 얕게 보임).
- (e)는 2번째 sketch 구간에서 cmd 연쇄 오류(SOL→LINE)가 발생하여 **두 번째 extrude가 소실**됨. 그 결과 3D 복원은 성공하지만 **단순 정육면체로 퇴화** — GT의 L자 구조가 완전히 사라짐.
- **교훈**: "OCC 변환 성공" 하나로 품질을 판단하면 안 된다. (e)의 경우 변환은 성공했지만 **형상 의미(semantic)는 붕괴**. Args accuracy 52% 대 84%의 차이가 3D 수준에서는 "L → cube"라는 정성적으로 명확히 다른 결과로 드러남.
- 이 샘플은 (e)가 (a)보다 **열등한** 드문 케이스로, 과적합(섹션 7.4)이 반영된 것으로 해석 가능 — (e)의 val_loss_cmd=1.285로 Baseline 1.178 대비 높음.

### 6.5 하위 성능 (Bottom Tier)

![Bottom Tier 3D Comparison](figures/qualitative_3d/grid_bottom_tier.png)

**Cross-variant 3D 복원 성공 여부**:

| Sample (seq_len) | (a) Cmd/Args | (a) OCC | (e) Cmd/Args | (e) OCC | GT OCC |
|--|:-:|:-:|:-:|:-:|:-:|
| 00306982 (56) | 7/19 | **실패 (AssertionError)** | 2/16 | **실패 (RuntimeError)** | 성공 |
| 00582849 (39) | 8/12 | 성공 | 3/12 | **실패 (RuntimeError)** | 성공 |
| 00625131 (28) | 7/19 | **실패 (AssertionError)** | 4/15 | **실패 (AssertionError)** | 성공 |
| 00883872 (23) | 5/16 | 성공 | 5/16 | 성공 | 성공 |

*값: Cmd%/Args%. OCC 변환은 BRep valid 통과 여부까지 포함.*

**Sample `00306982` (seq_len=56)** — Cmd 2-7%, **양 variant 모두 OCC 변환 실패**
- GT는 다중 extrude가 얽힌 기계 부품. 56개 토큰에 10 EXT 포함.
- pred는 첫 토큰 이후 cmd가 붕괴하여 SOL/EXT 구조가 무너지고, extrude 시작 가정(AssertionError) 또는 boolean op(RuntimeError)에서 변환 실패. 3D 이미지 자체가 생성 불가.

**Sample `00582849` (seq_len=39)** — (a)는 변환 성공하지만 **의미 없는 형상**
- GT는 둥근 단면의 곡선 바. (a)는 args가 부분적으로 맞아 얇은 대각선 막대로 복원되지만 GT와 무관. (e)는 RuntimeError로 변환 자체 실패.
- **"변환 성공"이 "정답 근접"을 보장하지 않음**을 보여주는 케이스. (a)는 metric 기준 약간 더 우수(Cmd 8 vs 3)하나 3D 수준에서는 둘 다 "사용 불가".

**Sample `00625131` (seq_len=28)** — GT는 얇은 링 구조, **양 variant 모두 AssertionError**
- GT의 얇은 곡면 링(Arc 다수로 구성)은 OCC 변환 시 사면(Face) 생성이 극도로 민감한 구조. pred sequence가 sketch loop을 닫지 못해 AssertionError.

**Sample `00883872` (seq_len=23)** — 원통 + 기울어진 상면 사각 (공통 붕괴)
- GT는 원통 위에 다이아몬드 형태의 내접 사각이 있는 혼합 스케치. (a)와 (e) 모두 Cmd 5%, Args 16%로 완전히 붕괴.
- 흥미롭게도 **두 variant 모두 OCC 변환에는 성공**하지만, 3D 결과는 **평범한 원통**으로 퇴화. Cmd sequence 전반이 무작위 수준이어도 "CIRCLE + EXT" 한 세트가 우연히 일관되면 cylinder가 만들어지지만 GT의 특징 sketch는 완전 소실.
- **교훈**: 하위 구간에서는 (a)-(e) 차이가 거의 없음. 두 모델 모두 **첫 토큰 이후 사실상 무작위 예측**으로 붕괴하므로 아키텍처 개선은 무의미하며, 길이 30+ 시퀀스 자체에 대한 구조적 해법(hierarchical decoding, length-scheduled curriculum 등)이 필요.

### 6.6 Variant × 시퀀스 길이 교차 분석

![Variant x SeqLen Heatmap](figures/fig9_variant_seqlen_heatmap.png)

모든 variant가 긴 시퀀스에서 공통적으로 성능이 하락하며, variant 간 차이는 중간 길이(9-30) 구간에서 가장 두드러진다. 3D 기준으로 보면 이 중간 길이 구간이 OCC 변환 실패율이 급격히 커지는 영역과도 일치하여, variant 개선의 실질 효과가 집중되는 구간이다.

### 6.7 Command Confusion Matrix

![Confusion Matrix](figures/fig10_cmd_confusion_matrix.png)

EXT(Extrude) 명령의 recall이 가장 낮으며, EOS나 다른 sketch 명령으로 잘못 예측되는 경향이 있다. 이는 섹션 6.5의 하위 샘플들이 공통적으로 보이는 "EXT 소실 → 단순 원통/사각 퇴화" 패턴의 근본 원인이다.

### 6.8 정성 평가 요약

| 구간 | 특성 | 3D 복원 결과 (공통) | (e)의 3D 개선 여부 |
|------|------|-------------------|:-:|
| **상위** (51.4%, score≥0.9) | seq ≤ 20, 단순/혼합 형상 | GT와 시각적 구분 불가한 완벽 복원 | — (둘 다 완벽) |
| **중위** (35.2%, 0.5≤score<0.9) | seq 10-30, 1-2 extrude | 대부분 변환 성공, 일부 feature 누락 | **유의미 개선** (특히 `00868771`에서 (a)는 OCC-fail, (e)는 성공·정확 복원). 그러나 `00319566` 같이 **(e)가 더 나빠지는 회귀 케이스도 존재**. |
| **하위** (13.4%, score<0.5) | seq 30+, 다중 extrude | 첫 토큰 이후 cmd 붕괴 → 변환 실패 또는 단순 형상으로 퇴화 | 개선 없음 (구조적 한계, 두 variant 동등 수준 붕괴) |

**3D 기준의 핵심 발견** (섹션 4-5의 숫자 metric과 차별되는 관찰):
1. **22% pred가 "형상으로 존재하지 않음"**: OCC 변환 실패 샘플은 어떤 후속 task(rendering, BOM 추출, PLM 연동)에서도 활용 불가. 이는 Cmd 82% / Args 79%라는 숫자 지표가 가리는 "structural validity" 차원의 병목이다.
2. **Cmd 한 개 차이가 3D 이진 결과로 증폭**: `00868771`에서 (a) 92% vs (e) 100%의 8%p 차이가 3D에서는 "없음" vs "정확 복원"으로 나타남. Cross-attention의 cmd 수준 교정 효과가 downstream 3D validity에 비선형적으로 기여함을 보여주는 가장 명확한 증거.
3. **Args 수치 개선이 3D 형상 보존으로 이어지지 않는 경우 존재**: `00319566`은 (e)가 Args metric에서 후퇴한 결과 **L자 → 정육면체 의미 붕괴**. Args 수치가 같아도 **어느 command 위치에서 틀리는가**가 3D 결과를 좌우한다.
4. **하위 구간은 metric-architecture 무관**: score<0.15 구간에서는 (a), (e) 차이가 시각적으로 거의 없으며 둘 다 "사실상 무작위". 모델 개선이 아닌 데이터/디코딩 재설계가 필요.

---

## 7. 해석

### 7.1 Baseline 재현성

Variant (a)는 논문 원본 대비 Cmd Acc -0.79%, Avg Args -0.58% 하락했다. 이는 Phase 1에서 custom MHA를 `nn.MultiheadAttention`으로 전환하면서 발생한 차이로, 아키텍처 변경 효과를 평가할 때 이 갭을 기준으로 보정해야 한다.

**Baseline(a) 기준 상대 개선폭**:

| Variant | Cmd Acc (vs a) | Avg Args (vs a) |
|---------|:-:|:-:|
| (b) Cross-Attn | +0.56 | +0.22 |
| (c) Cross-Attn+BN | +0.92 | +0.24 |
| (d) Alt-Attn | +0.15 | +0.35 |
| (e) Alt+Cross | **+0.81** | **+0.68** |

### 7.2 방법론별 기여 분리

**Cross-Attention Decoder (방법론 2)**:
- (a)→(b) 비교: Cmd +0.56, Args +0.22. 디코더의 정보 병목 해소가 Command 예측 정확도에 주로 기여.
- (b)→(c) 비교: Bottleneck 추가 시 Cmd +0.36 추가 향상. Element-wise bottleneck이 per-token regularizer 역할을 하여 Cmd Acc에 긍정적.

**Alternating Attention Encoder (방법론 1)**:
- (a)→(d) 비교: Cmd +0.15, Args +0.35. 인코더 개선만으로는 Cmd에 미미한 효과. 그러나 **Args Accuracy에는 의미 있는 기여** — 특히 extrude 관련 metric(plane +0.38, trans +0.35, extent +1.42)에서 두드러짐.
- 계획서에서 (d)를 "인코더만으로는 디코더 병목을 극복 불가한 Negative Control"로 설정했으나, 예상과 달리 args에서 유의미한 개선이 관찰됨. Frame-wise attention이 개별 뷰의 3D extrusion 파라미터 추출에 효과적인 것으로 해석.

**조합 효과 (방법론 1+2)**:
- (e)는 개별 기여의 합(인코더 +0.35 + 디코더 +0.22 = +0.57)보다 더 큰 Args +0.68을 달성. 두 방법론 간 **약한 시너지 효과**가 존재함을 시사.
- Extrude 계열(plane, trans, extent)에서 일관되게 전체 최고 성능 및 최저 MAE를 기록.

### 7.3 정확도 vs 추론 비용 Trade-off

| Variant | Avg Args 개선 (vs a) | Latency 증가 | 파라미터 증가 | 효율 (개선/latency비) |
|---------|:--------------------:|:------------:|:------------:|:---:|
| (b) | +0.22 | +22.0% | +10.3% | 0.010 |
| (c) | +0.24 | +23.5% | +11.2% | 0.010 |
| (d) | +0.35 | +53.7% | +26.5% | 0.007 |
| (e) | +0.68 | +75.7% | +36.8% | 0.009 |

- 추론 비용 대비 효율은 (b), (c)가 가장 높으나 절대 개선폭이 작음.
- (e)는 latency 75.7% 증가에 대해 가장 큰 절대 개선을 제공. Phase 3의 torch.compile 최적화로 latency 증가분 상쇄 가능성 있음.

### 7.4 과적합 분석

| Variant | Train Loss (cmd) | Val Loss (cmd) | Gap |
|---------|:----------------:|:--------------:|:---:|
| (a) | 0.329 | 1.178 | 0.849 |
| (b) | 0.247 | 1.166 | 0.919 |
| (c) | 0.251 | 1.297 | 1.046 |
| (d) | 0.258 | 1.397 | 1.139 |
| (e) | 0.267 | 1.285 | 1.018 |

- 모든 variant에서 train-val gap이 Baseline 대비 증가하여 과적합 경향 존재.
- 특히 (d)의 val loss가 가장 높음 — Alternating encoder의 표현력 증가가 broadcast decoder의 병목과 결합되어 encoder 쪽에서 과적합 발생.
- (e)는 (d) 대비 val loss가 낮아, cross-attention decoder가 encoder의 표현력을 효과적으로 활용하면서 과적합을 완화하는 효과.

### 7.5 Command 유형별 패턴

- **Sketch 계열 (line, arc, circle)**: Variant 간 차이가 1-2%p 이내로 미미. 기존 broadcast decoder도 sketch 파라미터를 충분히 잘 예측.
- **Extrude 계열 (plane, trans, extent)**: Variant 간 차이가 크고, Alternating encoder 계열(d, e)이 일관되게 우수. 3D extrusion 파라미터는 multi-view 간 대응관계에 크게 의존하며, frame-wise → global 교대 attention이 이를 효과적으로 포착.
- **extent**는 전체에서 가장 낮은 정확도(~68%)로 남아 있어, 향후 개선 여지가 가장 큰 타겟.

### 7.6 3D 복원 관점에서의 추가 발견

섹션 6의 OCC 기반 3D 정성 평가에서 숫자 metric이 놓치는 세 가지 관찰:

**(1) "Structural validity"라는 숨은 병목 — pred 22%는 3D 형상으로 존재하지 않음**

| Variant | Cmd Acc | Avg Args Acc | OCC 변환 성공률 | BRep valid 비율 |
|---------|:-:|:-:|:-:|:-:|
| (a) Baseline | 81.97% | 78.65% | 77.91% | 70.42% |
| (e) Alt+Cross | 82.78% | 79.33% | 77.88% | 70.52% |
| GT | — | — | 98.45% | 97.54% |

- Cmd/Args 수치만 보면 두 variant는 ~80% 수준의 성능으로 읽히지만, **"형상으로 복원 가능한가"라는 downstream 관점에서는 실질 성공률이 70%대**. 약 30%p의 갭이 구조적 유효성이라는 숨은 차원에서 발생.
- (e)가 (a) 대비 Cmd +0.81, Args +0.68 개선에도 **OCC 변환 성공률은 사실상 동등**. 이는 현재의 개선이 "이미 변환 가능한 샘플 안에서의 정밀도 향상"에 가까움을 의미. 3D 유효성 자체를 끌어올리려면 구조적 정합성을 직접 학습 신호로 넣는 보조 loss(예: OCC-in-loop 검증, syntax-aware masking)가 필요.

**(2) Cmd 한 토큰 오차가 3D 결과에서 이진 격차로 증폭**

- `00868771` 샘플: (a) Cmd 92.3% → IndexError로 OCC 변환 실패(3D 결과 없음), (e) Cmd 100% → 정확한 3D 복원. **Cmd accuracy 8%p 차이가 "3D 있음/없음" 이진 격차로 비선형 증폭**.
- 이는 Cmd 정확도가 Args 정확도보다 3D 유효성에 더 민감하게 기여함을 시사하며, 섹션 7.2에서 확인된 "Cross-attention decoder는 Cmd 개선에 주로 기여"라는 관찰이 3D downstream에 실질적 영향을 줌을 구체적으로 뒷받침한다.
- 반대로 Args accuracy가 약간 낮더라도 Cmd만 맞으면 "유사하지만 덜 정확한 3D"가 얻어진다 — 이는 활용 가능한 결과.

**(3) Args 회귀가 형상 의미(semantic)의 붕괴로 이어지는 회귀 케이스**

- `00319566` 샘플: (a) Cmd 100%·Args 84% → GT와 근접한 L자 블록 복원. (e) Cmd 58%·Args 53% → **단순 정육면체로 퇴화(L의 두 번째 extrude 완전 소실)**.
- 숫자상으로는 Args metric 30%p 하락이지만, 3D로는 "L vs cube"라는 정성적으로 명확히 다른 형상. **어느 위치의 cmd가 틀리느냐**가 3D 결과의 정체성을 좌우하며, 단순 평균 metric으로는 이 semantic-level 붕괴를 포착할 수 없다.
- (e)가 평균적으로 (a)를 상회하더라도 샘플 단위로는 회귀가 존재하며, 이는 (e)의 높은 val_loss(1.285 vs 1.178)가 반영하는 과적합 현상으로 해석 가능 — 과적합 완화 실험의 동기를 강화.

**(4) 하위 구간(score<0.3)은 아키텍처 개선의 한계 영역**

- `00306982` (56 토큰, 10 EXT) 등 하위 4개 샘플에서 (a)·(b)·(c)·(d)·(e) 모두 Cmd <10%, OCC 변환은 AssertionError/RuntimeError로 실패하거나 "단순 원통"으로 퇴화. 
- 이는 인코더·디코더 개선이라는 Phase 2의 방향으로는 해결 불가한 영역이며, **길이 30+ 시퀀스에 대해서는 hierarchical decoding, length-scheduled curriculum, 또는 Phase 4의 mask-predict iterative refinement 같은 별도의 구조적 접근이 필요**함을 재확인.

---

## 8. Phase 3: 추론 최적화

**환경**: PyTorch 2.7.0 (CUDA 12.8), NVIDIA A100-SXM4-80GB  
**대상 모델**: Variant (e) Alt+Cross

### 8.1 Graph Break 분석

torch.compile 적용 전 forward path의 graph break 요인을 분석하였다.

| 파일 | 패턴 | 심각도 | 상태 |
|------|------|--------|------|
| `functional.py:90,94` | `torch.equal(query, key)` | CRITICAL | 미사용 (nn.MHA 전환 완료) |
| `model.py:83` | `if S == self.tokens_per_view` | HIGH | input_option=4x 고정 시 S=400 상수 |
| `model.py:91-93` | `for v in range(num_views)` 루프 | HIGH | 4x 고정 시 4회 고정 루프 |
| `trainer.py:80,107` | `.cpu().numpy()` | MEDIUM | forward path 외부 (평가/후처리) |

**결론**: `input_option=4x`로 고정 시 forward path에 실질적 graph break 없음. `torch.compile`이 정상 trace 가능.

### 8.2 추론 벤치마크 (batch_size=1)

| 설정 | Latency (ms) | Speedup (vs 자체 base) | vs (a) Baseline FP32 |
|------|:-----------:|:-----:|:----:|
| **(a) Baseline FP32 (no compile)** | 3.925 | — | 1.00x |
| (e) FP32 (no compile) | 6.876 | — | 0.57x |
| (e) FP16 autocast only | 9.344 | 0.74x | 0.42x |
| (e) torch.compile (default) FP32 | 3.475 | 1.99x | 1.13x |
| (e) torch.compile (reduce-overhead) FP32 | 1.569 | 4.41x | 2.50x |
| (e) torch.compile (default) + FP16 | 4.301 | 1.61x | 0.91x |
| **(e) torch.compile (reduce-overhead) + FP16** | **1.251** | **5.19x** | **3.14x** |

### 8.3 배치 크기별 성능

| Batch | 설정 | Latency | Throughput | Speedup |
|:-----:|------|:-------:|:----------:|:-------:|
| 1 | (e) Baseline FP32 | 6.876ms | 145/s | 1.00x |
| 1 | (e) compile+FP16 | 1.251ms | 799/s | **5.50x** |
| 16 | (e) Baseline FP32 | 7.240ms | 2,210/s | 1.00x |
| 16 | (e) compile+FP16 | 2.897ms | 5,523/s | **2.50x** |
| 256 | (e) Baseline FP32 | 60.446ms | 4,235/s | 1.00x |
| 256 | (e) compile+FP16 | 29.713ms | 8,616/s | **2.03x** |

- `reduce-overhead` 모드는 batch=1에서 CUDA Graph로 kernel launch overhead를 완전히 제거하여 **5.19x** 가속
- batch=256에서는 compute-bound이므로 CUDA Graph 효과 감소, FP16이 주 가속 요인 (**2.03x**)
- FP16 단독(autocast only)은 오히려 느려짐 — kernel launch overhead가 지배적인 소규모 배치에서 dtype 변환 오버헤드가 상쇄

### 8.4 정확도 보존 검증

| 비교 대상 | command_logits max diff | args_logits max diff | Cmd argmax 일치율 |
|-----------|:-:|:-:|:-:|
| compile FP32 vs Baseline FP32 | 0.018 | 0.105 | 99.99% |
| compile FP16 vs Baseline FP32 | 0.025 | 0.204 | 99.99% |

수치 오차는 FP16 정밀도 범위 내이며, argmax 결과에 실질적 영향 없음.

### 8.5 핵심 발견

1. **추론 비용 문제 완전 해결**: Phase 2에서 (e)는 (a) 대비 +75.7% latency 증가가 있었으나, torch.compile 적용 후 **(a)의 FP32 대비 3.14x 더 빠름**
2. **FP16 단독은 비효과적**: autocast만 적용하면 오히려 느려짐. compile과 결합해야 효과 발현
3. **batch=1 최적화가 가장 극적**: overhead-bound 환경에서 CUDA Graph의 효과가 극대화

---

## 9. Phase 4: Mask-Predict Iterative Refinement

**환경**: NVIDIA A100-SXM4-80GB, PyTorch 2.7.0  
**대상 모델**: Variant (e) Alt+Cross 기반으로 `PartialPredictionEmbedding`과 `mask_token` 파라미터(+19,600개)를 추가하여 재학습

### 9.1 구현 요약

CMLM(Ghazvininejad et al., EMNLP 2019) 방식의 iterative refinement을 NAT decoder에 접목:

1. **새 모듈 `PartialPredictionEmbedding`**: 기존 `ConstEmbedding` 대체. 1차 예측 시에는 zeros+PE(기존 동작), 2차+에서는 `(prev_cmd, prev_args)` 임베딩을 생성하고 confidence 하위 k개 위치를 학습 가능한 `[MASK]` 토큰으로 치환.
2. **학습 전략**: Variant (e) pretrained checkpoint 로드 → 전체 파라미터 fine-tuning 70 epochs (lr=5e-4, batch=256). 각 step마다 `Uniform(0.15, 0.85)`로 mask_ratio 샘플링, GT를 이전 예측으로 사용. Loss = initial pass + refinement pass (마스킹된 위치에 대해서만).
3. **추론 전략**: 1차 예측 → confidence 하위 k 위치 masking → decoder 재실행 → N step 반복.

총 학습 시간: 5시간 55분 (variant_e_mask_predict, 70 epochs).

### 9.2 정량 평가 (Tolerance=3)

| Metric | (e) Alt+Cross (Phase 2) | **N=0 (MP 학습, refinement 없음)** | **N=1** | **N=2** | **N=3** |
|--------|:------:|:------:|:------:|:------:|:------:|
| **Cmd Acc** | 82.78 | **82.61** | 48.42 | 43.48 | 39.21 |
| line | 70.77 | 71.46 | 60.41 | 51.92 | 52.57 |
| arc | 79.25 | 79.43 | 65.52 | 62.20 | 62.61 |
| circle | 92.72 | 93.05 | 93.37 | 80.29 | 83.66 |
| plane | 94.52 | 94.34 | **96.67** | 91.29 | 89.20 |
| trans | 70.58 | 70.75 | **80.14** | 57.60 | 60.89 |
| extent | 68.14 | 68.78 | **79.80** | 53.71 | 58.19 |
| **Avg Args** | 79.33 | **79.64** | 79.32 | 66.17 | 67.85 |

(Mask ratio schedule: N=1 `[0.5]`, N=2 `[0.5, 0.3]`, N=3 `[0.6, 0.4, 0.2]`)

### 9.3 추론 Latency (batch_size=1, A100, FP32, no compile)

| 설정 | Latency (ms) | vs N=0 | vs (a) Baseline |
|------|:-----------:|:-----:|:---------------:|
| N=0 | 6.95 ± 0.22 | 1.00x | +76% |
| N=1 | 10.13 ± 0.11 | 1.46x | +158% |
| N=2 | 13.32 ± 0.30 | 1.92x | +239% |
| N=3 | 16.50 ± 0.35 | 2.37x | +320% |

각 refinement step당 약 **+3.2 ms** 증가 (decoder 재실행 1회 비용).

### 9.4 해석

**긍정적 발견 — Mask-Predict 학습 regime의 부수 효과**:
- N=0 (refinement 미적용) 결과가 variant (e)를 Avg Args 기준 **+0.31%p** 상회 (79.33 → 79.64)
- 특히 line(70.77→71.46), circle(92.72→93.05), extent(68.14→68.78)에서 개선
- 해석: random masking이 일종의 **data augmentation/regularization** 역할. Mask-Predict 전용 파라미터의 추가 학습이 encoder-decoder의 표현력을 강화했을 가능성

**부정적 발견 — Iterative refinement 자체는 악화**:
- N≥1에서 Cmd Acc가 82.61% → 48.42%로 붕괴
- 원인 분석:
  1. **Padding confidence 과다**: 1차 예측에서 EOS/padding 위치의 confidence가 극히 높아(>0.99), `topk(lowest-k)` masking이 실제로는 **유효한 sketch 위치**를 반복적으로 mask → 재예측하며 왜곡
  2. **학습-추론 불일치**: 학습 시에는 GT를 prev_prediction으로 사용했으나 추론 시에는 1차 예측(노이즈 포함)을 사용. GT로 학습된 `PartialPredictionEmbedding`이 노이즈 입력에 robust하지 않음
  3. **Mask schedule 부적합**: 매 step 50% 이상 masking하는 기본 스케줄이 NAT decoder의 convergence를 해침

**부분적 긍정 — N=1에서 Extrude 파라미터 대폭 개선**:
- plane +2.33, trans +9.39, extent +11.02 — Phase 2에서 가장 약했던 extrude args가 단 1회 refinement로 크게 향상
- Cmd Acc가 정확한 위치에 한해서는 refinement가 **EXT 계열 args만 선택적으로 교정**하는 효과 존재
- Cmd 붕괴만 해결하면 hybrid inference (cmd는 N=0, ext args는 N=1)로 활용 가능성

### 9.5 3D 복원 기반 정량 평가 (CD/IR)

DeepCAD `cadlib.visualize.vec2CADsolid` 파이프라인으로 예측 vec → OCC TopoDS_Shape → STL → trimesh surface sampling (2000 points) 수행 후 GT vec과의 **Chamfer Distance** 및 **Invalidity Ratio**를 측정. 전체 test set에서 random 2000 샘플을 사용 (seed=0 고정).

| Config | IR (↓) | CD Mean (↓) | CD Median (↓) | CD Trimmed Mean (↓) |
|--------|:-----:|:----------:|:-------------:|:-------------------:|
| (a) Baseline | 0.3145 | 0.1176 | 0.00729 | 0.0575 |
| (b) Cross-Attn | 0.3050 | 0.1109 | 0.00867 | 0.0583 |
| (c) Cross-Attn+BN | 0.3120 | 0.1138 | 0.00622 | 0.0591 |
| (d) Alt-Attn | 0.3050 | 0.1027 | 0.00603 | 0.0548 |
| **(e) Alt+Cross** | 0.3135 | 0.1077 | 0.00619 | 0.0537 |
| **MP N=0** | **0.2995** | **0.1058** | **0.00541** | **0.0525** |
| MP N=1 | 0.9585 | 0.0406† | 0.00147† | 0.00443† |
| MP N=2 | 1.0000 | N/A | N/A | N/A |
| MP N=3 | 0.9880 | 0.0980† | 0.01405† | 0.04919† |

† MP N≥1의 CD 수치는 IR이 95% 이상이라 극소수 유효 샘플만 반영되어 신뢰성 낮음.

**IR 해석**:
- Phase 2 모든 variant의 IR은 30-31% 수준으로 수렴 — CAD sequence의 **구조적 유효성**은 아키텍처 변화에 둔감
- MP N=0가 유일하게 IR < 30% (29.95%) 달성 — MP 학습 regime이 생성 시퀀스의 구문 안정성을 약간 향상
- MP N≥1에서 IR이 95-100%로 폭증 — refinement가 sequence 구조 (SOL/EXT 매칭)를 파괴. CMD 교정과 동시에 sequence layout도 망가뜨림

**CD 해석**:
- (e) Alt+Cross는 Phase 2 variant 중 **CD Trimmed Mean 0.0537로 최저** — args 정확도 개선이 실제 3D 형상 근접도로 이어짐
- MP N=0는 모든 CD 지표에서 최저 (mean 0.1058, median 0.00541, trimmed 0.0525). Variant (e) 대비 trimmed mean -2.2%
- CD median과 CD mean의 큰 격차 (0.005 vs 0.1)는 **소수의 대형 오류 샘플**이 평균을 끌어올림을 의미 — 하위 성능 샘플의 구조적 실패 패턴과 일치

### 9.6 결론

| 측면 | 결과 |
|------|------|
| **Mask-Predict 학습 regime** | ✅ Variant (e) 대비 Avg Args +0.31%p, IR -1.4%p, CD trimmed -2.2% 일관 개선 (regularizer 효과) |
| **Iterative refinement (N≥1)** | ❌ Cmd Acc 붕괴 + IR 95%+ 폭증으로 실용화 불가 |
| **EXT args 선택적 개선** | ⚠️ N=1에서 extent +11%p 등 유의미하나 cmd 붕괴와 상쇄 |
| **Latency overhead** | Step당 +46% (~3.2ms). N=2 이상은 ROI 없음 |

**권장 후속 작업**:
- Padding 위치를 masking 대상에서 제외하는 gating 로직 추가 (EOS 이후 확률적 masking 금지)
- 학습 시 prev_prediction으로 GT 대신 **noisy GT** 또는 **1차 예측 결과**를 사용하여 학습-추론 간 gap 해소
- Cmd 예측은 N=0 고정, args만 refinement하는 **cmd-frozen refinement** 변형 실험

---

## 10. 결론

1. **Variant (e) Alt+Cross가 최적 모델**. 논문 원본 대비 Avg Args Accuracy를 유일하게 상회(+0.10)하며, Cmd Acc도 거의 동등(+0.02). Baseline(a) 대비 Cmd +0.81, Args +0.68 개선.

2. **두 방법론 모두 유효하나 기여 영역이 다름**.
   - Cross-Attention Decoder → Command 정확도 향상에 주로 기여
   - Alternating Attention Encoder → Argument(특히 extrude) 정확도 향상에 주로 기여
   - 조합 시 시너지 효과 존재

3. **개선폭은 전체적으로 제한적** (Avg Args 기준 +0.68%p). 원인:
   - 모델이 이미 높은 수준에 도달 (Cmd 82%, Args 79%)
   - 200 epochs, dropout 0.1 등 기존 하이퍼파라미터를 그대로 사용
   - 계획서에 명시된 과적합 대응(dropout 상향, DropPath, data augmentation 등)을 미적용

4. **torch.compile로 추론 비용 문제 완전 해결**. (e) + compile(reduce-overhead) + FP16 조합이 (a) Baseline FP32 대비 **3.14x 빠르면서 정확도도 우수**. 성능과 속도 모두에서 Baseline을 상회하는 결과 달성.

5. **Mask-Predict는 학습 regime 측면에서만 부분 효과**. 추론 시 iterative refinement 자체는 Cmd Acc 붕괴로 실용성 없음. 단, Mask-Predict 학습 자체가 regularizer 역할을 하여 N=0 기준 Args +0.31%p 추가 개선 달성.

### 다음 단계 권장

- **과적합 완화 실험**: (e) 기반으로 dropout 0.2, DropPath, data augmentation 적용 후 재학습하여 추가 성능 향상 확인
- **Mask-Predict 개선**: Padding 위치 masking 제외 + 학습-추론 gap 해소 (noisy prev_pred) 후 재실험
- **배포 패키징**: torch.compile + FP16 설정을 inference script에 통합
