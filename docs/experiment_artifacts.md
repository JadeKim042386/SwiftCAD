# Drawing2CAD 실험 산출물 매니페스트

**작성일**: 2026-04-18
**작성자**: Artifact Inventory Agent
**베이스 경로**: `/home/work/Drawing2CAD`
**Git HEAD**: `51f8daf` — feat: add Phase 4 Mask-Predict refinement + OCC-based 3D qualitative/CD/IR evaluation

이 문서는 Drawing2CAD 프로젝트의 모든 실험 산출물(학습 체크포인트, 테스트 결과, 평가 JSON, 시각화, 로그, 스크립트, 코드 변경, 외부 로그)을 실측 기반으로 정리한 매니페스트입니다. 다른 에이전트(Timeline Writer, Master Report Writer)의 참조용 색인입니다.

---

## 1. 학습 체크포인트

모든 체크포인트 경로는 `/home/work/Drawing2CAD/proj_log/<exp_name>/model/` 기준. 각 variant는 `config.txt`(JSON)와 `model/latest.pth`(그리고 epoch 체크포인트 2개)를 가집니다.

### 1.1 Phase 2 variants — 3x (기존)

| Variant | 경로 | latest.pth 크기 | latest.pth 생성 | 보유 체크포인트 | encoder_type | decoder_type | use_bottleneck | input_option | nr_epochs | lr | batch_size |
|---|---|---|---|---|---|---|---|---|---|---|---|
| a_baseline | `proj_log/variant_a_baseline` | 92,002,552 B (87.7 MB) | 2026-04-14 13:30 | latest + epoch100 + epoch200 | standard | broadcast | false | 3x | 200 | 1e-3 | 256 |
| b_cross_attn | `proj_log/variant_b_cross_attn` | 101,519,928 B (96.8 MB) | 2026-04-15 03:03 | latest + epoch100 + epoch200 | standard | cross_attention | false | 3x | 200 | 1e-3 | 256 |
| c_cross_attn_bn | `proj_log/variant_c_cross_attn_bn` | 102,315,720 B (97.6 MB) | 2026-04-15 15:17 | latest + epoch100 + epoch200 | standard | cross_attention | **true** | 3x | 200 | 1e-3 | 256 |
| d_alt_attn | `proj_log/variant_d_alt_attn` | 116,746,296 B (111.3 MB) | 2026-04-15 21:15 | latest + epoch100 + epoch200 | alternating | broadcast | false | 3x | 200 | 1e-3 | 256 |
| e_alt_cross | `proj_log/variant_e_alt_cross` | 126,263,608 B (120.4 MB) | 2026-04-15 21:31 | **latest만** (epoch100/200 없음) | alternating | cross_attention | false | 3x | 200 | 1e-3 | 256 |

주: `variant_e_alt_cross`는 epoch 체크포인트가 저장되지 않고 latest만 존재. `train_logs/variant_e.log` 크기(2.6 MB)가 다른 3x 로그(~36 MB) 대비 작음 — 학습은 완료(progress.log의 `[DONE] Variant (e) Alt+Cross` 기록)되었으나 로그 streaming이 중간에 truncate된 것으로 보임.

### 1.2 Phase 2 variants — 4x (재학습)

모두 `num_workers=2`(3x는 8), 나머지 하이퍼파라미터 동일(nr_epochs=200, batch_size=256, lr=1e-3).

| Variant | 경로 | latest.pth 크기 | latest.pth 생성 | encoder_type | decoder_type | use_bottleneck |
|---|---|---|---|---|---|---|
| a_baseline_4x | `proj_log/variant_a_baseline_4x` | 92,310,584 B (88.0 MB) | 2026-04-17 04:04 | standard | broadcast | false |
| b_cross_attn_4x | `proj_log/variant_b_cross_attn_4x` | 101,827,960 B (97.1 MB) | 2026-04-17 07:58 | standard | cross_attention | false |
| c_cross_attn_bn_4x | `proj_log/variant_c_cross_attn_bn_4x` | 102,623,752 B (97.9 MB) | 2026-04-17 07:58 | standard | cross_attention | **true** |
| d_alt_attn_4x | `proj_log/variant_d_alt_attn_4x` | 116,746,296 B (111.3 MB) | 2026-04-17 11:08 | alternating | broadcast | false |
| e_alt_cross_4x | `proj_log/variant_e_alt_cross_4x` | 126,263,608 B (120.4 MB) | 2026-04-17 12:01 | alternating | cross_attention | false |

각 4x variant는 epoch100, epoch200, latest 세 체크포인트를 모두 보유. 로그상 모두 `EPOCH[199]`까지 도달 확인.

### 1.3 Phase 4 Mask-Predict

| 항목 | 값 |
|---|---|
| 경로 | `/home/work/Drawing2CAD/proj_log/variant_e_mask_predict` |
| latest.pth 크기 | 126,504,700 B (120.6 MB) |
| latest.pth 생성 | 2026-04-18 05:13 |
| encoder_type | alternating |
| decoder_type | cross_attention |
| use_mask_predict | **true** |
| n_refinement_steps (config) | 0 |
| mask_ratios (config) | "0.5,0.3" |
| freeze_pretrained | false |
| input_option | 4x |
| nr_epochs | 70 |
| lr | 5e-4 |
| batch_size | 256 |

사전학습 체크포인트 `proj_log/variant_e_alt_cross_4x/model/latest.pth` 로드 후 새 파라미터 5개(mask-predict embedding+mask token)만 추가 학습. 로그상 `EPOCH[69]` 도달 → 70 epoch 완주.

### 1.4 임시 MP 테스트 디렉토리 (`_test_mp*`)

모두 Mask-Predict 개발 중 생성한 스모크 테스트 artifact. 공통 설정: `nr_epochs=1`, `freeze_pretrained=true`, `use_mask_predict=true`, `encoder_type=alternating`, `decoder_type=cross_attention`, `input_option=4x`. **model/, log/ 디렉토리가 모두 비어 있음** (checkpoint 저장 전 중단된 디버그 실행).

| 디렉토리 | batch_size | num_workers | 생성 | 상태 |
|---|---|---|---|---|
| `_test_mp` | 4 | 0 | 2026-04-17 23:14 | 모델 없음 |
| `_test_mp2` | 2 | 0 | 2026-04-17 23:15 | 모델 없음 |
| `_test_mp3` | 2 | 0 | 2026-04-17 23:16 | 모델 없음 |
| `_test_mp4` | 2 | 0 | 2026-04-17 23:16 | 모델 없음 |
| `_test_mp5` | 2 | 0 | 2026-04-17 23:16 | 모델 없음 |
| `_test_mp6` | 2 | 0 | 2026-04-17 23:17 | 모델 없음 |
| `_test_mp7` | 4 | 0 | 2026-04-17 23:18 | 모델 없음 |

---

## 2. 테스트 결과 파일 (`test_results/*_vec.h5`)

각 `_vec.h5`는 `out_vec`, `gt_vec` 두 데이터셋(int32, shape `(seq_len, 17)` — cmd+16 args) 보유.

### 2.1 variant별 test_results

| Variant | 경로 | 파일 수 | 디렉토리 크기 |
|---|---|---|---|
| variant_a_baseline | `proj_log/variant_a_baseline/test_results` | **7881** | 40 MB |
| variant_a_baseline_4x | `proj_log/variant_a_baseline_4x/test_results` | **7881** | ~40 MB |
| variant_b_cross_attn | `proj_log/variant_b_cross_attn/test_results` | **7881** | ~40 MB |
| variant_b_cross_attn_4x | `proj_log/variant_b_cross_attn_4x/test_results` | **7881** | ~40 MB |
| variant_c_cross_attn_bn | `proj_log/variant_c_cross_attn_bn/test_results` | **7881** | ~40 MB |
| variant_c_cross_attn_bn_4x | `proj_log/variant_c_cross_attn_bn_4x/test_results` | **7881** | ~40 MB |
| variant_d_alt_attn | `proj_log/variant_d_alt_attn/test_results` | **7881** | ~40 MB |
| variant_d_alt_attn_4x | `proj_log/variant_d_alt_attn_4x/test_results` | **7881** | ~40 MB |
| variant_e_alt_cross | `proj_log/variant_e_alt_cross/test_results` | **0** (디렉토리 없음) | - |
| variant_e_alt_cross_4x | `proj_log/variant_e_alt_cross_4x/test_results` | **7881** | ~40 MB |

주: `variant_e_alt_cross`(3x)의 test_results는 애초에 생성되지 않음 → 3x 완전 비교를 위한 9개 분석에서 제외되는 이유.

### 2.2 variant_e_mask_predict (N별)

| N | 경로 | 파일 수 | mask_ratios |
|---|---|---|---|
| 0 | `proj_log/variant_e_mask_predict/test_results_n0` | **7881** | 0.5 (사용 안 함; n_steps=0) |
| 1 | `proj_log/variant_e_mask_predict/test_results_n1` | **7881** | 0.5 |
| 2 | `proj_log/variant_e_mask_predict/test_results_n2` | **7881** | 0.5, 0.3 |
| 3 | `proj_log/variant_e_mask_predict/test_results_n3` | **7881** | 0.6, 0.4, 0.2 |

---

## 3. 평가 결과 JSON

### 3.1 `docs/phase4_accuracy.json` (1.34 KB, 2026-04-18 10:51)

7881 샘플 × N={0,1,2,3}에 대한 Command Accuracy와 Args Accuracy(line/arc/circle/plane/trans/extent).

| N | cmd_acc | args_avg | line | arc | circle | plane | trans | extent |
|---|---|---|---|---|---|---|---|---|
| 0 | **0.8261** | **0.7964** | 0.7146 | 0.7943 | 0.9305 | 0.9434 | 0.7075 | 0.6878 |
| 1 | 0.4842 | 0.7932 | 0.6041 | 0.6552 | 0.9337 | 0.9667 | 0.8014 | 0.7980 |
| 2 | 0.4348 | 0.6617 | 0.5192 | 0.6220 | 0.8029 | 0.9129 | 0.5760 | 0.5371 |
| 3 | 0.3921 | 0.6785 | 0.5257 | 0.6261 | 0.8366 | 0.8920 | 0.6089 | 0.5819 |

핵심 관찰: **N=0이 cmd_acc 최고**(0.826), N>=1에서 cmd_acc 급락(refinement가 cmd에는 역효과). args 계열은 plane/trans/extent가 N=1에서 오히려 상승.

### 3.2 `docs/phase4_latency.json` (959 B, 2026-04-18 11:13)

50샘플 GPU inference latency (ms). `bench_mp_latency.py`로 측정.

| N | n_refinement_steps | mask_schedule | mean (ms) | std | median |
|---|---|---|---|---|---|
| 0 | 0 | null | **6.946** | 0.222 | 6.915 |
| 1 | 1 | [0.5] | 10.131 | 0.109 | 10.120 |
| 2 | 2 | [0.5, 0.3] | 13.324 | 0.302 | 13.276 |
| 3 | 3 | [0.6, 0.4, 0.2] | 16.497 | 0.353 | 16.416 |

각 refinement step당 약 +3.2 ms. N=0 대비 N=3은 2.37× 느림.

### 3.3 `docs/phase4_cd_ir.json` (4.28 KB, 2026-04-18 12:12)

2000 subset (랜덤 서브샘플)에 대한 Chamfer Distance / Invalidity Ratio. `eval_cd_ir.py`로 산출. 9개 config(5 variants 4x + MP×4 N).

| Config | n_valid | IR | cd_mean | cd_median | cd_trimmed_mean |
|---|---|---|---|---|---|
| variant_a_baseline_4x | 1371 | 0.3145 | 0.1176 | 0.00729 | 0.05746 |
| variant_b_cross_attn_4x | 1390 | 0.3050 | 0.1109 | 0.00867 | 0.05833 |
| variant_c_cross_attn_bn_4x | 1376 | 0.3120 | 0.1138 | 0.00622 | 0.05914 |
| variant_d_alt_attn_4x | **1390** | **0.3050** | **0.1027** | 0.00603 | 0.05484 |
| variant_e_alt_cross_4x | 1373 | 0.3135 | 0.1077 | 0.00619 | **0.05368** |
| mp_n0 | **1401** | **0.2995** | 0.1058 | **0.00541** | 0.05249 |
| mp_n1 | 83 | 0.9585 | 0.0406 | 0.00147 | 0.00443 |
| mp_n2 | 0 | **1.0000** | null | null | null |
| mp_n3 | 24 | 0.9880 | 0.0980 | 0.01405 | 0.04919 |

핵심 관찰: **mp_n0가 IR 0.2995로 최저**(가장 많이 유효). N>=1에서 대다수 샘플이 `IndexError`(pred:convert:IndexError)로 실패 → refinement가 유효 BRep 산출에는 오히려 치명적. mp_n0 cd_median(0.00541)이 전체 최저.

### 3.4 `docs/_cd_smoke.json` (431 B, 2026-04-18 12:09)

50 샘플 smoke test (variant_e_alt_cross_4x): n_valid=36, IR=0.28, cd_mean=0.0851, cd_median=0.00433. 정식 2000 평가 전 파이프라인 검증용.

---

## 4. 3D 정성 평가 산출물

경로: `/home/work/Drawing2CAD/docs/figures/qualitative_3d/`

### 4.1 전체 7881 샘플 IR 집계 JSON (success_variant_*.json)

`qualitative_eval.py --success-count` 모드. 각 variant의 전체 7881 샘플에 대해 vec→OCC 변환 성공률을 재평가하여 기록.

| 파일 | pred ok | pred ok_valid | pred fail | gt ok | gt ok_valid | 소요 시간 |
|---|---|---|---|---|---|---|
| success_variant_a_baseline_4x.json | 6140 | 5550 | 1741 | 7759 | 7687 | 138.4s |
| success_variant_b_cross_attn_4x.json | 6177 | 5604 | 1704 | 7759 | 7687 | 170.4s |
| success_variant_c_cross_attn_bn_4x.json | 6183 | 5579 | 1698 | 7759 | 7687 | 139.5s |
| success_variant_d_alt_attn_4x.json | **6213** | **5664** | 1668 | 7759 | 7687 | 137.8s |
| success_variant_e_alt_cross_4x.json | 6138 | 5558 | 1743 | 7759 | 7687 | 137.8s |

주: GT는 모든 variant에서 동일한 테스트셋이므로 GT 값이 모두 동일(7759/7687/122).

### 4.2 렌더링 JSON (render_variant_*.json)

샘플 8개 × 4 views(512×512) 렌더. variant_a와 e 각각 작성.

- `render_variant_a_baseline_4x.json`: 8 샘플 (00008056, 00017379, 00868771, 00319566, 00306982, 00582849, 00625131, 00883872), pred ok=5/8(convert_failed 3), gt ok=8/8. elapsed 1.41s.
- `render_variant_e_alt_cross_4x.json`: 동일 8 샘플. pred ok=5/8(convert_failed 3), gt ok=8/8. elapsed 1.40s.

### 4.3 `qualitative_eval_summary.json` (2026-04-18 12:18)

Top-level 집계. variant_b/c/d의 success-count 집계만 포함(최종 통합 버전). a/e는 개별 JSON 참조.

### 4.4 PNG 개별 이미지

- `variant_a_baseline_4x/`: **76개 PNG** (11 unique sample_id × 최대 8개 뷰 = 7×8+20 혼합; 실제로는 pred/gt × v0-v3 기준 최대 88 중 변환 실패분 제외)
  - 샘플 ID: 00000134, 00000392, 00000559, 00008056, 00017379, 00306982, 00319566, 00582849, 00625131, 00868771, 00883872
- `variant_e_alt_cross_4x/`: **76개 PNG** (동일 11 sample_id)

### 4.5 Grid PNG

| 파일 | 크기 | 용도 |
|---|---|---|
| `grid_top_tier.png` | 251 KB | Top tier(잘 된 것) 비교 그리드 |
| `grid_mid_tier.png` | 207 KB | Mid tier 비교 그리드 |
| `grid_bottom_tier.png` | 365 KB | Bottom tier(실패 케이스) 비교 그리드 |
| `grid_variant_a_baseline_4x.png` | 330 KB | variant_a 단독 그리드 |
| `grid_variant_e_alt_cross_4x.png` | 247 KB | variant_e 단독 그리드 |

### 4.6 Phase 2 정량 시각화 (`docs/figures/fig*.png`)

| 파일 | 크기 | 용도 |
|---|---|---|
| fig1_cmd_args_accuracy.png | 110 KB | Cmd/Args accuracy bar chart |
| fig2_per_type_accuracy.png | 73 KB | 명령 타입별 accuracy |
| fig3_mae_comparison.png | 62 KB | MAE 비교 |
| fig4_accuracy_vs_latency.png | 75 KB | accuracy/latency 산점도 |
| fig5_delta_accuracy.png | 60 KB | 델타 accuracy |
| fig6_score_distribution.png | 91 KB | score 분포 |
| fig7_score_vs_seqlen.png | 278 KB | score vs 시퀀스 길이 |
| fig8_qualitative_samples.png | 314 KB | 정성 샘플 |
| fig9_variant_seqlen_heatmap.png | 84 KB | seqlen × variant heatmap |
| fig10_cmd_confusion_matrix.png | 95 KB | 명령 confusion matrix |

---

## 5. 학습/테스트 로그 (`train_logs/`)

총 36개 파일, ~648 MB. 주요 파일:

### 5.1 Progress 로그 (요약 타임스탬프)

| 파일 | 크기 | 주요 이벤트 |
|---|---|---|
| `progress.log` | 609 B | `2026-04-14 02:00:00 Starting all 5 variants` → `2026-04-15 21:42:21 ALL 5 VARIANTS COMPLETED` (3x 순차 학습, ~43 시간) |
| `progress_4x.log` | 377 B | `2026-04-15 21:49:15 Starting 5 variants parallel (4x, num_workers=2)` → `2026-04-17 12:02:04 ALL VARIANTS FINISHED (4x)` (4x 병렬, ~38 시간) |
| `progress_mp.log` | 111 B | `2026-04-17 23:18:27 [START] Mask-Predict training (70 epochs)` → `2026-04-18 05:13:28 [DONE]` (~6 시간) |

### 5.2 학습 raw 로그

| 파일 | 크기 | 마지막 epoch |
|---|---|---|
| `variant_a.log` ~ `variant_d.log` | 각 ~36 MB | EPOCH[199] |
| `variant_e.log` | 2.59 MB | EPOCH[15] (로그 truncated; 학습은 완료) |
| `variant_a_4x.log` ~ `variant_e_4x.log` | 각 ~36 MB | EPOCH[199] |
| `mask_predict.log` | 11.7 MB | EPOCH[69] |
| `nohup.log` | 140 MB | 3x 학습 배경 실행 aggregate |
| `nohup_4x.log` | 172 MB | 4x 학습 배경 실행 aggregate |
| `nohup_mp.log` | 11.7 MB | MP 학습 배경 실행 aggregate |

### 5.3 테스트 실행 로그

- `test_variant_a.log` ~ `test_variant_e_4x.log` (각 ~6 KB) — 각 variant의 test.py stdout. 마지막 줄은 wandb 실행 URL (예: `https://wandb.ai/jujoo/Drawing2CAD/runs/2k2idbs7`).
- `test_variant_e.log`은 없음 (variant_e 3x test 미실행).

### 5.4 Mask-Predict 테스트 로그

| 파일 | 크기 | 내용 |
|---|---|---|
| `mp_test.log` | 739 B | N=0..3 순차 실행 START/DONE 타임스탬프. `2026-04-18 10:49:49` ~ `10:50:38` (~50초) |
| `mp_test_n0.log` ~ `mp_test_n3.log` | 각 ~5.9 KB | N별 test.py stdout, wandb URL 포함 |
| `mp_test_nohup.log` | 24 KB | 배경 실행 aggregate |

### 5.5 Phase 4 평가 로그

- `cd_ir.log` (25 KB) — `eval_cd_ir.py` 실행 로그. 9 config 순차 처리. 각 config당 2000 샘플 × n_points=2000 × n_jobs=8. OCC null-triangulation 경고 다수. 마지막: `Saved: /home/work/Drawing2CAD/docs/phase4_cd_ir.json`.
- `success_bcd.log` (3.5 KB) — variant_b/c/d에 대한 success-count 실행 로그. 중간 진행률 `[N/7881] pred_ok=X gt_ok=Y` 포맷. 마지막: `Top-level summary -> docs/figures/qualitative_3d/qualitative_eval_summary.json`.
- `success_mp.log` (483 B) — mp에 대한 success-count. `SKIP (not a directory)` 메시지 다수(디렉토리 명명 불일치).

### 5.6 주요 에러/경고 패턴

- **OCC Null Triangulation 경고** (`cd_ir.log`): `Warning: N faces have been skipped due to null triangulation` — OCC의 메시화에서 빈 페이스 발생. IR 계산에는 영향 없음(pred:convert 에러 taxonomy로 잡힘).
- **pred:convert:IndexError** — MP N≥1에서 압도적. mp_n1: 1831/2000, mp_n2: 1459/2000, mp_n3: 1822/2000. Mask-predict refinement가 vec→OCC 변환 단계의 인덱스 경계를 깨뜨림.
- **pred:convert:AssertionError** — 모든 variant에서 200~900 범위로 발생(variant_c가 247로 최다, mp_n0는 228).
- **wandb API 키 이슈 (14일 00:44)**: `wandb.errors.errors.UsageError: No API key configured`. 이후 `~/.netrc`로 해결됨.

---

## 6. Git 이력

```
51f8daf feat: add Phase 4 Mask-Predict refinement + OCC-based 3D qualitative/CD/IR evaluation  (HEAD, 182 files, +3009/-129)
8461e6f docs: add Phase 3 inference optimization results to report  (1 file, +66/-4)
c5566ce docs: add Phase 2 ablation study report with quantitative/qualitative evaluation  (16 files, +653/-34)
0ba7faa docs: add experiment plan report  (1 file, +372)
67701ea feat: alternating attention encoder + cross-attention decoder  (4 files, +406/-42)
08e2823 refactor: replace custom MHA with nn.MultiheadAttention for SDPA support  (3 files, +404/-9)
9f223ed feat: visualize  (2 files, +308)
f0b2b08 feat: lambdaLR  (2 files, +11/-3)
fee204a feat: defined model for d_model 132  (12 files, +1557/-3)
4b080bc feat: removed mlp from embedding  (2 files, +5/-5)
beff03d feat: update experiments
794a36b feat: added svg converter script and modified loss
835c5a8 feat: EMD loss
5424664 init commit
```

현재 작업 branch 상태: HEAD=51f8daf. 브랜치 정보 및 `git status`는 별도 조회 필요.

---

## 7. 스크립트 및 도구

### 7.1 학습/테스트 셸 스크립트 (`/home/work/Drawing2CAD/`)

| 스크립트 | 목적 | 사용법 요약 |
|---|---|---|
| `run_all_variants.sh` | Phase 2 3x **순차** 5-variant 학습 | `bash run_all_variants.sh` → 각 variant 200 epoch 순차 train. batch_size=256, lr=1e-3. 로그는 `train_logs/variant_[a-e].log`. |
| `run_all_variants_parallel.sh` | Phase 2 4x **병렬** 5-variant 학습 (초기 시도, num_workers=2) | `bash run_all_variants_parallel.sh` → 5 variant를 동시에 background 실행. 로그는 `variant_[a-e]_4x.log`. |
| `run_4x_parallel.sh` | Phase 2 4x **병렬** 5-variant 학습 (간결 버전) | 동일 목적, 스크립트가 더 간결. `COMMON` 변수로 공통 인자. |
| `run_4x_wave1.sh` | Phase 2 4x **웨이브 분할** (a/b/c/d 병렬 후 e 단독) | GPU 메모리 제약 시 5 variant 동시 실행 불가 → 4개 + 1개로 분할. |
| `run_mask_predict_train.sh` | Phase 4 Mask-Predict 학습 | `bash run_mask_predict_train.sh` → variant_e_mask_predict 70 epoch 학습, lr=5e-4, batch_size=256. 사전학습 weight는 `variant_e_alt_cross_4x/model/latest.pth`. |
| `run_mask_predict_test.sh` | Phase 4 MP N=0..3 순차 테스트 | `bash run_mask_predict_test.sh` → 각 N에 대해 test.py 실행 후 `test_results` → `test_results_n{N}`로 rename. ratios: N0=0.5, N1=0.5, N2=0.5,0.3, N3=0.6,0.4,0.2. |
| `train.sh` / `test.sh` | 초기 single-variant 래퍼 | `nohup python train.py/test.py ... &` 데몬화. 더 이상 사용 안 함(Phase 2 이후 run_* 스크립트로 대체). |

### 7.2 Python 도구 (`tools/`)

| 파일 | 크기 | 목적 | 사용법 요약 |
|---|---|---|---|
| `eval_mp.py` | 3.6 KB | Mask-Predict 테스트 결과(`test_results_n{N}`)에 대한 cmd/args accuracy 집계 | `python tools/eval_mp.py` → `phase4_accuracy.json` 생성. `CAD_*_IDX` 상수 사용해 per-command 정확도 분리. |
| `eval_cd_ir.py` | 5.3 KB | 모든 variant의 vec→OCC→STL→trimesh sampling → CD/IR 계산 (2000 subset) | `python tools/eval_cd_ir.py` → `phase4_cd_ir.json` 생성. joblib Parallel(n_jobs=8), KDTree CD. DeepCAD pipeline 재사용. |
| `bench_mp_latency.py` | 3.1 KB | MP 모델 GPU inference latency 측정 (N=0..3 × 50 샘플) | `python tools/bench_mp_latency.py` → `phase4_latency.json` 생성. argv patching으로 Config 재사용. |
| `render_cad.py` | 12.3 KB | vec → OCC TopoDS_Shape → STL → Viewer3d offscreen PNG 렌더 | `xvfb-run -a python tools/render_cad.py --h5 <path> --out <dir>` — BRepCheck_Analyzer로 invalid 판별. |
| `qualitative_eval.py` | 9.6 KB | End-to-end 정성 평가: 렌더 + 그리드 조합 + 에러 taxonomy JSON | `xvfb-run -a python tools/qualitative_eval.py --variant <name> --sample-ids ...` — `render_variant_*.json` / `success_variant_*.json` 작성. |
| `make_tier_grids.py` | 5.3 KB | GT + variant_a + variant_e 병렬 비교 그리드 PNG 생성 (tier별) | `python tools/make_tier_grids.py` → `grid_{top,mid,bottom}_tier.png`. 변환 실패 타일은 회색 placeholder. matplotlib backend='Agg'. |

---

## 8. 핵심 코드 변경 (HEAD~1..HEAD = 8461e6f..51f8daf)

### 8.1 `config/config.py` (+10 lines)

Phase 4 MP 인자 4개 추가:

```python
parser.add_argument('--use_mask_predict', action='store_true', default=False)
parser.add_argument('--n_refinement_steps', type=int, default=0)
parser.add_argument('--mask_ratios', type=str, default='0.5,0.3')
parser.add_argument('--freeze_pretrained', action='store_true', default=False)
```

### 8.2 `model/model.py` (+91 / -? lines)

- **`PartialPredictionEmbedding` 클래스 신규** (112~151 lines). `ConstEmbedding`을 대체; refinement pass 시 prev_cmd/prev_args embed + learnable mask_token으로 masked position 대체.
- `Decoder.__init__`: `use_mask_predict` flag로 embedding 분기.
- `Decoder.forward`: `prev_cmd`, `prev_args`, `mask_positions` 인자 추가.
- `SVG2CADTransformer.forward`: `n_refinement_steps`, `mask_ratio_schedule` 인자 추가. 각 step마다 confidence-based masking (`torch.topk(cmd_confidence, k, largest=False)`)으로 저신뢰 위치를 재마스크 후 re-decode.

### 8.3 `trainer/trainer.py` (+96 / -? lines)

- **사전학습 로드**: `use_mask_predict=True and is_train`이면 `proj_log/variant_e_alt_cross_4x/model/latest.pth`를 `strict=False`로 로드. missing keys는 신규 MP params.
- **`freeze_pretrained` 모드**: MP 신규 파라미터 이름(`decoder.embedding.{command_embed,args_embed,args_proj,mask_token}`)만 `requires_grad=True`.
- **`_forward_mask_predict` 메서드 신규**: 초기 pass + refinement pass 2-pass 학습. Refinement는 GT에 random mask ratio(0.15~0.85)로 random 마스킹 후 decoder 재실행. Loss는 init+ref 합산.

### 8.4 `trainer/loss.py` (+20 / -? lines)

- `NewCADLoss.forward(outputs, cad_data, refinement_mask=None)` 시그니처 변경.
- `refinement_mask` 제공 시 masked positions에만 loss 계산 (`padding_mask *= refinement_mask.float()`).
- 기존 주석 처리되었던 EMD loss branch 제거.

### 8.5 `test.py` (+13 / -1 lines)

- `tr_agent.forward(data)` 직접 호출 → `tr_agent.net(sv, sc, sa, n_refinement_steps=..., mask_ratio_schedule=...)` 변경.
- `cfg.n_refinement_steps`, `cfg.mask_ratios` 파싱 후 schedule list 구성 및 전달.

---

## 9. DeepCAD 통합 (`cadlib_deepcad/`)

### 9.1 파일 목록 (6 module + __init__)

| 파일 | 크기 | 역할 |
|---|---|---|
| `__init__.py` | 0 B | 패키지 마커 |
| `curves.py` | 17 KB | Line/Arc/Circle curve 클래스. vec 표현 ↔ geometry 변환. |
| `extrude.py` | 14 KB | CoordSystem, Extrude, CADSequence 클래스. profile normalization/numericalize. |
| `macro.py` | 1.8 KB | `ALL_COMMANDS=['Line','Arc','Circle','EOS','SOL','Ext']`, 각종 IDX 상수 및 PAD/SOL 값. |
| `math_utils.py` | 3.2 KB | `cartesian2polar`, `polar_parameterization`, angle 유틸. |
| `sketch.py` | 10 KB | SketchBase, Loop, Profile 클래스. matplotlib backend='Agg'. |
| `visualize.py` | 5.7 KB | `vec2CADsolid`, `create_CAD`, `CADsolid2pc` (OCC 기반 TopoDS_Shape 생성 및 STL export). |

### 9.2 원본 DeepCAD 대비 패치 내역

기준: `/home/work/Drawing2CAD/deepcad_ref/cadlib/` (원본 참조 복사본)

| 파일 | 패치 | 내용 |
|---|---|---|
| `curves.py` | `np.int` → `int` | Line.numericalize(2회), Arc.numericalize(5회), Circle.numericalize(2회). NumPy 1.24+에서 `np.int` deprecated. |
| `extrude.py` | `np.int` → `int` | CoordSystem.numericalize(2회), Extrude.numericalize(4회), random_transform(1회). |
| `sketch.py` | `matplotlib.use('TkAgg')` → `matplotlib.use('Agg')` | headless 환경(Xvfb/Docker)에서 TkAgg 백엔드 에러 회피. |
| `macro.py`, `math_utils.py`, `visualize.py` | 변경 없음 | 원본 그대로. |

---

## 10. WandB 및 외부 로그

### 10.1 로그인 상태 (2026-04-18 기준)

- `~/.netrc`에 `api.wandb.ai` credential 저장됨 (entity: `jujoo`).
- `wandb login --verify` 성공: `Currently logged in as: jujoo to https://api.wandb.ai.`
- `wandb status`는 CLI 설정(엔티티/조직 null) 반환.

### 10.2 WandB 런 목록 (`/home/work/Drawing2CAD/wandb/`)

총 **44개 런** (1개 offline + 43개 online). 디스크 사용량 929 MB.

| 날짜 그룹 | 런 수 | 용도 추정 |
|---|---|---|
| 2026-04-14 | 5 | 초기 setup / 인코더 구조 실험 (offline-ds1zpvza, vssudr5p, euy4zwzk, 8sbd2gms, id1mf16y) |
| 2026-04-15 | 20 | Phase 2 3x 학습 (cliir603, pas4vr8e, 4wprn1uh) + 3x test (2k2idbs7, 6eu4t3il, 1ushy0wc, j3idbl7z) + 4x 1차 시도 (57sb0ku2 외 4) + 4x wave1 (8uvwpiz6 외 3) + 4x 최종 (i7sflqnm 외 4) |
| 2026-04-17 | 14 | 4x test (y93a2fbf 외 4), Phase 3 최적화 (c5u8mnh7 외 5, pdhn0xpo), MP 디버그 (u80bl0zs, 68r03j9z) |
| 2026-04-18 | 5 | MP 테스트 N=0..3 (5h8gyh5q, zj5shy33, g952fpyv, 9j1jjiv1) + benchmark (aiq36i2u) |

최신 런(latest-run 심볼릭): `run-20260418_111341-aiq36i2u` (MP latency 벤치).

### 10.3 WandB debug 로그

- `wandb/debug-cli.work.log` (2.2 KB, 2026-04-14 00:44) — 초기 login 실패 traceback 기록. 이후 `~/.netrc` 설정으로 해결.

---

## 부록 A: 전체 파일 트리 (핵심 경로만)

```
/home/work/Drawing2CAD/
├── README.md, LICENSE
├── config/ (config.py, macro.py, file_utils.py)
├── dataset/, model/, trainer/, evaluate/
├── cadlib_deepcad/               # DeepCAD 패치 버전 (np.int→int, Agg backend)
│   ├── curves.py, extrude.py, macro.py, math_utils.py, sketch.py, visualize.py
├── tools/                        # Phase 3/4 평가 스크립트
│   ├── eval_mp.py, eval_cd_ir.py, bench_mp_latency.py
│   ├── render_cad.py, qualitative_eval.py, make_tier_grids.py
├── deepcad_ref/                  # 원본 DeepCAD 참조 (패치 비교용)
├── proj_log/                     # 학습 결과 (3.4 GB)
│   ├── variant_[a-e]_baseline/, variant_[a-e]_*_4x/  # Phase 2 (3x + 4x)
│   ├── variant_e_mask_predict/                        # Phase 4 MP
│   │   ├── config.txt, model/latest.pth
│   │   └── test_results_n{0,1,2,3}/ (각 7881 h5)
│   └── _test_mp[1-7]/            # MP 개발 smoke test (모델 없음)
├── docs/                         # 4.8 MB
│   ├── phase4_accuracy.json, phase4_latency.json, phase4_cd_ir.json
│   ├── _cd_smoke.json, report_phase2_ablation.md
│   ├── experiment_artifacts.md   # ← 이 문서
│   └── figures/
│       ├── fig{1..10}_*.png (Phase 2 정량 시각화)
│       └── qualitative_3d/       # Phase 4 정성 3D
│           ├── grid_{top,mid,bottom}_tier.png
│           ├── grid_variant_{a,e}_*_4x.png
│           ├── success_variant_{a,b,c,d,e}_*_4x.json
│           ├── render_variant_{a,e}_*_4x.json
│           ├── qualitative_eval_summary.json
│           └── variant_{a,e}_*_4x/ (각 76 PNG, 11 샘플 × gt/pred × 4 views)
├── train_logs/                   # 648 MB
│   ├── progress*.log (요약), variant_*.log (raw, ~36MB/각)
│   ├── mask_predict.log, mp_test_n{0-3}.log, mp_test.log
│   ├── cd_ir.log, success_{bcd,mp}.log
│   └── nohup{_4x,_mp}.log
├── wandb/                        # 929 MB, 44 런
│   └── run-*, offline-run-*
├── run_*.sh                      # 학습/테스트 셸 스크립트 (7개)
├── train.py, test.py
└── *.py (sanity_check, inspect_ckpt, visualize_*, verify_*, convert_*)
```

## 부록 B: 요약 통계

- **체크포인트**: 11개 실질 variant (5×3x + 5×4x + 1×MP) + 7개 빈 디버그 디렉토리. 총 32개 `.pth` 파일. 전체 proj_log 3.4 GB.
- **테스트 결과 `_vec.h5`**: 9 variant × 7881 = 70,929개 (e 3x 제외) + MP × 4 N × 7881 = 31,524개. **총 102,453개 h5**.
- **평가 JSON**: 4개 (phase4_accuracy, phase4_latency, phase4_cd_ir, _cd_smoke) + 7개 qualitative (5 success + 2 render + 1 summary) = **11개**.
- **PNG**: Phase 2 fig 10개 + Phase 4 grid 5개 + individual 152개 = **167개**.
- **로그**: 36개 (3x 학습 5 + 4x 학습 5 + 3x test 4 + 4x test 5 + MP 8 + eval 2 + progress 3 + nohup 3 + success 2 - variant_e 3x test 제외).
- **스크립트**: 7 sh + 6 tool + train.py/test.py.
- **WandB 런**: 44개 (online 43 + offline 1).
- **코드 변경 (최신 commit)**: 182 파일 +3009/-129 lines (Phase 4 + DeepCAD 통합 + qualitative pipeline 전체).
