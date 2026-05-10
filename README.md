<div align="center">

# SwiftCAD: Efficient Parametric CAD Generation with Shared Decoder Transformers

<h4>
  Juyoung Kim<sup>1*</sup>
  ·
  Seongjun Choi<sup>2*</sup>
  ·
  Jiyeon Lim<sup>3</sup>
  ·
  Soomok Lee<sup>4</sup>
</h4>

<h5>
  <sup>1</sup>Metacle  <sup>2</sup>Yonsei University  <sup>3</sup>Samsung Electronics  <sup>4</sup>Kennesaw State University
</h5>

<h4>
  CVPR 2026 Workshop 3D4S
</h4>

<h5>
  *Equal contribution.
</h5>

[![Conference](https://img.shields.io/badge/CVPRW-2026-4b44ce)]()
[![Project Page](https://img.shields.io/badge/Project-Page-1f8a3e?logo=github&logoColor=white)](https://jadekim042386.github.io/SwiftCAD/)
[![Google Drive](https://img.shields.io/badge/Drive-Dataset-4285F4?logo=googledrive&logoColor=fff)](https://drive.google.com/drive/folders/1t9uO2iFh1eVDXRCKUEonKPBu8WGYA8wU?usp=sharing)

</div>

![SwiftCAD teaser](assets/figure_1.png)

SwiftCAD is a streamlined Transformer architecture for parametric CAD generation from vector engineering drawings, building on the SVG-based pipeline of Drawing2CAD. By removing redundant MLP layers in the embedding stage and unifying command and parameter prediction into a single weight-shared decoder with task-specific heads, SwiftCAD achieves a **64.74% reduction in parameters** (from 10.1M to 3.6M) and faster inference, while maintaining accuracy within 0.5% (command) and 0.9% (parameter) of the Drawing2CAD baseline on the CAD-VGDrawing benchmark.

## 🐍 Installation

- Python 3.9
- Cuda 11.8+

Install python package dependencies through pip:

```bash
pip install -r requirements.txt
```

## 📥 Dataset

We use the **CAD-VGDrawing** dataset from Drawing2CAD. Download data from [here](https://drive.google.com/drive/folders/1t9uO2iFh1eVDXRCKUEonKPBu8WGYA8wU?usp=sharing) and extract them under the `data` folder.

- `svg_raw` contains the engineering drawings of each CAD model in SVG format, including four views: `Front`, `Top`, `Right`, and `FrontTopRight`. Each SVG file has been preprocessed through path simplification and deduplication, path reordering, and viewbox normalization. To obtain engineering drawings in PNG format, you can simply convert them using [CairoSVG](https://cairosvg.org/) with a single line of code:

  ```python
  import cairosvg
  
  cairosvg.svg2png(url=svg_path, write_to=png_path, output_width=224, output_height=224, background_color='white')
  ```

- `svg_vec` contains vectorized representations of SVG drawing sequences. Each file stores the stacked drawing sequences for the four views (`Front`, `Top`, `Right`, and `FrontTopRight`), saved in `.npy` format to enable fast data loading.

- `cad_vec` contains the vectorized representation for CAD sequences, saved in `.h5` format to enable fast data loading.

## 🚀 Training

The main SwiftCAD configuration (Shared Decoder + no MLP embedding, `d_model=144`) can be trained with:

```bash
python train.py --input_option 4x --d_model 144 --exp_name swiftcad_main
```

To reproduce all five rows of Table 2, use the following commands. The relevant flags are:

- `--use_shared_decoder` (default: on) — shared decoder architecture (paper main). Pass `--no-use_shared_decoder` to use the dual-decoder baseline.
- `--use_mlp_embedding` (default: off) — pass to enable the legacy MLP embedding layer (Drawing2CAD baseline).
- `--d_model` (default: `144`, choices: `144` / `192` / `256`) — Transformer embedding dimension.

```bash
# (1) Baseline (Drawing2CAD): dual decoder + MLP embedding
python train.py --input_option 4x --no-use_shared_decoder --use_mlp_embedding --d_model 256 --exp_name baseline

# (2) Shared Decoder only (with MLP embedding)
python train.py --input_option 4x --use_mlp_embedding --d_model 256 --exp_name shared_decoder

# (3) Shared + w/o MLP, d_model = 144  (main config)
python train.py --input_option 4x --d_model 144 --exp_name shared_nomlp_144

# (4) Shared + w/o MLP, d_model = 192
python train.py --input_option 4x --d_model 192 --exp_name shared_nomlp_192

# (5) Shared + w/o MLP, d_model = 256
python train.py --input_option 4x --d_model 256 --exp_name shared_nomlp_256
```

Since different configurations produce different model checkpoints, it is recommended to specify a unique `--exp_name` for each run. For more configurable parameters, please refer to `config/config.py`.

## 📍 Evaluation

After training, run inference on all test data with the corresponding configuration. Make sure the inference flags match the training flags.

```bash
# (1) Baseline (Drawing2CAD)
python test.py --input_option 4x --no-use_shared_decoder --use_mlp_embedding --d_model 256 --exp_name baseline

# (2) Shared Decoder only
python test.py --input_option 4x --use_mlp_embedding --d_model 256 --exp_name shared_decoder

# (3) Shared + w/o MLP, d_model = 144  (main config)
python test.py --input_option 4x --d_model 144 --exp_name shared_nomlp_144

# (4) Shared + w/o MLP, d_model = 192
python test.py --input_option 4x --d_model 192 --exp_name shared_nomlp_192

# (5) Shared + w/o MLP, d_model = 256
python test.py --input_option 4x --d_model 256 --exp_name shared_nomlp_256
```

After inference, the final results will be saved under `proj_log/your_exp_name/test_results`. To export and visualize the final CAD models, please refer to the code from [DeepCAD](https://github.com/ChrisWu1997/DeepCAD).

## 📐 CD / IR Evaluation

Quantitative evaluation against the Drawing2CAD protocol: **Mean Chamfer Distance (MCD = mean CD × 10²)** between generated and ground-truth point clouds, and **Invalidity Ratio (IR, %)** — the fraction of test samples for which the predicted CAD vector cannot be reconstructed into a valid solid. Both are lower-is-better.

This step requires `pythonocc-core` (used by `evaluate/cadlib`). It is **not** in `requirements.txt`; install it via conda:

```bash
conda install -c conda-forge pythonocc-core=7.7.0
```

### Step 1 — Run inference for the five paper configs

Place the trained checkpoints under `proj_log/<exp_name>/model/` and run the five `python test.py ...` commands listed in the *🚀 Training* section above (one per `exp_name`). Each writes `proj_log/<exp_name>/test_results/<data_id>_vec.h5` containing both `out_vec` (prediction) and `gt_vec` (ground truth).

### Step 2 — Precompute ground-truth point clouds (one-time)

This is the canonical Drawing2CAD-style step: reconstruct each GT CAD vector via OCC and freeze a point cloud sample to disk. The orchestrator below loads these `.ply` files at evaluation time.

```bash
# (Recommended) Build GT pcs from gt_vec stored in test_results — no extra dataset download required.
python evaluate/precompute_pc.py \
    --src_test_results proj_log/baseline/test_results \
    --output data/cad_vec_pc --n_points 2000 --n_jobs 8

# (Alternative) Build from the CAD-VGDrawing release. Functionally equivalent (see note below).
python evaluate/precompute_pc.py \
    --src data/cad_vec --split_json data/train_val_test_split.json --phase test \
    --output data/cad_vec_pc --n_points 2000 --n_jobs 8
```

`gt_vec` (the field test.py writes into each `*_vec.h5`) is byte-equivalent to `cad_vec/<id>.h5` "vec" content after EOS truncation — verified empirically — so the two precompute paths produce identical point clouds. The `--src_test_results` path is the simpler default because it does not require the dataset to be available locally.

### Step 3 — Evaluate CD and IR for the five paper configs

```bash
python evaluate/eval_paper_models.py --gt_pc_root data/cad_vec_pc --n_jobs 8 --seed 0

# Subset
python evaluate/eval_paper_models.py --gt_pc_root data/cad_vec_pc --only shared_nomlp_144,baseline
```

The orchestrator runs the five `exp_name`s reported in Table 2 (`baseline`, `shared_decoder`, `shared_nomlp_256`, `shared_nomlp_192`, `shared_nomlp_144`), parses each per-config `cd_ir.txt.json`, and writes a Markdown summary to `proj_log/cd_ir_summary.md`. Results for each config are cached at `proj_log/<exp_name>/cd_ir.txt` (per-sample TSV + summary footer) and `cd_ir.txt.json` (machine-readable). Re-running skips configs whose JSON already matches the requested settings; pass `--force` to recompute.

For ad-hoc single-experiment evaluation, call the underlying script directly:

```bash
python evaluate/eval_cd_ir.py --src proj_log/swiftcad_main/test_results --gt_pc_root data/cad_vec_pc --n_jobs 8
```

A self-contained `--gt_from_h5` mode exists for quick experiments without precomputing — it regenerates GT point clouds at evaluation time. Be aware that this mode reports IR over the full test set and includes GT-side OCC failures in the numerator: `IR = (n_pred_invalid + n_gt_invalid) / N_total`. The numbers in this README come from the precomputed-`--gt_pc_root` mode (`IR = n_pred_invalid / N_evaluated`, GT failures excluded from both).

**Notes**
- `--n_points 2000` matches the Drawing2CAD `collect_gen_pc.py` default. Use `--n_points 8192` for a denser GT pool to be downsampled at runtime — but absolute MCD values may shift.
- Default normalization is **asymmetric** (`out_pc` rescaled when `max(|·|) > 2`, `gt_pc` left as-is). This matches Drawing2CAD; pass `--normalize_gt` for the symmetric variant.
- `--n_jobs 1` is the safest fallback for debugging (avoids OCC + multiprocessing edge cases) and is faster for very small `--num`.

## 📊 Results

Quantitative comparison on CAD-VGDrawing (Table 2 of the paper). All entries use the `4x` input option.

| Method | Parameters | File Size (MB) | Inf. Time (s) | ACC<sub>cmd</sub> | ACC<sub>param</sub> | IR ↓ | MCD ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline (Drawing2CAD) | 10,109,894 | 38.57 | 8.4815 | 82.76 | 79.23 | 20.44 | 11.16 |
| Shared Decoder | 7,722,438 | 29.46 | 8.4329 | 82.42 | 79.13 | 21.52 | 11.29 |
| Shared + w/o MLP (256) | 7,745,850 | 29.55 | 8.3925 | 82.20 | 79.21 | 21.55 | 11.30 |
| Shared + w/o MLP (192) | 5,204,234 | 19.85 | 8.4447 | 82.17 | 78.87 | 21.75 | 11.66 |
| Shared + w/o MLP (144) | 3,564,806 | 13.60 | 7.7575 | 82.37 | 78.55 | 21.89 | 11.90 |

The two contributions are complementary. The **shared decoder** alone removes ~24% of the parameters by replacing the dual command/parameter decoders with a single weight-shared decoder and task-specific output heads. Stacking the **streamlined encoding** (removing the embedding-stage MLP, letting Transformer self-attention alone capture cross-field interactions between view, command, and parameter tokens) and reducing `d_model` to 144 yields the main config: a 64.74% parameter reduction, the smallest file size, and the fastest inference, while staying within 0.5% of the baseline on command accuracy and 0.9% on parameter accuracy.

## 🌹 Acknowledgement

This repository builds upon the following awesome datasets and projects:

- [Drawing2CAD](https://arxiv.org/abs/2508.18733) (Qin et al., 2025)
- [DeepCAD](https://github.com/ChrisWu1997/DeepCAD)
- [FreeCAD](https://github.com/FreeCAD/FreeCAD)
- [CairoSVG](https://cairosvg.org/)

## 📝 Citation

If you find this project useful for your research, please use the following BibTeX entry.

```
@inproceedings{kim2026swiftcad,
  title={SwiftCAD: Efficient Parametric CAD Generation with Shared Decoder Transformers},
  author={Juyoung Kim and Seongjun Choi and Jiyeon Lim and Soomok Lee},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)},
  year={2026}
}
```

(BibTeX will be updated once camera-ready proceedings are published.)
