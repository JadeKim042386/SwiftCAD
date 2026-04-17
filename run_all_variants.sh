#!/bin/bash
set -e

export AUTO_OVERWRITE=1
cd /home/work/Drawing2CAD

LOG_DIR="/home/work/Drawing2CAD/train_logs"
mkdir -p "$LOG_DIR"

echo "$(date '+%Y-%m-%d %H:%M:%S') Starting all 5 variants training" | tee "$LOG_DIR/progress.log"

# ========================================
# Variant (a): Baseline
# ========================================
echo "$(date '+%Y-%m-%d %H:%M:%S') [START] Variant (a) Baseline" | tee -a "$LOG_DIR/progress.log"
python train.py \
    --exp_name variant_a_baseline \
    --encoder_type standard \
    --decoder_type broadcast \
    --input_option 3x \
    --nr_epochs 200 \
    --batch_size 256 \
    --lr 1e-3 \
    2>&1 | tee "$LOG_DIR/variant_a.log"
echo "$(date '+%Y-%m-%d %H:%M:%S') [DONE] Variant (a) Baseline" | tee -a "$LOG_DIR/progress.log"

# ========================================
# Variant (b): Cross-Attention Only
# ========================================
echo "$(date '+%Y-%m-%d %H:%M:%S') [START] Variant (b) Cross-attn" | tee -a "$LOG_DIR/progress.log"
python train.py \
    --exp_name variant_b_cross_attn \
    --encoder_type standard \
    --decoder_type cross_attention \
    --input_option 3x \
    --nr_epochs 200 \
    --batch_size 256 \
    --lr 1e-3 \
    2>&1 | tee "$LOG_DIR/variant_b.log"
echo "$(date '+%Y-%m-%d %H:%M:%S') [DONE] Variant (b) Cross-attn" | tee -a "$LOG_DIR/progress.log"

# ========================================
# Variant (c): Cross-Attention + Bottleneck
# ========================================
echo "$(date '+%Y-%m-%d %H:%M:%S') [START] Variant (c) Cross-attn+BN" | tee -a "$LOG_DIR/progress.log"
python train.py \
    --exp_name variant_c_cross_attn_bn \
    --encoder_type standard \
    --decoder_type cross_attention \
    --use_bottleneck \
    --input_option 3x \
    --nr_epochs 200 \
    --batch_size 256 \
    --lr 1e-3 \
    2>&1 | tee "$LOG_DIR/variant_c.log"
echo "$(date '+%Y-%m-%d %H:%M:%S') [DONE] Variant (c) Cross-attn+BN" | tee -a "$LOG_DIR/progress.log"

# ========================================
# Variant (d): Alt-Attention Only (Negative Control)
# ========================================
echo "$(date '+%Y-%m-%d %H:%M:%S') [START] Variant (d) Alt-attn only" | tee -a "$LOG_DIR/progress.log"
python train.py \
    --exp_name variant_d_alt_attn \
    --encoder_type alternating \
    --decoder_type broadcast \
    --input_option 3x \
    --nr_epochs 200 \
    --batch_size 256 \
    --lr 1e-3 \
    2>&1 | tee "$LOG_DIR/variant_d.log"
echo "$(date '+%Y-%m-%d %H:%M:%S') [DONE] Variant (d) Alt-attn only" | tee -a "$LOG_DIR/progress.log"

# ========================================
# Variant (e): Alt-Attention + Cross-Attention
# ========================================
echo "$(date '+%Y-%m-%d %H:%M:%S') [START] Variant (e) Alt+Cross" | tee -a "$LOG_DIR/progress.log"
python train.py \
    --exp_name variant_e_alt_cross \
    --encoder_type alternating \
    --decoder_type cross_attention \
    --input_option 3x \
    --nr_epochs 200 \
    --batch_size 256 \
    --lr 1e-3 \
    2>&1 | tee "$LOG_DIR/variant_e.log"
echo "$(date '+%Y-%m-%d %H:%M:%S') [DONE] Variant (e) Alt+Cross" | tee -a "$LOG_DIR/progress.log"

echo "$(date '+%Y-%m-%d %H:%M:%S') ALL 5 VARIANTS COMPLETED" | tee -a "$LOG_DIR/progress.log"
