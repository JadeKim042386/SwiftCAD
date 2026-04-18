#!/bin/bash
export AUTO_OVERWRITE=1
cd /home/work/Drawing2CAD

LOG_DIR="/home/work/Drawing2CAD/train_logs"
mkdir -p "$LOG_DIR"

echo "$(date '+%Y-%m-%d %H:%M:%S') Starting all 5 variants training (PARALLEL, 4x)" | tee "$LOG_DIR/progress_4x.log"

# Variant (a): Baseline
echo "$(date '+%Y-%m-%d %H:%M:%S') [START] Variant (a) Baseline" | tee -a "$LOG_DIR/progress_4x.log"
python train.py \
    --exp_name variant_a_baseline_4x \
    --encoder_type standard \
    --decoder_type broadcast \
    --input_option 4x \
    --nr_epochs 200 \
    --batch_size 256 \
    --lr 1e-3 \
    2>&1 | tee "$LOG_DIR/variant_a_4x.log" &
PID_A=$!

# Variant (b): Cross-Attention Only
echo "$(date '+%Y-%m-%d %H:%M:%S') [START] Variant (b) Cross-attn" | tee -a "$LOG_DIR/progress_4x.log"
python train.py \
    --exp_name variant_b_cross_attn_4x \
    --encoder_type standard \
    --decoder_type cross_attention \
    --input_option 4x \
    --nr_epochs 200 \
    --batch_size 256 \
    --lr 1e-3 \
    2>&1 | tee "$LOG_DIR/variant_b_4x.log" &
PID_B=$!

# Variant (c): Cross-Attention + Bottleneck
echo "$(date '+%Y-%m-%d %H:%M:%S') [START] Variant (c) Cross-attn+BN" | tee -a "$LOG_DIR/progress_4x.log"
python train.py \
    --exp_name variant_c_cross_attn_bn_4x \
    --encoder_type standard \
    --decoder_type cross_attention \
    --use_bottleneck \
    --input_option 4x \
    --nr_epochs 200 \
    --batch_size 256 \
    --lr 1e-3 \
    2>&1 | tee "$LOG_DIR/variant_c_4x.log" &
PID_C=$!

# Variant (d): Alt-Attention Only
echo "$(date '+%Y-%m-%d %H:%M:%S') [START] Variant (d) Alt-attn only" | tee -a "$LOG_DIR/progress_4x.log"
python train.py \
    --exp_name variant_d_alt_attn_4x \
    --encoder_type alternating \
    --decoder_type broadcast \
    --input_option 4x \
    --nr_epochs 200 \
    --batch_size 256 \
    --lr 1e-3 \
    2>&1 | tee "$LOG_DIR/variant_d_4x.log" &
PID_D=$!

# Variant (e): Alt-Attention + Cross-Attention
echo "$(date '+%Y-%m-%d %H:%M:%S') [START] Variant (e) Alt+Cross" | tee -a "$LOG_DIR/progress_4x.log"
python train.py \
    --exp_name variant_e_alt_cross_4x \
    --encoder_type alternating \
    --decoder_type cross_attention \
    --input_option 4x \
    --nr_epochs 200 \
    --batch_size 256 \
    --lr 1e-3 \
    2>&1 | tee "$LOG_DIR/variant_e_4x.log" &
PID_E=$!

echo "$(date '+%Y-%m-%d %H:%M:%S') All 5 variants launched: PID_A=$PID_A PID_B=$PID_B PID_C=$PID_C PID_D=$PID_D PID_E=$PID_E" | tee -a "$LOG_DIR/progress_4x.log"

wait $PID_A && echo "$(date '+%Y-%m-%d %H:%M:%S') [DONE] Variant (a)" | tee -a "$LOG_DIR/progress_4x.log"
wait $PID_B && echo "$(date '+%Y-%m-%d %H:%M:%S') [DONE] Variant (b)" | tee -a "$LOG_DIR/progress_4x.log"
wait $PID_C && echo "$(date '+%Y-%m-%d %H:%M:%S') [DONE] Variant (c)" | tee -a "$LOG_DIR/progress_4x.log"
wait $PID_D && echo "$(date '+%Y-%m-%d %H:%M:%S') [DONE] Variant (d)" | tee -a "$LOG_DIR/progress_4x.log"
wait $PID_E && echo "$(date '+%Y-%m-%d %H:%M:%S') [DONE] Variant (e)" | tee -a "$LOG_DIR/progress_4x.log"

echo "$(date '+%Y-%m-%d %H:%M:%S') ALL 5 VARIANTS COMPLETED (4x)" | tee -a "$LOG_DIR/progress_4x.log"
