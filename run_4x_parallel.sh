#!/bin/bash
export AUTO_OVERWRITE=1
cd /home/work/Drawing2CAD

LOG_DIR="/home/work/Drawing2CAD/train_logs"
mkdir -p "$LOG_DIR"

COMMON="--input_option 4x --nr_epochs 200 --batch_size 256 --lr 1e-3 --num_workers 2"

echo "$(date '+%Y-%m-%d %H:%M:%S') Starting 5 variants parallel (4x, num_workers=2)" | tee "$LOG_DIR/progress_4x.log"

# Variant (a): Baseline
python train.py --exp_name variant_a_baseline_4x --encoder_type standard --decoder_type broadcast $COMMON \
    2>&1 | tee "$LOG_DIR/variant_a_4x.log" &
PID_A=$!

# Variant (b): Cross-Attention
python train.py --exp_name variant_b_cross_attn_4x --encoder_type standard --decoder_type cross_attention $COMMON \
    2>&1 | tee "$LOG_DIR/variant_b_4x.log" &
PID_B=$!

# Variant (c): Cross-Attention + Bottleneck
python train.py --exp_name variant_c_cross_attn_bn_4x --encoder_type standard --decoder_type cross_attention --use_bottleneck $COMMON \
    2>&1 | tee "$LOG_DIR/variant_c_4x.log" &
PID_C=$!

# Variant (d): Alt-Attention
python train.py --exp_name variant_d_alt_attn_4x --encoder_type alternating --decoder_type broadcast $COMMON \
    2>&1 | tee "$LOG_DIR/variant_d_4x.log" &
PID_D=$!

# Variant (e): Alt + Cross-Attention
python train.py --exp_name variant_e_alt_cross_4x --encoder_type alternating --decoder_type cross_attention $COMMON \
    2>&1 | tee "$LOG_DIR/variant_e_4x.log" &
PID_E=$!

echo "$(date '+%Y-%m-%d %H:%M:%S') PIDs: A=$PID_A B=$PID_B C=$PID_C D=$PID_D E=$PID_E" | tee -a "$LOG_DIR/progress_4x.log"

for PID_VAR in "A:$PID_A" "B:$PID_B" "C:$PID_C" "D:$PID_D" "E:$PID_E"; do
    NAME="${PID_VAR%%:*}"
    PID="${PID_VAR##*:}"
    wait $PID && echo "$(date '+%Y-%m-%d %H:%M:%S') [DONE] Variant ($NAME)" | tee -a "$LOG_DIR/progress_4x.log" \
             || echo "$(date '+%Y-%m-%d %H:%M:%S') [FAIL] Variant ($NAME)" | tee -a "$LOG_DIR/progress_4x.log"
done

echo "$(date '+%Y-%m-%d %H:%M:%S') ALL VARIANTS FINISHED (4x)" | tee -a "$LOG_DIR/progress_4x.log"
