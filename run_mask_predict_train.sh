#!/bin/bash
export AUTO_OVERWRITE=1
cd /home/work/Drawing2CAD

LOG_DIR="/home/work/Drawing2CAD/train_logs"
mkdir -p "$LOG_DIR"

COMMON="--encoder_type alternating --decoder_type cross_attention --input_option 4x --use_mask_predict --batch_size 256 --num_workers 2"

echo "$(date '+%Y-%m-%d %H:%M:%S') [START] Mask-Predict training (70 epochs)" | tee "$LOG_DIR/progress_mp.log"

python train.py \
    --exp_name variant_e_mask_predict \
    $COMMON \
    --nr_epochs 70 \
    --lr 5e-4 \
    2>&1 | tee "$LOG_DIR/mask_predict.log"

echo "$(date '+%Y-%m-%d %H:%M:%S') [DONE] Mask-Predict training" | tee -a "$LOG_DIR/progress_mp.log"
