#!/bin/bash
cd /home/work/Drawing2CAD

LOG_DIR="/home/work/Drawing2CAD/train_logs"
RESULTS_BASE="/home/work/Drawing2CAD/proj_log/variant_e_mask_predict/test_results"

COMMON="--exp_name variant_e_mask_predict --encoder_type alternating --decoder_type cross_attention --input_option 4x --use_mask_predict --batch_size 256 --num_workers 2"

for N in 0 1 2 3; do
    case $N in
        0) RATIOS="0.5" ;;
        1) RATIOS="0.5" ;;
        2) RATIOS="0.5,0.3" ;;
        3) RATIOS="0.6,0.4,0.2" ;;
    esac

    echo "$(date '+%Y-%m-%d %H:%M:%S') [START] test n_refinement_steps=$N (ratios=$RATIOS)" | tee -a "$LOG_DIR/mp_test.log"

    # Clean previous results
    rm -rf "$RESULTS_BASE"

    python test.py \
        $COMMON \
        --n_refinement_steps $N \
        --mask_ratios "$RATIOS" \
        2>&1 | tee "$LOG_DIR/mp_test_n${N}.log"

    # Archive results with suffix
    if [ -d "$RESULTS_BASE" ]; then
        mv "$RESULTS_BASE" "${RESULTS_BASE}_n${N}"
        echo "$(date '+%Y-%m-%d %H:%M:%S') [DONE] Saved to ${RESULTS_BASE}_n${N}" | tee -a "$LOG_DIR/mp_test.log"
    else
        echo "$(date '+%Y-%m-%d %H:%M:%S') [FAIL] No test_results dir for N=$N" | tee -a "$LOG_DIR/mp_test.log"
    fi
done

echo "$(date '+%Y-%m-%d %H:%M:%S') [ALL DONE]" | tee -a "$LOG_DIR/mp_test.log"
