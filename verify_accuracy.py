import sys
import os
import numpy as np

# Add project root to path to import config
if "/workspace/Drawing2CAD" not in sys.path:
    sys.path.append("/workspace/Drawing2CAD")

from config.macro import CAD_CMD_ARGS_MASK

def calculate_accuracy(out_vec, gt_vec, tolerance=3):
    # out_vec, gt_vec: (S, 1 + N_ARGS)
    
    # Command Accuracy
    pred_cmd = out_vec[:, 0]
    gt_cmd = gt_vec[:, 0]
    cmd_match = (pred_cmd == gt_cmd) # (S,)
    cmd_acc = np.mean(cmd_match)
    
    # Parameter Accuracy
    pred_args = out_vec[:, 1:]
    gt_args = gt_vec[:, 1:]
    
    # Get mask for valid parameters based on Ground Truth commands
    # CAD_CMD_ARGS_MASK is (N_CMD, N_ARGS)
    # We assume gt_cmd contains valid command indices
    mask = CAD_CMD_ARGS_MASK[gt_cmd] # Shape: (S, N_ARGS)
    
    # Check for matches within tolerance
    diff = np.abs(pred_args - gt_args)
    
    # Condition: Within tolerance AND Command must be correct
    # tolerance * cmd_match[:, None] broadcasts cmd_match to (S, 1)
    # If cmd_match is False (0), threshold is 0, so diff < 0 is always False.
    # If cmd_match is True (1), threshold is tolerance.
    matches = (diff < tolerance * cmd_match[:, np.newaxis])
    
    # Filter by mask
    valid_matches = matches * mask
    
    # Calculate accuracy
    total_valid_params = np.sum(mask)
    total_correct_params = np.sum(valid_matches)
    
    if total_valid_params == 0:
        param_acc = 0.0
    else:
        param_acc = total_correct_params / total_valid_params
        
    return cmd_acc, param_acc

# Test case
# Command 0: Line (args 0, 1 valid)
# Command 3: EOS (no args valid)
gt_vec = np.array([
    [0, 10, 20, 0, 0, 0, 0, 0], # Line, args 10, 20 are valid
    [3, 0, 0, 0, 0, 0, 0, 0]    # EOS, no args valid
])

# Case 1: Perfect match
out_vec_perfect = np.array([
    [0, 10, 20, 99, 99, 99, 99, 99], # Extra args shouldn't matter
    [3, 99, 99, 99, 99, 99, 99, 99]
])

cmd_acc, param_acc = calculate_accuracy(out_vec_perfect, gt_vec)
print(f"Perfect Match - Cmd: {cmd_acc}, Param: {param_acc}")

# Case 2: Within tolerance (tolerance=3)
out_vec_tolerance = np.array([
    [0, 12, 18, 99, 99, 99, 99, 99], # 10->12 (diff 2), 20->18 (diff 2) -> Should match (diff < 3)
    [3, 99, 99, 99, 99, 99, 99, 99]
])
cmd_acc, param_acc = calculate_accuracy(out_vec_tolerance, gt_vec, tolerance=3)
print(f"Tolerance Match (diff=2) - Cmd: {cmd_acc}, Param: {param_acc}")
assert param_acc == 1.0, "Should be 1.0 within tolerance"

# Case 3: Wrong Command, Correct Params (should fail)
out_vec_wrong_cmd = np.array([
    [1, 10, 20, 99, 99, 99, 99, 99], # Wrong command (1!=0), but params match perfectly
    [3, 99, 99, 99, 99, 99, 99, 99]
])
cmd_acc, param_acc = calculate_accuracy(out_vec_wrong_cmd, gt_vec, tolerance=3)
print(f"Wrong Command - Cmd: {cmd_acc}, Param: {param_acc}")
assert cmd_acc == 0.5, "Cmd acc should be 0.5"
assert param_acc == 0.0, "Param acc should be 0.0 because command is wrong"

# Case 4: Mixed
# Row 0: Correct Cmd, Correct Params
# Row 1: Wrong Cmd (EOS vs Line), Params irrelevant
gt_vec_mixed = np.array([
    [0, 10, 20, 0, 0, 0, 0, 0],
    [0, 10, 20, 0, 0, 0, 0, 0]
])
out_vec_mixed = np.array([
    [0, 10, 20, 0, 0, 0, 0, 0],
    [1, 10, 20, 0, 0, 0, 0, 0]
])
cmd_acc, param_acc = calculate_accuracy(out_vec_mixed, gt_vec_mixed, tolerance=3)
print(f"Mixed - Cmd: {cmd_acc}, Param: {param_acc}")
assert cmd_acc == 0.5
assert param_acc == 0.5 # 2 valid params in row 0 (correct), 2 valid params in row 1 (incorrect cmd -> incorrect) -> 2/4 = 0.5
