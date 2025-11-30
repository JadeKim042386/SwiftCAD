import torch
import sys

try:
    ckpt_path = 'pretrained/model/ckpt_epoch1000.pth'
    print(f"Loading {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location='cpu')
    
    state_dict = None
    if 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
    elif 'net' in ckpt:
        state_dict = ckpt['net']
    elif isinstance(ckpt, dict):
        state_dict = ckpt
    
    if state_dict:
        print("\nChecking for 'fcn' keys and shapes:")
        fcn_keys = [k for k in state_dict.keys() if 'fcn' in k]
        for k in fcn_keys:
            print(f"{k}: {state_dict[k].shape}")

except Exception as e:
    print(f"Error: {e}")
