import torch
import sys

try:
    ckpt_path = 'pretrained/model/ckpt_epoch1000.pth'
    print(f"Loading {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location='cpu')
    print("Keys in checkpoint:", ckpt.keys())
    
    state_dict = None
    if 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
    elif 'net' in ckpt:
        state_dict = ckpt['net']
    elif isinstance(ckpt, dict):
        state_dict = ckpt
    
    if state_dict:
        print(f"\nTotal keys in state_dict: {len(state_dict)}")
        print("First 20 keys:")
        keys = list(state_dict.keys())
        for k in keys[:20]:
            print(k)
            
        print("\nChecking for 'decoder' keys:")
        decoder_keys = [k for k in keys if 'decoder' in k]
        print(f"Found {len(decoder_keys)} keys with 'decoder'.")
        for k in decoder_keys[:20]:
            print(k)
            
        print("\nChecking for 'bottleneck' keys:")
        bottleneck_keys = [k for k in keys if 'bottleneck' in k]
        print(f"Found {len(bottleneck_keys)} keys with 'bottleneck'.")
        for k in bottleneck_keys[:20]:
            print(k)

except Exception as e:
    print(f"Error: {e}")
