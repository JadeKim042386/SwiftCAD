import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Add current directory to path
sys.path.append(os.getcwd())

from dataset.bi_sequence_dataset import get_dataloader
from config.config import Config
from trainer.trainer import TrainerED

def visualize_encoder_attention(cfg, save_name="encoder_attention_map.png"):
    print(f"Initializing model for experiment: {cfg.exp_name}...")
    
    tr_agent = TrainerED(cfg)
    
    print(f"Loading checkpoint: {cfg.ckpt}...")
    name = cfg.ckpt if cfg.ckpt == 'latest' else "ckpt_epoch{}".format(cfg.ckpt)
    load_path = os.path.join(tr_agent.model_dir, "{}.pth".format(name))
    
    if os.path.exists(load_path):
        checkpoint = torch.load(load_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            tr_agent.net.load_state_dict(checkpoint['model_state_dict'])
        else:
            tr_agent.net.load_state_dict(checkpoint)
    else:
        print(f"Warning: Checkpoint {load_path} not found. Using randomly initialized weights.")
        
    tr_agent.net.eval()
    if torch.cuda.is_available():
        tr_agent.net.cuda()

    # Forward hook to capture attention weights from the first encoder layer
    captured_attn = {}
    
    def get_attn_hook(name):
        def hook_fn(module, input, output):
            # output of MultiheadAttention is (attn_output, attn_output_weights)
            # where attn_output_weights is (N, L, S) -> (bsz, seq_len, seq_len)
            if isinstance(output, tuple) and len(output) == 2:
                captured_attn[name] = output[1].detach().cpu()
        return hook_fn

    # Hook the first layer of the encoder
    # tr_agent.net is SVG2CADTransformer
    # .encoder is Encoder
    # .encoder.encoder is TransformerEncoder
    # .encoder.encoder.layers[0] is TransformerEncoderLayerImproved
    # .encoder.encoder.layers[0].self_attn is MultiheadAttention
    target_module = tr_agent.net.encoder.encoder.layers[0].self_attn
    print("Hooking the self_attn of the first encoder layer...")
    hook = target_module.register_forward_hook(get_attn_hook("layer1_attn"))

    test_loader = get_dataloader("test", cfg)
    print("Fetching one batch of data to visualize attention...")

    # Get a single batch
    data = next(iter(test_loader))
    
    with torch.no_grad():
        # This will trigger the hook
        outputs, _ = tr_agent.forward(data)

    hook.remove()

    if "layer1_attn" not in captured_attn:
        print("Failed to capture attention weights!")
        return

    attn_weights = captured_attn["layer1_attn"] # Shape: (bsz, seq_len, seq_len)
    print(f"Captured attention weights with shape: {attn_weights.shape}")
    
    # We will visualize the average attention map over the batch (or a single sample)
    # Let's average over the batch to see the general attention pattern
    avg_attn = attn_weights.mean(dim=0).numpy() # Shape: (seq_len, seq_len)
    
    # Alternatively, take the first valid instance in the batch
    # avg_attn = attn_weights[0].numpy()
    
    seq_len = avg_attn.shape[0]

    # Plot
    plt.figure(figsize=(12, 10))
    
    # Mask out the padding (padding has 0 weight effectively) 
    # but seaborn heatmap might show zeros as dark. We can just show it.
    
    # We apply vmin/vmax to make structure clearer
    ax = sns.heatmap(avg_attn, cmap="viridis", vmin=0, vmax=np.percentile(avg_attn, 99))
    
    plt.title(f"Encoder Layer 1 Self-Attention Map\nSingle Decoder")
    
    # If input_option is 4x, seq_len is 400.
    # The tokens correspond to 4 views (100 tokens each).
    if seq_len == 400:
        ticks = [50, 150, 250, 350]
        labels = ["View 1 (TOP)", "View 2 (FRONT)", "View 3 (RIGHT)", "View 4 (ISO)"]
        plt.xticks(ticks, labels, rotation=0)
        plt.yticks(ticks, labels, rotation=90)
        
        # Add visual separator lines
        plt.axhline(100, color='white', linewidth=1.5, linestyle='--')
        plt.axhline(200, color='white', linewidth=1.5, linestyle='--')
        plt.axhline(300, color='white', linewidth=1.5, linestyle='--')
        plt.axvline(100, color='white', linewidth=1.5, linestyle='--')
        plt.axvline(200, color='white', linewidth=1.5, linestyle='--')
        plt.axvline(300, color='white', linewidth=1.5, linestyle='--')
        
    elif seq_len == 300:
        ticks = [50, 150, 250]
        labels = ["View 1", "View 2", "View 3"]
        plt.xticks(ticks, labels, rotation=0)
        plt.yticks(ticks, labels, rotation=90)
        plt.axhline(100, color='white', linewidth=1.5, linestyle='--')
        plt.axhline(200, color='white', linewidth=1.5, linestyle='--')
        plt.axvline(100, color='white', linewidth=1.5, linestyle='--')
        plt.axvline(200, color='white', linewidth=1.5, linestyle='--')
        
    plt.xlabel("Key Positions (Source)")
    plt.ylabel("Query Positions (Target)")
    
    # Ensure save directory exists
    os.makedirs(os.path.dirname(save_name), exist_ok=True)
    plt.savefig(save_name, dpi=300, bbox_inches='tight')
    print(f"Saved visualization to {save_name}")


if __name__ == "__main__":
    os.environ["WANDB_MODE"] = os.environ.get("WANDB_MODE", "offline")
    
    # Parse existing config
    cfg = Config('test')
    
    save_path = os.path.join(cfg.exp_dir, f"encoder_layer1_attention_{os.path.basename(cfg.ckpt).replace('.pth', '')}.png")
    
    visualize_encoder_attention(cfg, save_name=save_path)

"""
python -u visualize_attention.py --input_option 4x --exp_name share_decoder_soft_target_4x_train --batch_size 64 --num_workers 0
"""
