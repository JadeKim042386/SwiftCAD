import torch
import numpy as np
import umap
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
import sys
from sklearn.preprocessing import Normalizer

# Add current directory to path
sys.path.append(os.getcwd())

from dataset.bi_sequence_dataset import get_dataloader
from config.config import Config
from trainer.trainer import TrainerED
from config.macro import *

def visualize_umap(cfg, n_batches=10, save_name="umap_visualization.png"):
    print(f"Initializing model for experiment: {cfg.exp_name}...")
    
    tr_agent = TrainerED(cfg)
    
    print(f"Loading checkpoint: {cfg.ckpt}...")
    # Determine load path precisely like BaseTrainer.load_ckpt
    name = cfg.ckpt if cfg.ckpt == 'latest' else "ckpt_epoch{}".format(cfg.ckpt)
    load_path = os.path.join(tr_agent.model_dir, "{}.pth".format(name))
    if not os.path.exists(load_path):
        raise ValueError("Checkpoint {} not exists.".format(load_path))

    checkpoint = torch.load(load_path)
    print("Loading model weights from {} ...".format(load_path))
    if 'model_state_dict' in checkpoint:
        tr_agent.net.load_state_dict(checkpoint['model_state_dict'])
    else:
        tr_agent.net.load_state_dict(checkpoint)
    tr_agent.net.eval()

    embeddings = []
    labels = []

    # Forward hook to capture "out" from the decoder
    captured_out = {}
    
    def get_hook(name):
        def hook_fn(module, input, output):
            # output of TransformerDecoder is (S, N, d_model)
            captured_out[name] = output.detach().cpu()
        return hook_fn

    # Hooking the TransformerDecoder inside Decoder
    target_module = tr_agent.net.decoder.decoder
    print("Hooking tr_agent.net.decoder.decoder")
    
    hook = target_module.register_forward_hook(get_hook("out"))

    test_loader = get_dataloader("test", cfg)
    print(f"Total number of test batches: {len(test_loader)}")

    # Collect embeddings
    for i, data in enumerate(tqdm(test_loader, total=n_batches, desc="Collecting embeddings")):
        if i >= n_batches:
            break
        
        with torch.no_grad():
            tr_agent.forward(data)
            
        if "out" not in captured_out:
            print("Hook was not triggered!")
            continue
            
        # captured_out: shape (S, N, d_model)
        out = captured_out["out"] # (S, N, d_model)
        
        # Get labels from GT commands
        # Matching test.py's data handling:
        cad_data = data['cad']
        # In test.py: cad_data['command'].unsqueeze(-1) suggests cad_data['command'] is (N, S) or (N, 1, S)
        # In trainer.py evaluate: cad_command.squeeze(1) suggests (N, 1, S)
        cad_command = cad_data['command']
        if cad_command.dim() == 3:
            cad_command = cad_command.squeeze(1) # (N, S)
        
        # Reshape: (S*N, d_model)
        S, N, D = out.shape
        # Permute to (N, S, D) then reshape to (N*S, D)
        out = out.permute(1, 0, 2).reshape(-1, D)
        
        # Reshape labels to (N*S)
        cmd_labels = cad_command.reshape(-1)
        
        embeddings.append(out.numpy())
        labels.append(cmd_labels.cpu().numpy())

    hook.remove()

    if not embeddings:
        print("No embeddings collected.")
        return

    embeddings = np.concatenate(embeddings, axis=0)
    labels = np.concatenate(labels, axis=0)

    # Filter out EOS and SOL for clearer primitives visualization if needed
    # CAD_COMMANDS = ['Line', 'Arc', 'Circle', 'EOS', 'SOL', 'Ext']
    # For now keep all to see full distribution

    print(f"Total points collected: {embeddings.shape[0]}")
    
    # Subsample if too many points for UMAP/Plotting
    if embeddings.shape[0] > 3000:
        idx = np.random.choice(embeddings.shape[0], 3000, replace=False)
        embeddings = embeddings[idx]
        labels = labels[idx]
        print(f"Subsampled to 3,000 points.")

    print("Running UMAP dimensionality reduction...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.05, n_components=2, random_state=42, metric='cosine')
    normalized_embeddings = Normalizer().fit(embeddings).transform(embeddings)
    embedding_2d = reducer.fit_transform(normalized_embeddings)

    # Plot
    plt.figure(figsize=(12, 10))
    unique_labels = np.unique(labels)
    
    # Use seaborn palette for consistent colors across classes
    palette = sns.color_palette("hls", len(CAD_COMMANDS))
    
    CAD_EOS_IDX = 3
    CAD_SOL_IDX = 4

    for label in unique_labels:
        if label >= len(CAD_COMMANDS) or label < 0: continue
        # Exclude EOS, SOL
        if label == CAD_EOS_IDX or label == CAD_SOL_IDX: continue
        mask = labels == label
        plt.scatter(embedding_2d[mask, 0], embedding_2d[mask, 1], 
                    label=CAD_COMMANDS[label], s=10, alpha=0.6, color=palette[label])

    plt.legend()
    plt.title(f"UMAP Visualization of Decoder Embeddings\nExp: {cfg.exp_name} | CKPT: {cfg.ckpt}")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.savefig(save_name, dpi=300, bbox_inches='tight')
    print(f"Saved visualization to {save_name}")

if __name__ == "__main__":
    # Ensure offline mode for wandb unless specified
    os.environ["WANDB_MODE"] = os.environ.get("WANDB_MODE", "offline")
    
    # Parse config
    cfg = Config('test')
    
    # Default values for visualization
    n_batches = 10
    save_path = os.path.join(cfg.exp_dir, f"umap_vis_{os.path.basename(cfg.ckpt).replace('.pth', '')}.png")
    
    visualize_umap(cfg, n_batches=n_batches, save_name=save_path)

"""
python -u visualize_umap.py --input_option 4x --exp_name share_decoder_soft_target_4x_train --batch_size 64 --num_workers 0
"""