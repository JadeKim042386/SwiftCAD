import torch
import sys
import os
from config.macro import *

# Mock config
class Config:
    def __init__(self):
        self.input_option = "3x"
        self.svg_n_commands = 10
        self.svg_n_args = 10
        self.args_dim = 256
        self.d_model = 256
        self.n_heads = 4
        self.dim_feedforward = 512
        self.dropout = 0.1
        self.n_layers = 2
        self.cad_max_total_len = 100
        self.svg_max_total_len = 100
        self.dim_z = 256
        self.n_layers_decode = 2
        self.cad_n_commands = 10
        self.cad_n_args = 10
        self.log_dir = "proj_log"
        self.model_dir = "proj_log"
        self.batch_size = 1
        self.lr = 1e-4
        self.warmup_step = 100
        self.grad_clip = 1.0
        self.lr_step_size = 100
        self.loss_weights = {'cmd': 1.0, 'args': 1.0}

cfg = Config()

# Mock BaseTrainer to avoid abstract method error if instantiated directly
# But we are importing TrainerED which inherits from BaseTrainer
from trainer.trainer import TrainerED

# Create dummy directories if they don't exist
os.makedirs(cfg.log_dir, exist_ok=True)

print("Instantiating TrainerED...")
try:
    trainer = TrainerED(cfg)
    print("TrainerED instantiated successfully.")
except Exception as e:
    print(f"Error instantiating TrainerED: {e}")
    import traceback
    traceback.print_exc()
