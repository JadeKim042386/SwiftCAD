import os
import argparse
import json
import shutil
from .file_utils import ensure_dirs
from .macro import *

class Config(object):
    def __init__(self, phase):
        self.is_train = phase == "train"

        # init hyperparameters and parse from command-line
        parser, args = self.parse()

        self.set_configuration()

        # set as attributes
        print("----Experiment Configuration-----")
        for k, v in args.__dict__.items():
            print("{0:20}".format(k), v)
            self.__setattr__(k, v)

        # Override d_model / dim_z from CLI (paper ablation supports 144, 192, 256).
        self.d_model = args.d_model
        self.dim_z = args.d_model

        # experiment paths
        self.exp_dir = os.path.join(self.proj_dir, self.exp_name)
        if phase == "train" and args.cont is not True and os.path.exists(self.exp_dir):
            response = input('Experiment log/model already exists, overwrite? (y/n) ')
            if response != 'y':
                exit()
            shutil.rmtree(self.exp_dir)

        self.log_dir = os.path.join(self.exp_dir, 'log')
        self.model_dir = os.path.join(self.exp_dir, 'model')
        ensure_dirs([self.log_dir, self.model_dir])

        # GPU usage
        if args.gpu_ids is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_ids)

        # save this configuration
        if self.is_train:
            with open('{}/config.txt'.format(self.exp_dir), 'w') as f:
                json.dump(args.__dict__, f, indent=2)

    def set_configuration(self):
        self.args_dim = ARGS_DIM # 256
        self.cad_n_args = CAD_N_ARGS
        self.svg_n_args = SVG_N_ARGS
        self.cad_n_commands = len(CAD_COMMANDS)  # Line, Arc, Circle, EOS, SOL, Ext
        self.svg_n_commands = len(SVG_COMMANDS)  # SOS, EOS, L, C

        self.n_layers = 4                # Number of Encoder blocks
        self.n_layers_decode = 4         # Number of Decoder blocks
        self.n_heads = 8                 # Transformer config: number of heads (works for 144, 192, 256)
        self.dim_feedforward = 512       # Transformer config: FF dimensionality
        self.dropout = 0.1               # Dropout rate used in basic layers and Transformers
        # NOTE: self.d_model and self.dim_z are set in __init__ from --d_model CLI arg.

        self.cad_max_n_ext = CAD_MAX_N_EXT
        self.cad_max_n_loops = CAD_MAX_N_LOOPS
        self.cad_max_n_curves = CAD_MAX_N_CURVES

        self.cad_max_total_len = CAD_MAX_TOTAL_LEN
        self.svg_max_total_len = SVG_MAX_TOTAL_LEN

        self.loss_weights = {
            "loss_cmd_weight": 1.0,
            "loss_args_weight": 2.0
        }

        
    def parse(self):
        """initiaize argument parser. Define default hyperparameters and collect from command-line arguments."""
        parser = argparse.ArgumentParser()

        parser.add_argument('--proj_dir', type=str, default="proj_log", help="path to project folder where models and logs will be saved")
        parser.add_argument('--data_root', type=str, default="data", help="path to source data folder")
        parser.add_argument('--exp_name', type=str, default=os.getcwd().split('/')[-1], help="name of this experiment")
        parser.add_argument('-g', '--gpu_ids', type=str, default='0', help="gpu to use, e.g. 0  0,1,2. CPU not supported.")        
        
        parser.add_argument('--batch_size', type=int, default=256, help="batch size")
        parser.add_argument('--num_workers', type=int, default=8, help="number of workers for data loading")

        parser.add_argument('--nr_epochs', type=int, default=200, help="total number of epochs to train")
        parser.add_argument('--lr', type=float, default=1e-3, help="initial learning rate")
        parser.add_argument('--grad_clip', type=float, default=1.0, help="initial learning rate")
        parser.add_argument('--warmup_step', type=int, default=2000, help="step size for learning rate warm up")
        parser.add_argument('--continue', dest='cont',  action='store_true', help="continue training from checkpoint")
        parser.add_argument('--ckpt', type=str, default='latest', required=False, help="desired checkpoint to restore")
        parser.add_argument('--vis', action='store_true', default=False, help="visualize output in training")
        parser.add_argument('--save_frequency', type=int, default=100, help="save models every x epochs")
        parser.add_argument('--val_frequency', type=int, default=50, help="run validation every x iterations")
        parser.add_argument('--vis_frequency', type=int, default=2000, help="visualize output every x iterations")

        parser.add_argument('--input_option', type=str, default="3x", help="number of input views (1x, 3x, 4x)")

        # --- Paper ablation flags (Table 2) ---
        # Shared decoder (paper's main contribution). Default True.
        # Use --no-use_shared_decoder to disable (dual decoder baseline).
        parser.add_argument('--use_shared_decoder', action=argparse.BooleanOptionalAction,
                            default=True, help="use shared decoder (paper main); pass --no-use_shared_decoder for dual decoder baseline")
        # MLP embedding flag. Default False -> SVGEmbeddingNoMlp (paper main).
        # Pass --use_mlp_embedding to use the legacy SVGEmbedding (with MLP).
        parser.add_argument('--use_mlp_embedding', action='store_true', default=False,
                            help="use legacy SVGEmbedding with MLP (Drawing2CAD baseline). Default: SVGEmbeddingNoMlp.")
        # Model dimensionality (paper ablates 144 / 192 / 256).
        parser.add_argument('--d_model', type=int, default=144, choices=[144, 192, 256],
                            help="transformer model dimensionality (paper ablates 144/192/256)")

        args = parser.parse_args()
        return parser, args
