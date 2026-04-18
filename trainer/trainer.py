import wandb
import os
import math
import torch
import torch.optim as optim
from tqdm import tqdm
from model.model import SVG2CADTransformer
from .base import BaseTrainer
from .loss import NewCADLoss
from .scheduler import GradualWarmupScheduler
from config.macro import *
import torch.nn as nn


def one_cycle(y1=0.0, y2=1.0, steps=100):
    # lambda function for sinusoidal ramp from y1 to y2 https://arxiv.org/pdf/1812.01187.pdf
    return lambda x: ((1 - math.cos(x * math.pi / steps)) / 2) * (y2 - y1) + y1

class TrainerED(BaseTrainer):
    def build_net(self, cfg):
        self.net = SVG2CADTransformer(cfg).cuda()

        # Load pretrained checkpoint for mask-predict training
        use_mp = getattr(cfg, 'use_mask_predict', False)
        pretrained_path = 'proj_log/variant_e_alt_cross_4x/model/latest.pth'
        if use_mp and cfg.is_train and os.path.exists(pretrained_path):
            print(f"Loading pretrained weights from {pretrained_path} for mask-predict...")
            ckpt = torch.load(pretrained_path, map_location='cuda', weights_only=False)
            missing, unexpected = self.net.load_state_dict(ckpt['model_state_dict'], strict=False)
            print(f"  Missing keys (new params): {len(missing)}")
            print(f"  Unexpected keys: {len(unexpected)}")

        # Freeze pretrained parameters if requested
        freeze = getattr(cfg, 'freeze_pretrained', False)
        if freeze and use_mp:
            new_param_names = {'decoder.embedding.command_embed', 'decoder.embedding.args_embed',
                               'decoder.embedding.args_proj', 'decoder.embedding.mask_token'}
            for name, param in self.net.named_parameters():
                is_new = any(name.startswith(prefix) for prefix in new_param_names)
                param.requires_grad = is_new
            trainable = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
            frozen = sum(p.numel() for p in self.net.parameters() if not p.requires_grad)
            print(f"Freeze mode: trainable={trainable:,} frozen={frozen:,}")

        # Total number of model parameters
        total_params = sum(p.numel() for p in self.net.parameters())
        print(f"Total parameters: {total_params:,} ({total_params * 4 / (1024**2):.2f} MB with float32)")

    def set_optimizer(self, cfg):
        """set optimizer and lr scheduler used in training"""
        self.optimizer = optim.Adam(self.net.parameters(), cfg.lr)

        # self.scheduler = GradualWarmupScheduler(self.optimizer, 1.0, cfg.warmup_step)
        lf = one_cycle(1, 1e-3, cfg.nr_epochs)
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lf)

    def set_loss_function(self):
        self.loss_func = NewCADLoss(self.cfg).cuda()


    def forward(self, data):
        cad_data = data['cad']
        svg_data = data['svg']
        svg_view = svg_data['view'].cuda()
        svg_command = svg_data['command'].cuda()
        svg_args = svg_data['args'].cuda()

        use_mp = getattr(self.cfg, 'use_mask_predict', False)

        if use_mp and self.net.training:
            return self._forward_mask_predict(svg_view, svg_command, svg_args, cad_data)

        outputs = self.net(svg_view, svg_command, svg_args)
        loss_dict = self.loss_func(outputs, cad_data)
        return outputs, loss_dict

    def _forward_mask_predict(self, svg_view, svg_command, svg_args, cad_data):
        """Mask-Predict training: initial pass + one refinement pass with random masking."""
        # Initial pass (no refinement)
        outputs = self.net(svg_view, svg_command, svg_args)
        loss_dict_init = self.loss_func(outputs, cad_data)

        # Build refinement input from GT with random masking
        tgt_commands = cad_data['command'].cuda()  # (N, S)
        tgt_args = cad_data['args'].cuda()  # (N, S, n_args)
        N, S = tgt_commands.shape

        # Random mask ratio per sample
        mask_ratio = torch.empty(N, 1, device=tgt_commands.device).uniform_(0.15, 0.85)
        rand_scores = torch.rand(N, S, device=tgt_commands.device)
        refinement_mask = rand_scores < mask_ratio  # True = masked

        # Convert to seq-first for decoder
        prev_cmd_T = tgt_commands.transpose(0, 1)  # (S, N)
        prev_args_T = tgt_args.permute(1, 0, 2)  # (S, N, n_args)
        mask_T = refinement_mask.transpose(0, 1)  # (S, N)

        # Encoder (reuse)
        from model.model_utils import _make_seq_first, _make_batch_first
        views_enc_, commands_enc_, args_enc_ = _make_seq_first(svg_view, svg_command, svg_args)
        memory_or_z, enc_key_padding_mask = self.net.encoder(views_enc_, commands_enc_, args_enc_)
        if self.net.use_bottleneck and self.net.decoder_type == 'cross_attention':
            memory_or_z = self.net.bottleneck(memory_or_z)

        # Refinement pass
        cmd_logits_ref, args_logits_ref = self.net.decoder(
            memory_or_z, enc_key_padding_mask,
            prev_cmd=prev_cmd_T, prev_args=prev_args_T, mask_positions=mask_T
        )
        cmd_logits_ref = _make_batch_first(cmd_logits_ref)
        args_logits_ref = _make_batch_first(args_logits_ref)
        outputs_ref = {"command_logits": cmd_logits_ref, "args_logits": args_logits_ref}

        loss_dict_ref = self.loss_func(outputs_ref, cad_data, refinement_mask=refinement_mask)

        # Combine losses: initial + refinement
        loss_dict = {
            "loss_cmd": loss_dict_init["loss_cmd"] + loss_dict_ref["loss_cmd"],
            "loss_args": loss_dict_init["loss_args"] + loss_dict_ref["loss_args"],
        }

        return outputs, loss_dict
    

    def logits2vec(self, outputs, refill_pad=True, to_numpy=True):
        """network outputs (logits) to final CAD vector"""
        out_command = torch.argmax(torch.softmax(outputs['command_logits'], dim=-1), dim=-1)  # (N, S)
        out_args = torch.argmax(torch.softmax(outputs['args_logits'], dim=-1), dim=-1) - 1  # (N, S, N_ARGS)
        if refill_pad: # fill all unused element to -1
            mask = ~torch.tensor(CAD_CMD_ARGS_MASK).bool().cuda()[out_command.long()]
            out_args[mask] = -1

        out_cad_vec = torch.cat([out_command.unsqueeze(-1), out_args], dim=-1)
        if to_numpy:
            out_cad_vec = out_cad_vec.detach().cpu().numpy()
        return out_cad_vec

    def evaluate(self, test_loader):
        """evaluatinon during training"""
        self.net.eval()
        pbar = tqdm(test_loader)
        pbar.set_description("EVALUATE[{}]".format(self.clock.epoch))

        all_ext_args_comp = []
        all_line_args_comp = []
        all_arc_args_comp = []
        all_circle_args_comp = []

        for i, data in enumerate(pbar):
            cad_data = data['cad']
            svg_data = data['svg']
            with torch.no_grad():
                svg_view = svg_data['view'].cuda()
                svg_command = svg_data['command'].cuda()
                svg_args = svg_data['args'].cuda()
                cad_command = cad_data['command']
                cad_args = cad_data['args']
                outputs = self.net(svg_view, svg_command, svg_args)

            out_args = torch.argmax(torch.softmax(outputs['args_logits'], dim=-1), dim=-1) - 1
            out_args = out_args.long().detach().cpu().numpy()  # (N, S, n_args)

            gt_commands = cad_command.squeeze(1).long().detach().cpu().numpy() # (N, S)
            gt_args = cad_args.squeeze(1).long().detach().cpu().numpy() # (N, S, n_args)

            ext_pos = np.where(gt_commands == CAD_EXT_IDX)
            line_pos = np.where(gt_commands == CAD_LINE_IDX)
            arc_pos = np.where(gt_commands == CAD_ARC_IDX)
            circle_pos = np.where(gt_commands == CAD_CIRCLE_IDX)

            args_comp = (gt_args == out_args).astype(np.int32)
            all_ext_args_comp.append(args_comp[ext_pos][:, -CAD_N_ARGS_EXT:])
            all_line_args_comp.append(args_comp[line_pos][:, :2])
            all_arc_args_comp.append(args_comp[arc_pos][:, :4])
            all_circle_args_comp.append(args_comp[circle_pos][:, [0, 1, 4]])

        all_ext_args_comp = np.concatenate(all_ext_args_comp, axis=0)
        sket_plane_acc = np.mean(all_ext_args_comp[:, :CAD_N_ARGS_PLANE])
        sket_trans_acc = np.mean(all_ext_args_comp[:, CAD_N_ARGS_PLANE:CAD_N_ARGS_PLANE+CAD_N_ARGS_TRANS])
        extent_one_acc = np.mean(all_ext_args_comp[:, -CAD_N_ARGS_EXT_PARAM])
        line_acc = np.mean(np.concatenate(all_line_args_comp, axis=0))
        arc_acc = np.mean(np.concatenate(all_arc_args_comp, axis=0))
        circle_acc = np.mean(np.concatenate(all_circle_args_comp, axis=0))

        wandb.log({
            "args_acc/line": line_acc, 
            "args_acc/arc": arc_acc, 
            "args_acc/circle": circle_acc,
            "args_acc/plane": sket_plane_acc, 
            "args_acc/trans": sket_trans_acc, 
            "args_acc/extent": extent_one_acc
        }, step=self.clock.step)
