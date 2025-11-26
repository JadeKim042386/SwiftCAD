import torch
import torch.nn as nn
import torch.nn.functional as F
from model.model_utils import _get_padding_mask_cad, _get_visibility_mask
from config.macro import CAD_CMD_ARGS_MASK


class NewCADLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.n_commands = cfg.cad_n_commands
        self.args_dim = cfg.args_dim + 1
        self.weights = cfg.loss_weights

        self.register_buffer("cmd_args_mask", torch.tensor(CAD_CMD_ARGS_MASK))

    def forward(self, outputs, cad_data):
        # Target
        tgt_commands = cad_data["command"].cuda()
        tgt_args = cad_data["args"].cuda()

        visibility_mask = _get_visibility_mask(tgt_commands, seq_dim=-1)
        padding_mask = _get_padding_mask_cad(tgt_commands, seq_dim=-1, extended=True) * visibility_mask.unsqueeze(-1)

        # Prediction
        command_logits = outputs["command_logits"]
        args_logits = outputs["args_logits"]

        mask = self.cmd_args_mask[tgt_commands.long()]

        loss_cmd = F.cross_entropy(command_logits[padding_mask.bool()].reshape(-1, self.n_commands), tgt_commands[padding_mask.bool()].reshape(-1).long())
        # loss_args = gumbel_loss(args_logits, tgt_args, mask)

        # args_logits: (batchsize, 60, 16, 257)
        # tgt_args: (batchsize, 60, 16)
        # mask: (batchsize, 60, 16)
        tgt_args_masked = tgt_args.clone()
        tgt_args_masked[tgt_args_masked == -1] = 0 # mask를 통해 어차피 무시되기 때문에 -1을 0으로 변환 
        loss_args = squared_emd_loss(
            logits=args_logits, 
            labels=tgt_args_masked, 
            num_classes=args_logits.shape[-1], 
            mask=mask
        )

        loss_cmd = self.weights["loss_cmd_weight"] * loss_cmd
        loss_args = loss_args.pow(self.weights["loss_args_weight"])

        res = {"loss_cmd": loss_cmd, "loss_args": loss_args}
        return res
    
def gumbel_loss(pred, target, mask, tolerance=3, alpha=2.0):
    B, S, N_ARGS, N_CLASS = pred.shape
    target += 1

    pred_probs = F.softmax(pred, dim=-1)  # (batchsize, 60, 16, 257)

    target_dist = torch.zeros_like(pred_probs)  # (batchsize, 60, 16, 257)

    for shift in range(-tolerance, tolerance + 1):
        shifted_target = torch.clamp(target + shift, 0, N_CLASS - 1)
        weight = torch.exp(torch.tensor(-alpha * abs(shift), dtype=torch.float32, device='cuda'))
        weight_tensor = weight.unsqueeze(0).expand(B, S, N_ARGS)  # (batchsize, 60, 16)
        target_dist.scatter_(3, shifted_target.unsqueeze(-1), weight_tensor.unsqueeze(-1))

    target_dist = target_dist / target_dist.sum(dim=-1, keepdim=True)

    loss_per_position = -torch.sum(target_dist * torch.log(pred_probs + 1e-9), dim=-1)  # (batchsize, 60, 16)
    loss_valid = (loss_per_position * mask).sum() / mask.sum()

    return loss_valid

def squared_emd_loss_one_hot_labels(y_pred, y_true, mask=None):
    """
    Squared EMD loss that considers the distance between classes as opposed to the cross-entropy
    loss which only considers if a prediction is correct/wrong.

    Squared Earth Mover's Distance-based Loss for Training Deep Neural Networks.
    Le Hou, Chen-Ping Yu, Dimitris Samaras
    https://arxiv.org/abs/1611.05916

    Args:
        y_pred (torch.FloatTensor): Predicted probabilities of shape (batch_size x ... x num_classes)
        y_true (torch.FloatTensor): Ground truth one-hot labels of shape (batch_size x ... x num_classes)
        mask (torch.FloatTensor): Binary mask of shape (batch_size x ...) to ignore elements (e.g. padded values)
                                  from the loss
    
    Returns:
        torch.tensor: Squared EMD loss
    """
    tmp = torch.mean(torch.square(torch.cumsum(y_true, dim=-1) - torch.cumsum(y_pred, dim=-1)), dim=-1)
    if mask is not None:
        tmp = tmp * mask
    return torch.sum(tmp) / tmp.shape[0]

def squared_emd_loss(logits, labels, num_classes=-1, mask=None):
    """
    Squared EMD loss that considers the distance between classes as opposed to the cross-entropy
    loss which only considers if a prediction is correct/wrong.

    Squared Earth Mover's Distance-based Loss for Training Deep Neural Networks.
    Le Hou, Chen-Ping Yu, Dimitris Samaras
    https://arxiv.org/abs/1611.05916

    Args:
        logits (torch.FloatTensor): Predicted logits of shape (batch_size x ... x num_classes)
        labels (torch.LongTensor): Ground truth class labels of shape (batch_size x ...)
        mask (torch.FloatTensor): Binary mask of shape (batch_size x ...) to ignore elements (e.g. padded values)
                                  from the loss
    
    Returns:
        torch.tensor: Squared EMD loss
    """
    y_pred = torch.softmax(logits, dim=-1) # (batchsize, 60, 16, 257)
    y_true = F.one_hot(labels, num_classes=num_classes).float() # (batchsize, 60, 16)
    return squared_emd_loss_one_hot_labels(y_pred, y_true, mask=mask)
