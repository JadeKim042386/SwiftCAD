from .layers.transformer import *
from .layers.improved_transformer import *
from .layers.positional_encoding import *
from .model_utils import _make_seq_first, _make_batch_first, \
    _get_padding_mask_svg, _get_key_padding_mask_svg


class SVGEmbedding(nn.Module):
    """Embedding: view embed + command embed + parameter embed + positional embed"""
    def __init__(self, cfg, seq_len):
        super().__init__()

        """concatenation-based"""
        # 3x or 4x
        if cfg.input_option == "3x" or cfg.input_option == "4x":
            self.view_embed = nn.Embedding(4, 4)
            self.command_embed = nn.Embedding(cfg.svg_n_commands, 8)
        # 1x: keep dimension constant with other input option
        if cfg.input_option == "1x":
            self.command_embed = nn.Embedding(cfg.svg_n_commands, 12)

        args_dim = cfg.args_dim + 1
        self.args_embed = nn.Embedding(args_dim, 64, padding_idx=0)
        self.args_mlp = nn.Linear(64 * cfg.svg_n_args, 128)
        self.mlp = nn.Linear(4 + 8 + 128, cfg.d_model)
        self.pos_encoding = PositionalEncodingLUT(cfg.d_model, max_len=seq_len + 2)
        
    
    def forward(self, view, command, args):
        assert command.shape == view.shape
        S, N = command.shape

        command_embedding = self.command_embed(command.long())
        args_embedding = self.args_mlp(self.args_embed((args + 1).long()).view(S, N, -1))

        """concatenation-based"""
        # 1x
        if S == 100:
            src = torch.cat([command_embedding, args_embedding], dim=-1)
        # 3x or 4x
        if S > 100:
            view_embedding = self.view_embed(view.long())
            src = torch.cat([view_embedding, command_embedding, args_embedding], dim=-1)
        
        src = self.mlp(src)
        src = self.pos_encoding(src)

        return src

class SVGEmbeddingNoMlp(nn.Module):
    def __init__(self, cfg, seq_len):
        super().__init__()

        if cfg.input_option in ["3x", "4x"]:
            self.view_embed = nn.Embedding(4, 4)
            self.command_embed = nn.Embedding(cfg.svg_n_commands, 8)

        if cfg.input_option == "1x":
            self.command_embed = nn.Embedding(cfg.svg_n_commands, 12)

        args_dim = cfg.args_dim + 1
        self.args_embed = nn.Embedding(args_dim, 64, padding_idx=0)
        # Dynamic: d_model - view_embed(4) - command_embed(8) = args_mlp_out
        args_mlp_out = cfg.d_model - 12  # 12 = view(4) + cmd(8), same as 1x: cmd(12)
        self.args_mlp = nn.Linear(64 * cfg.svg_n_args, args_mlp_out)

        self.decomposed_pe = getattr(cfg, 'encoder_type', 'alternating') == 'alternating'
        self.tokens_per_view = cfg.svg_max_total_len  # 100
        self.input_option = cfg.input_option

        if self.decomposed_pe:
            # Decomposed PE: intra-view PE (0-99) repeated per view
            self.pos_encoding = PositionalEncodingLUT(cfg.d_model, max_len=cfg.svg_max_total_len + 2)
        else:
            # Global PE: unique position for each token (0-299 for 3x)
            self.pos_encoding = PositionalEncodingLUT(cfg.d_model, max_len=seq_len + 2)

    def forward(self, view, command, args):
        S, N = command.shape
        command_embedding = self.command_embed(command.long())
        args_embedding = self.args_mlp(self.args_embed((args + 1).long()).view(S, N, -1))

        if S == self.tokens_per_view:  # 1x
            src = torch.cat([command_embedding, args_embedding], dim=-1)
            src = self.pos_encoding(src)
        else:  # 3x or 4x
            view_embedding = self.view_embed(view.long())
            src = torch.cat([view_embedding, command_embedding, args_embedding], dim=-1)
            if self.decomposed_pe:
                # Apply intra-view PE: same PE(0-99) repeated for each view
                num_views = S // self.tokens_per_view
                src_views = src.view(num_views, self.tokens_per_view, N, -1)
                for v in range(num_views):
                    src_views[v] = self.pos_encoding(src_views[v])
                src = src_views.view(S, N, -1)
            else:
                # Apply global PE over full sequence
                src = self.pos_encoding(src)
        return src
    
class ConstEmbedding(nn.Module):
    """learned constant embedding"""
    def __init__(self, cfg):
        super().__init__()

        self.d_model = cfg.d_model
        self.PE = PositionalEncodingLUT(cfg.d_model, max_len=cfg.cad_max_total_len)
        self.seq_len = cfg.cad_max_total_len

    def forward(self, z):
        N = z.size(1)
        src = self.PE(z.new_zeros(self.seq_len, N, self.d_model))
        return src

class Encoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.view_num = int(cfg.input_option[0])
        seq_len = self.view_num * cfg.svg_max_total_len
        self.encoder_type = getattr(cfg, 'encoder_type', 'alternating')
        self.decoder_type = getattr(cfg, 'decoder_type', 'cross_attention')

        self.embedding = SVGEmbeddingNoMlp(cfg, seq_len)

        encoder_norm = LayerNorm(cfg.d_model)

        if self.encoder_type == 'alternating':
            # Alternating attention encoder: frame-wise + global layers
            frame_layers = [
                TransformerEncoderLayerImproved(cfg.d_model, cfg.n_heads, cfg.dim_feedforward, cfg.dropout)
                for _ in range(cfg.n_layers)
            ]
            global_layers = [
                TransformerEncoderLayerImproved(cfg.d_model, cfg.n_heads, cfg.dim_feedforward, cfg.dropout)
                for _ in range(cfg.n_layers)
            ]
            self.encoder = AlternatingTransformerEncoder(frame_layers, global_layers, encoder_norm)
        else:
            # Standard encoder: single stack of self-attention layers
            encoder_layer = TransformerEncoderLayerImproved(cfg.d_model, cfg.n_heads, cfg.dim_feedforward, cfg.dropout)
            self.encoder = TransformerEncoder(encoder_layer, cfg.n_layers, encoder_norm)

    def forward(self, view, command, args):
        assert command.shape == view.shape
        padding_mask = _get_padding_mask_svg(command, seq_dim=0)       # (S, N, 1), True=valid
        key_padding_mask = _get_key_padding_mask_svg(command, seq_dim=0)  # (N, S), True=ignore

        src = self.embedding(view, command, args)

        if self.encoder_type == 'alternating':
            memory = self.encoder(src, num_views=self.view_num, src_key_padding_mask=key_padding_mask)
        else:
            memory = self.encoder(src, src_key_padding_mask=key_padding_mask)

        if self.decoder_type == 'broadcast':
            # Mean pool (padding-aware) → z (1, N, D)
            mask = padding_mask.float()  # (S, N, 1), 1=valid, 0=pad
            z = (memory * mask).sum(dim=0, keepdim=True) / mask.sum(dim=0, keepdim=True).clamp(min=1)
            return z, None
        else:
            return memory, key_padding_mask

class CommandFCN(nn.Module):
    def __init__(self, d_model, n_commands):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, n_commands)
        )
    def forward(self, out):
        command_logits = self.mlp(out)  # Shape [S, N, n_commands]

        return command_logits

class ArgsFCN(nn.Module):
    def __init__(self, d_model, n_args, args_dim=256):
        super().__init__()

        self.n_args = n_args
        self.args_dim = args_dim

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, n_args * args_dim)
        )
    def forward(self, out):
        S, N, _ = out.shape

        args_logits = self.mlp(out)  # Shape [S, N, n_args * args_dim]
        args_logits = args_logits.reshape(S, N, self.n_args, self.args_dim)  # Shape [S, N, n_args, args_dim]

        return args_logits

class CommandDecoder(nn.Module):
    def __init__(self, cfg):
        super(CommandDecoder, self).__init__()

        self.embedding = ConstEmbedding(cfg)

        decoder_layer = TransformerDecoderLayerGlobalImproved(cfg.d_model, cfg.dim_z, cfg.n_heads, cfg.dim_feedforward, cfg.dropout)
        decoder_norm = LayerNorm(cfg.d_model)
        self.decoder = TransformerDecoder(decoder_layer, cfg.n_layers_decode, decoder_norm)

        self.fcn = CommandFCN(cfg.d_model, cfg.cad_n_commands)

    def forward(self, z):
        src = self.embedding(z)
        out = self.decoder(src, z, tgt_mask=None, tgt_key_padding_mask=None)

        command_logits = self.fcn(out)

        # guidance
        return command_logits, out

class ArgsDecoder(nn.Module):
    def __init__(self, cfg):
        super(ArgsDecoder, self).__init__()

        self.embedding = ConstEmbedding(cfg)

        decoder_layer = TransformerDecoderLayerGlobalImproved(cfg.d_model, cfg.dim_z, cfg.n_heads, cfg.dim_feedforward, cfg.dropout)
        decoder_norm = LayerNorm(cfg.d_model)
        self.decoder = TransformerDecoder(decoder_layer, cfg.n_layers_decode, decoder_norm)

        args_dim = cfg.args_dim + 1
        self.fcn = ArgsFCN(cfg.d_model, cfg.cad_n_args, args_dim)

    def forward(self, z, guidance):
        src = self.embedding(z)
        out = self.decoder(src, z, tgt_mask=None, tgt_key_padding_mask=None)

        # guidance
        out = out + guidance

        args_logits = self.fcn(out)

        return args_logits

class Decoder(nn.Module):
    """Cross-attention decoder: attends to full encoder memory."""
    def __init__(self, cfg):
        super(Decoder, self).__init__()

        self.embedding = ConstEmbedding(cfg)

        decoder_layer = TransformerDecoderLayerImproved(cfg.d_model, cfg.n_heads, cfg.dim_feedforward, cfg.dropout)
        decoder_norm = LayerNorm(cfg.d_model)
        self.decoder = TransformerDecoder(decoder_layer, cfg.n_layers_decode, decoder_norm)

        self.fcn_cmd = CommandFCN(cfg.d_model, cfg.cad_n_commands)
        args_dim = cfg.args_dim + 1
        self.fcn_args = ArgsFCN(cfg.d_model, cfg.cad_n_args, args_dim)

    def forward(self, memory, memory_key_padding_mask=None):
        src = self.embedding(memory)
        out = self.decoder(src, memory, tgt_mask=None, tgt_key_padding_mask=None,
                           memory_key_padding_mask=memory_key_padding_mask)

        command_logits = self.fcn_cmd(out)
        args_logits = self.fcn_args(out)

        return command_logits, args_logits


class DecoderBroadcast(nn.Module):
    """Broadcast decoder: receives mean-pooled z and broadcasts to all positions."""
    def __init__(self, cfg):
        super(DecoderBroadcast, self).__init__()

        self.embedding = ConstEmbedding(cfg)

        decoder_layer = TransformerDecoderLayerGlobalImproved(cfg.d_model, cfg.dim_z, cfg.n_heads, cfg.dim_feedforward, cfg.dropout)
        decoder_norm = LayerNorm(cfg.d_model)
        self.decoder = TransformerDecoder(decoder_layer, cfg.n_layers_decode, decoder_norm)

        self.fcn_cmd = CommandFCN(cfg.d_model, cfg.cad_n_commands)
        args_dim = cfg.args_dim + 1
        self.fcn_args = ArgsFCN(cfg.d_model, cfg.cad_n_args, args_dim)

    def forward(self, z, memory_key_padding_mask=None):
        src = self.embedding(z)
        out = self.decoder(src, z, tgt_mask=None, tgt_key_padding_mask=None)

        command_logits = self.fcn_cmd(out)
        args_logits = self.fcn_args(out)

        return command_logits, args_logits

class Bottleneck(nn.Module):
    def __init__(self, cfg):
        super(Bottleneck, self).__init__()

        self.bottleneck = nn.Sequential(nn.Linear(cfg.d_model, cfg.d_model // 2),
                                        nn.GELU(),
                                        nn.Linear(cfg.d_model // 2, cfg.d_model))

    def forward(self, z):
        return z + self.bottleneck(z)

class SVG2CADTransformer(nn.Module):
    def __init__(self, cfg):
        super(SVG2CADTransformer, self).__init__()

        self.args_dim = cfg.args_dim + 1
        self.decoder_type = getattr(cfg, 'decoder_type', 'cross_attention')
        self.use_bottleneck = getattr(cfg, 'use_bottleneck', False)

        self.encoder = Encoder(cfg)

        if self.decoder_type == 'cross_attention':
            self.decoder = Decoder(cfg)
        else:
            self.decoder = DecoderBroadcast(cfg)

        if self.use_bottleneck and self.decoder_type == 'cross_attention':
            self.bottleneck = Bottleneck(cfg)

    def forward(self, views_enc, commands_enc, args_enc):
        views_enc_, commands_enc_, args_enc_ = _make_seq_first(views_enc, commands_enc, args_enc)

        memory_or_z, enc_key_padding_mask = self.encoder(views_enc_, commands_enc_, args_enc_)

        if self.use_bottleneck and self.decoder_type == 'cross_attention':
            memory_or_z = self.bottleneck(memory_or_z)

        command_logits, args_logits = self.decoder(memory_or_z, enc_key_padding_mask)
        command_logits = _make_batch_first(command_logits)
        args_logits = _make_batch_first(args_logits)

        res = {
            "command_logits": command_logits,
            "args_logits": args_logits
        }

        return res
