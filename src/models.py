import torch
from torch import nn
from torchvision.ops import FeaturePyramidNetwork as FPN
from transformers import DistilBertModel

from collections import OrderedDict

RECONSTRUCTION = {
    "av": {"input": "audio", "output": "video"},
    "va": {"input": "video", "output": "audio"},
    "vv": {"input": "video", "output": "video"},
    "aa": {"input": "audio", "output": "audio"},
}


class Mask(nn.Module):

    def __init__(self, num_tokens, win_size, device):
        super().__init__()
        self.num_tokens = num_tokens
        self.win_size = win_size
        self.device = device
        self.mask = self.construct_mask(num_tokens, win_size).to(device)

    def construct_mask(self, num_tokens, win_size):
        return torch.stack(
            [
                torch.cat(
                    (
                        torch.ones((i,), dtype=torch.bool),
                        torch.zeros((win_size,), dtype=torch.bool),
                        torch.ones(num_tokens - i, dtype=torch.bool),
                    )
                )
                for i in range(num_tokens)
            ],
            dim=0,
        )[:, win_size // 2 : -win_size // 2]

    def forward(self, factor):
        if self.win_size > 0:
            return self.mask[::factor, ::factor]
        else:
            return None


class DownsamplingTransformerBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_heads,
        attn_mask=None,
        kernel_size=2,
        stride=2,
        dropout=0.0,
        attn_dropout=0.0,
        ff_expansion=4,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(in_channels)
        self.attn = nn.MultiheadAttention(in_channels, num_heads, dropout=attn_dropout, batch_first=True)
        self.attn_mask = attn_mask
        self.norm2 = nn.LayerNorm(in_channels)
        self.ff = nn.Sequential(
            nn.Linear(in_channels, in_channels * ff_expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_channels * ff_expansion, in_channels),
            nn.Dropout(dropout),
        )

        # Downsampling layer (using Conv1d)
        self.downsample = nn.Conv1d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride
        )  # 1D convolution for sequence
        self.norm3 = nn.LayerNorm(out_channels)

    def forward(self, x):
        # Input shape: (batch_size, seq_len, in_channels)

        # Attention
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, attn_mask=self.attn_mask)
        x = x + attn_out

        # Feed-forward
        x_norm = self.norm2(x)
        ff_out = self.ff(x_norm)
        x = x + ff_out

        # Downsampling (with Conv1d)
        x = x.permute(0, 2, 1)  # Reshape for Conv1d: (B, C, L)
        x = self.downsample(x)  # Downsample along the sequence dimension
        x = x.permute(0, 2, 1)  # Reshape back: (B, L_new, C_new)
        x = self.norm3(x)

        return x


class UpsamplingTransformerBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        num_heads,
        attn_mask=None,
        dropout=0.0,
        attn_dropout=0.0,
        ff_expansion=4,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(in_channels)
        self.attn = nn.MultiheadAttention(in_channels, num_heads, dropout=attn_dropout, batch_first=True)
        self.attn_mask = attn_mask
        self.norm2 = nn.LayerNorm(in_channels)
        self.ff = nn.Sequential(
            nn.Linear(in_channels, in_channels * ff_expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_channels * ff_expansion, in_channels),
            nn.Dropout(dropout),
        )

        # Downsampling layer (using Conv1d)
        self.upsample = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=2, stride=2)
        self.norm3 = nn.LayerNorm(out_channels)

    def forward(self, x):
        # Input shape: (batch_size, seq_len, in_channels)

        # Attention
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, attn_mask=self.attn_mask)
        x = x + attn_out

        # Feed-forward
        x_norm = self.norm2(x)
        ff_out = self.ff(x_norm)
        x = x + ff_out

        # Downsampling (with Conv1d)
        x = x.permute(0, 2, 1)  # Reshape for Conv1d: (B, C, L)
        x = self.upsample(x)  # Downsample along the sequence dimension
        x = x.permute(0, 2, 1)  # Reshape back: (B, L_new, C_new)
        x = self.norm3(x)

        return x


class ConvBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride, padding, timesteps, dropout, use_ln, use_rl, use_do
    ):
        super(ConvBlock, self).__init__()
        self.use_ln = use_ln
        self.use_rl = use_rl
        self.use_do = use_do
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        if use_ln:
            self.norm = nn.LayerNorm([out_channels, timesteps])
        if use_rl:
            self.relu = nn.ReLU()
        if use_do:
            self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.conv(x)
        if self.use_ln:
            x = self.norm(x)
        if self.use_rl:
            x = self.relu(x)
        if self.use_do:
            x = self.dropout(x)
        return x


class ConvTransposeBlock(nn.Module):

    def __init__(
        self, in_channels, out_channels, kernel_size, stride, padding, timesteps, dropout, use_ln, use_rl, use_do
    ):
        super(ConvTransposeBlock, self).__init__()
        self.use_ln = use_ln
        self.use_rl = use_rl
        self.use_do = use_do
        self.deconv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding, output_padding=1)
        if self.use_ln:
            self.norm = nn.LayerNorm([out_channels, timesteps])
        if self.use_rl:
            self.relu = nn.ReLU()
        if self.use_do:
            self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.deconv(x)
        if self.use_ln:
            x = self.norm(x)
        if self.use_rl:
            x = self.relu(x)
        if self.use_do:
            x = self.dropout(x)
        return x


class Projection(nn.Module):

    def __init__(self, nlayers, in_channels, out_channels, timesteps, use_ln, use_rl, use_do, dropout, pre):
        super().__init__()
        self.projection = nn.Sequential(
            *nn.ModuleList(
                [
                    ConvBlock(
                        in_channels=in_channels if not pre or (pre and i == 0) else out_channels,
                        out_channels=out_channels if pre or (not pre and i == nlayers - 1) else in_channels,
                        kernel_size=3,
                        stride=1,
                        padding="same",
                        timesteps=timesteps,
                        dropout=dropout,
                        use_ln=use_ln,
                        use_rl=use_rl,
                        use_do=use_do,
                    )
                    for i in range(nlayers)
                ]
            )
        )

    def forward(self, x):
        return self.projection(x)


class TransformerDownUpCrossModalReconstructionModel(nn.Module):

    def __init__(
        self,
        input_dimension,
        d_model,
        nlayers,
        num_heads,
        max_length,
        win_size,
        use_ln,
        use_rl,
        use_do,
        dropout,
        device,
    ):
        super().__init__()
        self.projection_pre = Projection(
            nlayers=nlayers["pre"],
            in_channels=input_dimension,
            out_channels=d_model,
            timesteps=max_length,
            use_ln=use_ln,
            use_rl=use_rl,
            use_do=use_do,
            dropout=dropout,
            pre=True,
        )
        self.projection_post = Projection(
            nlayers=nlayers["post"],
            in_channels=d_model,
            out_channels=input_dimension,
            timesteps=max_length,
            use_ln=use_ln,
            use_rl=use_rl,
            use_do=use_do,
            dropout=dropout,
            pre=False,
        )
        self.attn_mask = Mask(num_tokens=max_length, win_size=win_size, device=device)
        self.pos_embed = nn.Embedding(max_length, d_model)
        self.positions = torch.arange(0, max_length).unsqueeze(0).to(device)
        self.encoder = nn.Sequential(
            *nn.ModuleList(
                [
                    DownsamplingTransformerBlock(
                        in_channels=d_model,
                        out_channels=d_model,
                        num_heads=num_heads,
                        attn_mask=self.attn_mask(2**i),
                        dropout=dropout,
                    )
                    for i in range(nlayers["downsample"])
                ]
            )
        )
        self.decoder = nn.Sequential(
            *nn.ModuleList(
                [
                    UpsamplingTransformerBlock(
                        in_channels=d_model,
                        out_channels=d_model,
                        num_heads=num_heads,
                        attn_mask=self.attn_mask(2 ** (nlayers["upsample"] - i)),
                        dropout=dropout,
                    )
                    for i in range(nlayers["upsample"])
                ]
            )
        )

    def forward(self, x):
        x = self.projection_pre(x)
        x = x.permute(0, 2, 1)
        x = x + self.pos_embed(self.positions)
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.permute(0, 2, 1)
        x = self.projection_post(x)
        return x


class PreTrainedTransformerCrossModalReconstructionModel(nn.Module):

    def __init__(
        self,
        input_dimension,
        d_model,
        nlayers,
        max_length,
        win_size,
        use_ln,
        use_rl,
        use_do,
        dropout,
        device,
    ):
        super().__init__()
        self.projection_pre = Projection(
            nlayers=nlayers["pre"],
            in_channels=input_dimension,
            out_channels=d_model,
            timesteps=max_length,
            use_ln=use_ln,
            use_rl=use_rl,
            use_do=use_do,
            dropout=dropout,
            pre=True,
        )
        self.projection_post = Projection(
            nlayers=nlayers["post"],
            in_channels=d_model,
            out_channels=input_dimension,
            timesteps=max_length,
            use_ln=use_ln,
            use_rl=use_rl,
            use_do=use_do,
            dropout=dropout,
            pre=False,
        )
        self.attn_mask = Mask(num_tokens=max_length, win_size=win_size, device=device)
        self.pos_embed = nn.Embedding(max_length, d_model)
        self.positions = torch.arange(0, max_length).unsqueeze(0).to(device)
        self.transformer = DistilBertModel.from_pretrained(
            "distilbert-base-uncased", torch_dtype=torch.float32, attn_implementation="sdpa"
        ).transformer

    def forward(self, x):
        x = self.projection_pre(x)
        x = x.permute(0, 2, 1)
        x = x + self.pos_embed(self.positions)
        x = self.transformer(x, attn_mask=self.attn_mask(1), head_mask=[None] * 6)[0]
        x = x.permute(0, 2, 1)
        x = self.projection_post(x)
        return x


class AutoRegressiveTransformerEncoderCrossModalReconstructionModel(nn.Module):

    def __init__(
        self,
        input_dimension,
        d_model,
        nlayers,
        num_heads,
        max_length,
        win_size,
        use_ln,
        use_rl,
        use_do,
        dropout,
        device,
    ):
        super().__init__()
        self.projection_pre = Projection(
            nlayers=nlayers["pre"],
            in_channels=input_dimension,
            out_channels=d_model,
            timesteps=max_length,
            use_ln=use_ln,
            use_rl=use_rl,
            use_do=use_do,
            dropout=dropout,
            pre=True,
        )
        self.projection_post = Projection(
            nlayers=nlayers["post"],
            in_channels=d_model,
            out_channels=input_dimension,
            timesteps=max_length,
            use_ln=use_ln,
            use_rl=use_rl,
            use_do=use_do,
            dropout=dropout,
            pre=False,
        )
        self.src_mask = self.get_src_mask(max_length, win_size, device)
        self.pos_embed = nn.Embedding(max_length, d_model)
        self.positions = torch.arange(0, max_length).unsqueeze(0).to(device)
        self.model = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=2 * d_model, batch_first=True),
            num_layers=nlayers["downsample"] + nlayers["upsample"],
        )

    def get_src_mask(self, num_tokens, win_size, device):
        mask = torch.stack(
            [
                torch.cat(
                    (
                        torch.ones((i,), dtype=torch.bool),
                        torch.zeros((win_size,), dtype=torch.bool),
                        torch.ones(num_tokens - i, dtype=torch.bool),
                    )
                )
                for i in range(num_tokens)
            ],
            dim=0,
        )[:, win_size:]
        mask[0, 0] = False
        return mask.to(device)

    def forward(self, x):
        x = self.projection_pre(x)
        x = x.permute(0, 2, 1)
        x = x + self.pos_embed(self.positions)
        x = self.model(x, mask=self.src_mask)
        x = x.permute(0, 2, 1)
        x = self.projection_post(x)
        return x


class CNNDownUpCrossModalReconstructionModel(nn.Module):

    def __init__(self, input_dimension, d_model, nlayers, win_size, max_length, use_ln, use_rl, use_do, dropout):
        super().__init__()
        self.projection_pre = Projection(
            nlayers=nlayers["pre"],
            in_channels=input_dimension,
            out_channels=d_model,
            timesteps=max_length,
            use_ln=use_ln,
            use_rl=use_rl,
            use_do=use_do,
            dropout=dropout,
            pre=True,
        )
        self.projection_post = Projection(
            nlayers=nlayers["post"],
            in_channels=d_model,
            out_channels=input_dimension,
            timesteps=max_length,
            use_ln=use_ln,
            use_rl=use_rl,
            use_do=use_do,
            dropout=dropout,
            pre=False,
        )
        self.encoder = nn.Sequential(
            *nn.ModuleList(
                [
                    ConvBlock(
                        in_channels=d_model,
                        out_channels=d_model,
                        kernel_size=win_size,
                        stride=2,
                        padding=win_size // 2,
                        timesteps=max_length // 2 ** (i + 1),
                        dropout=dropout,
                        use_ln=use_ln,
                        use_rl=use_rl,
                        use_do=use_do,
                    )
                    for i in range(nlayers["downsample"])
                ]
            )
        )
        self.decoder = nn.Sequential(
            *nn.ModuleList(
                [
                    ConvTransposeBlock(
                        in_channels=d_model,
                        out_channels=d_model,
                        kernel_size=win_size,
                        stride=2,
                        padding=win_size // 2,
                        timesteps=max_length // 2 ** (nlayers["upsample"] - i - 1),
                        dropout=dropout,
                        use_ln=use_ln,
                        use_rl=use_rl,
                        use_do=use_do,
                    )
                    for i in range(nlayers["upsample"])
                ]
            )
        )

    def forward(self, x):
        x = self.projection_pre(x)
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.projection_post(x)
        return x


class ReconstructionModel(nn.Module):

    def __init__(
        self,
        model_type,
        input_dimension,
        d_model,
        nlayers,
        num_heads,
        use_ln,
        use_rl,
        use_do,
        dropout,
        max_length,
        win_size,
        device,
    ):
        super().__init__()
        if model_type in ["cnn"]:
            self.model = CNNDownUpCrossModalReconstructionModel(
                input_dimension=input_dimension,
                d_model=d_model,
                nlayers=nlayers,
                win_size=win_size,
                max_length=max_length,
                use_ln=use_ln,
                use_rl=use_rl,
                use_do=use_do,
                dropout=dropout,
            )
        elif model_type in ["transformer"]:
            self.model = TransformerDownUpCrossModalReconstructionModel(
                input_dimension=input_dimension,
                d_model=d_model,
                nlayers=nlayers,
                num_heads=num_heads,
                max_length=max_length,
                win_size=win_size,
                use_ln=use_ln,
                use_rl=use_rl,
                use_do=use_do,
                dropout=dropout,
                device=device,
            )
        elif model_type in ["distilbert"]:
            self.model = PreTrainedTransformerCrossModalReconstructionModel(
                input_dimension=input_dimension,
                d_model=768,
                nlayers=nlayers,
                max_length=max_length,
                win_size=win_size,
                use_ln=use_ln,
                use_rl=use_rl,
                use_do=use_do,
                dropout=dropout,
                device=device,
            )
        elif model_type in ["autoregressive"]:
            self.model = AutoRegressiveTransformerEncoderCrossModalReconstructionModel(
                input_dimension=input_dimension,
                d_model=d_model,
                nlayers=nlayers,
                num_heads=num_heads,
                max_length=max_length,
                win_size=win_size,
                use_ln=use_ln,
                use_rl=use_rl,
                use_do=use_do,
                dropout=dropout,
                device=device,
            )
        else:
            raise Exception(f"model type {model_type} not suported")

    def forward(self, x):
        return self.model(x)


class TransformerEncoderModel(nn.Module):

    def __init__(self, input_dimension, d_model, nlayers, num_heads, fpn, dropout, max_length, win_size, device):
        super().__init__()
        self.attn_mask = Mask(num_tokens=max_length, win_size=win_size, device=device)
        self.model = nn.ModuleList(
            [
                DownsamplingTransformerBlock(
                    in_channels=input_dimension if i == 0 else d_model,
                    out_channels=d_model,
                    num_heads=num_heads,
                    attn_mask=self.attn_mask(1),
                    kernel_size=1,
                    stride=1,
                    dropout=dropout,
                )
                for i in range(nlayers["retain"])
            ]
            + [
                DownsamplingTransformerBlock(
                    in_channels=d_model,
                    out_channels=d_model,
                    num_heads=num_heads,
                    attn_mask=self.attn_mask(2**i if fpn else 1),
                    kernel_size=2 if fpn else 1,
                    stride=2 if fpn else 1,
                    dropout=dropout,
                )
                for i in range(nlayers["downsample"])
            ]
        )
        self.fpn = fpn
        if self.fpn:
            self.pyramid = FPN(
                in_channels_list=[d_model] * (nlayers["retain"] + nlayers["downsample"]), out_channels=d_model
            )

    def forward(self, x):
        intermediate = []
        out = x
        for i, m in enumerate(self.model):
            out = m(out)
            intermediate.append((f"feat{i}", out.permute(0, 2, 1).unsqueeze(-1)))
        if self.fpn:
            out = self.pyramid(OrderedDict(intermediate))
            return [out[x].squeeze() for x in out]
        else:
            return [out.permute(0, 2, 1)]


class CNNEncoderModel(nn.Module):

    def __init__(self, input_dimension, d_model, nlayers, win_size, max_length, fpn, use_ln, use_rl, use_do, dropout):
        super().__init__()
        self.model = nn.ModuleList(
            [
                ConvBlock(
                    in_channels=input_dimension if i == 0 else d_model,
                    out_channels=d_model,
                    kernel_size=win_size,
                    stride=1,
                    padding="same",
                    timesteps=max_length,
                    dropout=dropout,
                    use_ln=use_ln,
                    use_rl=use_rl,
                    use_do=use_do,
                )
                for i in range(nlayers["retain"])
            ]
            + [
                ConvBlock(
                    in_channels=d_model,
                    out_channels=d_model,
                    kernel_size=win_size,
                    stride=2 if fpn else 1,
                    padding=win_size // 2 if fpn else "same",
                    timesteps=max_length // 2 ** (i + 1) if fpn else max_length,
                    dropout=dropout,
                    use_ln=use_ln,
                    use_rl=use_rl,
                    use_do=use_do,
                )
                for i in range(nlayers["downsample"])
            ]
        )
        self.fpn = fpn
        if self.fpn:
            self.pyramid = FPN(
                in_channels_list=[d_model] * (nlayers["retain"] + nlayers["downsample"]), out_channels=d_model
            )

    def forward(self, x):
        intermediate = []
        out = x
        for i, m in enumerate(self.model):
            out = m(out)
            intermediate.append((f"feat{i}", out.unsqueeze(-1)))
        if self.fpn:
            out = self.pyramid(OrderedDict(intermediate))
            return [out[x].squeeze(dim=-1) for x in out]
        else:
            return [out]


class EncoderModel(nn.Module):

    def __init__(
        self,
        model_type,
        input_dimension,
        d_model,
        nlayers,
        num_heads,
        fpn,
        use_ln,
        use_rl,
        use_do,
        dropout,
        max_length,
        win_size,
        device,
    ):
        super().__init__()
        if model_type in ["cnn"]:
            self.model = CNNEncoderModel(
                input_dimension=input_dimension,
                d_model=d_model,
                nlayers=nlayers,
                win_size=win_size,
                max_length=max_length,
                fpn=fpn,
                use_ln=use_ln,
                use_rl=use_rl,
                use_do=use_do,
                dropout=dropout,
            )
        elif model_type in ["transformer"]:
            self.model = TransformerEncoderModel(
                input_dimension=input_dimension,
                d_model=d_model,
                nlayers=nlayers,
                num_heads=num_heads,
                fpn=fpn,
                dropout=dropout,
                max_length=max_length,
                win_size=win_size,
                device=device,
            )
        else:
            raise Exception(f"model type {model_type} not suported")

    def forward(self, x):
        return self.model(x)


class HeadModule(nn.Module):

    def __init__(self, d_model, output_size, timesteps, use_ln, use_rl, use_do, dropout, final_relu):
        super().__init__()
        submodules = [
            ConvBlock(
                in_channels=d_model,
                out_channels=d_model,
                kernel_size=3,
                stride=1,
                padding="same",
                timesteps=timesteps,
                dropout=dropout,
                use_ln=use_ln,
                use_rl=use_rl,
                use_do=use_do,
            ),
            ConvBlock(
                in_channels=d_model,
                out_channels=d_model,
                kernel_size=3,
                stride=1,
                padding="same",
                timesteps=timesteps,
                dropout=dropout,
                use_ln=use_ln,
                use_rl=use_rl,
                use_do=use_do,
            ),
            nn.Conv1d(
                in_channels=d_model,
                out_channels=output_size,
                kernel_size=3,
                padding="same",
            ),
        ]
        if final_relu:
            submodules.append(nn.ReLU())
        self.module = nn.Sequential(*submodules)

    def forward(self, x):
        return self.module(x)


class HeadModel(nn.Module):

    def __init__(self, d_model, factor, output_size, max_length, fpn, use_ln, use_rl, use_do, dropout, final_relu):
        super().__init__()
        self.fpn = fpn
        if self.fpn:
            self.model = nn.ModuleList(
                [
                    HeadModule(
                        d_model=d_model,
                        output_size=output_size,
                        timesteps=max_length // f,
                        use_ln=use_ln,
                        use_rl=use_rl,
                        use_do=use_do,
                        dropout=dropout,
                        final_relu=final_relu,
                    )
                    for f in factor
                ]
            )
        else:
            self.model = nn.ModuleList(
                [
                    HeadModule(
                        d_model=d_model,
                        output_size=output_size,
                        timesteps=max_length,
                        use_ln=use_ln,
                        use_rl=use_rl,
                        use_do=use_do,
                        dropout=dropout,
                        final_relu=final_relu,
                    )
                ]
            )

    def forward(self, x):
        return [self.model[i](f) for i, f in enumerate(x)]


class Model(nn.Module):

    def __init__(
        self,
        max_length,
        d_model,
        win_size,
        num_heads,
        operation,
        reconstruction,
        encoder,
        dropout,
        use_ln,
        use_rl,
        use_do,
        model_type,
        factor,
        device,
    ):
        super().__init__()
        self.device = device
        self.input_dimension = 768
        self.cls_output_size = 1
        self.reg_output_size = 2
        self.max_length = max_length
        self.model_type = model_type
        self.d_model = d_model
        self.win_size = win_size
        self.num_heads = num_heads
        self.operation = operation
        self.reconstruction_nlayers = reconstruction["nlayers"]
        self.reconstruction_modality = [RECONSTRUCTION[m] for m in reconstruction["modality"]]
        self.encoder_nlayers = encoder["nlayers"]
        self.fpn = encoder["fpn"]
        self.use_ln = use_ln
        self.use_rl = use_rl
        self.use_do = use_do
        self.dropout_main = dropout["main"]
        self.dropout_head = dropout["head"]
        self.factor = factor
        self.reconstruction_model = ReconstructionModel(
            model_type=self.model_type["reconstruction"],
            input_dimension=self.input_dimension,
            d_model=self.d_model,
            nlayers=self.reconstruction_nlayers,
            num_heads=self.num_heads,
            use_ln=self.use_ln,
            use_rl=self.use_rl,
            use_do=self.use_do,
            dropout=self.dropout_main,
            max_length=self.max_length,
            win_size=self.win_size,
            device=self.device,
        )
        self.encoder_model = EncoderModel(
            model_type=self.model_type["encoder"],
            input_dimension=self.input_dimension * len(self.reconstruction_modality),
            d_model=self.d_model,
            nlayers=self.encoder_nlayers,
            num_heads=self.num_heads,
            fpn=self.fpn,
            use_ln=self.use_ln,
            use_rl=self.use_rl,
            use_do=self.use_do,
            dropout=self.dropout_main,
            max_length=self.max_length,
            win_size=self.win_size,
            device=self.device,
        )
        self.classification = HeadModel(
            d_model=self.d_model,
            factor=self.factor,
            output_size=self.cls_output_size,
            max_length=self.max_length,
            fpn=self.fpn,
            use_ln=self.use_ln,
            use_rl=self.use_rl,
            use_do=self.use_do,
            dropout=self.dropout_head,
            final_relu=False,
        )
        self.regression = HeadModel(
            d_model=self.d_model,
            factor=self.factor,
            output_size=self.reg_output_size,
            max_length=self.max_length,
            fpn=self.fpn,
            use_ln=self.use_ln,
            use_rl=self.use_rl,
            use_do=self.use_do,
            dropout=self.dropout_head,
            final_relu=True,
        )

    def get_model_inputs(self, x):
        return x[0].permute((0, 2, 1)), x[1].permute((0, 2, 1))

    def get_reconstruction_pairs(self, audio, video):
        reconstructed = [
            self.reconstruction_model(audio) if modality["input"] == "audio" else self.reconstruction_model(video)
            for modality in self.reconstruction_modality
        ]
        pairs = [
            (
                {"target": audio, "prediction": reconstructed_}
                if self.reconstruction_modality[i]["output"] == "audio"
                else {"target": video, "prediction": reconstructed_}
            )
            for i, reconstructed_ in enumerate(reconstructed)
        ]
        if self.operation == "subtraction":
            dissimilarity = (
                torch.stack([torch.abs(pair["target"] - pair["prediction"]) for pair in pairs], dim=0)
                .sum(dim=0)
                .mean(dim=-2)
            )
        elif self.operation == "multiplication":
            dissimilarity = torch.stack(
                [
                    1 - torch.nn.functional.cosine_similarity(pair["target"], pair["prediction"], dim=-2)
                    for pair in pairs
                ],
                dim=0,
            ).sum(dim=0)
        else:
            raise Exception(f"operation {self.operation} is not supported")
        return pairs, dissimilarity

    def get_encoder_inputs(self, pairs):
        if self.operation == "subtraction":
            return torch.cat([pair["prediction"] - pair["target"] for pair in pairs], dim=-2)
        elif self.operation == "multiplication":
            return torch.cat([pair["prediction"] * pair["target"] for pair in pairs], dim=-2)
        else:
            raise Exception(f"operation {self.operation} is not supported")

    def forward(self, src):
        # Inputs
        video, audio = self.get_model_inputs(src)

        # Reconstruction target/prediction pairs & corresponding dissimilarity
        pairs, dissimilarity = self.get_reconstruction_pairs(audio, video)

        # Encoder inputs
        encoder_inputs = self.get_encoder_inputs(pairs)

        # Processing
        if self.model_type["encoder"] in ["transformer"]:
            encoder_inputs = encoder_inputs.permute((0, 2, 1))
        features = self.encoder_model(encoder_inputs)

        # Outputs
        logits = self.classification(features)
        boundaries = self.regression(features)
        outputs = [torch.cat((l, b), dim=1).permute((0, 2, 1)) for l, b in zip(logits, boundaries)]
        return outputs, dissimilarity

    def get_features(self, src):
        # Inputs
        video, audio = self.get_model_inputs(src)

        # Reconstruction target/prediction pairs & corresponding dissimilarity
        pairs, dissimilarity = self.get_reconstruction_pairs(audio, video)

        # Encoder inputs
        encoder_inputs = self.get_encoder_inputs(pairs)

        # Processing
        if self.model_type["encoder"] in ["transformer"]:
            encoder_inputs = encoder_inputs.permute((0, 2, 1))
        features = self.encoder_model(encoder_inputs)

        # Outputs
        logits = self.classification(features)
        boundaries = self.regression(features)
        outputs = [torch.cat((l, b), dim=1).permute((0, 2, 1)) for l, b in zip(logits, boundaries)]
        return outputs, dissimilarity, features
