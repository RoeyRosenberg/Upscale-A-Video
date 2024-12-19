# Adapted from https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/resnet.py

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class InflatedConv3d(nn.Conv2d):
    def forward(self, x):
        video_length = x.shape[2]

        x = rearrange(x, "b c f h w -> (b f) c h w")

        x = super().forward(x)

        x = rearrange(x, "(b f) c h w -> b c f h w", f=video_length)

        return x


class Upsample3D(nn.Module):
    def __init__(self, channels, use_conv=False, use_conv_transpose=False, out_channels=None,
                 name="conv", interpolate_mode="nearest"):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_conv_transpose = use_conv_transpose
        self.name = name
        self.interpolate_mode = interpolate_mode

        conv = None
        if use_conv_transpose:
            raise NotImplementedError
        elif use_conv:
            conv = InflatedConv3d(self.channels, self.out_channels, 3, padding=1)

        if name == "conv":
            self.conv = conv
        else:
            self.Conv2d_0 = conv

    def forward(self, hidden_states, output_size=None):
        assert hidden_states.shape[1] == self.channels

        if self.use_conv_transpose:
            raise NotImplementedError

        # Cast to float32 to as 'upsample_nearest2d_out_frame' op does not support bfloat16
        dtype = hidden_states.dtype
        if dtype == torch.bfloat16:
            hidden_states = hidden_states.to(torch.float32)

        # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984
        if hidden_states.shape[0] >= 64:
            hidden_states = hidden_states.contiguous()

        # if `output_size` is passed we force the interpolation output
        # size and do not make use of `scale_factor=2`
        if output_size is None:
            hidden_states = F.interpolate(hidden_states, scale_factor=[1.0, 2.0, 2.0], mode=self.interpolate_mode)
        else:
            hidden_states = F.interpolate(hidden_states, size=output_size, mode=self.interpolate_mode)

        # If the input is bfloat16, we cast back to bfloat16
        if dtype == torch.bfloat16:
            hidden_states = hidden_states.to(dtype)

        if self.use_conv:
            if self.name == "conv":
                hidden_states = self.conv(hidden_states)
            else:
                hidden_states = self.Conv2d_0(hidden_states)

        return hidden_states


class Downsample3D(nn.Module):
    """A 2D downsampling layer with an optional convolution.

    Parameters:
        channels (`int`):
            number of channels in the inputs and outputs.
        use_conv (`bool`, default `False`):
            option to use a convolution.
        out_channels (`int`, optional):
            number of output channels. Defaults to `channels`.
        padding (`int`, default `1`):
            padding for the convolution.
        name (`str`, default `conv`):
            name of the downsampling 2D layer.
    """

    def __init__(
        self,
        channels: int,
        use_conv: bool = False,
        out_channels: Optional[int] = None,
        padding: int = 1,
        name: str = "conv",
        kernel_size=3,
        bias=True,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.padding = padding
        stride = 2
        self.name = name

        if use_conv:
            conv = InflatedConv3d(
                self.channels, self.out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias
            )
        else:
            assert self.channels == self.out_channels
            conv = nn.AvgPool2d(kernel_size=stride, stride=stride)

        self.conv = conv

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        assert hidden_states.shape[1] == self.channels

        if self.use_conv and self.padding == 0:
            pad = (0, 1, 0, 1)
            hidden_states = F.pad(hidden_states, pad, mode="constant", value=0)

        assert hidden_states.shape[1] == self.channels

        hidden_states = self.conv(hidden_states)

        return hidden_states


class InflatedResnetBlock2D(nn.Module):
    def __init__(
            self,
            *,
            in_channels,
            out_channels=None,
            conv_shortcut=False,
            dropout=0.0,
            temb_channels=512,
            groups=32,
            groups_out=None,
            pre_norm=True,
            eps=1e-6,
            non_linearity="swish",
            time_embedding_norm="default",
            output_scale_factor=1.0,
            use_in_shortcut=None,
            add_3d_layer=False
    ):
        super().__init__()
        self.pre_norm = pre_norm
        self.pre_norm = True
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.time_embedding_norm = time_embedding_norm
        self.output_scale_factor = output_scale_factor

        if groups_out is None:
            groups_out = groups

        self.norm1 = torch.nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps, affine=True)

        self.conv1 = InflatedConv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if temb_channels is not None:
            if self.time_embedding_norm == "default":
                time_emb_proj_out_channels = out_channels
            elif self.time_embedding_norm == "scale_shift":
                time_emb_proj_out_channels = out_channels * 2
            else:
                raise ValueError(f"unknown time_embedding_norm : {self.time_embedding_norm} ")

            self.time_emb_proj = torch.nn.Linear(temb_channels, time_emb_proj_out_channels)
        else:
            self.time_emb_proj = None

        self.norm2 = torch.nn.GroupNorm(num_groups=groups_out, num_channels=out_channels, eps=eps, affine=True)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = InflatedConv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if non_linearity == "swish":
            self.nonlinearity = nn.SiLU()
        elif non_linearity == "mish":
            self.nonlinearity = nn.Mish()
        elif non_linearity == "silu":
            self.nonlinearity = nn.SiLU()

        self.use_in_shortcut = self.in_channels != self.out_channels if use_in_shortcut is None else use_in_shortcut

        self.conv_shortcut = None
        if self.use_in_shortcut:
            self.conv_shortcut = InflatedConv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

        self.add_3d_layer = add_3d_layer

        # This conv layer wasn't mentioned in the paper, but it exists in the state dictionary of the pretrained model.
        if self.add_3d_layer:
            self.norm_3d = nn.GroupNorm(num_groups=groups_out, num_channels=out_channels, eps=eps, affine=True)
            self.conv_3d = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, padding_mode="zeros")

    def forward(self, input_tensor, temb=None):
        hidden_states = input_tensor

        hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)

        hidden_states = self.conv1(hidden_states)

        if temb is not None:
            temb = self.time_emb_proj(self.nonlinearity(temb))[:, :, None, None, None]

        if temb is not None and self.time_embedding_norm == "default":
            hidden_states = hidden_states + temb

        hidden_states = self.norm2(hidden_states)

        if temb is not None and self.time_embedding_norm == "scale_shift":
            scale, shift = torch.chunk(temb, 2, dim=1)
            hidden_states = hidden_states * (1 + scale) + shift

        hidden_states = self.nonlinearity(hidden_states)

        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor)

        output_tensor = (input_tensor + hidden_states) / self.output_scale_factor

        if self.add_3d_layer:
            output_tensor_3d = self.norm_3d(output_tensor)
            output_tensor_3d = self.nonlinearity(output_tensor_3d)
            output_tensor_3d = self.conv_3d(output_tensor_3d)

            output_tensor = (output_tensor + output_tensor_3d) / self.output_scale_factor
        return output_tensor


class ResnetBlock3D(nn.Module):
    def __init__(
            self,
            *,
            in_channels,
            out_channels=None,
            kernel=(3, 1, 1),
            conv_shortcut=False,
            dropout=0.0,
            temb_channels=512,
            groups=32,
            groups_out=None,
            pre_norm=True,
            eps=1e-5,
            non_linearity="swish",
            time_embedding_norm="default",
            output_scale_factor=1.0,
            use_in_shortcut=None,
    ):
        super().__init__()
        self.pre_norm = pre_norm
        self.pre_norm = True
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.time_embedding_norm = time_embedding_norm
        self.output_scale_factor = output_scale_factor

        if groups_out is None:
            groups_out = groups

        self.norm1 = torch.nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps, affine=True)

        padding = ((kernel[i] - 1) // 2 for i in range(3))
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=kernel, stride=(1, 1, 1), padding=padding)

        if temb_channels is not None:
            if self.time_embedding_norm == "default":
                time_emb_proj_out_channels = out_channels
            elif self.time_embedding_norm == "scale_shift":
                time_emb_proj_out_channels = out_channels * 2
            else:
                raise ValueError(f"unknown time_embedding_norm : {self.time_embedding_norm} ")

            self.time_emb_proj = torch.nn.Linear(temb_channels, time_emb_proj_out_channels)
        else:
            self.time_emb_proj = None

        self.norm2 = torch.nn.GroupNorm(num_groups=groups_out, num_channels=out_channels, eps=eps, affine=True)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0))

        if non_linearity == "swish":
            self.nonlinearity = lambda x: F.silu(x)
        elif non_linearity == "mish":
            self.nonlinearity = nn.Mish()
        elif non_linearity == "silu":
            self.nonlinearity = nn.SiLU()

        self.use_in_shortcut = self.in_channels != self.out_channels if use_in_shortcut is None else use_in_shortcut

        self.conv_shortcut = None
        if self.use_in_shortcut:
            self.conv_shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1,
                                           padding=0)

    def forward(self, input_tensor: torch.Tensor, temb=None):
        hidden_states = input_tensor

        hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)

        hidden_states = self.conv1(hidden_states)

        if temb is not None:
            temb = self.time_emb_proj(self.nonlinearity(temb))[:, :, None, None, None]

        if temb is not None and self.time_embedding_norm == "default":
            hidden_states = hidden_states + temb

        hidden_states = self.norm2(hidden_states)

        if temb is not None and self.time_embedding_norm == "scale_shift":
            scale, shift = torch.chunk(temb, 2, dim=1)
            hidden_states = hidden_states * (1 + scale) + shift

        hidden_states = self.nonlinearity(hidden_states)

        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor)

        output_tensor = (input_tensor + hidden_states) / self.output_scale_factor

        return output_tensor


class EmptyResnetBlock(nn.Module):
    def forward(self, hidden_states, temb=None):
        return hidden_states


# This model wasn't mentioned in the paper, but it exists in the state dictionary of the pretrained model.
class TempResnet3DBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels=None,
                 temb_channels=None,
                 groups=32,
                 ):
        super().__init__()

        self.resblocks_3d_spatial = InflatedResnetBlock2D(in_channels=in_channels, out_channels=in_channels, temb_channels=temb_channels, groups=groups)
        self.resblocks_3d_temporal = ResnetBlock3D(in_channels=in_channels, out_channels=out_channels, temb_channels=temb_channels, kernel=(5, 1, 1))

        self.shift_conv = zero_module(InflatedConv3d(in_channels=in_channels, out_channels=in_channels, kernel_size=1))

    def forward(self, hidden_states, temb=None):

        input_tensor = hidden_states

        hidden_states = self.resblocks_3d_temporal(hidden_states, temb)
        hidden_states = self.resblocks_3d_spatial(hidden_states, temb)

        hidden_states = self.shift_conv(hidden_states)
        hidden_states += input_tensor

        return hidden_states
