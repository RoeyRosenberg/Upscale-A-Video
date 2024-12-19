from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch.nn
from diffusers import ConfigMixin
from diffusers.configuration_utils import register_to_config
from diffusers.models.embeddings import Timesteps, TimestepEmbedding
from diffusers.utils import BaseOutput
from rotary_embedding_torch import RotaryEmbedding

from .model_helpers import TemporalModelMixin
from .resnet import InflatedConv3d, TempResnet3DBlock, EmptyResnetBlock
from .unet_blocks import get_down_block, get_up_block, UNetMidBlock3DCrossAttn


@dataclass
class UNet3DConditionOutput(BaseOutput):
    sample: torch.FloatTensor


class UNet3DConditionModel(TemporalModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True
    _no_split_modules = ["BasicTransformerBlock", "ResnetBlock2D", "CrossAttnUpBlock2D"]

    @register_to_config
    def __init__(
            self,
            in_channels: int = 4,
            out_channels: int = 4,
            flip_sin_to_cos: bool = True,
            freq_shift: int = 0,
            down_block_types: Tuple[str] = (
                    "CrossAttnDownBlock3D",
                    "CrossAttnDownBlock3D",
                    "CrossAttnDownBlock3D",
                    "DownBlock3D",
            ),
            mid_block_type: Optional[str] = "UNetMidBlock2DCrossAttn",
            up_block_types: Tuple[str] = (
            "UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"),
            only_cross_attention: Union[bool, Tuple[bool]] = False,
            block_out_channels: Tuple[int] = (320, 640, 1280, 1280),
            layers_per_block: Union[int, Tuple[int]] = 2,
            downsample_padding: int = 1,
            mid_block_scale_factor: float = 1,
            dropout: float = 0.0,
            act_fn: str = "silu",
            norm_num_groups: Optional[int] = 32,
            norm_eps: float = 1e-5,
            cross_attention_dim: Union[int, Tuple[int]] = 1280,

            attention_head_dim: Union[int, Tuple[int]] = 8,

            dual_cross_attention: bool = False,
            use_linear_projection: bool = False,
            class_embed_type: Optional[str] = None,
            num_class_embeds: Optional[int] = None,
            upcast_attention: bool = False,
            resnet_time_scale_shift: str = "default",
            time_embedding_dim: Optional[int] = None,
            timestep_post_act: Optional[str] = None,
            time_cond_proj_dim: Optional[int] = None,
            conv_in_kernel: int = 3,
            conv_out_kernel: int = 3,

    ):
        super().__init__()

        self.conv_in = InflatedConv3d(in_channels, block_out_channels[0], kernel_size=conv_in_kernel, padding=1)

        time_embed_dim = time_embedding_dim or block_out_channels[0] * 4
        self.time_proj = Timesteps(block_out_channels[0], flip_sin_to_cos, freq_shift)
        timestep_input_dim = block_out_channels[0]

        self.time_embedding = TimestepEmbedding(timestep_input_dim,
            time_embed_dim,
            act_fn=act_fn,
            post_act_fn=timestep_post_act,
            cond_proj_dim=time_cond_proj_dim)

        if class_embed_type is None and num_class_embeds is not None:
            self.class_embedding = torch.nn.Embedding(num_class_embeds, time_embed_dim)
        elif class_embed_type == "timestep":
            self.class_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim)
        elif class_embed_type == "identity":
            self.class_embedding = torch.nn.Identity(time_embed_dim, time_embed_dim)
        else:
            self.class_embedding = None

        self.down_blocks = torch.nn.ModuleList([])
        self.up_blocks = torch.nn.ModuleList([])

        # temp blocks weren't mentioned in the paper, but they exists in the state dictionary of the pretrained model.

        self.use_temp_blocks = False
        self.down_temp_blocks = torch.nn.ModuleList([])
        self.up_temp_blocks = torch.nn.ModuleList([])

        self.temporal_rotary_emb = RotaryEmbedding(32)

        if type(attention_head_dim) is int:
            attention_head_dim = [attention_head_dim] * len(down_block_types)

        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = get_down_block(
                down_block_type,
                num_layers=layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=time_embed_dim,
                add_downsample=not is_final_block,
                attn_num_head_channels=attention_head_dim[i],
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                cross_attention_dim=cross_attention_dim,
                downsample_padding=downsample_padding,
                dual_cross_attention=dual_cross_attention,
                use_linear_projection=use_linear_projection,
                only_cross_attention=only_cross_attention[i],
                upcast_attention=upcast_attention,
                dropout=dropout,
                resnet_time_scale_shift=resnet_time_scale_shift,
                rotary_emb=self.temporal_rotary_emb
            )

            down_temp_block = TempResnet3DBlock(
                in_channels=output_channel,
                out_channels=output_channel,
                temb_channels=time_embed_dim,
            ) if self.use_temp_blocks else EmptyResnetBlock()
            self.down_temp_blocks.append(down_temp_block)
            self.down_blocks.append(down_block)

        self.mid_block = UNetMidBlock3DCrossAttn(
            in_channels=block_out_channels[-1],
            temb_channels=time_embed_dim,
            resnet_eps=norm_eps,
            resnet_act_fn=act_fn,
            dropout=dropout,
            output_scale_factor=mid_block_scale_factor,
            resnet_time_scale_shift=resnet_time_scale_shift,
            cross_attention_dim=cross_attention_dim,
            attn_num_head_channels=attention_head_dim[-1],
            resnet_groups=norm_num_groups,
            dual_cross_attention=dual_cross_attention,
            use_linear_projection=use_linear_projection,
            upcast_attention=upcast_attention,
            rotary_emb=self.temporal_rotary_emb
        )

        self.mid_temp_block = TempResnet3DBlock(
            in_channels=block_out_channels[-1],
            out_channels=block_out_channels[-1],
            temb_channels=time_embed_dim,
        ) if self.use_temp_blocks else EmptyResnetBlock()

        self.num_upsamplers = 0

        reversed_block_out_channels = list(reversed(block_out_channels))

        only_cross_attention = list(reversed(only_cross_attention))

        for i, up_block_type in enumerate(up_block_types):
            is_final_block = i == len(block_out_channels) - 1

            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[min(i + 1, len(block_out_channels) - 1)]

            # add upsample block for all BUT final layer
            if not is_final_block:
                add_upsample = True
                self.num_upsamplers += 1
            else:
                add_upsample = False

            up_block = get_up_block(
                up_block_type,
                num_layers=layers_per_block + 1,
                in_channels=input_channel,
                out_channels=output_channel,
                dropout=dropout,
                prev_output_channel=prev_output_channel,
                temb_channels=time_embed_dim,
                add_upsample=add_upsample,
                attn_num_head_channels=attention_head_dim[i],
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                cross_attention_dim=cross_attention_dim,
                dual_cross_attention=dual_cross_attention,
                use_linear_projection=use_linear_projection,
                only_cross_attention=only_cross_attention[i],
                upcast_attention=upcast_attention,
                resnet_time_scale_shift=resnet_time_scale_shift,
                rotary_emb=self.temporal_rotary_emb
            )

            up_temp_block = TempResnet3DBlock(
                in_channels=output_channel,
                out_channels=output_channel,
                temb_channels=time_embed_dim,
            ) if self.use_temp_blocks else EmptyResnetBlock()

            self.up_temp_blocks.append(up_temp_block)
            self.up_blocks.append(up_block)

        self.conv_norm_out = torch.nn.GroupNorm(
            num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=norm_eps
        )

        self.conv_act = torch.nn.SiLU()
        self.conv_out = InflatedConv3d(block_out_channels[0], out_channels, kernel_size=conv_out_kernel, padding=1)

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor = None,
        class_labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        **cross_attention_kwargs,
    ) -> Union[UNet3DConditionOutput, Tuple]:
        r"""
        Args:
            sample (`torch.FloatTensor`): (batch, channel, height, width) noisy inputs tensor
            timestep (`torch.FloatTensor` or `float` or `int`): (batch) timesteps
            encoder_hidden_states (`torch.FloatTensor`): (batch, sequence_length, feature_dim) encoder hidden states
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`models.unet_2d_condition.UNet2DConditionOutput`] instead of a plain tuple.

        Returns:
            [`~models.unet_2d_condition.UNet2DConditionOutput`] or `tuple`:
            [`~models.unet_2d_condition.UNet2DConditionOutput`] if `return_dict` is True, otherwise a `tuple`. When
            returning a tuple, the first element is the sample tensor.
        """
        # By default samples have to be AT least a multiple of the overall upsampling factor.
        # The overall upsampling factor is equal to 2 ** (# num of upsampling layears).
        # However, the upsampling interpolation output size can be forced to fit any upsampling size
        # on the fly if necessary.

        if sample.ndim == 4:
            sample = sample.unsqueeze(0)
            sample = sample.permute(0, 2, 1, 3, 4)

        default_overall_up_factor = 2**self.num_upsamplers

        # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
        forward_upsample_size = False
        upsample_size = None

        if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
            forward_upsample_size = True

        # prepare attention_mask
        if attention_mask is not None:
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)



        # time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=self.dtype)
        emb = self.time_embedding(t_emb)
        if self.class_embedding is not None:
            if class_labels is None:
                raise ValueError("class_labels should be provided when num_class_embeds > 0")

            if self.config.class_embed_type == "timestep":

                class_labels = self.time_proj(class_labels)

            class_emb = self.class_embedding(class_labels).to(dtype=self.dtype)

            emb = emb + class_emb

        sample = self.conv_in(sample)

        # down
        down_block_res_samples = (sample,)
        for i, (downsample_block,  down_temp_block)in enumerate(zip(self.down_blocks, self.down_temp_blocks)):
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:

                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                )

            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

            down_block_res_samples += res_samples

            sample = down_temp_block(hidden_states=sample, temb=emb)



        sample = self.mid_block(
            sample, emb, encoder_hidden_states=encoder_hidden_states, attention_mask=attention_mask
        )
        sample = self.mid_temp_block(hidden_states=sample, temb=emb)

        # up
        for i, (upsample_block, up_temp_block) in enumerate(zip(self.up_blocks, self.up_temp_blocks)):
            is_final_block = i == len(self.up_blocks) - 1

            res_samples = down_block_res_samples[-len(upsample_block.resnets):]

            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            # if we have not reached the final block and need to forward the
            # upsample size, we do it here
            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:

                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    upsample_size=upsample_size,
                    attention_mask=attention_mask,
                )

            else:

                sample = upsample_block(
                    hidden_states=sample, temb=emb, res_hidden_states_tuple=res_samples, upsample_size=upsample_size
                )

            sample = up_temp_block(hidden_states=sample, temb=emb)

        # post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)


        if not return_dict:
            return (sample,)
        sample = UNet3DConditionOutput(sample=sample)
        return sample


