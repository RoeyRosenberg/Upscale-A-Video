import argparse
import math
import os.path

from einops import rearrange
import imageio
import torch
from tqdm import tqdm
from models import UNet3DConditionModel, AutoencoderKL, DDIMScheduler
from pipeline import UpscaleAVideoPipeline
from utils import get_video_frames, blip_it, tensor_to_np, get_optical_flows, convert_to_list
from wavelet_color_fix import wavelet_reconstruction, adaptive_instance_normalization


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--input", type=str, default=r"inputs/old_animation_1.mp4", help="Path to the input video")

    parser.add_argument("-o", "--output", type=str, default="output", help="Output folder, the video name remains")
    parser.add_argument('-n', '--noise_level', type=int, default=120,
            help='Noise level [0, 200] applied to the input video. A higher noise level typically results in better \
                video quality but lower fidelity. Default value: 120')
    parser.add_argument('-g', '--guidance_scale',
                        type=int,
                        default=9,
                        help='Classifier-free guidance scale for prompts')
    parser.add_argument('-s', '--inference_steps',
                        type=int,
                        default=30,
                        help='Number of denoise steps')
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        choices=["no", "bf16", "fp16"],
        help=(
            "Whether to use mixed precision. Choose between fp32 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )

    parser.add_argument("--checkpoint_steps", type=int, default=2, help="Save the output for each x checkpoint")
    parser.add_argument("--caption", "--use_image_caption", type=bool, default=None, help="Caption frames at each batch")
    parser.add_argument("--n_frames", type=int, default=None, help="Number of frames to process each step")
    parser.add_argument("--a_prompt", type=str, default='best quality, extremely detailed')
    parser.add_argument("--neg_prompt", type=str, default='blur, worst quality')
    parser.add_argument("--pretrained_model_path", type=str, default="pretrained_models/upscale_a_video", help="Path to the pretrained folder")
    parser.add_argument("--cut_to_tiles", type=bool, default=False, help="Upscale in tiles, not recommended due to block artifacts")
    parser.add_argument("--tile_size", type=int, default=120, help="Size of tile if upscales images in pieces")
    parser.add_argument("-p", "--prop_steps", type=convert_to_list, default=[14, 15, 16, 17], help="Index of propagation steps")
    parser.add_argument("--colorfix_type", type=str, choices=["adain", "wavelet", "nofix"], default="adain", help="Color fix type to adjust the color of HR result according to LR input: adain ; wavelet; nofix")

    args = parser.parse_args()
    return args


def main():

    args = parse_args()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    pipeline = UpscaleAVideoPipeline.from_pretrained(args.pretrained_model_path, torch_dtype=weight_dtype)

    # Needs to configure vae and unet manually because they aren`t diffusers modules
    pipeline.unet = UNet3DConditionModel.from_pretrained(args.pretrained_model_path, subfolder="unet").to(weight_dtype)

    pipeline.vae = AutoencoderKL.from_pretrained(args.pretrained_model_path, subfolder="vae").to(weight_dtype)
    pipeline.scheduler = DDIMScheduler.from_pretrained(args.pretrained_model_path, subfolder="scheduler")
    pipeline.eval()

    pipeline = pipeline.to(device)

    v_frames, fps = get_video_frames(args.input)

    if args.n_frames is None:
        args.n_frames = v_frames.shape[0]
    if args.checkpoint_steps == 0:
        args.checkpoint_steps = v_frames.shape[0]

    if args.caption:
        prompt = blip_it(v_frames[0]) + args.a_prompt
    else:
        prompt = args.a_prompt
    negative_prompt = args.neg_prompt

    f, c, h, w = v_frames.shape
    output_video = torch.zeros(f, c, h*4, w*4)
    for i in range(0, f, args.n_frames):
        end = min(f, i+args.n_frames)

        frames = v_frames[i:end]

        frames = frames.unsqueeze(0)
        frames = rearrange(frames, "b f c h w -> b c f h w")

        flows = get_optical_flows(frames, dtype=weight_dtype, device=device)

        frames = frames.to(weight_dtype)

        if args.cut_to_tiles:
            tile_size = args.tile_size

            b, c, f, h, w = frames.shape
            output_w, output_h = w * 4, h * 4

            output = torch.zeros((b, c, f, output_h, output_w))
            tile_x = math.ceil(w / tile_size)
            tile_y = math.ceil(h / tile_size)
            pipeline.set_progress_bar_config(disable=True)

            progress_bar = tqdm(total=tile_y*tile_x, desc="Tile Progress")
            for y in range(tile_y):
                for x in range(tile_x):

                    start_x = x * tile_size
                    start_y = y * tile_size

                    end_x = min((x+1) * tile_size, w)
                    end_y = min((y+1) * tile_size, h)

                    tile_frames = frames[:, :, :, start_y:end_y, start_x:end_x]

                    tile_flows = [flow[:, :, :, start_y:end_y, start_x:end_x] for flow in flows]
                    tile_output = pipeline(prompt=prompt,
                                           image=tile_frames,
                                           flows=tile_flows,
                                           num_inference_steps=args.inference_steps,
                                           negative_prompt=negative_prompt
                                           ).images
                    output[:, :, :, start_y*4:end_y*4, start_x*4:end_x*4] = tile_output
                    progress_bar.update(1)

        else:
            output = pipeline(prompt=prompt,
                              image=frames,
                              flows=flows,
                              num_inference_steps=args.inference_steps,
                              negative_prompt=negative_prompt,
                              noise_level=args.noise_level,
                              propagation_steps=args.prop_steps
                              ).images
        output = output.squeeze(0).permute(1, 0, 2, 3).to(device)
        frames = frames.squeeze(0).permute(1, 0, 2, 3).to(device)

        # Color-fix mentioned in the paper, Copied from https://github.com/IceClear/StableSR/blob/main/scripts/wavelet_color_fix.py
        if args.colorfix_type == "wavelet":
            output = wavelet_reconstruction(output, torch.nn.functional.interpolate(frames, scale_factor=4, mode="bicubic"))
        elif args.colorfix_type == "adain":
            output = adaptive_instance_normalization(output, torch.nn.functional.interpolate(frames, scale_factor=4, mode="bicubic"))
        elif args.colorfix_type == "nofix":
            print("Skipping color fix")

        output_video[i:end] = output.cpu()
        print(output_video[i:end].shape)
        filename = os.path.basename(args.input)
        if not os.path.isdir(args.output):
            os.makedirs(args.output)
        output_file = os.path.join(args.output, filename)

        writer = imageio.get_writer(output_file, fps=fps)
        for frame in output_video:
            frame = tensor_to_np(frame)
            writer.append_data(frame)
        writer.close()
        print("Output video saved in:", os.path.join(args.output, filename))


if __name__ == "__main__":
    main()
