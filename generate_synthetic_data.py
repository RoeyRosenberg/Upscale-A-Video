import argparse
import glob
import os.path

import torch
import yaml
from diffusers import DDPMScheduler, DDIMScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from dataset import VideoPairsDataset
from models import AutoencoderKL, UNet3DConditionModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",
                        default="pretrained_models/upscale_a_video",
                        type=str,
                        help="Path to pretrained model")
    parser.add_argument("-o", "--output", default="VAE_dataset", type=str, help="Path to output folder")
    parser.add_argument("--config", default="configs/generate_video_pairs.yaml", type=str, help="Path to config")
    parser.add_argument("-b", "--batch_size", default=1, type=int, help="Batch size of synthetic data")
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    batch_size = args.batch_size

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dataset = VideoPairsDataset(config["data"], device)

    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_path,
        subfolder="vae",
        config_type="3d"
    )

    low_res_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_path, subfolder="low_res_scheduler")
    scheduler = DDIMScheduler.from_pretrained(args.pretrained_model_path, subfolder="scheduler")
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_path, subfolder="text_encoder"
    )
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_path, subfolder="tokenizer")
    unet = UNet3DConditionModel.from_pretrained(args.pretrained_model_path, subfolder="unet")

    def tokenize_captions(captions):

        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids

    weight_dtype = torch.float32
    if args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16

    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)
    vae.requires_grad_(False)

    unet.to(device, dtype=weight_dtype)
    vae.to(device, dtype=weight_dtype)
    text_encoder.to(device, dtype=weight_dtype)

    outpath = args.output
    input_path = os.path.join(outpath, "inputs")
    os.makedirs(input_path, exist_ok=True)
    gt_path = os.path.join(outpath, "gts")
    os.makedirs(gt_path, exist_ok=True)
    latent_path = os.path.join(outpath, "latents")
    os.makedirs(latent_path, exist_ok=True)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    steps = min(config["n_pairs"], len(dataset))
    
    global_step = len(glob.glob(os.path.join(gt_path, "*.png"))) // config["data"]["n_frames"]

    progress_bar = tqdm(total=steps, desc="Progress", initial=global_step)
    print(f"Resuming from {global_step} step")
    while global_step < steps:
        for data in loader:

            torch.cuda.empty_cache()

            lq = data["lq"].to(weight_dtype)
            gt = data["gt"].to(weight_dtype)

            caption = data["caption"]

            encoder_hidden_states = text_encoder(tokenize_captions(caption).to(device), return_dict=False)[0]

            gt_latents = vae.encode(gt).latent_dist.sample() * vae.config.scaling_factor
            noise = torch.randn_like(gt_latents)
            noise_level = torch.randint(0, low_res_scheduler.config.num_train_timesteps, (lq.shape[0],),
                                        device=device)
            timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (gt_latents.shape[0],),
                                      device=device).long()

            lq_noise = torch.randn_like(lq)
            lq_noisy = low_res_scheduler.add_noise(lq, noise=lq_noise, timesteps=noise_level)

            gt_latents = scheduler.add_noise(gt_latents, noise, timesteps)
            xc = torch.cat([gt_latents, lq_noisy], dim=1)

            pred = unet(sample=xc,
                        timestep=timesteps,
                        encoder_hidden_states=encoder_hidden_states,
                        class_labels=noise_level,
                        return_dict=False)[0]

            n_frames = gt.shape[2]
            for i in range(batch_size):
                for f in range(n_frames):

                    index = i * n_frames + f

                    # x_input = tensor_to_np(lq)
                    # Image.fromarray(x_input[index].astype(np.uint8)).save(
                    #     os.path.join(input_path, f"{global_step:06}_{f}.png"))
                    #
                    # x_gt = tensor_to_np(gt)
                    # Image.fromarray(x_gt[index].astype(np.uint8)).save(
                    #     os.path.join(gt_path, f"{global_step:06}_{f}.png"))
                    #
                    # x_latent = tensor_to_np(pred)
                    # np.save(os.path.join(latent_path, f"{global_step:06}_{f}.npy"), x_latent[index])

                global_step += 1
            progress_bar.update(batch_size)


if __name__ == "__main__":
    main()
