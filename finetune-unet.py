
import argparse
import os
import shutil

import matplotlib.pyplot as plt
import torch
from accelerate import Accelerator
from diffusers import DDPMScheduler, DDIMScheduler
from diffusers.optimization import get_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from dataset import YouHQDataset
from models.Unet_3d import UNet3DConditionModel
from models.autoencoder_kl import AutoencoderKL


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path",
                        default="pretrained_models/upscale_a_video",
                        type=str,
                        help="Path to pretrained model")

    parser.add_argument("--gradient_accumulation_steps",
                        default=1,
                        type=int,
                        help="Number of updates steps to accumulate before performing a backward/update pass."
                        )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument("--train_steps",
                        type=int,
                        default=10_000,
                        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
                        )

    parser.add_argument("--lr",
                        default=1e-4,
                        type=float,
                        help="Initial learning rate"
                        )
    parser.add_argument("--resolution",
                        default=320,
                        type=int,
                        help="The resolution of high-resolution images will be automatically reduced by a factor of 4 for low-resolution processing.")

    parser.add_argument("--lr_scheduler",
                        default="cosine",
                        type=str,
                        help='The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
                        )
    parser.add_argument("--batch_size",
                        default=1,
                        type=int,
                        help="Batch size (per device) for the training dataloader."
                        )

    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--n_frames", type=int, default=8, help="Number of frames per patch")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints",
        help="The output directory where the checkpoints will be written.",
    )
    parser.add_argument("-d", "--data_path", type=str, default="VAE_dataset", help="Data root path for training data.")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default="latest",
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )

    args = parser.parse_args()
    return args


def main():

    args = parse_args()

    dataset = YouHQDataset("YouHQ-Train",  n_frames=args.n_frames)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    low_res_scheduler = DDPMScheduler.from_pretrained(args.model_path, subfolder="low_res_scheduler")
    scheduler = DDIMScheduler.from_pretrained(args.model_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(args.model_path, subfolder="tokenizer")

    vae = AutoencoderKL.from_pretrained(args.model_path, subfolder="vae", config_type="3d")
    unet = UNet3DConditionModel.from_pretrained(args.model_path, subfolder="unet", keywords="temporal")  # pretrained layers has been frozen

    text_encoder = CLIPTextModel.from_pretrained(
        args.model_path, subfolder="text_encoder"
    )

    gradient_accumulation_steps = args.gradient_accumulation_steps
    accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation_steps, mixed_precision=args.mixed_precision)

    # freeze text encoder and vae
    text_encoder.requires_grad_(False)
    vae.requires_grad_(False)

    weight_dtype = torch.float32
    if args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16

    batch_size = args.batch_size

    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    train_layers = list(filter(lambda p: p.requires_grad, unet.parameters()))

    def tokenize_captions(captions):

        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids

    optimizer = torch.optim.AdamW(
        train_layers,
        lr=args.lr,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    max_train_steps = args.train_steps
    num_training_steps_for_scheduler = max_train_steps * accelerator.num_processes

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps_for_scheduler,
    )

    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("unet")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])
    else:
        global_step = 0

    progress_bar = tqdm(
        range(0, max_train_steps),
        initial=global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )
    losses = []
    while global_step < max_train_steps:
        for batch, lr_batch in train_dataloader:
            torch.cuda.empty_cache()
            with accelerator.accumulate(unet):

                batch = batch.to(device, dtype=weight_dtype)

                low_res_batch = lr_batch.to(device, dtype=weight_dtype)
                caption = text_encoder(tokenize_captions("").to(device), return_dict=False)[0].to(dtype=weight_dtype)

                latent = vae.encode(batch).latent_dist.sample() * vae.config.scaling_factor

                noise = torch.randn_like(latent)

                timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (latent.shape[0],), device=latent.device).long()
                latent_noisy = scheduler.add_noise(latent, noise=noise, timesteps=timesteps)

                lr_noise = torch.randn_like(low_res_batch)

                noise_level = torch.randint(0, low_res_scheduler.config.num_train_timesteps, (low_res_batch.shape[0],), device=latent.device)
                lr_noisy = low_res_scheduler.add_noise(low_res_batch, timesteps=noise_level, noise=lr_noise)

                xc = torch.cat([latent_noisy, lr_noisy], dim=1)
                pred = unet(sample=xc, timestep=timesteps, encoder_hidden_states=caption, class_labels=noise_level, return_dict=False)[0]

                if scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif scheduler.config.prediction_type == "v_prediction":
                    target = scheduler.get_velocity(latent, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {scheduler.config.prediction_type}")

                loss = torch.nn.functional.mse_loss(pred.float(), target.float(), reduction="mean")
                losses.append(loss.item())

                progress_bar.set_postfix({"Loss": loss.detach().item()})
                global_step += 1
                progress_bar.update(1)
                if global_step % 500 == 0:
                    plt.plot(losses)
                    plt.show()

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), 1)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                if accelerator.sync_gradients:
                    if global_step % args.checkpointing_steps == 0:
                        if accelerator.is_main_process:

                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("unet")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= 1:
                                num_to_remove = len(checkpoints)
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                            save_path = os.path.join(args.output_dir, f"unet-{global_step}")
                            accelerator.save_state(save_path)


if __name__ == "__main__":
    main()
