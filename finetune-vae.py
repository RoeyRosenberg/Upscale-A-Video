import argparse
import os
import shutil

import accelerate
import matplotlib.pyplot as plt
import torch.optim
from einops import rearrange
from torch.utils.data import DataLoader
from tqdm import tqdm

from contperceptual import LPIPSWithDiscriminator
from dataset import SingleImageNPDataset
from models import AutoencoderKL


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path",
                        default="pretrained_models/upscale_a_video",
                        type=str,
                        help="Path to pretrained model")

    parser.add_argument("--train_steps",
                        type=int,
                        default=100_000,
                        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
                        )
    parser.add_argument("--gradient_accumulation_steps",
                        default=1,
                        type=int,
                        help="Number of updates steps to accumulate before performing a backward/update pass."
                        )
    parser.add_argument("--batch_size",
                        default=1,
                        type=int,
                        help="Batch size (per device) for the training dataloader."
                        )
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
    parser.add_argument("-d", "--data_path", type=str, default="VAE_dataset", help="Data root path for training data.")
    parser.add_argument("--n_frames", type=int, default=8, help="Number of frames per single patch")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints",
        help="The output directory where the checkpoints will be written.",
    )
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

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    vae = AutoencoderKL.from_pretrained(
        args.model_path,
        subfolder="vae",
        config_type="video",
        pretrained_type="3d",
        keywords=["3d", "condition"]
    )

    weight_dtype = torch.float32
    if args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
        print("NOTE: Not recommended, using torch.float16 may potentially result in NaN values. ")

    accelerator = accelerate.Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps,
                                         mixed_precision=args.mixed_precision)
    vae.to(accelerator.device, dtype=weight_dtype)

    batch_size = args.batch_size
    n_frames = args.n_frames

    train_dataset = SingleImageNPDataset(gt_path=args.gt_path, n_frames=n_frames)
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    lr = 1e-4

    dual_loss = LPIPSWithDiscriminator(disc_start=501, disc_weight=0.025, disc_factor=1.0).to(device, dtype=weight_dtype)

    train_layers = list(filter(lambda p: p.requires_grad, vae.parameters()))

    opt_ae = torch.optim.Adam(train_layers,
                              lr=lr, betas=(0.9, 0.99))
    opt_disc = torch.optim.Adam(dual_loss.discriminator.parameters(),
                                lr=lr, betas=(0.9, 0.99))

    train_steps = args.train_steps

    # Prepare everything with our `accelerator`.
    opt_ae, opt_disc, train_dataloader, vae = accelerator.prepare(
         opt_ae, opt_disc, dataloader, vae
    )

    optimizers = [opt_ae, opt_disc]

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)

            dirs = [d for d in dirs if d.startswith("vae")]
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
        total=train_steps,
        initial=global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )
    losses = []
    with progress_bar:
        for batch in train_dataloader:
            torch.cuda.empty_cache()
            lq = batch["lq"].to(device, dtype=weight_dtype)
            gt = batch["gt"].to(device, dtype=weight_dtype)
            latent = batch["latent"].to(device, dtype=weight_dtype) / vae.config.scaling_factor

            reconstructions = vae.decode(latent, condition=lq).sample
            gt = rearrange(gt, "b c f h w -> (b f) c h w")
            reconstructions = rearrange(reconstructions, "b c f h w -> (b f) c h w")
            for optimizer_idx, optimizer in enumerate(optimizers):

                loss = dual_loss(gt, reconstructions, optimizer_idx, global_step,
                                    last_layer=vae.get_last_layer(), split="train")

                if optimizer_idx == 0:
                    ae_loss = loss.item()
                    losses.append(ae_loss)
                if optimizer_idx == 1:
                    disc_loss = loss.item()

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(vae.parameters(), 1)
                optimizer.step()

                optimizer.zero_grad()
            if global_step > 50:
                plt.plot(losses)
                plt.show()
            progress_bar.set_postfix({"ae_loss": ae_loss, "disc_loss": disc_loss})
            global_step += 1
            progress_bar.update(1)
            if accelerator.sync_gradients:
                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:

                        checkpoints = os.listdir(args.output_dir)
                        checkpoints = [d for d in checkpoints if d.startswith("vae")]
                        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                        # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                        if len(checkpoints) >= 1:
                            num_to_remove = len(checkpoints)
                            removing_checkpoints = checkpoints[0:num_to_remove]

                            for removing_checkpoint in removing_checkpoints:
                                removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"vae-{global_step}")
                        accelerator.save_state(save_path)


if __name__ == "__main__":
    main()
