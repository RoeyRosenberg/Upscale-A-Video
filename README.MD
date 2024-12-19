# Upscale-A-Video (Unofficial Implementation)

This repository provides an unofficial implementation of the [Upscale-A-Video](https://github.com/sczhou/Upscale-A-Video) model, designed for video super-resolution tasks.

---
## License
This project is an unofficial implementation of [Original Project Name] by S-Lab. It is licensed under the S-Lab License 1.0, which allows redistribution and use for non-commercial purposes under specific conditions.

For more information, refer to the [LICENSE](./LICENSE) file.

### Disclaimer
This implementation is **unofficial** and is not affiliated with or endorsed by S-Lab or the original authors.

---
## Features
- High-quality video upscaling using pre-trained models.
- Options for fine-tuning U-Net and VAE models.
- Seamless integration for inference on your own videos.

---

## Prerequisites
- Download the pre-trained Upscale-A-Video model from the [official repository](https://github.com/sczhou/Upscale-A-Video).

---

## Training Pipeline

### 1. Generating Training Data

Before fine-tuning, you need to download the YouHQ-Train dataset from the [official repository](https://github.com/sczhou/Upscale-A-Video).



### 2. Finetune

#### U-net:

```bash
python finetune-unet.py --model_path pretrained_models/upscale_a_video --data_path YouHQ-Train --resolution 320 
```

#### Generate synthetic data:
Before fine-tuning vae model, you need to generate synthetic data using the fine tuned U-net model:
```bash
python generate_synthetic_data.py --config configs/generate_video_pairs.yaml -o VAE_dataset
```

#### vae:

```bash
python finetune-vae.py --model_path pretrained_models/upscale_a_video --data_path VAE_dataset
```

### 3. Inference

```bash
python predict.py -i ./inputs/old_animation_1.mp4 -o ./results -n 120 -g 8 -s 20
```
