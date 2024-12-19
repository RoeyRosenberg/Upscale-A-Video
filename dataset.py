import glob
import math
import os.path
import random

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from einops import rearrange
from torch.utils.data import Dataset

from basicsr.data.degradations import random_add_poisson_noise_pt, random_add_gaussian_noise_pt, \
    apply_random_video_compression
from basicsr.data.degradations import random_mixed_kernels, circular_lowpass_kernel
from basicsr.data.transforms import augment
from basicsr.data.transforms import random_crop
from basicsr.utils import USMSharp, DiffJPEG
from basicsr.utils.img_process_util import filter2D


class VideoPairsDataset(Dataset):
    def __init__(self, opt, device):
        super().__init__()

        self.opt = opt
        self.device = device

        self.n_frames = opt["n_frames"]

        # self.usm_sharpener = USMSharp().cuda()
        self.jpeger = DiffJPEG(differentiable=False).cuda()
        self.usm_sharpener = USMSharp().cuda()



        self.kernel_range = [2 * v + 1 for v in range(3, 11)]  # kernel size ranges from 7 to 21

        self.pulse_tensor = torch.zeros(21, 21).float()  # convolving with pulse tensor brings no blurry effect
        self.pulse_tensor[10, 10] = 1

        self.paths = glob.glob(opt["dataset_path"]+"/**/*.mp4", recursive=True)

        if len(self.paths) == 0:
            print(f"NOTE: dataset is empty, path: {opt['dataset_path']}/*/*/**.mp4")

        self._index = 0
        self._start = 0

    def __len__(self):
        return len(self.paths) * (30 // self.n_frames)  # the maximum frames in video is 30

    def _preprocess(self, gt):

        opt = self.opt["preprocess"]
        # -------------------- Do augmentation for training: flip, rotation -------------------- #
        img_gt = augment(gt, opt['use_hflip'], opt['use_rot'])

        # crop or pad to 400
        # TODO: 400 is hard-coded. You may change it accordingly
        # h, w = img_gt.shape[0:2]
        crop_pad_size = 320
        # pad
        # if h < crop_pad_size or w < crop_pad_size:
        #     pad_h = max(0, crop_pad_size - h)
        #     pad_w = max(0, crop_pad_size - w)
        #     img_gt = cv2.copyMakeBorder(img_gt, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)

        # cropping same position for each frame in feed_data function

        # if img_gt.shape[0] > crop_pad_size or img_gt.shape[1] > crop_pad_size:
        #     h, w = img_gt.shape[0:2]
        #     # randomly choose top and left coordinates
        #     top = random.randint(0, h - crop_pad_size)
        #     left = random.randint(0, w - crop_pad_size)
        #     img_gt = img_gt[top:top + crop_pad_size, left:left + crop_pad_size, ...]

        # ------------------------ Generate kernels (used in the first degradation) ------------------------ #
        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < opt['sinc_prob']:
            # this sinc filter setting is for kernels ranging from [7, 21]
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel = random_mixed_kernels(
                opt["kernel_list"],
                opt["kernel_prob"],
                kernel_size,
                opt["self.blur_sigma"],
                opt["blur_sigma"], [-math.pi, math.pi],
                opt["betag_range"],
                opt["betap_range"],
                noise_range=None)
        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))

        # ------------------------ Generate kernels (used in the second degradation) ------------------------ #
        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < opt['sinc_prob2']:
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel2 = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel2 = random_mixed_kernels(
                opt["kernel_list2"],
                opt["kernel_prob2"],
                opt["kernel_size"],
                opt["self.blur_sigma2"],
                opt["blur_sigma2"], [-math.pi, math.pi],
                opt["betag_range2"],
                opt["betap_range2"],
                noise_range=None)

        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel2 = np.pad(kernel2, ((pad_size, pad_size), (pad_size, pad_size)))

        # ------------------------------------- the final sinc kernel ------------------------------------- #
        if np.random.uniform() < opt['final_sinc_prob']:
            kernel_size = random.choice(self.kernel_range)
            omega_c = np.random.uniform(np.pi / 3, np.pi)
            sinc_kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=21)
            sinc_kernel = torch.FloatTensor(sinc_kernel)
        else:
            sinc_kernel = self.pulse_tensor

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt = img_gt.to(self.device)

        kernel = torch.FloatTensor(kernel).to(self.device)
        kernel2 = torch.FloatTensor(kernel2).to(self.device)
        sinc_kernel = sinc_kernel.to(self.device)
        d = {'gt': img_gt,
             'kernel': kernel,
             'kernel2': kernel2,
             'sinc_kernel': sinc_kernel,
             }
        return d

    @torch.no_grad()
    def feed_data(self, data):

        opt = self.opt["degradation"]
        # training data synthesis
        gt = data['gt'].cuda()

        gt = self.usm_sharpener(gt)
        self.kernel1 = data['kernel']
        self.kernel2 = data['kernel2']
        self.sinc_kernel = data['sinc_kernel']

        ori_h, ori_w = gt.size()[2:4]

        # ----------------------- The first degradation process ----------------------- #
        # blur
        out = filter2D(gt, self.kernel1)
        # random resize
        updown_type = random.choices(['up', 'down', 'keep'], opt['resize_prob'])[0]
        if updown_type == 'up':
            scale = np.random.uniform(1, opt['resize_range'][1])
        elif updown_type == 'down':
            scale = np.random.uniform(opt['resize_range'][0], 1)
        else:
            scale = 1
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        out = F.interpolate(out, scale_factor=scale, mode=mode)
        # add noise
        gray_noise_prob = opt['gray_noise_prob']
        if np.random.uniform() < opt['gaussian_noise_prob']:
            out = random_add_gaussian_noise_pt(
                out, sigma_range=opt['noise_range'], clip=True, rounds=False, gray_prob=gray_noise_prob)
        else:
            out = random_add_poisson_noise_pt(
                out,
                scale_range=opt['poisson_scale_range'],
                gray_prob=gray_noise_prob,
                clip=True,
                rounds=False)
        # JPEG compression + Video compression
        jpeg_p = out.new_zeros(out.size(0)).uniform_(*opt['jpeg_range'])
        out = torch.clamp(out, 0, 1)  # clamp to [0, 1], otherwise JPEGer will result in unpleasant artifacts
        out = self.jpeger(out, quality=jpeg_p)

        mode = ['libx264', 'h264', 'mpeg4']
        codec = random.choices(mode, [1 / 3., 1 / 3., 1 / 3.])[0]
        bitrate = np.random.randint(1e4, 1e5 + 1)

        out = apply_random_video_compression(out, codec, bitrate)

        # ----------------------- The second degradation process ----------------------- #
        # blur
        if np.random.uniform() < opt['second_blur_prob']:
            out = filter2D(out, self.kernel2)
        # random resize
        updown_type = random.choices(['up', 'down', 'keep'], opt['resize_prob2'])[0]
        if updown_type == 'up':
            scale = np.random.uniform(1, opt['resize_range2'][1])
        elif updown_type == 'down':
            scale = np.random.uniform(opt['resize_range2'][0], 1)
        else:
            scale = 1
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        out = F.interpolate(
            out, size=(int(ori_h / opt['scale'] * scale), int(ori_w / opt['scale'] * scale)), mode=mode)
        # add noise
        gray_noise_prob = opt['gray_noise_prob2']
        if np.random.uniform() < opt['gaussian_noise_prob2']:
            out = random_add_gaussian_noise_pt(
                out, sigma_range=opt['noise_range2'], clip=True, rounds=False, gray_prob=gray_noise_prob)
        else:
            out = random_add_poisson_noise_pt(
                out,
                scale_range=opt['poisson_scale_range2'],
                gray_prob=gray_noise_prob,
                clip=True,
                rounds=False)

        # JPEG compression + Video compression+ the final sinc filter
        # We also need to resize images to desired sizes. We group [resize back + sinc filter] together
        # as one operation.
        # We consider two orders:
        #   1. [resize back + sinc filter] + JPEG compression + Video compression
        #   2. JPEG compression + Video compression + [resize back + sinc filter]
        # Empirically, we find other combinations (sinc + JPEG + Video + Resize) will introduce twisted lines.
        if np.random.uniform() < 0.5:
            # resize back + the final sinc filter
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(out, size=(ori_h // opt['scale'], ori_w // opt['scale']), mode=mode)
            out = filter2D(out, self.sinc_kernel)
            # JPEG compression
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*opt['jpeg_range2'])
            out = torch.clamp(out, 0, 1)
            out = self.jpeger(out, quality=jpeg_p)

            mode = ['libx264', 'h264', 'mpeg4']
            codec = random.choices(mode, [1 / 3., 1 / 3., 1 / 3.])[0]

            bitrate = np.random.randint(1e4, 1e5 + 1)

            out = apply_random_video_compression(out, codec, bitrate)

        else:
            # JPEG compression + Video compression
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*opt['jpeg_range2'])
            out = torch.clamp(out, 0, 1)
            out = self.jpeger(out, quality=jpeg_p)

            mode = ['libx264', 'h264', 'mpeg4']
            codec = random.choices(mode, [1 / 3., 1 / 3., 1 / 3.])[0]

            bitrate = np.random.randint(1e4, 1e5 + 1)

            out = apply_random_video_compression(out, codec, bitrate)
            # resize back + the final sinc filter
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(out, size=(ori_h // opt['scale'], ori_w // opt['scale']), mode=mode)
            out = filter2D(out, self.sinc_kernel)

        # clamp and round
        lq = (out - .5) * 2
        gt = (gt - .5) * 2
        data["gt"] = rearrange(gt, "f c h w -> c f h w")
        data["lq"] = rearrange(lq, "f c h w -> c f h w")

        return data

    def __getitem__(self, index):

        path = self.paths[self._index]

        video, _, fps_info = torchvision.io.read_video(path, pts_unit="sec", output_format="TCHW")
        video_frames = video.shape[0]

        video = video[self._start:self._start+self.n_frames]

        self._start += self.n_frames
        if self._start + self.n_frames > video_frames:
            self._index += 1
            self._start = 0

        # because YouHQ dataset has no captions and WebVid10M is no longer available, the captions are null.
        batch = {"caption": ""}

        video = random_crop(video, self.opt["preprocess"]["gt_size"]) / 255

        for image in video:
            d = self._preprocess(image)

            for key in d.keys():
                d[key] = d[key].unsqueeze(0)

                if key not in batch.keys():
                    batch.update({key: d[key]})
                else:
                    batch[key] = torch.cat((batch[key], d[key]), dim=0)

        batch = self.feed_data(batch)

        return batch


class SingleImageNPDataset(Dataset):
    """Read only lq images in the test phase.

    Read diffusion generated data for training CFW.

    Args:

        gt_path: Data root path for training data. The path needs to contain the following folders:
            gts: Ground-truth images.
            inputs: Input LQ images.
            latents: The corresponding HQ latent code generated by diffusion model given the input LQ image.

    """

    def __init__(self, gt_path: str, n_frames: int = 8, image_type: str = "png"):
        super(SingleImageNPDataset, self).__init__()

        self.file_client = None
        self.n_frames = n_frames

        self.gt_paths = glob.glob(os.path.join(gt_path, "gts", f"*.{image_type}"))
        self.np_paths = glob.glob(os.path.join(gt_path, "latents", "*.npy"))

        self._index = 0

        assert len(self.gt_paths) > 0, "Dataset is empty"

    def __len__(self):
        return len(self.gt_paths) // self.n_frames

    def _load_data(self, index):

        # load lq image
        gt_path = self.gt_paths[index]

        lq_path = gt_path.replace("gts", "inputs")
        np_path = self.np_paths[index]

        img_lq = torchvision.io.read_image(lq_path)
        img_gt = torchvision.io.read_image(gt_path)

        latent_np = np.load(np_path)
        img_lq = img_lq / 255 * 2 - 1
        img_gt = img_gt / 255 * 2 - 1

        latent_np = torch.from_numpy(latent_np).permute(2, 0, 1) / 255 * 2 - 1

        return {"lq": img_lq, "gt": img_gt, "latent": latent_np}

    def __getitem__(self, item):

        batch = {}
        for f in range(self.n_frames):
            d = self._load_data(item)

            for key in d.keys():
                d[key] = d[key].unsqueeze(0)
                d[key] = rearrange(d[key], "f c h w->c f h w")

                if key not in batch.keys():
                    batch.update({key: d[key]})
                else:
                    batch[key] = torch.cat((batch[key], d[key]), dim=1)

        return batch


class YouHQDataset(Dataset):
    def __init__(self, dataset_path, n_frames: int = 8, size: int = 320):
        self.dataset_path = dataset_path
        self.n_frames = n_frames
        self.videos = glob.glob(os.path.join(dataset_path, "**/*.mp4"), recursive=True)

        self.size = size
        if len(self.videos) == 0:
            print("NOTE: dataset is empty")

        self._index = 0
        self._start = 0

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, item):
        path = self.videos[self._index]
        frames, _, fps = torchvision.io.read_video(path, output_format="TCHW", pts_unit="sec")
        frames = frames[self._start:self._start+self.n_frames]

        frames = F.interpolate(frames, size=(self.size, self.size)) / 255 * 2 - 1
        frames = rearrange(frames, "t c h w -> c t h w")

        c, t, h, w = frames.shape

        self._start += self.n_frames
        if self._start + self.n_frames > t:
            self._index += 1
            self._start = 0

        lr_frames = F.interpolate(frames, size=(h//4, w//4))

        return frames, lr_frames








