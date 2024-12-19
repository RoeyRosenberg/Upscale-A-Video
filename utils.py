import os.path
from typing import Union, List, Tuple

import numpy as np
import torch
import torchvision.io
from PIL import Image
from einops import rearrange
from safetensors.torch import load_file
from torchvision.models.optical_flow import raft_large
from transformers import pipeline


def convert_to_list(*args):

    return list(args)


def get_optical_flows(frames, dtype, device="cuda"):
    """
      Computes the forward and backward optical flows for a sequence of video frames using the RAFT model.

      Parameters
      ----------
      frames : torch.Tensor
          A 5D tensor representing a batch of video frames with dimensions `(batch_size, channels, num_frames, height, width)`.

      dtype : torch.dtype
          The data type for the computed optical flow tensors (e.g., `torch.float32` or `torch.float16`).

      device : str, optional
          The device to perform the computation on. Default is `"cuda"` for GPU acceleration.

      Returns
      -------
      backward_flows : torch.Tensor
          A 5D tensor representing the backward optical flows with dimensions
          `(batch_size, 2, num_frames - 1, height, width)`.
          - `2` represents the flow in the x and y directions.

      forward_flows : torch.Tensor
          A 5D tensor representing the forward optical flows with dimensions
          `(batch_size, 2, num_frames - 1, height, width)`.
          - `2` represents the flow in the x and y directions.
    """
    raft = raft_large("Raft_Large_Weights.C_T_SKHT_K_V2").to(device)
    raft.requires_grad_(False)

    last_frame, current_frame = frames[:, :, :-1, :, :], frames[:, :, 1:, :, :]
    last_frame = rearrange(last_frame, "b c f h w -> (b f) c h w").to(device)
    current_frame = rearrange(current_frame, "b c f h w -> (b f) c h w").to(device)

    n_frames = frames.shape[2]
    forward_flows = raft(last_frame, current_frame)[-1].to(dtype)
    backward_flows = raft(current_frame, last_frame)[-1].to(dtype)

    forward_flows = rearrange(forward_flows, "(b f) c h w -> b c f h w", f=n_frames - 1)
    backward_flows = rearrange(backward_flows, "(b f) c h w -> b c f h w", f=n_frames - 1)

    # Clear some gpu memory
    del last_frame, current_frame, raft
    torch.cuda.empty_cache()

    return backward_flows, forward_flows


def get_video_frames(path: str) -> Tuple[torch.Tensor, float]:
    """
    Extracts video frames and their corresponding frame rate from a video file.

    Parameters
    ----------
    path : str
        The file path to the video. Supported formats include `.mp4`, `.mov`, and `.avi`
        (case insensitive). The function will raise an `AssertionError` if the file extension is not supported.

    Returns
    -------
    frames : torch.Tensor
        A normalized 4D tensor containing the video frames with dimensions `(time, channels, height, width)`:

    fps : float
        The frames per second (FPS) of the video.
    """

    VIDEO_EXTENSIONS = ('.mp4', '.mov', '.avi', '.MP4', '.MOV', '.AVI')
    assert path.endswith(VIDEO_EXTENSIONS)

    frames, _, info = torchvision.io.read_video(path, pts_unit="sec", output_format="TCHW")

    fps = info["video_fps"]
    frames = frames / 255. * 2 - 1  # TCHW

    return frames, fps


def load_model(
    model: torch.nn.Module,
    filename: Union[str, os.PathLike],
    strict: bool = False,
    device: Union[str, int] = "cuda",
    keywords: Union[str, List[str]] = None,
) -> torch.nn.Module:
    """
    Loads a given filename onto a torch model.
    This method exists specifically to avoid tensor sharing issues which are
    not allowed in `safetensors`. [More information on tensor sharing](../torch_shared_tensors)

    Args:
        model (`torch.nn.Module`):
            The model to load onto.

        filename (`str`, or `os.PathLike`):
            The filename location to load the file from.

        strict (`bool`, *optional*, defaults to True):
            Whether to fail if you're missing keys or having unexpected ones.
            When false, the function simply returns missing and unexpected names.

        device (`Union[str, int]`, *optional*, defaults to `cuda`):
            The device where the tensors need to be located after load.
            available options are all regular torch device locations.

        keywords(`Union[str, List[str]`, *optional*) Train only the parameters
            that contain one of the specified keywords in their name.

    """
    if filename.endswith(".safetensors"):
        state_dict = load_file(filename, device=device)
    elif filename.endswith(".bin"):
        state_dict = torch.load(filename, map_location=device, weights_only=True)

    # In older versions of Diffusers, the to_q, to_k, and to_v in attention blocks were called query, key, and value.
    for name, parameter in state_dict.copy().items():
        if "query" in name:
            new_name = name.replace("query", "to_q")
            state_dict[new_name] = state_dict.pop(name)
        if "key" in name:
            new_name = name.replace("key", "to_k")
            state_dict[new_name] = state_dict.pop(name)
        if "value" in name:
            new_name = name.replace("value", "to_v")
            state_dict[new_name] = state_dict.pop(name)

    if keywords is not None:
        param_names = [name for name, _ in model.named_parameters()]

        if isinstance(keywords, str):
            keywords = [keywords]

        keyword_names = list(filter(lambda name: any(sub in name for sub in keywords), param_names))

        named_parameters = dict(model.named_parameters())
        model.requires_grad_(False)
        for name in keyword_names:
            named_parameters[name].requires_grad = True

    model.load_state_dict(state_dict, strict=strict)

    return model


def denormalize_tensor(tensor: torch.Tensor) -> np.ndarray:
    """
    Converts a normalized PyTorch tensor into a denormalized NumPy array.

    Parameters
    ----------
    tensor : torch.Tensor
        A PyTorch tensor representing a single image with dimensions `(channels, height, width)`:
        The tensor should have values in the range `[-1, 1]`.

    Returns
    -------
    np.ndarray
        A NumPy array representing the denormalized image with dimensions `(height, width, channels)`:

    """
    tensor = ((tensor / 2) + .5) * 255
    tensor = tensor.detach().cpu().float().numpy().transpose(1, 2, 0).astype(np.uint8)

    return tensor


def tensor_to_np(tensor: torch.Tensor) -> np.ndarray:
    """
    Converts a PyTorch tensor to a NumPy array. Supports tensors with 3, 4, or 5 dimensions.
    Handles denormalization of pixel values when necessary.

    Parameters
    ----------
    tensor : torch.Tensor
        Input PyTorch tensor.

    Returns
    -------
    np.ndarray
        Corresponding NumPy array representation of the input tensor.
    """
    if tensor.ndim == 5:
        tensor = rearrange(tensor, "b c f h w -> (b f) c h w")

    if tensor.ndim == 4:
        b, c, h, w = tensor.shape
        array = np.zeros((b, h, w, c), dtype=np.uint8)
        for i, t in enumerate(tensor):
            t = denormalize_tensor(t)
            array[i] = t
    elif tensor.ndim == 3:
        array = denormalize_tensor(tensor)

    return array


def blip_it(src: Union[str, Image.Image, torch.Tensor]) -> str:
    """
    Generates a caption for the input image using the BLIP (Bootstrapping Language-Image Pre-training) model.
    The input can be a file path (string), an image (PIL Image), or a tensor (PyTorch tensor).

    Parameters
    ----------
    src : Union[str, Image.Image, torch.Tensor]
        The input image in one of the following formats:
        - A string (file path to the image),
        - A PIL Image object,
        - A PyTorch tensor representing the image.

    Returns
    -------
    str
        A generated caption describing the content of the image.
    """
    if isinstance(src, str):
        src = Image.open(src)

    if isinstance(src, torch.Tensor):
        src = Image.fromarray((src.detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8))

    captioner = pipeline("image-to-text", model="Salesforce/blip-image-captioning-large", device="cuda")

    caption = captioner(src)[0]["generated_text"]
    return caption
