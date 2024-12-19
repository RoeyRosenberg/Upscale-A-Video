import torch
import torch.nn.functional as F


def image_warp(im, flow, interpolation="bilinear", padding_mode="zeros", align_corners=True):
    """
    Performs a backward warp of an image using the predicted flow.

    Args:
        im: Batch of images. [num_batch, channels, height, width]
        flow: Batch of flow vectors. [num_batch, 2, height, width] (flow in x and y directions)
        interpolation: 'bilinear' | 'nearest' | 'bicubic'. Default: 'bilinear'
        padding_mode (str): padding mode for outside grid values
            'zeros' | 'border' | 'reflection'. Default: 'zeros'
        align_corners (bool): Before pytorch 1.3, the default value is
            align_corners=True. After pytorch 1.3, the default value is
            align_corners=False. Here, we use the True as default.

    Returns:
        warped: Transformed image of the same shape as the input image.
    """
    num_batch, channels, height, width = im.shape

    # Get the device from the input image
    device = im.device

    # Create a mesh grid for pixel coordinates on the same device as the input
    grid_y, grid_x = torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device),
                                    indexing='ij')
    grid = torch.stack((grid_x, grid_y), dim=0)  # [2, H, W]

    # Expand the grid to match the batch size
    grid = grid.unsqueeze(0).repeat(num_batch, 1, 1, 1)  # [num_batch, 2, H, W]

    # Add flow to the grid to get the displaced positions
    displaced_grid = grid + flow  # [num_batch, 2, H, W]

    # Normalize grid values to [-1, 1] range for torch grid_sample
    displaced_grid[:, 0, :, :] = 2.0 * displaced_grid[:, 0, :, :] / (width - 1) - 1.0  # Normalize x coordinates
    displaced_grid[:, 1, :, :] = 2.0 * displaced_grid[:, 1, :, :] / (height - 1) - 1.0  # Normalize y coordinates

    # Rearrange the grid for sampling: [num_batch, H, W, 2]
    displaced_grid = displaced_grid.permute(0, 2, 3, 1)

    warped = F.grid_sample(im, displaced_grid, mode=interpolation, padding_mode=padding_mode,
                           align_corners=align_corners)
    return warped


def length_sq(x):
    return torch.sum(torch.square(x), 1, keepdim=True)


def ConsistencyCheck(forward_flow, backward_flow, alpha1=.01, alpha2=.5):
    flow_bw_warped = image_warp(backward_flow, forward_flow)  # wb(wf(x))

    flow_diff_fw = length_sq(forward_flow + flow_bw_warped)

    mag_sq_fw = length_sq(forward_flow) + length_sq(flow_bw_warped)  # |wf| + |wb(wf(x))|

    fw_thresh = alpha1 * mag_sq_fw + alpha2

    fw_mask = (flow_diff_fw < fw_thresh).to(forward_flow)

    return fw_mask


def LatentPropagation(x, backward_flows, forward_flows, interpolation="nearest", alpha1=0.01, alpha2=0.5, beta=.5):

        b, c, f, h, w = x.shape

        latents_feat = []

        for frame in range(f):

            if frame == 0:
                latent_prop = x[:, :, 0, :, :]
            else:
                fw_flow = forward_flows[:, :, frame - 1, :, :]
                bw_flow = backward_flows[:, :, frame - 1, :, :]
                current_prop = x[:, :, frame, :, :]
                fw_mask = ConsistencyCheck(fw_flow, bw_flow, alpha1, alpha2)

                latent_warped = image_warp(latent_prop, bw_flow, interpolation=interpolation)

                latent_warped = (latent_warped * beta + current_prop * beta)

                latent_prop = latent_warped * fw_mask + (1 - fw_mask) * current_prop
            latents_feat.append(latent_prop)

        outputs = torch.stack(latents_feat, dim=2)

        return outputs
