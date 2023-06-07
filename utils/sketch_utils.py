import os
import wandb
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

from torchvision import transforms
from torchvision.utils import make_grid


def plot_attn_clip(attn, threshold_map, inputs, inds, use_wandb, output_path):
    # currently supports one image (and not a batch)
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 3, 1)
    main_im = make_grid(inputs, normalize=True, pad_value=2)
    main_im = np.transpose(main_im.cpu().numpy(), (1, 2, 0))
    plt.imshow(main_im, interpolation='nearest')
    plt.scatter(inds[:, 1], inds[:, 0], s=10, c='red', marker='o')
    plt.title("input im")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(attn, interpolation='nearest', vmin=0, vmax=1)
    plt.title("atn map")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    threshold_map_ = (threshold_map - threshold_map.min()) / \
        (threshold_map.max() - threshold_map.min())
    plt.imshow(threshold_map_, interpolation='nearest', vmin=0, vmax=1)
    plt.title("prob softmax")
    plt.scatter(inds[:, 1], inds[:, 0], s=10, c='red', marker='o')
    plt.axis("off")

    plt.tight_layout()
    if use_wandb:
        wandb.log({"attention_map": wandb.Image(plt)})
    plt.savefig(output_path)
    plt.close()


def plot_atten(attn, threshold_map, inputs, inds, use_wandb, output_path):
    plot_attn_clip(attn, threshold_map, inputs, inds,
                       use_wandb, output_path)