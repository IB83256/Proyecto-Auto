# -*- coding: utf-8 -*-
"""
Visualization utilities for color image generation and diffusion models.

Functions included:
- plot_image_grid: visualize batches of RGB images in a grid.
- plot_image_evolution: visualize evolution of multiple RGB images over time steps.
- animation_images: create an animation from the temporal evolution of a single RGB image.

Author: Álvaro Duro y Carlos Beti
Date: 2025-05-05
"""

from numpy.typing import ArrayLike
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

import torch
from torchvision.utils import make_grid
from torchvision.transforms import functional


def plot_image_grid_color( 
    images: torch.Tensor, 
    figsize: tuple,
    n_rows: int,
    n_cols: int,
    padding: int = 2,
    pad_value: int = 1.0,
    normalize: bool = False,
    axis_on_off: bool = "off",  
):
    """
    Display a grid of RGB images using matplotlib.

    Args:
        images (Tensor): List or batch of image tensors (C, H, W).
        figsize (tuple): Figure size in inches.
        n_rows (int): Number of rows in the grid.
        n_cols (int): Number of columns in the grid.
        padding (int): Padding between images.
        pad_value (float): Padding value.
        normalize (bool): Normalize images for display.
        axis_on_off (str): "on" or "off" to show/hide axes.
    """
    grid = make_grid(
        images, 
        nrow=n_cols, 
        padding=padding, 
        normalize=normalize,
        pad_value=pad_value,
    ) 

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(functional.to_pil_image(grid))  # RGB — no cmap
    ax.axis(axis_on_off)
    return fig, ax


def plot_image_evolution_color(
    images: torch.Tensor,
    n_images: int,
    n_intermediate_steps: ArrayLike,
    figsize: tuple,
):
    """
    Display the temporal evolution of several RGB images side by side.

    Args:
        images (Tensor): Tensor of shape (N, C, H, W, T).
        n_images (int): Number of images to show.
        n_intermediate_steps (ArrayLike): Time indices to display.
        figsize (tuple): Figure size.
    """
    fig, axs = plt.subplots(n_images, len(n_intermediate_steps), figsize=figsize)

    for n_image in np.arange(n_images):
        for i, ax in enumerate(axs[n_image, :]):
            ax.imshow(
                images[n_image, :, :, :, n_intermediate_steps[i]].permute(1, 2, 0)
            )
            ax.set_axis_off()
    return fig, axs

def animation_images_color(
    images_t: np.ndarray,
    interval: int,
    figsize: tuple,
):
    """
    Create an animation of a single RGB image evolving over time.

    Args:
        images_t (ndarray): Array of shape (H, W, C, T).
        interval (int): Delay between frames in milliseconds.
        figsize (tuple): Figure size.

    Returns:
        fig, ax, animation object
    """
    H, W, C, T = images_t.shape

    fig, ax = plt.subplots(figsize=figsize)
    img_display = ax.imshow(images_t[:, :, :, 0])
    ax.axis("off")

    def update(t):
        img_display.set_array(images_t[:, :, :, t])
        return [img_display]

    return (
        fig,
        ax,
        animation.FuncAnimation(
            fig,
            update,
            frames=T,
            interval=interval,
            blit=False
        )
    )
