# -*- coding: utf-8 -*-
"""
Visualization utilities for image generation and diffusion models.

Functions included:
- plot_image_grid: visualize batches of images in a grid.
- plot_image_evolution: visualize evolution of multiple images over time steps.
- animation_images: create an animation from the temporal evolution of a single image.

Author: Álvaro Duro y Carlos Beti
Date: 2025-04-23
"""

from numpy.typing import ArrayLike

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

from matplotlib.colors import Colormap

import torch
from  torchvision.utils import make_grid
from torchvision.transforms import functional

def plot_image_grid( 
    images: torch.Tensor, 
    figsize: tuple,
    n_rows: int,
    n_cols: int,
    padding: int = 2,
    pad_value: int = 1.0,
    cmap: Colormap = "gray",
    normalize: bool = False,
    axis_on_off: bool = "off",  
    ):
    """
    Display a grid of images using matplotlib.
    Args:
        images (Tensor): List or batch of image tensors (C, H, W).
        figsize (tuple): Figure size in inches.
        n_rows (int): Number of rows in the grid.
        n_cols (int): Number of columns in the grid.
        padding (int): Padding between images.
        pad_value (float): Padding value.
        cmap (Colormap): Colormap for grayscale images.
        normalize (bool): Normalize images for display.
        axis_on_off (str): "on" or "off" to show/hide axes.
    Returns:
        fig, ax: Figure and axes objects.
    """

    grid = make_grid(
        images, 
        nrow=n_cols, 
        padding=padding, 
        normalize=normalize,
        pad_value=pad_value,
    ) 

    # Convert to PIL Image and display

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(functional.to_pil_image(grid), cmap=cmap)
    ax.axis("off")
    return fig, ax


def plot_image_evolution(
    images: torch.Tensor,
    n_images: int,
    n_intermediate_steps: ArrayLike,
    figsize: tuple,
    cmap: Colormap = "gray",
    ):

    """
    Display the temporal evolution of several images side by side.
    Args:
        images (Tensor): Tensor of shape (N, C, H, W, T).
        n_images (int): Number of images to show.
        n_intermediate_steps (ArrayLike): Time indices to display.
        figsize (tuple): Figure size.
        cmap (Colormap): Colormap for grayscale images.
    Returns:
        fig, axs: Figure and axes objects.
    """

    fig, axs = plt.subplots(
        n_images, 
        len(n_intermediate_steps), 
        figsize=figsize,
    )

    for n_image in np.arange(n_images):
        for i, ax in enumerate(axs[n_image, :]):
            ax.imshow(
                images[n_image, 0,:, :, n_intermediate_steps[i]], 
                cmap="gray",
                )
            axs[n_image, i].set_axis_off()
    return fig, axs
    
def animation_images(
        images_t, 
        interval,
        figsize,
    ): 
    """
    Create an animation from the temporal evolution of a single image.
    Args:
        images_t (ndarray): Array of shape (H, W, C, T).
        interval (int): Delay between frames in milliseconds.
        figsize (tuple): Figure size.
    Returns:
        fig, ax, animation object
    """
    _, _, n_frames = np.shape(images_t)

    # Create a figure and axes.  
    fig, ax = plt.subplots(figsize=figsize)
    img_display = ax.imshow(images_t[:, :, 0], cmap="gray")

    def update(t):
        """Update function for the animation."""
        img_display.set_array(images_t[:, :, t])
        return [img_display]

    return ( 
        fig, 
        ax, 
        animation.FuncAnimation(
            fig, 
            update, 
            frames=n_frames, 
            interval=interval, 
            blit=False)
    )