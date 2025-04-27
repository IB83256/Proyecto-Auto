# -*- coding: utf-8 -*-
"""
Noise schedules package initialization.

Author: Álvaro Duro y Carlos Beti
Date: [Fecha]
"""

from .linear import linear_beta_schedule
from .cosine import cosine_beta_schedule

__all__ = [
    "linear_beta_schedule",
    "cosine_beta_schedule",
]