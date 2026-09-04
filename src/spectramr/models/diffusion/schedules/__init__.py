"""Diffusion forward-process schedules.

Houses non-standard forward processes that do not fit the plain
beta-schedule DDPM mould — e.g. the heat-equation (DCT) blurring
schedule of Hoogeboom & Salimans (ICLR 2023).
"""

from spectramr.models.diffusion.schedules.blurring_schedule import BlurringSchedule

__all__ = ["BlurringSchedule"]
