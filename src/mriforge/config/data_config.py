"""
Data Configuration Entry Point.
This module bridges the Schema layer (Pydantic definitions) to the Application layer.
It aliases the schema classes so the rest of the application imports from here.
"""

from mriforge.config.schemas.data import DataConfigSchema as DataConfig
from mriforge.config.schemas.data import DatasetSourceSchema as DataSourceConfig

__all__ = ["DataConfig", "DataSourceConfig"]
