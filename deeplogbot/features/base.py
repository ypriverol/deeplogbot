"""Base classes for extensible feature extractors."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
import pandas as pd

if TYPE_CHECKING:
    from .schema import LogSchema


class BaseFeatureExtractor(ABC):
    """
    Base class for feature extractors.
    
    Subclasses should implement extract() to add custom features.
    Feature extractors can be chained together to build complex feature sets.
    
    Example:
        class MyExtractor(BaseFeatureExtractor):
            def extract(self, df: pd.DataFrame, input_parquet_path: str, conn) -> pd.DataFrame:
                # Add custom features
                df['my_feature'] = ...
                return df
    """
    
    def __init__(self, schema: "LogSchema"):
        """
        Initialize feature extractor with a schema.
        
        Args:
            schema: LogSchema defining field mappings for the log format
        """
        self.schema = schema
    
    @abstractmethod
    def extract(self, df: pd.DataFrame, input_parquet_path: str, conn) -> pd.DataFrame:
        """
        Extract features from the data.
        
        Args:
            df: DataFrame with basic location-level stats already computed.
                 This will have columns like: geo_location, country, city,
                 unique_users, total_downloads, downloads_per_user, etc.
            input_parquet_path: Path to the input parquet file (already escaped for SQL)
            conn: Database connection (DuckDB) for executing queries
            
        Returns:
            DataFrame with additional features added. Must preserve all existing
            columns and add new feature columns.
        """
        pass
    
    def get_name(self) -> str:
        """Get the name of this feature extractor."""
        return self.__class__.__name__
