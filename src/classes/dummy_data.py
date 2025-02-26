"""
DummyDataframeInitializer class for generating test time series data.

This module provides functionality to create synthetic time series data
for testing and demonstration purposes.
"""

from typing import Dict, List, Optional
import logging

import pandas as pd

logger = logging.getLogger(__name__)


class DummyDataframeInitializer:
    """
    A class to initialize a DataFrame for time series analysis using dummy data.

    This class abstracts the process of generating dummy time series data and combines multiple time series
    into a single DataFrame, based on provided configuration details.

    Example:
        ```python
        # Configuration dictionaries
        array_config = {
            'series1': [1, 2, 3, 4, 5],
            'series2': [10, 20, 30, 40, 50]
        }
        forecast_config = {
            'input_series_length': 5
        }

        # Initialize and generate data
        initializer = DummyDataframeInitializer(array_config, forecast_config)
        df = initializer.initialize_series_df()
        ```

    Attributes:
        array_config (dict): Configuration for dummy data sequences.
        forecast_config (dict): Forecast configuration details including input series length.
        monthly_dates (pd.DatetimeIndex): Generated date range for the time series.
    """

    def __init__(self, array_config: Dict[str, List[float]], forecast_config: Dict[str, int]):
        """
        Initializes the DummyDataframeInitializer with array_config and forecast_config.

        Args:
            array_config: Configuration for the dummy data sequences.
                        Dictionary with series names as keys and lists of values.
            forecast_config: Forecast configuration, including input series length.
                           Must contain 'input_series_length' key.

        Raises:
            ValueError: If configurations are invalid or missing required keys.
        """
        # Validate configurations
        if not array_config:
            raise ValueError("array_config cannot be empty")
        
        if 'input_series_length' not in forecast_config:
            raise ValueError("forecast_config must contain 'input_series_length' key")
        
        # Validate that all series have the same length as input_series_length
        expected_length = forecast_config['input_series_length']
        for series_name, values in array_config.items():
            if len(values) != expected_length:
                raise ValueError(
                    f"Series '{series_name}' length ({len(values)}) does not match "
                    f"input_series_length ({expected_length})"
                )

        self.array_config = array_config
        self.forecast_config = forecast_config
        self.monthly_dates = None
        
        logger.info(
            f"Initialized DummyDataframeInitializer with {len(array_config)} series "
            f"of length {expected_length}"
        )

    def initialize_series_df(self) -> pd.DataFrame:
        """
        Initializes and returns the combined series DataFrame based on the provided configurations.

        Returns:
            pandas.DataFrame: Combined DataFrame with all time series data.
                            Index: DatetimeIndex named 'ds'
                            Columns: ['y', 'unique_id']

        Raises:
            ValueError: If date range hasn't been created yet.
        """
        logger.info("Initializing combined series DataFrame")
        self.create_date_range(periods=self.forecast_config["input_series_length"])
        return self.generate_combined_series()

    def create_date_range(self, periods: int, start: str = "2019-01-01") -> None:
        """
        Create a monthly date range starting from a given date.

        Args:
            periods: Number of periods (months) for the date range.
            start: Start date for the date range in 'YYYY-MM-DD' format.
                  Defaults to "2019-01-01".
        """
        logger.debug(f"Creating date range: {periods} periods starting from {start}")
        self.monthly_dates = pd.date_range(start=start, periods=periods, freq="MS")

    def create_time_series_df(self, sequence_key: str, unique_id: str) -> pd.DataFrame:
        """
        Create a DataFrame for a single time series sequence.

        Args:
            sequence_key: Key to retrieve time series data from the array_config.
            unique_id: Unique identifier for the time series.

        Returns:
            pandas.DataFrame: DataFrame with the time series data and a unique identifier.
                            Index: DatetimeIndex named 'ds'
                            Columns: ['y', 'unique_id']
        """
        logger.debug(f"Creating time series DataFrame for {unique_id}")
        series_df = pd.DataFrame(self.array_config[sequence_key], columns=["y"])
        series_df["unique_id"] = unique_id
        series_df.index = self.monthly_dates
        series_df.index.name = "ds"
        return series_df

    def generate_combined_series(self) -> pd.DataFrame:
        """
        Generate a combined DataFrame with all time series.

        Returns:
            pandas.DataFrame: Combined DataFrame with all time series.
                            Index: DatetimeIndex named 'ds'
                            Columns: ['y', 'unique_id']

        Raises:
            ValueError: If date range hasn't been created yet.
        """
        # Check if self.monthly_dates has been initialized and is not empty
        if self.monthly_dates is None or not len(self.monthly_dates):
            raise ValueError("Date range not set. Please call create_date_range() method first.")

        logger.info("Generating combined series DataFrame")
        
        # Create a DataFrame for each time series
        single_series_dfs = [
            self.create_time_series_df(key, f"id_{idx + 1}")
            for idx, key in enumerate(self.array_config.keys())
        ]

        # Concatenate all DataFrames into one
        combined_series_df = pd.concat(single_series_dfs)
        
        logger.info(
            f"Generated combined DataFrame with {len(self.array_config)} series "
            f"and {len(self.monthly_dates)} time points"
        )
        
        return combined_series_df 