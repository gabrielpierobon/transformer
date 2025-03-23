"""
Series detrending utilities for time series data.

This module provides functionality for detrending time series data using
both STL decomposition and linear regression methods.
"""

import logging
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.stats import linregress
from statsmodels.tsa.seasonal import STL

logger = logging.getLogger(__name__)


class SeriesDetrending:
    """
    A class for handling time series detrending operations.
    
    This class provides methods for:
    - STL decomposition (trend, seasonality, residuals)
    - Linear trend removal
    - Trend strength assessment
    - Trend reapplication
    - Zero padding detection and handling
    
    The class automatically selects the best detrending method based on
    data characteristics and availability.
    """

    def __init__(
        self, 
        min_points_stl: int = 12, 
        min_points_linear: int = 5, 
        remove_only_positive_trends: bool = False,
        disable_linear_detrending: bool = False
    ):
        """
        Initialize the SeriesDetrending class.

        Args:
            min_points_stl: Minimum points required for STL decomposition
            min_points_linear: Minimum points required for linear regression
            remove_only_positive_trends: If True, only remove positive trends and preserve negative ones
            disable_linear_detrending: If True, linear detrending will be skipped entirely
        """
        self.min_points_stl = min_points_stl
        self.min_points_linear = min_points_linear
        self.remove_only_positive_trends = remove_only_positive_trends
        self.disable_linear_detrending = disable_linear_detrending
        
        if self.disable_linear_detrending:
            logger.info("Linear detrending is disabled by configuration.")

    def detect_zero_padding(self, series: pd.Series, threshold: float = 1e-6) -> Tuple[pd.Series, Dict]:
        """
        Detect and extract zero-padded prefixes from a time series.
        
        This is used to handle time series that have been artificially padded
        with zeros to meet minimum length requirements.
        
        Args:
            series: Input time series
            threshold: Values below this threshold are considered zero padding
            
        Returns:
            Tuple of (non-padded series, padding info dictionary)
        """
        if len(series) == 0:
            return series, {"padding_length": 0, "has_padding": False}
            
        # Find the first non-zero value (considering the threshold)
        non_zero_indices = np.where(np.abs(series.values) > threshold)[0]
        
        if len(non_zero_indices) == 0:
            # All zeros case
            logger.warning("Series contains only zeros or near-zero values.")
            return series, {"padding_length": 0, "has_padding": False}
            
        first_non_zero_idx = non_zero_indices[0]
        
        if first_non_zero_idx == 0:
            # No padding detected
            return series, {"padding_length": 0, "has_padding": False}
            
        # Extract the non-padded portion
        non_padded_series = series.iloc[first_non_zero_idx:].copy()
        
        padding_info = {
            "padding_length": first_non_zero_idx,
            "has_padding": True,
            "original_length": len(series),
            "padding_indices": series.index[:first_non_zero_idx],
            "non_padding_indices": series.index[first_non_zero_idx:]
        }
        
        logger.info(f"Detected {first_non_zero_idx} zero-padded points at the beginning of the series.")
        return non_padded_series, padding_info

    def recombine_with_padding(
        self, 
        processed_series: pd.Series, 
        padding_info: Dict
    ) -> pd.Series:
        """
        Recombine a processed series with its original zero padding.
        
        Args:
            processed_series: The processed time series (after detrending, etc.)
            padding_info: Padding information from detect_zero_padding
            
        Returns:
            Series with padding reapplied
        """
        if not padding_info.get("has_padding", False):
            return processed_series
            
        # Create a series for the padding
        padding = pd.Series(
            [0] * padding_info["padding_length"],
            index=padding_info["padding_indices"]
        )
        
        # Combine padding with processed series
        result = pd.concat([padding, processed_series])
        
        return result

    def perform_stl_decomposition(
        self, series: pd.Series, period: int = 12
    ) -> Optional[Dict[str, Union[pd.Series, float]]]:
        """
        Perform STL decomposition on the time series.

        Args:
            series: Input time series
            period: Seasonal period for decomposition (default is 12 for monthly data)

        Returns:
            Dictionary containing decomposition components or None if series is too short
        """
        if len(series) >= self.min_points_stl:
            try:
                stl = STL(series, period=period)
                res = stl.fit()
                decomposition = {
                    "trend": res.trend,
                    "seasonal": res.seasonal,
                    "resid": res.resid,
                    "trend_strength": 1 - res.resid.var() / (res.trend + res.resid).var(),
                    "seasonal_strength": 1 - res.resid.var() / (res.seasonal + res.resid).var(),
                }
                logger.info(
                    f"STL decomposition successful. "
                    f"Trend strength: {decomposition['trend_strength']:.4f}, "
                    f"Seasonal strength: {decomposition['seasonal_strength']:.4f}"
                )
                return decomposition
            except Exception as e:
                logger.warning(f"STL decomposition failed: {str(e)}")
                return None
        else:
            logger.warning(
                f"Not enough data points for STL decomposition. "
                f"Minimum required: {self.min_points_stl}, Got: {len(series)}"
            )
            return None

    def remove_linear_trend(self, series: pd.Series) -> Tuple[pd.Series, Dict[str, Union[str, float]]]:
        """
        Remove trend using simple linear regression.

        Args:
            series: Input time series

        Returns:
            Tuple of (detrended series, trend parameters)
        """
        x = np.arange(len(series))
        slope, intercept, r_value, _, _ = linregress(x, series)
        
        # Check if we should skip negative trend removal
        if self.remove_only_positive_trends and slope < 0:
            logger.info(f"Skipping negative linear trend removal (slope: {slope:.4f})")
            return series, {"type": "none"}
            
        trend = slope * x + intercept
        detrended = series - trend
        trend_params = {
            "type": "linear",
            "slope": slope,
            "intercept": intercept,
            "r_squared": r_value ** 2
        }
        logger.info(f"Linear detrending completed. R-squared: {trend_params['r_squared']:.4f}")
        return pd.Series(detrended, index=series.index), trend_params

    def remove_trend(self, series: pd.Series, force_linear: bool = False) -> Tuple[pd.Series, Dict]:
        """
        Remove trend from the time series using the best available method.
        
        Handles zero-padded series by detecting and removing padding before
        detrending, then recombining after detrending.

        Args:
            series: Input time series
            force_linear: Force using linear detrending even if STL is possible

        Returns:
            Tuple of (detrended series, trend parameters)
        """
        # First, detect and handle any zero padding
        non_padded_series, padding_info = self.detect_zero_padding(series)
        
        # If the series is too short after removing padding, return unchanged
        if len(non_padded_series) < self.min_points_linear:
            logger.warning(
                f"Series too short after removing padding. "
                f"Non-padded length: {len(non_padded_series)}, "
                f"Min required: {self.min_points_linear}."
            )
            return series, {"type": "none", "padding_info": padding_info}
            
        # Perform detrending on the non-padded portion
        if not force_linear and len(non_padded_series) >= self.min_points_stl:
            decomposition = self.perform_stl_decomposition(non_padded_series)
            if decomposition:
                # Check if we should skip negative trend removal
                if self.remove_only_positive_trends:
                    # Determine if the trend is negative by checking if the end is lower than the start
                    # or by calculating the overall slope
                    trend = decomposition["trend"]
                    trend_start = trend.iloc[:min(12, len(trend))].mean()
                    trend_end = trend.iloc[-min(12, len(trend)):].mean()
                    trend_direction = trend_end - trend_start
                    
                    if trend_direction < 0:
                        logger.info(f"Skipping negative STL trend removal (trend direction: {trend_direction:.4f})")
                        
                        # Recombine with padding and return
                        if padding_info.get("has_padding", False):
                            return self.recombine_with_padding(non_padded_series, padding_info), {"type": "none", "padding_info": padding_info}
                        return series, {"type": "none"}
                
                detrended = non_padded_series - decomposition["trend"]
                trend_params = {
                    "type": "stl",
                    "trend": decomposition["trend"],
                    "trend_strength": decomposition["trend_strength"],
                    "seasonal_strength": decomposition["seasonal_strength"],
                    "padding_info": padding_info
                }
                
                # Recombine with padding if needed
                if padding_info.get("has_padding", False):
                    detrended = self.recombine_with_padding(detrended, padding_info)
                
                return detrended, trend_params

        # Check if linear detrending is disabled
        if self.disable_linear_detrending:
            logger.info("Skipping linear detrending as it is disabled in configuration")
            
            # Recombine with padding if needed and return original series
            if padding_info.get("has_padding", False):
                return self.recombine_with_padding(non_padded_series, padding_info), {"type": "none", "padding_info": padding_info}
            return series, {"type": "none", "padding_info": padding_info}

        # Fallback to linear detrending if enabled and series is long enough
        if len(non_padded_series) >= self.min_points_linear:
            detrended, trend_params = self.remove_linear_trend(non_padded_series)
            
            # Add padding info to trend params
            trend_params["padding_info"] = padding_info
            
            # Recombine with padding if needed
            if padding_info.get("has_padding", False):
                detrended = self.recombine_with_padding(detrended, padding_info)
            
            return detrended, trend_params
        
        logger.warning("Not enough data points for any detrending method. Using original series.")
        return series, {"type": "none", "padding_info": padding_info}

    def add_trend(
        self, 
        series: pd.Series, 
        trend_params: Dict, 
        future_index: pd.DatetimeIndex
    ) -> pd.Series:
        """
        Add trend back to the detrended series.

        Args:
            series: Detrended series
            trend_params: Trend parameters from remove_trend
            future_index: DatetimeIndex for future predictions

        Returns:
            Series with trend added back
        """
        if not isinstance(series, pd.Series):
            series = pd.Series(series, index=future_index)
            
        if trend_params["type"] == "stl":
            if "trend" not in trend_params:
                logger.error("STL trend parameters missing trend component")
                return series
                
            # Calculate trend continuation based on last n points
            n_points = min(60, len(trend_params["trend"])) #TODO it was 12 before
            last_trends = trend_params["trend"].iloc[-n_points:]
            trend_diff = last_trends.diff().mean()
            
            # Start from the last known trend value
            last_trend = trend_params["trend"].iloc[-1]
            
            # Generate future trend values
            future_trend = pd.Series(
                [last_trend + (i + 1) * trend_diff for i in range(len(future_index))],
                index=future_index
            )
            
            result = series + future_trend
            logger.debug(f"Added STL trend. Range: [{result.min():.4f}, {result.max():.4f}]")
            return result

        elif trend_params["type"] == "linear":
            slope = trend_params["slope"]
            intercept = trend_params["intercept"]
            
            # For linear trend, calculate x values continuing from training data
            if "original_index" in trend_params:
                start_idx = len(trend_params["original_index"])
            else:
                # Consider padding offset if present
                if "padding_info" in trend_params and trend_params["padding_info"].get("has_padding", False):
                    # We continue from the non-padded portion length, not the full series
                    start_idx = len(trend_params["padding_info"]["non_padding_indices"])
                else:
                    start_idx = 0
                
            x_values = np.arange(start_idx, start_idx + len(future_index))
            future_trend = pd.Series(
                slope * x_values + intercept,
                index=future_index
            )
            
            result = series + future_trend
            logger.debug(f"Added linear trend. Range: [{result.min():.4f}, {result.max():.4f}]")
            return result

        logger.warning("No trend parameters found, returning original series")
        return series

    def get_trend_strength(self, series: pd.Series) -> float:
        """
        Calculate the strength of the trend in the series.
        
        Handles zero-padded series by detecting and removing padding
        before calculating trend strength.

        Args:
            series: Input time series

        Returns:
            Float indicating trend strength (0 to 1, higher means stronger trend)
        """
        # First, detect and handle any zero padding
        non_padded_series, _ = self.detect_zero_padding(series)
        
        if len(non_padded_series) >= self.min_points_stl:
            decomposition = self.perform_stl_decomposition(non_padded_series)
            if decomposition:
                return decomposition["trend_strength"]
        
        # For shorter series, use R-squared from linear regression if linear detrending is enabled
        if not self.disable_linear_detrending:
            _, trend_params = self.remove_linear_trend(non_padded_series)
            return trend_params["r_squared"]
        
        # If linear detrending is disabled, return a low trend strength value
        logger.info("Linear detrending disabled; returning minimal trend strength")
        return 0.0 