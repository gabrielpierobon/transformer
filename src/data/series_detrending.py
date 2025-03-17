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
    
    The class automatically selects the best detrending method based on
    data characteristics and availability.
    """

    def __init__(self, min_points_stl: int = 12, min_points_linear: int = 5, remove_only_positive_trends: bool = False):
        """
        Initialize the SeriesDetrending class.

        Args:
            min_points_stl: Minimum points required for STL decomposition
            min_points_linear: Minimum points required for linear regression
            remove_only_positive_trends: If True, only remove positive trends and preserve negative ones
        """
        self.min_points_stl = min_points_stl
        self.min_points_linear = min_points_linear
        self.remove_only_positive_trends = remove_only_positive_trends

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

        Args:
            series: Input time series
            force_linear: Force using linear detrending even if STL is possible

        Returns:
            Tuple of (detrended series, trend parameters)
        """
        if not force_linear and len(series) >= self.min_points_stl:
            decomposition = self.perform_stl_decomposition(series)
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
                        return series, {"type": "none"}
                
                detrended = series - decomposition["trend"]
                trend_params = {
                    "type": "stl",
                    "trend": decomposition["trend"],
                    "trend_strength": decomposition["trend_strength"],
                    "seasonal_strength": decomposition["seasonal_strength"],
                }
                return detrended, trend_params

        if len(series) >= self.min_points_linear:
            return self.remove_linear_trend(series)
        
        logger.warning("Not enough data points for any detrending method. Using original series.")
        return series, {"type": "none"}

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

        Args:
            series: Input time series

        Returns:
            Float indicating trend strength (0 to 1, higher means stronger trend)
        """
        if len(series) >= self.min_points_stl:
            decomposition = self.perform_stl_decomposition(series)
            if decomposition:
                return decomposition["trend_strength"]
        
        # For shorter series, use R-squared from linear regression
        _, trend_params = self.remove_linear_trend(series)
        return trend_params["r_squared"] 