"""
ForecastGenerator class for generating probabilistic time series forecasts.

This class handles the generation of forecasts using a probabilistic transformer model,
including trend decomposition, scaling, and uncertainty estimation.
"""

import logging
import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import tensorflow as tf
from scipy import stats

from src.data.series_detrending import SeriesDetrending

logger = logging.getLogger(__name__)


class ForecastGenerator:
    """
    A class for generating probabilistic forecasts using a transformer model.
    
    This class handles:
    - Time series decomposition (trend, seasonality)
    - Data scaling and inverse scaling
    - Probabilistic forecasting with uncertainty bounds
    - Trend adjustment in forecasts
    
    Attributes:
        proba_model (tf.keras.Model): The probabilistic transformer model
        data_scaler: Scaler object for data normalization
        is_probabilistic (bool): Whether the model is probabilistic
        detrending (SeriesDetrending): Handler for series detrending operations
        apply_detrending (bool): Whether to apply detrending during inference
    """

    def __init__(
        self, 
        proba_model: tf.keras.Model, 
        data_scaler, 
        is_probabilistic: bool = False,
        min_points_stl: int = 12,
        min_points_linear: int = 5,
        config_path: str = None
    ):
        """
        Initialize the ForecastGenerator.

        Args:
            proba_model: Trained transformer model
            data_scaler: Scaler object for data normalization
            is_probabilistic: Whether the model outputs mean and std or just point predictions
            min_points_stl: Minimum points required for STL decomposition
            min_points_linear: Minimum points required for linear regression
            config_path: Path to the inference configuration file
        """
        self.proba_model = proba_model
        self.data_scaler = data_scaler
        self.is_probabilistic = is_probabilistic
        
        # Load inference configuration
        self.config = self._load_inference_config(config_path)
        
        # Override min_points parameters if provided in config
        min_points_stl = self.config.get('min_points_stl', min_points_stl)
        min_points_linear = self.config.get('min_points_linear', min_points_linear)
        
        # Get detrending configuration
        self.apply_detrending = self.config.get('apply_detrending', True)
        logger.info(f"Detrending during inference is {'enabled' if self.apply_detrending else 'disabled'}")
        
        # Get positive-only trend removal configuration
        self.remove_only_positive_trends = self.config.get('remove_only_positive_trends', False)
        logger.info(f"Positive-only trend removal is {'enabled' if self.remove_only_positive_trends else 'disabled'}")
        
        # Get disable linear detrending configuration
        self.disable_linear_detrending = self.config.get('disable_linear_detrending', False)
        logger.info(f"Linear detrending is {'disabled' if self.disable_linear_detrending else 'enabled'}")
        
        self.detrending = SeriesDetrending(
            min_points_stl=min_points_stl,
            min_points_linear=min_points_linear,
            remove_only_positive_trends=self.remove_only_positive_trends,
            disable_linear_detrending=self.disable_linear_detrending
        )

    def _load_inference_config(self, config_path: str = None) -> Dict:
        """
        Load inference configuration from YAML file.
        
        Args:
            config_path: Path to the inference configuration file
            
        Returns:
            Dictionary containing configuration parameters
        """
        # Default configuration
        default_config = {
            'apply_detrending': True,
            'min_points_stl': 12,
            'min_points_linear': 5,
            'remove_only_positive_trends': False,
            'disable_linear_detrending': False,
            'default_low_bound_conf': 30,
            'default_high_bound_conf': 70,
            'default_num_samples': 1000
        }
        
        # If no config path provided, look in standard location
        if config_path is None:
            # Try to find config in the standard location
            project_root = Path(__file__).resolve().parents[2]
            config_path = os.path.join(project_root, 'config', 'inference_config.yaml')
        
        # Load configuration if file exists
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as file:
                    config = yaml.safe_load(file)
                logger.info(f"Loaded inference configuration from {config_path}")
                return config
            except Exception as e:
                logger.warning(f"Error loading inference config: {str(e)}. Using default configuration.")
                return default_config
        else:
            logger.warning(f"Inference config file not found at {config_path}. Using default configuration.")
            return default_config

    def generate_single_step_forecast(
        self,
        series_scaled_flatten: np.ndarray,
        input_series_length: List[int],
        forecast_steps: int = 1,
        confidence_intervals: Optional[List[float]] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Generate forecasts with uncertainty bounds.

        Args:
            series_scaled_flatten: Flattened scaled series
            input_series_length: List containing required input sequence length
            forecast_steps: Number of steps to forecast
            confidence_intervals: List of confidence interval percentiles

        Returns:
            Dictionary containing mean predictions and confidence bounds
        """
        # Initialize arrays for predictions
        predictions = np.zeros(forecast_steps)
        lower_bounds = np.zeros(forecast_steps) if confidence_intervals else None
        upper_bounds = np.zeros(forecast_steps) if confidence_intervals else None
        
        # Make a copy of the input series for iterative prediction
        current_series = series_scaled_flatten.copy()
        
        # Generate predictions iteratively
        for step in range(forecast_steps):
            # Ensure input is properly shaped
            if len(current_series) < input_series_length[0]:
                raise ValueError(f"Input series length {len(current_series)} is less than required {input_series_length[0]}")
                
            series_reshaped = current_series[-input_series_length[0]:].reshape(1, input_series_length[0], 1)
            prediction = self.proba_model.predict(series_reshaped, verbose=0)  # Reduce TF logging noise

            if prediction is None or np.isnan(prediction).any():
                logger.error(f"Model returned None or NaN prediction at step {step}")
                raise ValueError("Invalid prediction from model")

            if self.is_probabilistic:
                # Split into mean and standard deviation for probabilistic models
                mu, sigma = np.split(prediction, 2, axis=-1)
                point_pred = mu.flatten()[0]
                sigma_val = sigma.flatten()[0]

                # Increase uncertainty with forecast horizon
                scaling_increment = 0.1
                uncertainty_scaling = 1 + (step * scaling_increment)
                scaled_sigma = sigma_val * uncertainty_scaling

                predictions[step] = point_pred
                if confidence_intervals:
                    for ci in confidence_intervals:
                        z_score = stats.norm.ppf(ci / 100)
                        if ci < 50:
                            lower_bounds[step] = point_pred - z_score * scaled_sigma
                        else:
                            upper_bounds[step] = point_pred + z_score * scaled_sigma
            else:
                # For point models, use the prediction directly
                point_pred = prediction.flatten()[0]
                predictions[step] = point_pred
                
                if confidence_intervals:
                    # Add artificial uncertainty that increases with horizon
                    base_uncertainty = 0.1  # 10% of the prediction
                    horizon_factor = 1 + (step * 0.05)  # 5% increase per step
                    uncertainty = abs(point_pred) * base_uncertainty * horizon_factor
                    
                    for ci in confidence_intervals:
                        if ci < 50:
                            lower_bounds[step] = point_pred - uncertainty
                        else:
                            upper_bounds[step] = point_pred + uncertainty

            # Update series for next prediction
            current_series = np.append(current_series[1:], point_pred)

        # Validate outputs
        if np.isnan(predictions).any():
            logger.error("NaN values in predictions")
            raise ValueError("NaN values in predictions")

        # Package results
        result = {"mean": predictions}
        if confidence_intervals:
            if any(ci < 50 for ci in confidence_intervals):
                result["lower"] = lower_bounds
            if any(ci > 50 for ci in confidence_intervals):
                result["upper"] = upper_bounds

        logger.debug(
            f"Predictions range: [{result['mean'].min():.4f}, {result['mean'].max():.4f}]"
        )

        return result

    def process_time_series(
        self,
        group_df: pd.DataFrame,
        n: int,
        num_samples: int,
        input_series_length: List[int],
        low_bound_conf: int,
        high_bound_conf: int,
        force_linear_detrend: bool = False
    ) -> pd.DataFrame:
        """
        Process a single time series to generate forecasts.

        Args:
            group_df: DataFrame containing the time series
            n: Number of steps to forecast
            num_samples: Number of samples for uncertainty estimation
            input_series_length: Length of input sequence
            low_bound_conf: Lower confidence bound percentile
            high_bound_conf: Upper confidence bound percentile
            force_linear_detrend: Force using linear detrending even if STL is possible

        Returns:
            DataFrame containing forecasts and confidence bounds
        """
        logger.info(f"Original data range: {group_df['y'].min()} to {group_df['y'].max()}")

        # Get the original series values
        original_series = group_df["y"].values
        
        if self.apply_detrending:
            # Get trend strength before detrending
            trend_strength = self.detrending.get_trend_strength(group_df["y"])
            logger.info(f"Detected trend strength: {trend_strength:.4f}")

            # 1. Remove trend
            detrended_series, trend_params = self.detrending.remove_trend(
                group_df["y"],
                force_linear=force_linear_detrend
            )
            logger.info(
                f"Detrending method: {trend_params['type']}, "
                f"Data range after detrending: [{detrended_series.min():.4f}, {detrended_series.max():.4f}]"
            )
            
            # Use detrended series for scaling
            series_to_scale = detrended_series.values
        else:
            # Skip detrending, use original series
            logger.info("Detrending is disabled, using original series")
            series_to_scale = original_series
            # Create dummy trend params for later
            trend_params = {'type': 'none'}
        
        # 2. Scale the data (detrended or original)
        series_scaled = self.data_scaler.scale(series_to_scale.reshape(-1, 1)).flatten()
        logger.info(f"Scaled data range: [{series_scaled.min():.4f}, {series_scaled.max():.4f}]")
        
        # Add padding if needed
        padding_length = max(0, input_series_length[0] - len(series_scaled))
        if padding_length > 0:
            logger.info(f"Padding input sequence with {padding_length} zeros")
            series_scaled = np.pad(series_scaled, (padding_length, 0), "constant")

        # Generate predictions
        predictions = self.generate_single_step_forecast(
            series_scaled,
            input_series_length,
            n,
            [low_bound_conf, high_bound_conf]
        )
        
        # Create future dates for predictions
        last_date = group_df.index[-1]
        future_dates = pd.date_range(start=last_date, periods=n + 1, freq="MS")[1:]

        # Inverse scale predictions and bounds
        predictions_unscaled = pd.Series(
            self.data_scaler.inverse_scale(predictions["mean"].reshape(-1, 1)).flatten(),
            index=future_dates
        )
        logger.debug(f"Unscaled predictions range: [{predictions_unscaled.min():.4f}, {predictions_unscaled.max():.4f}]")
        
        # Create forecast DataFrame
        forecast_df = pd.DataFrame({
            "ds": future_dates,
            "q_0.5": predictions_unscaled
        })
        
        # Handle confidence intervals
        if "lower" in predictions and "upper" in predictions:
            lower_unscaled = pd.Series(
                self.data_scaler.inverse_scale(predictions["lower"].reshape(-1, 1)).flatten(),
                index=future_dates
            )
            upper_unscaled = pd.Series(
                self.data_scaler.inverse_scale(predictions["upper"].reshape(-1, 1)).flatten(),
                index=future_dates
            )
            
            if self.apply_detrending:
                # Add trend back if detrending was applied
                forecast_df["q_0.5"] = self.detrending.add_trend(predictions_unscaled, trend_params, future_dates)
                forecast_df[f"q_0.{low_bound_conf}"] = self.detrending.add_trend(lower_unscaled, trend_params, future_dates)
                forecast_df[f"q_0.{high_bound_conf}"] = self.detrending.add_trend(upper_unscaled, trend_params, future_dates)
            else:
                # Use unscaled values directly if no detrending
                forecast_df["q_0.5"] = predictions_unscaled
                forecast_df[f"q_0.{low_bound_conf}"] = lower_unscaled
                forecast_df[f"q_0.{high_bound_conf}"] = upper_unscaled
        else:
            if self.apply_detrending:
                # Add trend back to point predictions if detrending was applied
                forecast_df["q_0.5"] = self.detrending.add_trend(predictions_unscaled, trend_params, future_dates)
            else:
                # Use unscaled values directly if no detrending
                forecast_df["q_0.5"] = predictions_unscaled
            
        # Add unique_id if present in original data
        if "unique_id" in group_df.columns:
            forecast_df["unique_id"] = group_df["unique_id"].iloc[0]
            
        # Validate final output
        if forecast_df["q_0.5"].isna().any():
            logger.error("NaN values found in final predictions")
            return pd.DataFrame()

        logger.info(f"Final predictions range: [{forecast_df['q_0.5'].min():.4f}, {forecast_df['q_0.5'].max():.4f}]")
        return forecast_df

    def generate_forecast(
        self,
        series_df: pd.DataFrame,
        n: int,
        num_samples: int,
        input_series_length: int,
        low_bound_conf: int,
        high_bound_conf: int,
        force_linear_detrend: bool = False,
    ) -> pd.DataFrame:
        """
        Generate forecasts for multiple time series.

        Args:
            series_df: DataFrame containing multiple time series
            n: Number of steps to forecast
            num_samples: Number of samples for uncertainty estimation
            input_series_length: Length of input sequence
            low_bound_conf: Lower confidence bound percentile
            high_bound_conf: Upper confidence bound percentile
            force_linear_detrend: Force using linear detrending even if STL is possible

        Returns:
            DataFrame containing forecasts for all series
        """
        combined_predictions = []
        for unique_id, group_df in series_df.groupby("unique_id"):
            logger.info(f"Transformer generating predictions for time-series: {unique_id}")
            try:
                predicted_series = self.process_time_series(
                    group_df,
                    n,
                    num_samples,
                    input_series_length,
                    low_bound_conf,
                    high_bound_conf,
                    force_linear_detrend
                )
                combined_predictions.append(predicted_series)
                logger.info(f"Predictions for {unique_id}:\n{predicted_series}")
            except Exception as e:
                logger.error(f"Error processing time series {unique_id}: {str(e)}")
                continue

        if not combined_predictions:
            raise ValueError("No predictions were generated for any time series.")

        return pd.concat(combined_predictions).reset_index(drop=True)

    def plot_forecast(
        self, 
        historical_df: pd.DataFrame, 
        forecast_df: pd.DataFrame,
        low_bound_conf: int = 30,
        high_bound_conf: int = 70
    ):
        """
        Plot historical data and forecast with confidence intervals.

        Args:
            historical_df: DataFrame containing historical data
            forecast_df: DataFrame containing forecast data
            low_bound_conf: Lower confidence bound percentile (default: 30)
            high_bound_conf: Upper confidence bound percentile (default: 70)
        """
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 6))

        # Plot historical data
        plt.plot(historical_df.index, historical_df["y"], label="Historical", color="blue")

        # Plot forecast
        plt.plot(forecast_df["ds"], forecast_df["q_0.5"], label="Forecast", color="red")

        # Plot confidence interval if available
        conf_cols = [f"q_0.{low_bound_conf}", f"q_0.{high_bound_conf}"]
        if all(col in forecast_df.columns for col in conf_cols):
            plt.fill_between(
                forecast_df["ds"],
                forecast_df[f"q_0.{low_bound_conf}"],
                forecast_df[f"q_0.{high_bound_conf}"],
                color="red",
                alpha=0.2,
                label="Confidence Interval",
            )

        plt.title("Time Series Forecast")
        plt.xlabel("Date")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)
        plt.show() 