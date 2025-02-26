"""
ForecastGenerator class for generating probabilistic time series forecasts.

This class handles the generation of forecasts using a probabilistic transformer model,
including trend decomposition, scaling, and uncertainty estimation.
"""

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import tensorflow as tf
from scipy import stats
from scipy.stats import linregress
from statsmodels.tsa.seasonal import STL

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
    """

    def __init__(self, proba_model: tf.keras.Model, data_scaler, is_probabilistic: bool = False):
        """
        Initialize the ForecastGenerator.

        Args:
            proba_model: Trained transformer model
            data_scaler: Scaler object for data normalization
            is_probabilistic: Whether the model outputs mean and std or just point predictions
        """
        self.proba_model = proba_model
        self.data_scaler = data_scaler
        self.is_probabilistic = is_probabilistic

    def perform_stl_decomposition(self, series: pd.Series, period: int = 12) -> Optional[dict[str, Any]]:
        """
        Perform STL decomposition on the time series.

        Args:
            series: Input time series
            period: Seasonal period for decomposition

        Returns:
            Dictionary containing decomposition components or None if series is too short
        """
        if len(series) >= period:
            stl = STL(series, period=period)
            res = stl.fit()
            return {
                "trend": res.trend,
                "seasonal": res.seasonal,
                "resid": res.resid,
                "trend_strength": 1 - res.resid.var() / (res.trend + res.resid).var(),
                "seasonal_strength": 1 - res.resid.var() / (res.seasonal + res.resid).var(),
            }
        else:
            logger.warning(
                f"Not enough data points for STL decomposition. Minimum required: {period}, Got: {len(series)}",
            )
            return None

    def remove_trend(self, series: pd.Series) -> tuple:
        """
        Remove trend from the time series using STL decomposition or linear regression.

        Args:
            series: Input time series

        Returns:
            Tuple of (detrended series, trend parameters)
        """
        decomposition = self.perform_stl_decomposition(series)
        if decomposition:
            detrended = series - decomposition["trend"]
            trend_params = {
                "type": "stl",
                "trend": decomposition["trend"],
                "trend_strength": decomposition["trend_strength"],
                "seasonal_strength": decomposition["seasonal_strength"],
            }
            logger.info(
                f"STL decomposition performed. Trend strength: {trend_params['trend_strength']:.4f}, Seasonal strength: {trend_params['seasonal_strength']:.4f}",
            )
            return detrended, trend_params
        elif len(series) >= 5:
            logger.info("Using simple linear regression for trend removal.")
            return self.simple_trend_removal(series)
        else:
            logger.info("Not enough data points for trend removal. Using original series.")
            return series, {"type": "none"}

    def simple_trend_removal(self, series: pd.Series) -> tuple:
        """
        Remove trend using simple linear regression.

        Args:
            series: Input time series

        Returns:
            Tuple of (detrended series, trend parameters)
        """
        x = np.arange(len(series))
        slope, intercept, _, _, _ = linregress(x, series)
        trend = slope * x + intercept
        detrended = series - trend
        return detrended, {"type": "linear", "slope": slope, "intercept": intercept}

    def add_trend(self, series: pd.Series, trend_params: dict, future_index: pd.DatetimeIndex) -> pd.Series:
        """
        Add trend back to the forecasted series.

        Args:
            series: Input series
            trend_params: Trend parameters from remove_trend
            future_index: DatetimeIndex for future predictions

        Returns:
            Series with trend added back
        """
        if trend_params["type"] == "stl":
            last_trend = trend_params["trend"].iloc[-1]
            trend_diff = trend_params["trend"].diff().mean()
            future_trend = pd.Series(
                [last_trend + i * trend_diff for i in range(1, len(series) + 1)],
                index=future_index,
            )
        elif trend_params["type"] == "linear":
            future_trend = trend_params["slope"] * np.arange(len(series)) + trend_params["intercept"]
        else:
            return series
        return series + future_trend

    def generate_single_step_forecast(
        self,
        series_scaled_flatten: np.ndarray,
        input_series_length: int,
        step: int,
        num_samples: int,
        low_bound_conf: int,
        high_bound_conf: int,
    ) -> tuple:
        """
        Generate a single step forecast with uncertainty bounds.

        Args:
            series_scaled_flatten: Flattened scaled series
            input_series_length: Length of input sequence
            step: Current forecast step
            num_samples: Number of samples for uncertainty estimation
            low_bound_conf: Lower confidence bound percentile
            high_bound_conf: Upper confidence bound percentile

        Returns:
            Tuple of (point prediction, lower bound, upper bound)
        """
        series_reshaped = series_scaled_flatten[-input_series_length[0]:].reshape(1, input_series_length[0], 1)
        prediction = self.proba_model.predict(series_reshaped)

        if self.is_probabilistic:
            # Split into mean and standard deviation for probabilistic models
            mu, sigma = np.split(prediction, 2, axis=-1)
            mu = mu.flatten()[0]
            sigma = sigma.flatten()[0]

            # Increase uncertainty with forecast horizon
            scaling_increment = 0.1
            uncertainty_scaling_factor = 1 + (step * scaling_increment)
            scaled_sigma = sigma * uncertainty_scaling_factor

            # Generate samples from normal distribution using numpy
            samples = stats.norm.rvs(loc=mu, scale=scaled_sigma, size=num_samples)

            # Calculate point prediction and confidence bounds
            point_prediction = np.median(samples)  # 0.5 quantile
            lower_bound = np.percentile(samples, low_bound_conf)
            upper_bound = np.percentile(samples, high_bound_conf)
        else:
            # For point models, use the prediction directly
            point_prediction = prediction.flatten()[0]
            # Add artificial uncertainty that increases with horizon
            base_uncertainty = 0.1  # 10% of the prediction
            horizon_factor = 1 + (step * 0.05)  # 5% increase per step
            uncertainty = point_prediction * base_uncertainty * horizon_factor
            lower_bound = point_prediction - uncertainty
            upper_bound = point_prediction + uncertainty

        return point_prediction, lower_bound, upper_bound

    def process_time_series(
        self,
        group_df: pd.DataFrame,
        n: int,
        num_samples: int,
        input_series_length: int,
        low_bound_conf: int,
        high_bound_conf: int,
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

        Returns:
            DataFrame containing forecasts and confidence bounds
        """
        logger.info(f"Original data range: {group_df['y'].min()} to {group_df['y'].max()}")

        # 1. Remove trend
        detrended_series, trend_params = self.remove_trend(group_df["y"])
        logger.info(f"Detrended data range: {detrended_series.min()} to {detrended_series.max()}")

        # 2. Scale the detrended data
        series_scaled = self.data_scaler.scale(detrended_series.to_numpy().reshape(-1, 1)).flatten()
        logger.info(f"Scaled detrended data range: {series_scaled.min()} to {series_scaled.max()}")

        predictions = []
        lower_bound_predictions = []
        upper_bound_predictions = []
        series_scaled_flatten = series_scaled

        padding_length = max(0, input_series_length[0] - len(series_scaled_flatten))
        series_scaled_flatten = np.pad(series_scaled_flatten, (padding_length, 0), "constant")

        # 3. Predict
        for step in range(n):
            prediction, lower_bound, upper_bound = self.generate_single_step_forecast(
                series_scaled_flatten,
                input_series_length,
                step,
                num_samples,
                low_bound_conf,
                high_bound_conf,
            )
            predictions.append(prediction)
            lower_bound_predictions.append(lower_bound)
            upper_bound_predictions.append(upper_bound)
            series_scaled_flatten = np.append(series_scaled_flatten, prediction)

        logger.info(f"Predictions range (scaled): {min(predictions)} to {max(predictions)}")

        prediction_dates = pd.date_range(start=group_df.index[-1], periods=n + 1, freq="MS")[1:]

        # 4. Rescale back
        rescaled_predictions = self.data_scaler.inverse_scale(np.array(predictions).reshape(-1, 1)).flatten()
        rescaled_lower = self.data_scaler.inverse_scale(np.array(lower_bound_predictions).reshape(-1, 1)).flatten()
        rescaled_upper = self.data_scaler.inverse_scale(np.array(upper_bound_predictions).reshape(-1, 1)).flatten()

        logger.info(f"Predictions range (rescaled): {min(rescaled_predictions)} to {max(rescaled_predictions)}")

        # 5. Add trend back
        predictions_with_trend = self.add_trend(
            pd.Series(rescaled_predictions, index=prediction_dates),
            trend_params,
            prediction_dates,
        )
        lower_bound_with_trend = self.add_trend(
            pd.Series(rescaled_lower, index=prediction_dates),
            trend_params,
            prediction_dates,
        )
        upper_bound_with_trend = self.add_trend(
            pd.Series(rescaled_upper, index=prediction_dates),
            trend_params,
            prediction_dates,
        )

        logger.info(f"Final predictions range: {predictions_with_trend.min()} to {predictions_with_trend.max()}")

        predicted_series = pd.DataFrame(
            {
                "ds": prediction_dates,
                "q_0.3": lower_bound_with_trend,
                "q_0.5": predictions_with_trend,
                "q_0.7": upper_bound_with_trend,
                "unique_id": group_df["unique_id"].iloc[0],
            },
        )

        return predicted_series

    def generate_forecast(
        self,
        series_df: pd.DataFrame,
        n: int,
        num_samples: int,
        input_series_length: int,
        low_bound_conf: int,
        high_bound_conf: int,
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
                )
                combined_predictions.append(predicted_series)
                logger.info(f"Predictions for {unique_id}:\n{predicted_series}")
            except Exception as e:
                logger.error(f"Error processing time series {unique_id}: {str(e)}")
                continue

        if not combined_predictions:
            raise ValueError("No predictions were generated for any time series.")

        return pd.concat(combined_predictions).reset_index(drop=True)

    def plot_forecast(self, historical_df: pd.DataFrame, forecast_df: pd.DataFrame):
        """
        Plot historical data and forecast with confidence intervals.

        Args:
            historical_df: DataFrame containing historical data
            forecast_df: DataFrame containing forecast data
        """
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 6))

        # Plot historical data
        plt.plot(historical_df.index, historical_df["y"], label="Historical", color="blue")

        # Plot forecast
        plt.plot(forecast_df["ds"], forecast_df["q_0.5"], label="Forecast", color="red")

        # Plot confidence interval
        plt.fill_between(
            forecast_df["ds"],
            forecast_df["q_0.3"],
            forecast_df["q_0.7"],
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