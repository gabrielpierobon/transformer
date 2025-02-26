import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import traceback
from typing import Any, Dict, Optional
from scipy.stats import linregress
from statsmodels.tsa.seasonal import STL

logger = logging.getLogger(__name__)

class ForecastGenerator:
    def __init__(self, model: tf.keras.Model, data_scaler):
        self.model = model
        self.data_scaler = data_scaler
        # Determine if model is probabilistic based on output shape
        self.is_probabilistic = model.output_shape[-1] == 2

    def perform_stl_decomposition(self, series: pd.Series) -> dict:
        try:
            # Get non-zero values for STL
            non_zero_mask = series != 0
            non_zero_series = series[non_zero_mask]
            
            # Perform STL decomposition with robust=True for better trend estimation
            stl = STL(non_zero_series, period=12, robust=True)
            result = stl.fit()
            
            # Calculate strengths
            total_variance = np.var(non_zero_series)
            detrended_variance = np.var(result.seasonal + result.resid)
            deseasonalized_variance = np.var(result.trend + result.resid)
            
            # Calculate strengths without forcing minimum/maximum
            trend_strength = 1 - detrended_variance / total_variance
            seasonal_strength = 1 - deseasonalized_variance / total_variance
            
            # Create full series with zeros where original had zeros
            full_trend = pd.Series(0, index=series.index, dtype=float)
            full_seasonal = pd.Series(0, index=series.index, dtype=float)
            
            # Fill in the decomposed values
            full_trend[non_zero_mask] = result.trend
            full_seasonal[non_zero_mask] = result.seasonal
            
            # Get last year's seasonal pattern
            last_seasonal = np.zeros(12)
            if len(result.seasonal) >= 12:
                last_seasonal = result.seasonal[-12:].values
            
            logger.info(f"STL Decomposition Stats:")
            logger.info(f"  - Trend range: {result.trend.min():.4f} to {result.trend.max():.4f}")
            logger.info(f"  - Seasonal range: {result.seasonal.min():.4f} to {result.seasonal.max():.4f}")
            logger.info(f"  - Residual range: {result.resid.min():.4f} to {result.resid.max():.4f}")
            logger.info(f"  - Trend strength: {trend_strength:.4f}")
            logger.info(f"  - Seasonal strength: {seasonal_strength:.4f}")
            logger.info(f"  - Total variance: {total_variance:.4f}")
            
            return {
                "trend": full_trend,
                "seasonal": full_seasonal,
                "trend_strength": trend_strength,
                "seasonal_strength": seasonal_strength,
                "last_seasonal_values": last_seasonal
            }
        except Exception as e:
            logger.error(f"STL decomposition failed: {str(e)}")
            logger.error(traceback.format_exc())
            return None

    def remove_trend(self, series: pd.Series) -> tuple:
        # Get non-zero values
        non_zero_mask = series != 0
        non_zero_series = series[non_zero_mask]
        
        # Minimum points required for each detrending method
        MIN_POINTS_STL = 12
        MIN_POINTS_LINEAR = 12  # Require at least 12 months for any detrending
        
        # Log the number of non-zero points
        logger.info(f"Number of non-zero data points: {len(non_zero_series)}")
        
        if len(non_zero_series) >= MIN_POINTS_STL:
            # Try STL first if we have enough points
            decomposition = self.perform_stl_decomposition(series)
            if decomposition:
                detrended = series - decomposition["trend"]
                trend_params = {
                    "type": "stl",
                    "trend": decomposition["trend"],
                    "seasonal": decomposition["seasonal"],  # Include seasonal component
                    "trend_strength": decomposition["trend_strength"],
                    "seasonal_strength": decomposition["seasonal_strength"],
                    "last_seasonal_values": decomposition["seasonal"][non_zero_mask].iloc[-12:].values  # Store last year's seasonal pattern
                }
                logger.info(
                    f"STL decomposition performed. Trend strength: {trend_params['trend_strength']:.4f}, "
                    f"Seasonal strength: {trend_params['seasonal_strength']:.4f}, "
                    f"Last seasonal range: {trend_params['last_seasonal_values'].min():.4f} to {trend_params['last_seasonal_values'].max():.4f}"
                )
                return detrended, trend_params
        
        if len(non_zero_series) >= MIN_POINTS_LINEAR:
            # Try linear regression if we have enough points but STL failed
            logger.info("Using simple linear regression for trend removal.")
            return self.simple_trend_removal(series)
        
        # If not enough points for either method, return original series
        logger.info(f"Not enough non-zero data points for detrending (minimum {MIN_POINTS_LINEAR} required). Using original series.")
        return series, {"type": "none"}

    def simple_trend_removal(self, series: pd.Series) -> tuple:
        # Get non-zero values and their indices
        non_zero_mask = series != 0
        non_zero_series = series[non_zero_mask]
        non_zero_indices = np.arange(len(series))[non_zero_mask]
        
        # Perform linear regression on non-zero values
        slope, intercept, _, _, _ = linregress(non_zero_indices, non_zero_series)
        
        # Calculate trend for all points
        all_indices = np.arange(len(series))
        trend = slope * all_indices + intercept
        
        # Set trend to zero where original series is zero
        trend[~non_zero_mask] = 0
        
        # Calculate detrended series
        detrended = series - trend
        
        return detrended, {"type": "linear", "slope": slope, "intercept": intercept}

    def add_trend(self, series: pd.Series, trend_params: dict, future_index: pd.DatetimeIndex) -> pd.Series:
        if trend_params["type"] == "stl":
            # Get the last non-zero trend value
            trend = trend_params["trend"]
            non_zero_mask = trend != 0
            last_trend = trend[non_zero_mask].iloc[-1]
            
            # Calculate trend difference using robust methods
            full_trend = trend[non_zero_mask]
            long_term_diff = np.median(np.diff(full_trend))
            recent_diffs = np.diff(full_trend[-12:]) if len(full_trend) >= 12 else np.diff(full_trend)
            short_term_diff = np.median(recent_diffs)
            
            # Log trend calculation details
            logger.info(f"Trend Calculation Details:")
            logger.info(f"  - Last trend value: {last_trend:.4f}")
            logger.info(f"  - Long-term median diff: {long_term_diff:.4f}")
            logger.info(f"  - Short-term median diff: {short_term_diff:.4f}")
            
            # Use trend strength to determine weights with stronger dampening
            trend_strength = trend_params.get('trend_strength', 0.5)
            trend_weight = min(0.6, max(0.2, trend_strength))  # Reduced maximum weight
            
            # Weighted combination of trends with additional dampening
            trend_diff = (trend_weight * long_term_diff + (1 - trend_weight) * short_term_diff) * 0.8
            
            # Calculate conservative growth limits
            historical_range = full_trend.max() - full_trend.min()
            max_allowed_trend = historical_range * 0.1  # Reduced to 10% of historical range
            
            # Get seasonal information with reduced impact
            last_seasonal_values = trend_params.get('last_seasonal_values', np.zeros(12))
            seasonal_strength = trend_params.get('seasonal_strength', 0) * 0.3  # Further reduced seasonal impact
            
            # Apply trend with stronger dampening
            future_trends = []
            current_value = last_trend
            cumulative_trend = 0
            
            for i in range(len(series)):
                # Stronger dampening factors
                horizon_dampening = 0.93 ** (1 + i/3)  # More aggressive dampening
                seasonal_idx = i % 12
                seasonal_adjustment = last_seasonal_values[seasonal_idx] * seasonal_strength
                
                base_step = trend_diff * horizon_dampening
                seasonal_factor = max(0.1, 1.0 - (i / len(series)))
                step_diff = base_step + (seasonal_adjustment * seasonal_factor * 0.2)
                
                if i > 6:  # Start stronger dampening earlier
                    step_diff *= 0.6
                
                # Stricter trend growth limit
                if abs(cumulative_trend + step_diff) > max_allowed_trend:
                    step_diff = 0  # Stop trend growth completely if limit reached
                
                current_value += step_diff
                cumulative_trend += step_diff
                future_trends.append(current_value)
            
            future_trend = pd.Series(future_trends, index=future_index)
            
        elif trend_params["type"] == "linear":
            slope = trend_params["slope"]
            intercept = trend_params["intercept"]
            last_idx = len(series) - 1
            
            # Generate future indices relative to the last point
            future_indices = np.arange(len(series)) + last_idx + 1
            
            # Apply stronger dampening to slope
            dampening_factors = 0.93 ** (np.arange(len(series)) / 6)  # More aggressive dampening
            dampened_slopes = slope * dampening_factors * 0.8  # Additional 20% reduction
            
            # Calculate trend values with dampened slope
            future_trend_values = np.zeros(len(series))
            current_value = slope * last_idx + intercept
            
            for i in range(len(series)):
                current_value += dampened_slopes[i]
                future_trend_values[i] = current_value
            
            future_trend = pd.Series(future_trend_values, index=future_index)
            
            logger.info(f"Linear Trend Continuation:")
            logger.info(f"  - Original slope: {slope:.4f}")
            logger.info(f"  - Final dampened slope: {dampened_slopes[-1]:.4f}")
            logger.info(f"  - Total change: {future_trend.iloc[-1] - future_trend.iloc[0]:.4f}")
        else:
            future_trend = pd.Series(0, index=future_index)
        
        return series + future_trend

    def generate_single_step_forecast(
        self,
        series_scaled_flatten: np.ndarray,
        input_series_length: int,
        n_non_zero: int
    ) -> tuple:
        # Prepare input for model
        series_reshaped = series_scaled_flatten[-input_series_length:].reshape(1, input_series_length, 1)
        
        # Get model prediction
        model_output = self.model.predict(series_reshaped, verbose=0)
        
        # Calculate historical statistics from non-zero values
        non_zero_values = series_scaled_flatten[series_scaled_flatten != 0]
        if len(non_zero_values) > 0:
            historical_mean = np.mean(non_zero_values)
            historical_std = np.std(non_zero_values) if len(non_zero_values) >= 2 else 0.1
        else:
            historical_mean = 0.5  # Default to middle of range
            historical_std = 0.1   # Conservative default
        
        # Handle probabilistic vs point prediction models
        if self.is_probabilistic:  # Probabilistic model
            mu = model_output[0, 0]
            sigma = np.abs(model_output[0, 1])  # Ensure positive
            
            # Clip mu to reasonable range based on historical data
            mu = np.clip(mu, historical_mean - 3 * historical_std, historical_mean + 3 * historical_std)
            
            # Scale uncertainty based on forecast horizon and data availability
            base_uncertainty = 0.2  # Base uncertainty level
            data_factor = max(0.5, min(1.0, n_non_zero / 12))  # Scale based on available data
            uncertainty_scaling_factor = base_uncertainty / data_factor
            
            scaled_sigma = sigma * uncertainty_scaling_factor
            
            # Generate samples for uncertainty estimation
            num_samples = 1000
            z_score = 1.96  # 95% confidence interval
            
            # Ensure bounds are within reasonable range
            lower_bound = np.clip(mu - (scaled_sigma * z_score), 0, 1)
            upper_bound = np.clip(mu + (scaled_sigma * z_score), 0, 1)
            point_prediction = mu
            
        else:  # Point prediction model
            point_prediction = model_output[0, 0]
            
            # Clip prediction to reasonable range
            point_prediction = np.clip(point_prediction, 0, 1)
            
            # If prediction deviates too far from historical mean, pull it back
            if abs(point_prediction - historical_mean) > 2 * historical_std:
                # Apply soft dampening
                direction = np.sign(point_prediction - historical_mean)
                excess = abs(point_prediction - historical_mean) - 2 * historical_std
                dampening_factor = 0.7  # Reduce excess by 30%
                point_prediction = historical_mean + direction * (2 * historical_std + excess * dampening_factor)
            
            # Calculate uncertainty based on prediction deviation
            deviation = abs(point_prediction - historical_mean) / historical_std
            uncertainty = historical_std * (1 + deviation * 0.5)  # Increase uncertainty for larger deviations
            
            # Ensure bounds are within reasonable range
            lower_bound = np.clip(point_prediction - uncertainty, 0, 1)
            upper_bound = np.clip(point_prediction + uncertainty, 0, 1)
        
        # Final sanity check
        point_prediction = np.clip(point_prediction, 0, 1)
        lower_bound = np.clip(lower_bound, 0, 1)
        upper_bound = np.clip(upper_bound, 0, 1)
        
        # Ensure bounds are properly ordered
        lower_bound = min(lower_bound, point_prediction)
        upper_bound = max(upper_bound, point_prediction)
        
        return point_prediction, lower_bound, upper_bound

    def plot_detrending(self, original_series: pd.Series, detrended_series: pd.Series, trend: pd.Series = None, title="Detrending Analysis"):
        """Plot original, trend, and detrended series for analysis."""
        plt.figure(figsize=(15, 8))
        
        # Plot original series
        plt.subplot(2, 1, 1)
        plt.plot(original_series.index, original_series.values, label='Original', color='blue')
        if trend is not None:
            plt.plot(trend.index, trend.values, label='Trend', color='red', linestyle='--')
        plt.title('Original Series and Trend')
        plt.grid(True)
        plt.legend()
        
        # Plot detrended series
        plt.subplot(2, 1, 2)
        plt.plot(detrended_series.index, detrended_series.values, label='Detrended', color='green')
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.title('Detrended Series')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        
        # Save figure
        figures_dir = Path('reports/figures')
        figures_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(figures_dir / 'detrending_analysis.png')
        plt.show()

    def plot_detrended_predictions(self, detrended_series: pd.Series, scaled_predictions: np.ndarray, scaled_lower: np.ndarray, scaled_upper: np.ndarray, title="Detrended Predictions Analysis"):
        """Plot detrended series with scaled predictions before trend reintroduction."""
        plt.figure(figsize=(15, 8))
        
        # Get the last date from historical data
        last_date = detrended_series.index[-1]
        next_month = last_date + pd.DateOffset(months=1)
        prediction_dates = pd.date_range(start=next_month, periods=len(scaled_predictions), freq="MS")
        
        # Rescale predictions and bounds back to detrended space
        rescaled_predictions = self.data_scaler.inverse_scale(scaled_predictions.reshape(-1, 1)).flatten()
        rescaled_lower = self.data_scaler.inverse_scale(scaled_lower.reshape(-1, 1)).flatten()
        rescaled_upper = self.data_scaler.inverse_scale(scaled_upper.reshape(-1, 1)).flatten()
        
        # Plot detrended historical data
        plt.plot(detrended_series.index, detrended_series.values, label='Detrended Historical', color='blue')
        
        # Plot detrended predictions
        plt.plot(prediction_dates, rescaled_predictions, label='Detrended Predictions', color='red')
        
        # Plot confidence interval
        plt.fill_between(
            prediction_dates,
            rescaled_lower,
            rescaled_upper,
            color='red',
            alpha=0.2,
            label='Confidence Interval'
        )
        
        # Add a horizontal line at y=0 for reference
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Add a vertical line at the forecast start
        plt.axvline(x=last_date, color='black', linestyle='--', alpha=0.7)
        
        plt.title('Detrended Series with Predictions')
        plt.grid(True)
        plt.legend()
        
        # Improve date formatting
        plt.gcf().autofmt_xdate()
        
        # Save figure
        figures_dir = Path('reports/figures')
        figures_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(figures_dir / 'detrended_predictions.png')
        plt.show()

    def process_time_series(
        self,
        group_df: pd.DataFrame,
        n: int,
        num_samples: int,
        input_series_length: int,
        low_bound_conf: int,
        high_bound_conf: int,
    ) -> pd.DataFrame:
        # Log initial statistics
        logger.info(f"Original data range: {group_df['y'].min():.4f} to {group_df['y'].max():.4f}")
        logger.info(f"Original data mean: {group_df['y'].mean():.4f}, std: {group_df['y'].std():.4f}")
        logger.info(f"Original data first 5 values: {group_df['y'].head().values}")
        logger.info(f"Original data last 5 values: {group_df['y'].tail().values}")

        # 1. Remove trend
        detrended_series, trend_params = self.remove_trend(group_df["y"])
        
        # Plot detrending analysis
        trend = trend_params.get("trend", None) if trend_params["type"] == "stl" else None
        self.plot_detrending(group_df["y"], detrended_series, trend)
        
        # Log detrending details
        logger.info(f"Detrended data range: {detrended_series.min():.4f} to {detrended_series.max():.4f}")
        logger.info(f"Detrended data mean: {detrended_series.mean():.4f}, std: {detrended_series.std():.4f}")
        logger.info(f"Detrended first 5 values: {detrended_series.head().values}")
        logger.info(f"Detrended last 5 values: {detrended_series.tail().values}")

        # 2. Scale the detrended data
        series_scaled = self.data_scaler.scale(detrended_series.to_numpy().reshape(-1, 1)).flatten()
        
        # Log scaling details
        logger.info(f"Scaled detrended data range: {series_scaled.min():.4f} to {series_scaled.max():.4f}")
        logger.info(f"Scaled detrended mean: {series_scaled.mean():.4f}, std: {series_scaled.std():.4f}")
        logger.info(f"Scaled first 5 values: {series_scaled[:5]}")
        logger.info(f"Scaled last 5 values: {series_scaled[-5:]}")

        predictions = []
        lower_bound_predictions = []
        upper_bound_predictions = []
        series_scaled_flatten = series_scaled

        padding_length = max(0, input_series_length - len(series_scaled_flatten))
        if padding_length > 0:
            logger.info(f"Adding {padding_length} padding zeros at the start")
        series_scaled_flatten = np.pad(series_scaled_flatten, (padding_length, 0), "constant")

        # Log model input details
        logger.info(f"Model input sequence length: {len(series_scaled_flatten)}")
        logger.info(f"Model input first 5 values: {series_scaled_flatten[:5]}")
        logger.info(f"Model input last 5 values: {series_scaled_flatten[-5:]}")

        # 3. Predict
        for step in range(n):
            prediction, lower_bound, upper_bound = self.generate_single_step_forecast(
                series_scaled_flatten,
                input_series_length,
                len(series_scaled_flatten[series_scaled_flatten != 0])
            )
            predictions.append(prediction)
            lower_bound_predictions.append(lower_bound)
            upper_bound_predictions.append(upper_bound)
            series_scaled_flatten = np.append(series_scaled_flatten, prediction)
            
            if step % 12 == 0:  # Log every 12 steps
                logger.info(f"Step {step}: prediction={prediction:.4f}, bounds=[{lower_bound:.4f}, {upper_bound:.4f}]")

        logger.info(f"Raw predictions range (scaled): {min(predictions):.4f} to {max(predictions):.4f}")
        logger.info(f"Raw predictions mean (scaled): {np.mean(predictions):.4f}, std: {np.std(predictions):.4f}")

        # Plot detrended predictions before adding trend back
        self.plot_detrended_predictions(
            detrended_series,
            np.array(predictions),
            np.array(lower_bound_predictions),
            np.array(upper_bound_predictions)
        )

        # Fix: Create proper date range that continues from the last historical date
        last_date = pd.Timestamp(group_df.index[-1])
        next_month = last_date + pd.DateOffset(months=1)
        prediction_dates = pd.date_range(start=next_month, periods=n, freq="MS")
        
        # 4. Rescale back to original space
        predictions_array = np.array(predictions).reshape(-1, 1)
        lower_bound_array = np.array(lower_bound_predictions).reshape(-1, 1)
        upper_bound_array = np.array(upper_bound_predictions).reshape(-1, 1)
        
        # Debug scaling
        logger.info(f"Shape before rescaling - predictions: {predictions_array.shape}")
        logger.info(f"Range before rescaling: {predictions_array.min():.4f} to {predictions_array.max():.4f}")
        
        rescaled_predictions = self.data_scaler.inverse_scale(predictions_array).flatten()
        rescaled_lower = self.data_scaler.inverse_scale(lower_bound_array).flatten()
        rescaled_upper = self.data_scaler.inverse_scale(upper_bound_array).flatten()
        
        logger.info(f"Rescaled predictions range: {rescaled_predictions.min():.4f} to {rescaled_predictions.max():.4f}")
        logger.info(f"Rescaled predictions mean: {np.mean(rescaled_predictions):.4f}, std: {np.std(rescaled_predictions):.4f}")
        logger.info(f"First 5 rescaled predictions: {rescaled_predictions[:5]}")
        logger.info(f"Last 5 rescaled predictions: {rescaled_predictions[-5:]}")

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

        logger.info(f"Final predictions range: {predictions_with_trend.min():.4f} to {predictions_with_trend.max():.4f}")
        logger.info(f"Final predictions mean: {predictions_with_trend.mean():.4f}, std: {predictions_with_trend.std():.4f}")
        logger.info(f"First 5 final predictions: {predictions_with_trend.head().values}")
        logger.info(f"Last 5 final predictions: {predictions_with_trend.tail().values}")
        logger.info(f"Confidence interval widths - First 5: {(upper_bound_with_trend.head() - lower_bound_with_trend.head()).values}")
        logger.info(f"Confidence interval widths - Last 5: {(upper_bound_with_trend.tail() - lower_bound_with_trend.tail()).values}")

        # Create predicted series DataFrame
        predicted_series = pd.DataFrame({
            "ds": prediction_dates,
            "q_0.3": lower_bound_with_trend.values,
            "q_0.5": predictions_with_trend.values,
            "q_0.7": upper_bound_with_trend.values,
            "unique_id": group_df["unique_id"].iloc[0],
        })

        return predicted_series

    def generate_forecast(
        self,
        df: pd.DataFrame,
        n: int,
        num_samples: int = 1000,
        input_series_length: int = 60,
        low_bound_conf: int = 25,
        high_bound_conf: int = 75,
    ) -> pd.DataFrame:
        """
        Generate forecast for all time series in the DataFrame.
        
        Args:
            df: DataFrame with time series data
            n: Number of steps to forecast
            num_samples: Number of samples for probabilistic forecast
            input_series_length: Length of input sequences
            low_bound_conf: Lower bound percentile
            high_bound_conf: Upper bound percentile
            
        Returns:
            DataFrame with forecast
        """
        # Ensure df has a copy of the date column as 'ds' if it's the index
        if 'ds' not in df.columns and df.index.name != 'ds':
            if isinstance(df.index, pd.DatetimeIndex):
                df = df.reset_index()
                if 'index' in df.columns:
                    df = df.rename(columns={'index': 'ds'})
        
        # Group by unique_id
        grouped = df.groupby("unique_id")
        
        all_predictions = []
        
        for group_name, group_df in grouped:
            try:
                logger.info(f"Processing time series {group_name}")
                
                # Process time series
                predicted_series = self.process_time_series(
                    group_df.set_index("ds") if "ds" in group_df.columns else group_df,
                    n,
                    num_samples,
                    input_series_length,
                    low_bound_conf,
                    high_bound_conf,
                )
                
                all_predictions.append(predicted_series)
                
                logger.info(f"Predictions for {group_name}:\n{predicted_series}")
            except Exception as e:
                logger.error(f"Error processing time series {group_name}: {str(e)}")
                logger.error(traceback.format_exc())
        
        if not all_predictions:
            raise ValueError("No predictions were generated for any time series.")
        
        # Concatenate all predictions
        result = pd.concat(all_predictions, ignore_index=True)
        
        # Ensure ds column is datetime type
        result['ds'] = pd.to_datetime(result['ds'])
        
        return result

    def plot_forecast(self, historical_df: pd.DataFrame, forecast_df: pd.DataFrame, title="Time Series Forecast"):
        """Plot the historical data and forecast with confidence intervals."""
        plt.figure(figsize=(12, 6))

        # Find the first non-zero value index
        first_non_zero_idx = historical_df[historical_df["y"] != 0].index[0]
        
        # Filter historical data from first non-zero value
        historical_df_filtered = historical_df[first_non_zero_idx:]

        # Plot historical data
        plt.plot(historical_df_filtered.index, historical_df_filtered["y"], label="Historical", color="blue")

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

        # Improve date formatting
        plt.gcf().autofmt_xdate()  # Auto-format the x-axis date labels
        
        # Set grid with date-specific formatting
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        
        # Add a vertical line at the forecast start
        plt.axvline(x=historical_df.index[-1], color='black', linestyle='--', alpha=0.7)

        plt.title(title)
        plt.xlabel("Date")
        plt.ylabel("Value")
        plt.legend()
        
        # Save figure
        figures_dir = Path('reports/figures')
        figures_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(figures_dir / 'advanced_forecast.png')
        
        plt.show() 