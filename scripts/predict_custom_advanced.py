import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import argparse
import sys
import logging
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import random
from typing import Any, Dict, Optional
from scipy.stats import linregress
from statsmodels.tsa.seasonal import STL
import datetime
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent.parent))

# Import custom loss functions and metrics
from src.models.transformer import gaussian_nll, smape_loss, hybrid_loss

def mae_prob(y_true, y_pred):
    """Custom MAE metric for probabilistic models that only uses mean prediction."""
    mu, _ = tf.split(y_pred, 2, axis=-1)
    return tf.keras.metrics.mean_absolute_error(y_true, mu)

class DataScaler:
    """Simple wrapper around MinMaxScaler to match the interface expected by ForecastGenerator."""
    
    def __init__(self, feature_range=(0, 1)):
        self.scaler = MinMaxScaler(feature_range=feature_range)
        self.is_fitted = False
        
    def scale(self, data):
        """Scale the data."""
        if not self.is_fitted:
            self.scaler.fit(data)
            self.is_fitted = True
        return self.scaler.transform(data)
    
    def inverse_scale(self, data):
        """Inverse transform the scaled data."""
        if not self.is_fitted:
            raise ValueError("Scaler must be fitted before inverse_scale can be called.")
        return self.scaler.inverse_transform(data)

class ForecastGenerator:
    def __init__(self, proba_model: tf.keras.Model, data_scaler):
        self.proba_model = proba_model
        self.data_scaler = data_scaler

    def perform_stl_decomposition(self, series: pd.Series, period: int = 12) -> Optional[dict[str, Any]]:
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
        x = np.arange(len(series))
        slope, intercept, _, _, _ = linregress(x, series)
        trend = slope * x + intercept
        detrended = series - trend
        return detrended, {"type": "linear", "slope": slope, "intercept": intercept}

    def add_trend(self, series: pd.Series, trend_params: dict, future_index: pd.DatetimeIndex) -> pd.Series:
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
        series_reshaped = series_scaled_flatten[-input_series_length:].reshape(1, input_series_length, 1)
        proba_prediction = self.proba_model.predict(series_reshaped, verbose=0)

        mu, log_var = np.split(proba_prediction, 2, axis=-1)
        sigma = np.exp(0.5 * log_var)
        
        # Scale uncertainty based on forecast horizon
        scaling_increment = 0.1
        uncertainty_scaling_factor = 1 + (step * scaling_increment)
        scaled_sigma = sigma * uncertainty_scaling_factor

        # Generate samples from the predicted distribution
        samples = np.random.normal(mu, scaled_sigma, (num_samples, 1, 1))
        
        point_prediction = np.median(samples, axis=0).flatten()[0]  # 0.5 quantile
        lower_bound = np.percentile(samples, low_bound_conf, axis=0).flatten()[0]
        upper_bound = np.percentile(samples, high_bound_conf, axis=0).flatten()[0]

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

        padding_length = max(0, input_series_length - len(series_scaled_flatten))
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

        # Fix: Create proper date range that continues from the last historical date
        last_date = pd.Timestamp(group_df.index[-1])  # Convert to pandas Timestamp
        next_month = last_date + pd.DateOffset(months=1)
        prediction_dates = pd.date_range(start=next_month, periods=n, freq="MS")
        
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

        # Fix: Ensure dates are properly formatted in the DataFrame
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

def generate_synthetic_time_series_df(
    length=100, 
    seasonality=12, 
    trend=0.01, 
    noise=0.1, 
    start_date="2020-01-01"
):
    """
    Generate a synthetic time series with trend, seasonality, and noise as a DataFrame.
    
    Args:
        length: Length of the time series
        seasonality: Seasonality period
        trend: Trend coefficient
        noise: Noise level
        start_date: Start date for the time series
        
    Returns:
        DataFrame with time series data
    """
    # Time component
    t = np.arange(length)
    
    # Trend component
    trend_component = trend * t
    
    # Seasonal component
    seasonal_component = np.sin(2 * np.pi * t / seasonality)
    
    # Noise component
    noise_component = noise * np.random.randn(length)
    
    # Combine components
    series = trend_component + seasonal_component + noise_component
    
    # Create date range
    date_range = pd.date_range(start=start_date, periods=length, freq="MS")
    
    # Create DataFrame
    df = pd.DataFrame({
        "ds": date_range,
        "y": series,
        "unique_id": "synthetic_series"
    })
    
    # Set index to date
    df.set_index("ds", inplace=True)
    
    return df

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate advanced forecasts for a custom time series')
    
    parser.add_argument(
        '--proba-model',
        type=str,
        default='models/final/transformer_1.0_directml_proba_hybrid_0.8_M1_M2',
        help='Path to probabilistic model'
    )
    
    parser.add_argument(
        '--loss-type',
        type=str,
        default='hybrid',
        choices=['gaussian_nll', 'smape', 'hybrid'],
        help='Loss type used for the probabilistic model'
    )
    
    parser.add_argument(
        '--loss-alpha',
        type=float,
        default=0.8,
        help='Alpha value for hybrid loss'
    )
    
    parser.add_argument(
        '--n-steps',
        type=int,
        default=36,
        help='Number of steps to forecast'
    )
    
    parser.add_argument(
        '--num-samples',
        type=int,
        default=1000,
        help='Number of samples for probabilistic forecast'
    )
    
    parser.add_argument(
        '--low-bound-conf',
        type=int,
        default=30,
        help='Lower bound percentile'
    )
    
    parser.add_argument(
        '--high-bound-conf',
        type=int,
        default=70,
        help='Upper bound percentile'
    )
    
    parser.add_argument(
        '--sequence-length',
        type=int,
        default=60,
        help='Length of input sequences'
    )
    
    parser.add_argument(
        '--series-length',
        type=int,
        default=100,
        help='Length of synthetic time series'
    )
    
    parser.add_argument(
        '--start-date',
        type=str,
        default='2020-01-01',
        help='Start date for synthetic time series'
    )
    
    return parser.parse_args()

def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Set random seed for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    random.seed(42)
    
    # Generate synthetic time series as DataFrame
    print("Generating synthetic time series...")
    historical_df = generate_synthetic_time_series_df(
        length=args.series_length,
        seasonality=12,  # Monthly seasonality
        trend=0.01,
        noise=0.1,
        start_date=args.start_date
    )
    
    print(f"Generated time series with {len(historical_df)} data points from {historical_df.index.min()} to {historical_df.index.max()}")
    
    # Prepare custom objects dictionary based on loss type
    custom_objects = {
        'mae_prob': mae_prob,
        'gaussian_nll': gaussian_nll,
        'smape_loss': smape_loss
    }
    
    # Add the appropriate loss function based on the specified loss type
    if args.loss_type == 'hybrid':
        hybrid_loss_fn = hybrid_loss(alpha=args.loss_alpha)
        custom_objects['hybrid_loss'] = hybrid_loss_fn
        custom_objects['loss'] = hybrid_loss_fn
        custom_objects['loss_fn'] = hybrid_loss_fn
    elif args.loss_type == 'gaussian_nll':
        custom_objects['loss'] = gaussian_nll
        custom_objects['loss_fn'] = gaussian_nll
    elif args.loss_type == 'smape':
        # For SMAPE with probabilistic model
        def smape_prob(y_true, y_pred):
            mu, _ = tf.split(y_pred, 2, axis=-1)
            return smape_loss(y_true, mu)
        custom_objects['loss'] = smape_prob
        custom_objects['smape_prob'] = smape_prob
        custom_objects['loss_fn'] = smape_prob
    
    # Load probabilistic model
    print(f"Loading probabilistic model from {args.proba_model}...")
    print(f"Using loss type: {args.loss_type}" + (f" with alpha={args.loss_alpha}" if args.loss_type == 'hybrid' else ""))
    proba_model = tf.keras.models.load_model(args.proba_model, custom_objects=custom_objects)
    
    # Create data scaler
    data_scaler = DataScaler()
    
    # Create forecast generator
    forecast_generator = ForecastGenerator(proba_model, data_scaler)
    
    # Generate forecast
    print(f"Generating forecast for {args.n_steps} steps...")
    forecast_df = forecast_generator.generate_forecast(
        historical_df.reset_index(),
        n=args.n_steps,
        num_samples=args.num_samples,
        input_series_length=args.sequence_length,
        low_bound_conf=args.low_bound_conf,
        high_bound_conf=args.high_bound_conf
    )
    
    # Plot forecast
    print("Plotting forecast...")
    forecast_generator.plot_forecast(
        historical_df,
        forecast_df,
        title=f"Time Series Forecast ({args.n_steps} steps ahead)"
    )
    
    print("Done! Forecast plot saved to reports/figures/advanced_forecast.png")

if __name__ == "__main__":
    main() 