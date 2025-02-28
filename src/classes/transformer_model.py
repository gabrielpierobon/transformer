"""
TransformerModel class for managing and executing time series forecasting.

This class provides a high-level interface for loading local transformer models
and generating forecasts with uncertainty quantification.
"""

import logging
import os
from pathlib import Path
import pandas as pd
import tensorflow as tf

from .data_scaler import DataScaler
from .forecast_generator import ForecastGenerator
from .model_loader import ModelLoader

logger = logging.getLogger(__name__)


class TransformerModel:
    """
    A class for managing and executing time series forecasting using Transformer models.

    This class provides functionalities to load and utilize Transformer-based models
    for time series forecasting. Models are loaded from the local 'models/final' directory.

    Example:
        ```python
        # Initialize with a model name
        model = TransformerModel(
            model_name="transformer_1.0_directml_proba_hybrid_0.8_M1_M2"
        )

        # Generate forecasts
        forecasts = model.predict(
            series_df=your_data,
            n=12,  # forecast horizon
            num_samples=1000
        )
        ```

    Attributes:
        proba_model (tf.keras.Model): The loaded probabilistic model used for forecasting.
        input_series_length (tuple[int]): The length of the input time series data required by the model.
        data_scaler (DataScaler): An instance for normalizing and denormalizing the time series data.
        forecast_generator (ForecastGenerator): An instance responsible for generating forecasts.
        is_probabilistic (bool): Whether the model is probabilistic or point-based.
    """

    def __init__(
        self,
        model_name: str,
        input_series_length: int = 60,
        inference_config_path: str = None,
    ):
        """
        Initializes the TransformerModel with a local model.

        Args:
            model_name: Name of the model directory in models/final/
                      (e.g., "transformer_1.0_directml_proba_hybrid_0.8_M1_M2")
            input_series_length: Expected length of input time series for the model.
                               Defaults to 60.
            inference_config_path: Path to the inference configuration file.
                                 If None, will look for config/inference_config.yaml.

        Raises:
            ValueError: If model loading fails or if model directory doesn't exist.
        """
        logger.info(f"Initializing TransformerModel with model: {model_name}")
        
        try:
            # Construct model path
            model_path = str(Path('models/final') / model_name)
            if not os.path.exists(model_path):
                raise ValueError(f"Model directory not found: {model_path}")

            # Determine if model is probabilistic from name
            self.is_probabilistic = "proba" in model_name.lower()
            logger.info(f"Model type: {'probabilistic' if self.is_probabilistic else 'point'}")

            # Load the model
            model_loader = ModelLoader(
                model_path=model_path,
                model_server_type='local',
                custom_loss="gaussian_nll" if self.is_probabilistic else None
            )
            self.proba_model = model_loader.load_model()
            logger.info("Successfully loaded model")

            # Set up other components
            self.input_series_length = (input_series_length,)
            self.data_scaler = DataScaler()
            self.forecast_generator = ForecastGenerator(
                proba_model=self.proba_model,
                data_scaler=self.data_scaler,
                is_probabilistic=self.is_probabilistic,
                config_path=inference_config_path,
            )
            logger.info("TransformerModel initialization complete")

        except Exception as e:
            logger.error(f"Failed to initialize TransformerModel: {str(e)}")
            raise ValueError(f"TransformerModel initialization failed: {str(e)}")

    def predict(
        self,
        series_df: pd.DataFrame,
        n: int,
        num_samples: int = 1000,
        low_bound_conf: int = 30,
        high_bound_conf: int = 70,
        force_linear_detrend: bool = False,
    ) -> pd.DataFrame:
        """
        Generates forecasts with confidence intervals for given time series data.

        Args:
            series_df: DataFrame containing the time series data.
                      Must have columns ['unique_id', 'y'] and datetime index.
            n: Number of future data points to forecast.
            num_samples: Number of samples for generating probabilistic forecasts.
                        Defaults to 1000.
            low_bound_conf: Lower percentile for confidence intervals.
                          Defaults to 30.
            high_bound_conf: Upper percentile for confidence intervals.
                           Defaults to 70.
            force_linear_detrend: Force using linear detrending even if STL is possible.
                                 Only applies if detrending is enabled in config.

        Returns:
            DataFrame containing the forecasted values and confidence intervals.
            Columns: ['ds', 'q_0.3', 'q_0.5', 'q_0.7', 'unique_id']

        Raises:
            ValueError: If input data is invalid or if forecasting fails.
        """
        logger.info(f"Generating forecasts for {len(series_df)} time series")
        
        try:
            # Validate input data
            required_columns = ['unique_id', 'y']
            if not all(col in series_df.columns for col in required_columns):
                raise ValueError(
                    f"Input DataFrame must contain columns: {required_columns}",
                )
            
            if not isinstance(series_df.index, pd.DatetimeIndex):
                raise ValueError("Input DataFrame must have a DatetimeIndex")

            # Generate forecasts
            forecasts = self.forecast_generator.generate_forecast(
                series_df=series_df,
                n=n,
                num_samples=num_samples,
                input_series_length=self.input_series_length,
                low_bound_conf=low_bound_conf,
                high_bound_conf=high_bound_conf,
                force_linear_detrend=force_linear_detrend,
            )
            
            logger.info("Successfully generated forecasts")
            return forecasts

        except Exception as e:
            logger.error(f"Error generating forecasts: {str(e)}")
            raise ValueError(f"Forecast generation failed: {str(e)}")

    def plot_forecast(
        self, 
        historical_df: pd.DataFrame, 
        forecast_df: pd.DataFrame,
        low_bound_conf: int = 30,
        high_bound_conf: int = 70
    ) -> None:
        """
        Plot historical data and forecast with confidence intervals.

        Args:
            historical_df: DataFrame containing historical data.
            forecast_df: DataFrame containing forecast data.
            low_bound_conf: Lower confidence bound percentile (default: 30)
            high_bound_conf: Upper confidence bound percentile (default: 70)
        """
        self.forecast_generator.plot_forecast(
            historical_df, 
            forecast_df,
            low_bound_conf,
            high_bound_conf
        ) 