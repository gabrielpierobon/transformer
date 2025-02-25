import numpy as np
import tensorflow as tf
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Union
import sys
from pathlib import Path
import logging
import uvicorn
from datetime import datetime, timedelta
import os

# Add the project root to the path
sys.path.append(str(Path(__file__).parent.parent))

# Import custom loss functions and metrics
from src.models.transformer import gaussian_nll, smape_loss, hybrid_loss

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Time Series Forecasting API",
    description="API for generating time series forecasts using transformer models",
    version="1.0.0"
)

# Define request and response models
class TimeSeriesRequest(BaseModel):
    values: List[float]
    model_type: str = "point"  # "point" or "probabilistic"
    n_steps: int = 36
    sequence_length: int = 60
    num_samples: Optional[int] = 1000
    low_bound_conf: Optional[int] = 25
    high_bound_conf: Optional[int] = 75
    loss_type: Optional[str] = "hybrid"
    loss_alpha: Optional[float] = 0.8
    
class TimeSeriesResponse(BaseModel):
    forecast: List[float]
    lower_bound: Optional[List[float]] = None
    upper_bound: Optional[List[float]] = None
    dates: Optional[List[str]] = None

# Define custom metric for probabilistic models
def mae_prob(y_true, y_pred):
    """Custom MAE metric for probabilistic models that only uses mean prediction."""
    mu, _ = tf.split(y_pred, 2, axis=-1)
    return tf.keras.metrics.mean_absolute_error(y_true, mu)

# Global model cache to avoid reloading models
model_cache = {}

def create_mock_point_model(sequence_length=60):
    """Create a simple mock model for point predictions when the real model is not available."""
    logger.info("Creating mock point prediction model for testing")
    inputs = tf.keras.Input(shape=(sequence_length, 1))
    x = tf.keras.layers.GlobalAveragePooling1D()(inputs)
    outputs = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mse')
    return model

def create_mock_probabilistic_model(sequence_length=60):
    """Create a simple mock model for probabilistic predictions when the real model is not available."""
    logger.info("Creating mock probabilistic model for testing")
    inputs = tf.keras.Input(shape=(sequence_length, 1))
    x = tf.keras.layers.GlobalAveragePooling1D()(inputs)
    mean = tf.keras.layers.Dense(1, name='mean')(x)
    log_var = tf.keras.layers.Dense(1, name='log_var')(x)
    outputs = tf.keras.layers.Concatenate()([mean, log_var])
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss=gaussian_nll)
    return model

def load_model(model_type: str, loss_type: str = None, loss_alpha: float = None):
    """Load model from cache or disk. Creates a mock model if the real model is not available."""
    model_key = f"{model_type}_{loss_type}_{loss_alpha}" if model_type == "probabilistic" else "point"
    
    if model_key in model_cache:
        return model_cache[model_key]
    
    # Get the project root directory
    root_dir = Path(__file__).parent.parent
    
    # Define model paths
    if model_type == "point":
        model_path = root_dir / "models" / "final" / "transformer_1.0_directml_point_M1_M2"
    else:
        if loss_type == "hybrid":
            model_path = root_dir / "models" / "final" / f"transformer_1.0_directml_proba_hybrid_{loss_alpha}_M1_M2"
        elif loss_type == "gaussian_nll":
            model_path = root_dir / "models" / "final" / "transformer_1.0_directml_proba_gaussian_nll_M1_M2"
        elif loss_type == "smape":
            model_path = root_dir / "models" / "final" / "transformer_1.0_directml_proba_smape_M1_M2"
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    # Prepare custom objects for probabilistic models
    if model_type == "probabilistic":
        custom_objects = {
            'mae_prob': mae_prob,
            'gaussian_nll': gaussian_nll,
            'smape_loss': smape_loss
        }
        
        # Add the appropriate loss function
        if loss_type == "hybrid":
            hybrid_loss_fn = hybrid_loss(alpha=loss_alpha)
            custom_objects['hybrid_loss'] = hybrid_loss_fn
            custom_objects['loss'] = hybrid_loss_fn
            custom_objects['loss_fn'] = hybrid_loss_fn
        elif loss_type == "gaussian_nll":
            custom_objects['loss'] = gaussian_nll
            custom_objects['loss_fn'] = gaussian_nll
        elif loss_type == "smape":
            def smape_prob(y_true, y_pred):
                mu, _ = tf.split(y_pred, 2, axis=-1)
                return smape_loss(y_true, mu)
            custom_objects['loss'] = smape_prob
            custom_objects['smape_prob'] = smape_prob
            custom_objects['loss_fn'] = smape_prob
            
        # Try to load model with custom objects
        try:
            if os.path.exists(model_path):
                logger.info(f"Loading probabilistic model from {model_path}")
                model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
            else:
                logger.warning(f"Model not found at {model_path}, creating mock model for testing")
                model = create_mock_probabilistic_model()
        except Exception as e:
            logger.warning(f"Error loading model {model_path}: {str(e)}, creating mock model for testing")
            model = create_mock_probabilistic_model()
    else:
        # Try to load point prediction model
        try:
            if os.path.exists(model_path):
                logger.info(f"Loading point prediction model from {model_path}")
                model = tf.keras.models.load_model(model_path)
            else:
                logger.warning(f"Model not found at {model_path}, creating mock model for testing")
                model = create_mock_point_model()
        except Exception as e:
            logger.warning(f"Error loading model {model_path}: {str(e)}, creating mock model for testing")
            model = create_mock_point_model()
    
    # Cache the model
    model_cache[model_key] = model
    
    return model

def preprocess_time_series(values: List[float], sequence_length: int = 60):
    """
    Preprocess time series for model input.
    
    Args:
        values: Input time series values
        sequence_length: Length of input sequences
        
    Returns:
        Preprocessed series and scaler
    """
    # Convert to numpy array
    series = np.array(values)
    
    # Reshape for scaler
    series_reshaped = series.reshape(-1, 1)
    
    # Scale the series
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_series = scaler.fit_transform(series_reshaped).flatten()
    
    # Prepare input sequence (take last sequence_length values or pad)
    if len(scaled_series) >= sequence_length:
        model_input = scaled_series[-sequence_length:]
    else:
        # Pad with zeros if series is shorter than sequence_length
        model_input = np.zeros(sequence_length)
        model_input[-len(scaled_series):] = scaled_series
    
    # Reshape for model input (batch_size, sequence_length, features)
    model_input = model_input.reshape(1, sequence_length, 1)
    
    return model_input, scaler, scaled_series

def generate_point_forecast(model, initial_series, n_steps, sequence_length=60):
    """
    Generate point forecasts recursively.
    
    Args:
        model: Trained point prediction model
        initial_series: Initial series values (scaled)
        n_steps: Number of steps to forecast
        sequence_length: Length of input sequences
        
    Returns:
        Array of predicted values
    """
    series = initial_series.copy()
    predictions = []
    
    for step in range(n_steps):
        # Create model input
        if len(series) >= sequence_length:
            model_input = series[-sequence_length:].reshape(1, sequence_length, 1)
        else:
            # Pad with zeros if series is shorter than sequence_length
            model_input = np.zeros((1, sequence_length, 1))
            model_input[0, -len(series):, 0] = series
        
        # Make prediction
        next_value = model.predict(model_input, verbose=0)[0, 0]
        
        # Store prediction
        predictions.append(next_value)
        
        # Update series with prediction
        series = np.append(series, next_value)
    
    return np.array(predictions)

def generate_probabilistic_forecast(model, initial_series, n_steps, sequence_length=60, num_samples=1000, low_bound_conf=25, high_bound_conf=75):
    """
    Generate probabilistic forecasts recursively.
    
    Args:
        model: Trained probabilistic model
        initial_series: Initial series values (scaled)
        n_steps: Number of steps to forecast
        sequence_length: Length of input sequences
        num_samples: Number of samples to generate for each step
        low_bound_conf: Lower bound percentile
        high_bound_conf: Upper bound percentile
        
    Returns:
        Dictionary with mean predictions, lower and upper bounds
    """
    series = initial_series.copy()
    mean_predictions = []
    lower_bounds = []
    upper_bounds = []
    
    for step in range(n_steps):
        # Create model input
        if len(series) >= sequence_length:
            model_input = series[-sequence_length:].reshape(1, sequence_length, 1)
        else:
            # Pad with zeros if series is shorter than sequence_length
            model_input = np.zeros((1, sequence_length, 1))
            model_input[0, -len(series):, 0] = series
        
        # Make prediction (mean and log variance)
        prediction = model.predict(model_input, verbose=0)[0]
        mean = prediction[0]
        log_var = prediction[1]
        std = np.exp(0.5 * log_var)
        
        # Scale uncertainty based on forecast horizon
        scaling_increment = 0.1
        uncertainty_scaling_factor = 1 + (step * scaling_increment)
        scaled_std = std * uncertainty_scaling_factor
        
        # Generate samples from the predicted distribution
        samples = np.random.normal(mean, scaled_std, num_samples)
        
        # Store mean prediction
        mean_predictions.append(mean)
        
        # Calculate confidence intervals
        lower_bound = np.percentile(samples, low_bound_conf)
        upper_bound = np.percentile(samples, high_bound_conf)
        
        lower_bounds.append(lower_bound)
        upper_bounds.append(upper_bound)
        
        # Update series with mean prediction
        series = np.append(series, mean)
    
    return {
        'mean': np.array(mean_predictions),
        'lower': np.array(lower_bounds),
        'upper': np.array(upper_bounds)
    }

def generate_forecast_dates(n_steps, start_date=None):
    """Generate dates for forecast."""
    if start_date is None:
        start_date = datetime.now()
    
    # Round to the beginning of the month
    start_date = datetime(start_date.year, start_date.month, 1)
    
    # Generate dates
    dates = []
    current_date = start_date
    for _ in range(n_steps):
        # Move to next month
        if current_date.month == 12:
            current_date = datetime(current_date.year + 1, 1, 1)
        else:
            current_date = datetime(current_date.year, current_date.month + 1, 1)
        dates.append(current_date.strftime("%Y-%m-%d"))
    
    return dates

@app.post("/forecast/", response_model=TimeSeriesResponse)
async def generate_forecast(request: TimeSeriesRequest):
    """
    Generate forecast for time series.
    
    Args:
        request: TimeSeriesRequest object
        
    Returns:
        TimeSeriesResponse object
    """
    # Validate request first, before any other processing
    if not request.values:
        raise HTTPException(status_code=400, detail="No time series values provided")
    
    if request.n_steps <= 0:
        raise HTTPException(status_code=400, detail="Number of steps must be positive")
    
    if request.sequence_length <= 0:
        raise HTTPException(status_code=400, detail="Sequence length must be positive")
    
    if request.model_type not in ["point", "probabilistic"]:
        raise HTTPException(status_code=400, detail="Model type must be 'point' or 'probabilistic'")
    
    if request.model_type == "probabilistic":
        if request.loss_type not in ["hybrid", "gaussian_nll", "smape"]:
            raise HTTPException(status_code=400, detail="Loss type must be 'hybrid', 'gaussian_nll', or 'smape'")
        
        if request.loss_type == "hybrid" and (request.loss_alpha < 0 or request.loss_alpha > 1):
            raise HTTPException(status_code=400, detail="Loss alpha must be between 0 and 1")
    
    try:
        # Preprocess time series
        model_input, scaler, scaled_series = preprocess_time_series(
            request.values, 
            request.sequence_length
        )
        
        # Load model
        if request.model_type == "point":
            model = load_model("point")
            
            # Generate forecast
            scaled_forecast = generate_point_forecast(
                model, 
                scaled_series, 
                request.n_steps, 
                request.sequence_length
            )
            
            # Inverse transform forecast
            forecast = scaler.inverse_transform(scaled_forecast.reshape(-1, 1)).flatten()
            
            # Generate response
            response = TimeSeriesResponse(
                forecast=forecast.tolist(),
                dates=generate_forecast_dates(request.n_steps)
            )
        else:
            model = load_model(
                "probabilistic", 
                request.loss_type, 
                request.loss_alpha
            )
            
            # Generate forecast
            forecast_dict = generate_probabilistic_forecast(
                model, 
                scaled_series, 
                request.n_steps, 
                request.sequence_length,
                request.num_samples,
                request.low_bound_conf,
                request.high_bound_conf
            )
            
            # Inverse transform forecasts
            mean_forecast = scaler.inverse_transform(forecast_dict['mean'].reshape(-1, 1)).flatten()
            lower_bound = scaler.inverse_transform(forecast_dict['lower'].reshape(-1, 1)).flatten()
            upper_bound = scaler.inverse_transform(forecast_dict['upper'].reshape(-1, 1)).flatten()
            
            # Generate response
            response = TimeSeriesResponse(
                forecast=mean_forecast.tolist(),
                lower_bound=lower_bound.tolist(),
                upper_bound=upper_bound.tolist(),
                dates=generate_forecast_dates(request.n_steps)
            )
        
        return response
    
    except HTTPException:
        # Re-raise HTTP exceptions without modification
        raise
    except Exception as e:
        logger.error(f"Error generating forecast: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating forecast: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run("forecast_api:app", host="0.0.0.0", port=8000, reload=True) 