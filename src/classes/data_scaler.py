import numpy as np
from sklearn.preprocessing import MinMaxScaler
import logging

# Set up logging
logger = logging.getLogger(__name__)

class DataScaler:
    """
    Wrapper around MinMaxScaler that handles initial padding zeros specially.
    Only the sequence of zeros before the first non-zero value is treated as padding.
    """
    
    def __init__(self, feature_range=(0.3, 0.7)):  # Use a more centered range
        self.scaler = MinMaxScaler(feature_range=feature_range)
        self.is_fitted = False
        self.first_non_zero_idx = None
        
    def _find_first_non_zero(self, data):
        """Find the index of the first non-zero value in the series."""
        # Ensure data is flattened for index finding
        flat_data = data.ravel()
        non_zero_indices = np.nonzero(flat_data)[0]
        return non_zero_indices[0] if len(non_zero_indices) > 0 else len(flat_data)
        
    def scale(self, data):
        """
        Scale the data while preserving initial padding zeros.
        
        Args:
            data: numpy array of shape (n_samples,) or (n_samples, 1)
            
        Returns:
            scaled_data: numpy array of same shape as input where initial padding zeros
                        are preserved and all other values are scaled
        """
        # Ensure input is 2D
        original_shape = data.shape
        if len(original_shape) == 1:
            data = data.reshape(-1, 1)
            
        if not self.is_fitted:
            # Find the first non-zero value
            self.first_non_zero_idx = self._find_first_non_zero(data)
            logger.debug(f"First non-zero value found at index {self.first_non_zero_idx}")
            
            # Get the actual data (excluding initial padding)
            actual_data = data[self.first_non_zero_idx:]
            
            if len(actual_data) > 0:
                # Get the data range
                data_min = actual_data.min()
                data_max = actual_data.max()
                data_range = data_max - data_min
                
                # If the data range is very small, add a small buffer
                if data_range < 0.1:
                    buffer = max(0.1 - data_range, 0.05)
                    data_min = min(0, data_min - buffer/2)  # Allow negative values
                    data_max = data_max + buffer/2
                
                # Create synthetic calibration data that includes the actual range
                # plus a small buffer to prevent edge effects
                calibration_data = np.array([
                    data_min * 1.1,  # Add 10% buffer on both sides
                    data_min,
                    data_max,
                    data_max * 1.1
                ]).reshape(-1, 1)
                
                logger.debug(f"Fitting scaler on calibrated range: {calibration_data.min()} to {calibration_data.max()}")
                self.scaler.fit(calibration_data)
                self.is_fitted = True
            else:
                logger.warning("No non-padding data points found for scaling")
                return data.reshape(original_shape)
        
        # Create output array
        scaled_data = np.zeros_like(data)
        
        # Scale only the actual data (after initial padding)
        if self.is_fitted and self.first_non_zero_idx < len(data):
            actual_data = data[self.first_non_zero_idx:]
            scaled_actual = self.scaler.transform(actual_data)
            scaled_data[self.first_non_zero_idx:] = scaled_actual
            logger.debug(f"Scaled actual data range: {scaled_actual.min()} to {scaled_actual.max()}")
        
        # Return data in original shape
        return scaled_data.reshape(original_shape)
    
    def inverse_scale(self, data):
        """
        Inverse transform the scaled data while preserving initial padding zeros.
        
        Args:
            data: numpy array of shape (n_samples,) or (n_samples, 1)
            
        Returns:
            inverse_scaled_data: numpy array of same shape as input where initial padding
                               zeros are preserved and all other values are inverse scaled
        """
        if not self.is_fitted:
            raise ValueError("Scaler must be fitted before inverse_scale can be called.")
            
        # Ensure input is 2D
        original_shape = data.shape
        if len(original_shape) == 1:
            data = data.reshape(-1, 1)
        
        # Create output array
        inverse_scaled_data = np.zeros_like(data)
        
        # Inverse scale the actual data
        if self.first_non_zero_idx < len(data):
            actual_data = data[self.first_non_zero_idx:]
            inverse_scaled_actual = self.scaler.inverse_transform(actual_data)
            inverse_scaled_data[self.first_non_zero_idx:] = inverse_scaled_actual
            logger.debug(f"Inverse scaled actual data range: {inverse_scaled_actual.min()} to {inverse_scaled_actual.max()}")
        
        # Return data in original shape
        return inverse_scaled_data.reshape(original_shape)
    
    def reset(self):
        """Reset the scaler's fitted state and first non-zero index."""
        self.is_fitted = False
        self.first_non_zero_idx = None 