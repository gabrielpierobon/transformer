"""
DataScaler class for handling data normalization in time series forecasting.

This class provides a wrapper around sklearn's MinMaxScaler with additional
error handling and convenience methods for time series data.
"""

import numpy as np
from sklearn.preprocessing import MinMaxScaler


class DataScaler:
    """
    A class to handle the scaling of numerical data using the MinMaxScaler.

    This class provides a simplified interface for data scaling operations,
    particularly useful for time series data. It uses sklearn's MinMaxScaler
    internally and adds error handling and convenience methods.

    Attributes:
        scaler (MinMaxScaler): Instance of the MinMaxScaler, pre-configured with a feature range of 0 to 1.
    """

    def __init__(self) -> None:
        """
        Initializes the DataScaler with a MinMaxScaler.

        The scaler is configured with a feature range of 0 to 1, which is
        appropriate for neural network inputs.
        """
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def scale(self, data: np.ndarray) -> np.ndarray:
        """
        Scales the data using the MinMaxScaler.

        This method fits the scaler to the data and transforms it. The scaler
        parameters are stored for later use in inverse scaling.

        Args:
            data (numpy.ndarray): The data to be scaled. Can be 1D or 2D array.

        Returns:
            numpy.ndarray: Scaled data with values between 0 and 1.

        Raises:
            ValueError: If the input data contains non-finite values (NaN or inf)
                      or if any other issue occurs during fitting.
        """
        try:
            return self.scaler.fit_transform(data.reshape(-1, 1))
        except ValueError as e:
            raise ValueError(f"Data scaling error: {e}")

    def inverse_scale(self, data: np.ndarray) -> np.ndarray:
        """
        Reverts the scaling operation, transforming scaled data back to its original distribution.

        This method should be used on data that was previously scaled using the same
        instance of DataScaler.

        Args:
            data (numpy.ndarray): The scaled data to be inverse transformed.
                                Expected to contain values between 0 and 1.

        Returns:
            numpy.ndarray: The data returned to its original scale.

        Note:
            The method automatically flattens the output array for convenience
            in time series applications.
        """
        return self.scaler.inverse_transform(data.reshape(-1, 1)).flatten() 