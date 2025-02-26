"""
ModelLoader class for loading transformer models from different sources.

This class handles loading models from either local storage or MLflow,
with support for custom loss functions and model configurations.
"""

import logging
import os
from typing import Optional

import tensorflow as tf

logger = logging.getLogger(__name__)


class ModelLoader:
    """
    A class for loading transformer models from different sources.

    This class provides functionality to load models from either local storage
    or MLflow, with support for custom loss functions and configurations.

    Attributes:
        model_path (str): Path to the model, either local path or MLflow URI
        model_server_type (str): Type of server hosting the model ('local' or 'mlflow')
        custom_loss (Optional[str]): Name of custom loss function if used
    """

    def __init__(
        self,
        model_path: str,
        model_server_type: str,
        custom_loss: Optional[str] = None,
    ):
        """
        Initialize the ModelLoader.

        Args:
            model_path: Path to the model (local path or MLflow URI)
            model_server_type: Type of server hosting the model ('local' or 'mlflow')
            custom_loss: Name of custom loss function if used
        """
        self.model_path = model_path
        self.model_server_type = model_server_type.lower()
        self.custom_loss = custom_loss

        if self.model_server_type not in ["local", "mlflow"]:
            raise ValueError(
                f"Invalid model_server_type: {model_server_type}. Must be either 'local' or 'mlflow'",
            )

    def _load_local_model(self) -> tf.keras.Model:
        """
        Load a model from local storage.

        Returns:
            tf.keras.Model: Loaded model

        Raises:
            FileNotFoundError: If the model file doesn't exist
            ValueError: If there's an error loading the model
        """
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at path: {self.model_path}")

        try:
            custom_objects = {}
            if self.custom_loss == "gaussian_nll":
                from src.models.transformer import gaussian_nll
                custom_objects["loss"] = gaussian_nll
                custom_objects["gaussian_nll"] = gaussian_nll

            model = tf.keras.models.load_model(
                self.model_path,
                custom_objects=custom_objects,
            )
            logger.info(f"Successfully loaded local model from {self.model_path}")
            return model

        except Exception as e:
            raise ValueError(f"Error loading local model: {str(e)}")

    def _load_mlflow_model(self) -> tf.keras.Model:
        """
        Load a model from MLflow.

        Returns:
            tf.keras.Model: Loaded model

        Raises:
            ImportError: If mlflow is not installed
            ValueError: If there's an error loading the model
        """
        try:
            import mlflow
        except ImportError:
            raise ImportError(
                "mlflow is required to load models from MLflow. Install it with: pip install mlflow",
            )

        try:
            custom_objects = {}
            if self.custom_loss == "gaussian_nll":
                from src.models.transformer import gaussian_nll
                custom_objects["loss"] = gaussian_nll
                custom_objects["gaussian_nll"] = gaussian_nll

            model = mlflow.tensorflow.load_model(
                self.model_path,
                custom_objects=custom_objects,
            )
            logger.info(f"Successfully loaded MLflow model from {self.model_path}")
            return model

        except Exception as e:
            raise ValueError(f"Error loading MLflow model: {str(e)}")

    def load_model(self) -> tf.keras.Model:
        """
        Load the model from the specified source.

        Returns:
            tf.keras.Model: Loaded model

        Raises:
            ValueError: If there's an error loading the model
        """
        logger.info(f"Loading model from {self.model_server_type} source: {self.model_path}")

        if self.model_server_type == "local":
            return self._load_local_model()
        else:  # mlflow
            return self._load_mlflow_model() 