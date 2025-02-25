import unittest
import numpy as np
import requests
import json
import sys
import os
from pathlib import Path
import subprocess
import time
import signal
import atexit

# Add the project root to the path
sys.path.append(str(Path(__file__).parent.parent))

# Import function to generate synthetic data
from scripts.predict_custom import generate_synthetic_time_series

class TestForecastAPI(unittest.TestCase):
    """Test cases for the forecast API."""
    
    @classmethod
    def setUpClass(cls):
        """Start the API server before running tests."""
        # Check if the server is already running
        try:
            response = requests.get("http://localhost:8000/health")
            if response.status_code == 200:
                print("API server is already running.")
                cls.server_process = None
                return
        except requests.exceptions.ConnectionError:
            pass
        
        # Start the server
        print("Starting API server...")
        api_file = Path(__file__).parent / "forecast_api.py"
        cls.server_process = subprocess.Popen(
            [sys.executable, str(api_file)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Register cleanup function to stop the server
        atexit.register(cls.tearDownClass.__func__, cls)
        
        # Wait for the server to start
        max_retries = 10
        for i in range(max_retries):
            try:
                response = requests.get("http://localhost:8000/health")
                if response.status_code == 200:
                    print("API server started successfully.")
                    break
            except requests.exceptions.ConnectionError:
                if i == max_retries - 1:
                    raise Exception("Failed to start API server.")
                time.sleep(1)
    
    @classmethod
    def tearDownClass(cls):
        """Stop the API server after running tests."""
        if cls.server_process is not None:
            print("Stopping API server...")
            cls.server_process.terminate()
            cls.server_process.wait(timeout=5)
            cls.server_process = None
    
    def test_health_endpoint(self):
        """Test the health endpoint."""
        response = requests.get("http://localhost:8000/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "healthy"})
    
    def test_point_forecast(self):
        """Test point forecast."""
        # Generate synthetic data
        series = generate_synthetic_time_series(length=100, seasonality=12, trend=0.01, noise=0.1)
        
        # Prepare request payload
        payload = {
            "values": series.tolist(),
            "model_type": "point",
            "n_steps": 12,
            "sequence_length": 60
        }
        
        # Make API call
        response = requests.post("http://localhost:8000/forecast/", json=payload)
        
        # Check response
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        # Validate response structure
        self.assertIn("forecast", data)
        self.assertIn("dates", data)
        self.assertEqual(len(data["forecast"]), 12)
        self.assertEqual(len(data["dates"]), 12)
        
        # Validate forecast values
        self.assertTrue(all(isinstance(x, (int, float)) for x in data["forecast"]))
    
    def test_probabilistic_forecast(self):
        """Test probabilistic forecast."""
        # Generate synthetic data
        series = generate_synthetic_time_series(length=100, seasonality=12, trend=0.01, noise=0.1)
        
        # Prepare request payload
        payload = {
            "values": series.tolist(),
            "model_type": "probabilistic",
            "n_steps": 12,
            "sequence_length": 60,
            "loss_type": "hybrid",
            "loss_alpha": 0.8,
            "low_bound_conf": 25,
            "high_bound_conf": 75,
            "num_samples": 1000
        }
        
        # Make API call
        response = requests.post("http://localhost:8000/forecast/", json=payload)
        
        # Check response
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        # Validate response structure
        self.assertIn("forecast", data)
        self.assertIn("lower_bound", data)
        self.assertIn("upper_bound", data)
        self.assertIn("dates", data)
        self.assertEqual(len(data["forecast"]), 12)
        self.assertEqual(len(data["lower_bound"]), 12)
        self.assertEqual(len(data["upper_bound"]), 12)
        self.assertEqual(len(data["dates"]), 12)
        
        # Validate forecast values
        self.assertTrue(all(isinstance(x, (int, float)) for x in data["forecast"]))
        self.assertTrue(all(isinstance(x, (int, float)) for x in data["lower_bound"]))
        self.assertTrue(all(isinstance(x, (int, float)) for x in data["upper_bound"]))
        
        # Validate confidence intervals
        for i in range(len(data["forecast"])):
            self.assertLessEqual(data["lower_bound"][i], data["forecast"][i])
            self.assertGreaterEqual(data["upper_bound"][i], data["forecast"][i])
    
    def test_invalid_request(self):
        """Test invalid request."""
        # Prepare invalid request payload (empty values)
        payload = {
            "values": [],
            "model_type": "point",
            "n_steps": 12
        }
        
        # Make API call
        response = requests.post("http://localhost:8000/forecast/", json=payload)
        
        # Check response
        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertIn("detail", data)
        self.assertEqual(data["detail"], "No time series values provided")
    
    def test_short_series(self):
        """Test with a short time series that needs padding."""
        # Generate short synthetic data
        series = generate_synthetic_time_series(length=10, seasonality=12, trend=0.01, noise=0.1)
        
        # Prepare request payload
        payload = {
            "values": series.tolist(),
            "model_type": "point",
            "n_steps": 12,
            "sequence_length": 60
        }
        
        # Make API call
        response = requests.post("http://localhost:8000/forecast/", json=payload)
        
        # Check response
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        # Validate response structure
        self.assertIn("forecast", data)
        self.assertEqual(len(data["forecast"]), 12)

if __name__ == "__main__":
    unittest.main() 