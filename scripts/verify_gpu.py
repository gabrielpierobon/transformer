import tensorflow as tf
import os
import sys
import platform
import time
import numpy as np

def verify_installations():
    print("\n=== System Information ===")
    print(f"Python version: {sys.version}")
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Platform: {platform.platform()}")
    
    print("\n=== DirectML Information ===")
    try:
        import tensorflow_directml as tfdml
        print("TensorFlow DirectML plugin is installed")
        print(f"DirectML plugin version: {tfdml.__version__}")
    except ImportError:
        print("TensorFlow DirectML plugin is not installed")
    
    print("\n=== Device Information ===")
    print("Physical Devices:", tf.config.list_physical_devices())
    print("GPU/DML Devices:", tf.config.list_physical_devices('GPU'))
    print("CPU Devices:", tf.config.list_physical_devices('CPU'))
    
    print("\n=== TensorFlow Build Information ===")
    print("Built with GPU support:", tf.test.is_built_with_gpu_support())
    
    print("\n=== Basic GPU Test ===")
    try:
        # Simple addition on GPU
        with tf.device('/GPU:0'):
            x = tf.constant([1.0, 2.0, 3.0])
            y = tf.constant([4.0, 5.0, 6.0])
            z = x + y
            print("\nGPU Test - Vector Addition:")
            print("x:", x.numpy())
            print("y:", y.numpy())
            print("x + y:", z.numpy())
            print("\nGPU test successful!")
    except Exception as e:
        print("\nGPU test failed with error:", str(e))
    
    print("\n=== Performance Comparison ===")
    size = 1000
    
    # Create data
    data = np.random.rand(size, size).astype(np.float32)
    
    def run_test(device, data):
        times = []
        with tf.device(device):
            tensor = tf.constant(data)
            # Warmup
            _ = tf.reduce_sum(tensor)
            
            # Timed runs
            for _ in range(3):
                start = time.time()
                result = tf.reduce_sum(tensor)
                _ = result.numpy()  # Force execution
                times.append(time.time() - start)
        return np.mean(times)
    
    try:
        # GPU test
        gpu_time = run_test('/GPU:0', data)
        print(f"\nGPU average time: {gpu_time:.4f} seconds")
        
        # CPU test
        cpu_time = run_test('/CPU:0', data)
        print(f"CPU average time: {cpu_time:.4f} seconds")
        
        if gpu_time < cpu_time:
            speedup = cpu_time / gpu_time
            print(f"GPU is {speedup:.2f}x faster than CPU")
        else:
            print("Warning: GPU is not faster than CPU")
            
    except Exception as e:
        print("\nPerformance test failed with error:", str(e))

if __name__ == "__main__":
    verify_installations() 