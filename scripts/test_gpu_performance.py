import tensorflow as tf
import time
import os
import sys
import platform

def print_environment_info():
    print("\n=== System Information ===")
    print(f"Python version: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"TensorFlow version: {tf.__version__}")
    
    print("\n=== TensorFlow Build Information ===")
    print(f"Is built with CUDA: {tf.test.is_built_with_cuda()}")
    print(f"Is built with GPU support: {tf.test.is_built_with_gpu_support()}")
    
    print("\n=== CUDA Environment Variables ===")
    cuda_vars = ['CUDA_PATH', 'CUDA_HOME', 'CUDNN_PATH', 'CUDA_VISIBLE_DEVICES']
    for var in cuda_vars:
        print(f"{var}: {os.environ.get(var, 'Not Set')}")
    
    print("\n=== GPU Information ===")
    physical_devices = tf.config.list_physical_devices()
    print("All physical devices:", physical_devices)
    
    gpus = tf.config.list_physical_devices('GPU')
    print("GPU devices:", gpus)
    
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("\nGPU memory growth enabled")
        except RuntimeError as e:
            print("\nError configuring GPU:", e)

def run_performance_test():
    print("\n=== Performance Test ===")
    matrix_size = 2000
    print(f"Performing {matrix_size}x{matrix_size} matrix multiplication...")
    
    # Generate matrices once to ensure fair comparison
    a = tf.random.normal([matrix_size, matrix_size])
    b = tf.random.normal([matrix_size, matrix_size])
    
    # Test on GPU
    try:
        with tf.device('/GPU:0'):
            # Warmup run
            _ = tf.matmul(a, b)
            
            # Timed run
            start = time.time()
            c_gpu = tf.matmul(a, b)
            gpu_time = time.time() - start
            print(f"\nGPU Time: {gpu_time:.4f} seconds")
    except Exception as e:
        print(f"\nGPU test failed: {str(e)}")
        c_gpu = None
        gpu_time = float('inf')

    # Test on CPU
    try:
        with tf.device('/CPU:0'):
            # Warmup run
            _ = tf.matmul(a, b)
            
            # Timed run
            start = time.time()
            c_cpu = tf.matmul(a, b)
            cpu_time = time.time() - start
            print(f"CPU Time: {cpu_time:.4f} seconds")
    except Exception as e:
        print(f"\nCPU test failed: {str(e)}")
        return

    # Compare results
    if c_gpu is not None:
        print(f"\nResults match: {tf.reduce_all(tf.abs(c_gpu - c_cpu) < 1e-5)}")
        if gpu_time < cpu_time:
            speedup = cpu_time / gpu_time
            print(f"GPU is {speedup:.2f}x faster than CPU")
        else:
            print("Warning: GPU is not faster than CPU, suggesting it might not be properly utilized")

if __name__ == "__main__":
    print_environment_info()
    run_performance_test() 