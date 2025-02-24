# Remove any existing CUDA paths from PATH
$env:PATH = ($env:PATH.Split(';') | Where-Object { 
    -not ($_ -like '*CUDA*' -or $_ -like '*CUDNN*')
}) -join ';'

# Set CUDA paths
$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8"
$env:CUDNN_PATH = "C:\Program Files\NVIDIA\CUDNN\v9.7\bin\11.8"

# Add CUDA and cuDNN to PATH (at the beginning to take precedence)
$env:PATH = "$env:CUDA_PATH\bin;$env:CUDNN_PATH;$env:PATH"

Write-Host "CUDA and cuDNN environment variables set:"
Write-Host "CUDA_PATH: $env:CUDA_PATH"
Write-Host "CUDNN_PATH: $env:CUDNN_PATH"

# Unset conflicting CUDA version variables
Remove-Item Env:CUDA_PATH_V12_5 -ErrorAction SilentlyContinue 