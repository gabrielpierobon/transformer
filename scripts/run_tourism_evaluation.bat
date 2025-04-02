@echo off
echo ----------------------------------------
echo Tourism Dataset Evaluation for Transformer Models
echo ----------------------------------------

REM Parse command line parameters
set LOG_TRANSFORM=
set INCLUDE_NAIVE2=
if /i "%1"=="--log-transform" set LOG_TRANSFORM=--log-transform
if /i "%2"=="--log-transform" set LOG_TRANSFORM=--log-transform
if /i "%1"=="--include-naive2" set INCLUDE_NAIVE2=--include-naive2
if /i "%2"=="--include-naive2" set INCLUDE_NAIVE2=--include-naive2

REM Create directories if they don't exist
mkdir data\processed 2>nul
mkdir evaluation\tourism 2>nul
mkdir evaluation\tourism\plots 2>nul

REM Convert TSF file to CSV format
echo Converting TSF file to CSV format...
python scripts/convert_tourism_tsf_to_csv.py

REM Check if conversion was successful
if not exist data\processed\tourism_monthly_dataset.csv (
    echo Error: Failed to convert tourism dataset to CSV format.
    exit /b 1
)

REM Use the correct model for evaluation
set MODEL_NAME=transformer_1.0_directml_point_mse_M1_M48000_sampled2101_full_4epoch

REM Run evaluation with all 366 monthly series as per the paper
echo Running evaluation with all 366 monthly series (as per the paper)...
echo Options: %LOG_TRANSFORM% %INCLUDE_NAIVE2%

python scripts/evaluate_tourism.py --model-name %MODEL_NAME% --sample-size 366 --forecast-horizon 24 %LOG_TRANSFORM% %INCLUDE_NAIVE2%

echo Tourism dataset evaluation complete.
echo Results have been saved to evaluation/tourism/ directory. 