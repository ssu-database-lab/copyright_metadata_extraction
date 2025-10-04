@echo off
REM Batch script to download Hugging Face models
REM Usage: download_hf_model.bat <model_id> [output_directory]

if "%1"=="" (
    echo Usage: download_hf_model.bat ^<model_id^> [output_directory]
    echo Example: download_hf_model.bat microsoft/DialoGPT-medium
    echo Example: download_hf_model.bat bert-base-uncased C:\my_models
    exit /b 1
)

set MODEL_ID=%1
set OUTPUT_DIR=%2

if "%OUTPUT_DIR%"=="" (
    set OUTPUT_DIR=C:\hf_models
)

echo Downloading model: %MODEL_ID%
echo Output directory: %OUTPUT_DIR%

python download_hf_model.py "%MODEL_ID%" --output-dir "%OUTPUT_DIR%"

if %ERRORLEVEL% EQU 0 (
    echo Download completed successfully!
) else (
    echo Download failed!
    exit /b 1
)
