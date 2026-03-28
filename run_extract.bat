@echo off
chcp 65001 >nul 2>&1
title Copyright Metadata Extraction

echo ======================================================================
echo   Copyright Metadata Extraction Tool
echo   Soongsil University Database Lab
echo ======================================================================
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python 3.9+ from:
    echo         https://www.python.org/downloads/
    echo         Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)

:: Check if requirements are installed
python -c "import fastapi" >nul 2>&1
if errorlevel 1 (
    echo [INFO] Installing dependencies... This may take a few minutes.
    pip install -r requirements.txt
    if errorlevel 1 (
        echo [ERROR] Failed to install dependencies.
        pause
        exit /b 1
    )
    echo [OK] Dependencies installed.
    echo.
)

:: Get input file
if "%~1"=="" (
    echo Usage: Drag a PDF file onto this .bat file, or enter the path below.
    echo.
    set /p INPUT_FILE="Enter file or folder path: "
) else (
    set INPUT_FILE=%~1
)

if "%INPUT_FILE%"=="" (
    echo [ERROR] No file specified.
    pause
    exit /b 1
)

:: Get document type
echo.
echo Document types:
echo   1. 계약서 (Contract)
echo   2. 동의서 (Consent)
echo   3. 저작재산권 양도동의서 (Transfer)
echo   4. 공공저작물 자유이용허락 동의서 (Public License)
echo   5. 기타문서 (Other) [default]
echo.
set /p DOC_CHOICE="Select document type (1-5) [5]: "

if "%DOC_CHOICE%"=="1" set DOC_TYPE=계약서
if "%DOC_CHOICE%"=="2" set DOC_TYPE=동의서
if "%DOC_CHOICE%"=="3" set DOC_TYPE=저작재산권 양도동의서
if "%DOC_CHOICE%"=="4" set DOC_TYPE=공공저작물 자유이용허락 동의서
if "%DOC_TYPE%"=="" set DOC_TYPE=기타문서

:: Get pipeline stages
echo.
echo Pipeline stages:
echo   1. Full pipeline (OCR + LLM + NER + Consolidation) [default]
echo   2. OCR only
echo   3. OCR + NER only
echo   4. OCR + LLM only
echo   5. OCR + LLM + NER (no consolidation)
echo.
set /p STAGE_CHOICE="Select stages (1-5) [1]: "

if "%STAGE_CHOICE%"=="2" set STAGES=ocr
if "%STAGE_CHOICE%"=="3" set STAGES=ocr+ner
if "%STAGE_CHOICE%"=="4" set STAGES=ocr+llm
if "%STAGE_CHOICE%"=="5" set STAGES=ocr+llm+ner
if "%STAGES%"=="" set STAGES=all

:: Run extraction
echo.
echo ======================================================================
echo   Starting extraction...
echo   File: %INPUT_FILE%
echo   Type: %DOC_TYPE%
echo   Stages: %STAGES%
echo ======================================================================
echo.

python extract.py "%INPUT_FILE%" -t "%DOC_TYPE%" -s %STAGES% -o ./extraction_results

echo.
echo ======================================================================
echo   Done! Results saved to: extraction_results\
echo ======================================================================

:: Open results folder
if exist extraction_results (
    explorer extraction_results
)

pause
