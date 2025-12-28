@echo off
REM ========================================
REM NER Training & Evaluation Docker 실행 스크립트 (Windows Batch)
REM RTX 5070 최적화
REM ========================================

chcp 65001 >nul
setlocal enabledelayedexpansion

REM 설정
set IMAGE_NAME=ner-cuda129
set CONTAINER_NAME=ner-training-container
REM 작업 디렉토리 설정
set WORK_DIR=%~dp0
REM 마지막 백슬래시 제거 (있는 경우)
if "%WORK_DIR:~-1%"=="\" set WORK_DIR=%WORK_DIR:~0,-1%
set APP_DIR=/app

REM 모드 선택 (기본값: both)
set MODE=%1
if "%MODE%"=="" set MODE=both

REM continue_training 옵션 (기본값: true - 기존 모델 이어서 학습)
set CONTINUE_TRAINING=%2
if "%CONTINUE_TRAINING%"=="" set CONTINUE_TRAINING=true

echo ========================================
echo   NER Training ^& Evaluation Docker
echo ========================================
echo Image: %IMAGE_NAME%
echo Container: %CONTAINER_NAME%
echo Mode: %MODE%
echo Continue Training: %CONTINUE_TRAINING%
echo ========================================
echo.

REM 기존 컨테이너가 실행 중이면 중지 및 제거
echo [INFO] Checking for existing container...
docker ps -a --filter "name=%CONTAINER_NAME%" --format "{{.Names}}" | findstr /C:"%CONTAINER_NAME%" >nul
if %errorlevel% equ 0 (
    echo [INFO] Stopping and removing existing container...
    docker stop %CONTAINER_NAME% >nul 2>&1
    docker rm %CONTAINER_NAME% >nul 2>&1
)

REM 모델 저장 디렉토리 생성 (호스트)
set MODEL_SAVE_DIR=%WORK_DIR%models\saved
if not exist "%MODEL_SAVE_DIR%" mkdir "%MODEL_SAVE_DIR%"

echo [INFO] Starting Docker container...
    echo [INFO] Mode: %MODE%
    echo [INFO] Continue Training: %CONTINUE_TRAINING%
    echo [INFO] Model will be saved in local: models\ner_bilstm_crf_tf
    echo [INFO] Model location: %WORK_DIR%\models\ner_bilstm_crf_tf
echo.

REM 도커 실행 명령어 구성 (--rm 제거: 모델 복사를 위해 컨테이너 유지)
echo [INFO] Executing Docker command...
echo.

REM continue_training 옵션 설정
REM --continue-training은 action="store_true"이므로, false일 때는 인자를 전달하지 않음
set CONTINUE_ARG=
if "%CONTINUE_TRAINING%"=="true" (
    set CONTINUE_ARG=--continue-training
)

REM 변수 값 확인 및 검증
if "%IMAGE_NAME%"=="" (
    echo [ERROR] IMAGE_NAME is not set!
    exit /b 1
)
if "%CONTAINER_NAME%"=="" (
    echo [ERROR] CONTAINER_NAME is not set!
    exit /b 1
)

echo [DEBUG] IMAGE_NAME=%IMAGE_NAME%
echo [DEBUG] CONTAINER_NAME=%CONTAINER_NAME%
echo [DEBUG] MODE=%MODE%
echo [DEBUG] CONTINUE_TRAINING=%CONTINUE_TRAINING%
echo [DEBUG] WORK_DIR=%WORK_DIR%
echo.

REM 도커 실행 (명령어를 직접 실행)
REM continue_training이 true일 때만 --continue-training 전달
REM USE_CONTAINER_MODEL=false: 모델을 로컬(볼륨 마운트)에 저장
if "%CONTINUE_TRAINING%"=="true" (
    docker run --gpus all --name %CONTAINER_NAME% -v "%WORK_DIR%":%APP_DIR% -w %APP_DIR% -e TF_USE_LEGACY_KERAS=1 -e TF_FORCE_GPU_ALLOW_GROWTH=true -e TF_CPP_MIN_LOG_LEVEL=3 -e CUDA_VISIBLE_DEVICES=0 -e USE_CONTAINER_MODEL=false -e AUTO_DELETE_SLOW_MODEL=true %IMAGE_NAME% python3 ner_test.py --mode %MODE% --continue-training
) else (
    docker run --gpus all --name %CONTAINER_NAME% -v "%WORK_DIR%":%APP_DIR% -w %APP_DIR% -e TF_USE_LEGACY_KERAS=1 -e TF_FORCE_GPU_ALLOW_GROWTH=true -e TF_CPP_MIN_LOG_LEVEL=3 -e CUDA_VISIBLE_DEVICES=0 -e USE_CONTAINER_MODEL=false -e AUTO_DELETE_SLOW_MODEL=true %IMAGE_NAME% python3 ner_test.py --mode %MODE%
)

set EXIT_CODE=%errorlevel%

if %EXIT_CODE% equ 0 (
    echo.
    echo [INFO] ========================================
    echo [INFO] Training/Evaluation completed successfully!
    echo [INFO] ========================================
    echo.
    
    REM 모델은 이미 로컬(볼륨 마운트)에 저장되어 있음
    echo [INFO] Model is saved in local: models\ner_bilstm_crf_tf
    echo [INFO] Model location: %WORK_DIR%\models\ner_bilstm_crf_tf
    
    REM 컨테이너 정리
    echo [INFO] Cleaning up container...
    docker rm %CONTAINER_NAME% >nul 2>&1
    if %errorlevel% equ 0 (
        echo [INFO] Container removed successfully
    )
    
    echo.
    echo [INFO] ========================================
    echo [INFO] Process completed successfully!
    echo [INFO] ========================================
) else (
    echo.
    echo [ERROR] ========================================
    echo [ERROR] Process exited with code %EXIT_CODE%
    echo [ERROR] ========================================
    exit /b %EXIT_CODE%
)

endlocal

