@echo off
setlocal enabledelayedexpansion

echo ===================================================
echo NER 시스템 - 저작권 동의서 및 계약서 자동 처리 시스템
echo ===================================================

REM 디렉토리 확인
if not exist "%~dp0ner" (
    echo 오류: ner 디렉토리가 존재하지 않습니다.
    echo 현재 스크립트 위치: %~dp0
    pause
    exit /b 1
)

cd %~dp0ner

REM 필요한 패키지 확인 및 설치
echo 필요한 패키지를 확인하고 설치합니다...

REM 필수 패키지 개별 확인 및 설치 (특정 버전 지정)
python -c "import importlib.util; pkg='transformers'; spec=importlib.util.find_spec(pkg); exit(0 if spec else 1)" 2>nul
if %errorlevel% neq 0 (
    echo transformers 패키지를 설치합니다...
    pip install transformers==4.28.1
)

python -c "import importlib.util; pkg='datasets'; spec=importlib.util.find_spec(pkg); exit(0 if spec else 1)" 2>nul
if %errorlevel% neq 0 (
    echo datasets 패키지를 설치합니다...
    pip install datasets==2.14.6
)

python -c "import importlib.util; pkg='torch'; spec=importlib.util.find_spec(pkg); exit(0 if spec else 1)" 2>nul
if %errorlevel% neq 0 (
    echo torch 패키지를 설치합니다...
    pip install torch
)

python -c "import importlib.util; pkg='pandas'; spec=importlib.util.find_spec(pkg); exit(0 if spec else 1)" 2>nul
if %errorlevel% neq 0 (
    echo pandas 패키지를 설치합니다...
    pip install pandas
)

python -c "import importlib.util; pkg='tqdm'; spec=importlib.util.find_spec(pkg); exit(0 if spec else 1)" 2>nul
if %errorlevel% neq 0 (
    echo tqdm 패키지를 설치합니다...
    pip install tqdm
)

python -c "import importlib.util; pkg='numpy'; spec=importlib.util.find_spec(pkg); exit(0 if spec else 1)" 2>nul
if %errorlevel% neq 0 (
    echo numpy 패키지를 설치합니다...
    pip install numpy
)

python -c "import importlib.util; pkg='scikit_learn'; spec=importlib.util.find_spec(pkg); exit(0 if spec else 1)" 2>nul
if %errorlevel% neq 0 (
    echo scikit_learn 패키지를 설치합니다...
    pip install scikit_learn
)

:menu
echo.
echo 작업 선택:
echo 1. NER 모델 학습 (전체)
echo 2. 예측 실행 (전체)
echo 0. 종료
echo.

set /p choice=선택하세요 (0, 1, 2): 

if "%choice%"=="1" (
    python train_model.py --doc-type all
    goto menu
)
if "%choice%"=="2" (
    python ner.py --predict-all
    goto menu
)
if "%choice%"=="0" (
    echo 프로그램을 종료합니다.
    goto end
)

echo 잘못된 선택입니다. 다시 시도하세요.
goto menu

:end
cd ..
endlocal
