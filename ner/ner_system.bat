@echo off
setlocal enabledelayedexpansion

echo ===================================================
echo NER 시스템 - 저작권 동의서 및 계약서 자동 처리 시스템
echo ===================================================

cd %~dp0ner

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
