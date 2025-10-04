#!/bin/bash

echo "==================================================="
echo "NER 시스템 - 저작권 동의서 및 계약서 자동 처리 시스템"
echo "==================================================="

# 디렉토리 경로
SCRIPT_DIR="$(dirname "$0")"
NER_DIR="$SCRIPT_DIR/ner"

# ner 디렉토리로 이동
cd "$NER_DIR" || exit

# 필요한 패키지 확인 및 설치 함수
check_and_install_package() {
    python3 -c "import $1" 2>/dev/null
    if [ $? -ne 0 ]; then
        if [ "$2" != "" ]; then
            echo "$1 패키지를 설치합니다 (버전 $2)..."
            pip install $1==$2
        else
            echo "$1 패키지를 설치합니다..."
            pip install $1
        fi
    fi
}

# 필수 패키지 확인 및 설치
echo "필요한 패키지를 확인하고 설치합니다..."
check_and_install_package transformers "4.28.1"
check_and_install_package datasets "2.14.6"
check_and_install_package torch ""
check_and_install_package pandas ""
check_and_install_package tqdm ""
check_and_install_package numpy ""
check_and_install_package scikit_learn ""

function show_menu {
    echo ""
    echo "작업 선택:"
    echo "1. NER 모델 학습 (전체)"
    echo "2. 예측 실행 (전체)"
    echo "0. 종료"
    echo ""
}

while true; do
    show_menu
    read -p "선택하세요 (0, 1, 2): " choice
    
    case $choice in
        1)
            python train_model.py --doc-type all
            ;;
        2)
            python ner.py --predict-all
            ;;
        0)
            echo "프로그램을 종료합니다."
            cd ..
            exit 0
            ;;
        *)
            echo "잘못된 선택입니다. 다시 시도하세요."
            ;;
    esac
done
