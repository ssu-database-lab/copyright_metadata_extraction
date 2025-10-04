#!/bin/bash

echo "==================================================="
echo "NER 시스템 - 저작권 동의서 및 계약서 자동 처리 시스템"
echo "==================================================="

# Change to the ner directory
cd "$(dirname "$0")/ner" || exit

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
