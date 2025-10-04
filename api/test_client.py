#!/usr/bin/env python3
"""
REST API 테스트 클라이언트

서버에 PDF 파일을 전송하고 NER 결과를 받습니다.

사용법:
    python test_client.py <pdf_file_path>
    
예시:
    python test_client.py document/7.저작물양도계약서.pdf
"""

import sys
import json
import requests
from pathlib import Path
from datetime import datetime


def test_health(base_url: str = "http://localhost:5000"):
    """서버 상태 확인"""
    print("="*60)
    print("서버 상태 확인")
    print("="*60)
    
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✓ 서버 정상 작동")
            print(f"  - Status: {data['status']}")
            print(f"  - Version: {data['version']}")
            print(f"  - Timestamp: {data['timestamp']}")
            return True
        else:
            print(f"✗ 서버 응답 오류: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ 서버 연결 실패: {e}")
        return False


def process_pdf(pdf_path: str, base_url: str = "http://localhost:5000"):
    """PDF 파일을 서버로 전송하고 결과 받기"""
    pdf_file = Path(pdf_path)
    
    if not pdf_file.exists():
        print(f"✗ 파일을 찾을 수 없습니다: {pdf_path}")
        return None
    
    print("\n" + "="*60)
    print(f"PDF 처리 요청: {pdf_file.name}")
    print("="*60)
    print(f"파일 크기: {pdf_file.stat().st_size / (1024*1024):.2f} MB")
    print()
    
    try:
        # 파일 업로드
        with open(pdf_file, 'rb') as f:
            files = {'file': (pdf_file.name, f, 'application/pdf')}
            
            print("⏳ 서버로 전송 중...")
            start_time = datetime.now()
            
            response = requests.post(
                f"{base_url}/process",
                files=files,
                timeout=300  # 5분 타임아웃
            )
            
            end_time = datetime.now()
            elapsed = (end_time - start_time).total_seconds()
        
        print(f"✓ 응답 수신 완료 ({elapsed:.1f}초)")
        print()
        
        if response.status_code == 200:
            result = response.json()
            
            print("="*60)
            print("처리 결과")
            print("="*60)
            print(f"✓ 성공: {result['success']}")
            print(f"✓ Request ID: {result['request_id']}")
            print(f"✓ 처리 시간: {result['processing_time_seconds']}초")
            print(f"✓ 로그 파일: {result['log_file']}")
            print()
            
            # 단계별 결과
            print("-"*60)
            print("단계별 처리 결과:")
            print("-"*60)
            
            steps = result.get('steps', {})
            
            if 'pdf_to_image' in steps:
                step = steps['pdf_to_image']
                print(f"1. PDF → 이미지: {'✓' if step['success'] else '✗'}")
                print(f"   - 생성된 이미지: {step.get('images_created', 0)}개")
            
            if 'ocr' in steps:
                step = steps['ocr']
                print(f"2. OCR 처리: {'✓' if step['success'] else '✗'}")
                print(f"   - 처리된 파일: {step.get('files_processed', 0)}개")
            
            if 'ner' in steps:
                step = steps['ner']
                print(f"3. NER 분석: {'✓' if step['success'] else '✗'}")
                print(f"   - 처리된 파일: {step.get('files_processed', 0)}개")
                print(f"   - 추출된 엔티티: {step.get('total_entities', 0)}개")
            
            print()
            
            # 엔티티 결과
            entities = result.get('entities', [])
            entity_count = result.get('entity_count', 0)
            
            if entity_count > 0:
                print("-"*60)
                print(f"추출된 엔티티: {entity_count}개")
                print("-"*60)
                
                # 타입별 그룹화
                entities_by_type = {}
                for entity, entity_type in entities:
                    if entity_type not in entities_by_type:
                        entities_by_type[entity_type] = []
                    entities_by_type[entity_type].append(entity)
                
                # 타입별 출력 (상위 5개 타입)
                for entity_type, items in sorted(
                    entities_by_type.items(), 
                    key=lambda x: len(x[1]), 
                    reverse=True
                )[:5]:
                    print(f"\n[{entity_type}] ({len(items)}개)")
                    for item in items[:10]:  # 각 타입당 최대 10개
                        item_preview = item[:50] + '...' if len(item) > 50 else item
                        item_preview = item_preview.replace('\n', ' ')
                        print(f"  - {item_preview}")
                    
                    if len(items) > 10:
                        print(f"  ... 외 {len(items) - 10}개")
                
                # 전체 결과를 JSON 파일로 저장
                output_file = Path("result") / f"{result['request_id']}_result.json"
                output_file.parent.mkdir(exist_ok=True)
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                
                print()
                print(f"✓ 전체 결과 저장: {output_file}")
            
            return result
            
        else:
            print(f"✗ 처리 실패: {response.status_code}")
            try:
                error_data = response.json()
                print(f"  오류: {error_data.get('error', 'Unknown error')}")
            except:
                print(f"  응답: {response.text}")
            return None
            
    except requests.exceptions.Timeout:
        print("✗ 요청 시간 초과 (5분 이상)")
        return None
    except Exception as e:
        print(f"✗ 오류 발생: {e}")
        return None


def view_logs(base_url: str = "http://localhost:5000", limit: int = 10):
    """서버 로그 조회"""
    print("\n" + "="*60)
    print(f"최근 요청 로그 (최대 {limit}개)")
    print("="*60)
    
    try:
        response = requests.get(f"{base_url}/logs?limit={limit}")
        
        if response.status_code == 200:
            data = response.json()
            logs = data.get('logs', [])
            
            if not logs:
                print("로그가 없습니다.")
                return
            
            for i, log in enumerate(logs, 1):
                print(f"\n[{i}] {log.get('timestamp', 'N/A')}")
                print(f"  - Request ID: {log.get('request_id', 'N/A')}")
                print(f"  - Filename: {log.get('filename', 'N/A')}")
                print(f"  - Success: {log.get('success', False)}")
                print(f"  - Entities: {log.get('entity_count', 0)}")
                print(f"  - Processing time: {log.get('processing_time_seconds', 0)}s")
                
                if log.get('error'):
                    print(f"  - Error: {log['error']}")
        else:
            print(f"✗ 로그 조회 실패: {response.status_code}")
            
    except Exception as e:
        print(f"✗ 오류 발생: {e}")


def main():
    """메인 함수"""
    base_url = "http://localhost:5000"
    
    print("="*60)
    print("REST API 테스트 클라이언트")
    print("="*60)
    print()
    
    # 서버 상태 확인
    if not test_health(base_url):
        print("\n서버가 실행 중인지 확인하세요:")
        print("  python call.py")
        return
    
    # PDF 파일 처리
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        result = process_pdf(pdf_path, base_url)
        
        if result:
            # 로그 조회
            view_logs(base_url, limit=5)
    else:
        print("\n사용법:")
        print("  python test_client.py <pdf_file_path>")
        print("\n예시:")
        print("  python test_client.py document/7.저작물양도계약서.pdf")
        print()
        
        # 로그만 조회
        view_logs(base_url, limit=10)


if __name__ == '__main__':
    main()
