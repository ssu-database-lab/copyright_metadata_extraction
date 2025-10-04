import io
import os
import json
import shutil
from google.cloud import vision
from google.protobuf.json_format import MessageToDict  # JSON 변환 모듈

os.environ["GRPC_DNS_RESOLVER"] = "native"

# ✅ 1. 서비스 계정 인증 설정 (Colab 환경)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "conversational_AI_order_agent/semiotic-pager-466612-t0-c587b9296fb8.json"
# path to the IMAGE folder
image_folder = "images"

# ✅ 2. Vision API 클라이언트 생성
client = vision.ImageAnnotatorClient()

# ✅ 3. OCR 적용할 이미지 경로
# getting the image from input and navigate to the directory
image_path = input("Enter the image path: ")  # ✅ OCR 적용할 이미지
# save the image to the directory images/   
os.makedirs(image_folder, exist_ok=True)
# save the image to the directory images/
shutil.copy(image_path, image_folder)

# navigate to the directory
os.chdir(image_folder) 

# ✅ 4. 이미지 로드
with io.open(image_path, "rb") as image_file:
    content = image_file.read()
image = vision.Image(content=content)


# ✅ 5. OCR 요청 (문서 내 텍스트 검출)
response = client.document_text_detection(image=image)

# ✅ 6. OCR 결과 추출 및 출력
if response.text_annotations:
    extracted_text = response.text_annotations[0].description  # ✅ 전체 텍스트 가져오기
    print("\n🔹 Extracted OCR Text:\n")
    print(extracted_text)

    # ✅ 7. JSON 변환 후 저장 (🚨 기존 오류 해결)
    response_dict = MessageToDict(response._pb)  # ✅ `_pb` 사용하여 변환

    # ✅ 8. OCR 결과 저장 (텍스트)
    with open("ocr_result.txt", "w", encoding="utf-8") as text_file:
        text_file.write(extracted_text)
    print(f"\n✅ OCR 텍스트 저장 완료: ocr_result.txt")

    # ✅ 9. OCR 전체 결과 저장 (JSON)
    with open("ocr_result.json", "w", encoding="utf-8") as json_file:
        json.dump(response_dict, json_file, indent=4, ensure_ascii=False)
    print(f"\n✅ OCR 결과 JSON 저장 완료: ocr_result.json")

else:
    print("\n❌ OCR 결과 없음")