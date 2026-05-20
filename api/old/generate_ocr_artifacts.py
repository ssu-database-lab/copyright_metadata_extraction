import os
import json
import random
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import sys
import pytesseract

# Import ocr module
try:
    import ocr
except ImportError:
    sys.path.append(os.getcwd())
    import ocr

def render_dense_document(title: str, content_lines: list, font_path: str, out_path: str, noise_level=200, blur_radius=0.1, angle_range=0.1):
    """
    Render a dense, formal-looking consent form.
    """
    # A4 High Res: 2480 x 3508 (300 DPI) -> Let's use a bit smaller for speed but high enough quality
    # 1240 x 1754 (150 DPI)
    W, H = 1240, 1754
    img = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(img)
    
    try:
        title_font = ImageFont.truetype(font_path, 60)
        header_font = ImageFont.truetype(font_path, 32)
        body_font = ImageFont.truetype(font_path, 28)
        small_font = ImageFont.truetype(font_path, 22)
    except:
        title_font = ImageFont.load_default()
        header_font = ImageFont.load_default()
        body_font = ImageFont.load_default()
        small_font = ImageFont.load_default()

    # Margins
    margin_x = 100
    current_y = 100

    # 1. Title (Centered)
    bbox = draw.textbbox((0, 0), title, font=title_font)
    title_w = bbox[2] - bbox[0]
    draw.text(((W - title_w) // 2, current_y), title, fill="black", font=title_font)
    current_y += 120

    # 2. Header Table-like info
    draw.line((margin_x, current_y, W - margin_x, current_y), fill="black", width=2)
    current_y += 20
    header_text = "문서번호 : 2025-SEC-001   |   보존기간 : 5년   |   부서 : 법무팀"
    draw.text((margin_x, current_y), header_text, fill="black", font=header_font)
    current_y += 50
    draw.line((margin_x, current_y, W - margin_x, current_y), fill="black", width=2)
    current_y += 60

    # 3. Body Content (Dense)
    line_spacing = 20
    
    for line_text in content_lines:
        # Check if it's a section header (starts with number)
        if line_text[0].isdigit() and line_text[1] == '.':
            current_y += 20 # Extra space before section
            font_to_use = header_font
        else:
            font_to_use = body_font
            
        # Word wrap
        words = line_text.split()
        line_buffer = ""
        for word in words:
            test_line = line_buffer + word + " "
            bbox = draw.textbbox((0, 0), test_line, font=font_to_use)
            if bbox[2] > (W - margin_x * 2):
                draw.text((margin_x, current_y), line_buffer, fill="black", font=font_to_use)
                line_buffer = word + " "
                current_y += (40 + line_spacing)
            else:
                line_buffer = test_line
        
        if line_buffer:
            draw.text((margin_x, current_y), line_buffer, fill="black", font=font_to_use)
            current_y += (40 + line_spacing)

    # 4. Footer / Signature (Bottom Right)
    footer_y = H - 400
    
    date_text = "2025년  12월  05일"
    draw.text((W - margin_x - 300, footer_y), date_text, fill="black", font=header_font)
    
    footer_y += 60
    sign_text = "성명 :   홍  길  동     (인)"
    draw.text((W - margin_x - 300, footer_y), sign_text, fill="black", font=header_font)
    
    footer_y += 60
    sign_text2 = "서명 : _________________"
    draw.text((W - margin_x - 300, footer_y), sign_text2, fill="black", font=header_font)

    # 5. Apply Noise (Subtle Scan Effect)
    # We want it to look like a scan but still be readable (80-85% acc)
    # Slight rotation
    angle = random.uniform(-angle_range, angle_range)
    img = img.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor="white")
    
    # Blur
    if blur_radius > 0:
        img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    
    # Noise
    # Fast noise
    if noise_level > 0:
        px = img.load()
        for _ in range(noise_level):
            x = random.randint(0, W - 1)
            y = random.randint(0, H - 1)
            px[x, y] = (100, 100, 100) # Gray noise

    img.save(out_path)
    return img

def generate_artifacts():
    output_dir = os.path.join("out", "asdf")
    os.makedirs(output_dir, exist_ok=True)
    print(f"[Info] Output directory: {output_dir}")

    # Setup OCR
    try:
        ocr.setup_tesseract()
        font_path = ocr.pick_font_path() # Malgun Gothic
    except Exception as e:
        print(f"[Error] Setup failed: {e}")
        return

    # Define 3 Documents
    docs = [
        {
            "title": "개인정보 수집 및 이용 동의서",
            "content": [
                "1. 수집하는 개인정보 항목",
                "회사는 회원가입, 상담, 서비스 신청 등을 위해 아래와 같은 개인정보를 수집하고 있습니다.",
                "- 수집항목 : 이름, 생년월일, 성별, 로그인ID, 비밀번호, 자택 전화번호, 자택 주소, 휴대전화번호, 이메일, 직업, 회사명, 부서, 직책, 회사전화번호, 취미, 결혼여부, 기념일, 법정대리인정보, 주민등록번호, 신용카드 정보, 은행계좌 정보, 서비스 이용기록, 접속 로그, 쿠키, 접속 IP 정보, 결제기록",
                "- 개인정보 수집방법 : 홈페이지(회원가입), 서면양식, 팩스, 전화, 상담 게시판, 이메일, 이벤트 응모, 배송요청",
                "2. 개인정보의 수집 및 이용목적",
                "회사는 수집한 개인정보를 다음의 목적을 위해 활용합니다.",
                "- 서비스 제공에 관한 계약 이행 및 서비스 제공에 따른 요금정산 콘텐츠 제공, 구매 및 요금 결제, 물품배송 또는 청구지 등 발송",
                "- 회원 관리 : 회원제 서비스 이용에 따른 본인확인, 개인 식별, 불량회원의 부정 이용 방지와 비인가 사용 방지, 가입 의사 확인, 연령확인, 만14세 미만 아동 개인정보 수집 시 법정 대리인 동의여부 확인, 불만처리 등 민원처리, 고지사항 전달",
                "3. 개인정보의 보유 및 이용기간",
                "회사는 개인정보 수집 및 이용목적이 달성된 후에는 예외 없이 해당 정보를 지체 없이 파기합니다.",
                "귀하는 위와 같은 개인정보 수집 및 이용에 동의하지 않을 권리가 있으며, 동의를 거부할 경우 서비스 이용에 제한이 있을 수 있습니다."
            ]
        },
        {
            "title": "저작물 이용 허락 계약서",
            "content": [
                "1. 계약의 목적",
                "본 계약은 저작권자(이하 '갑')가 본인의 저작물을 이용자(이하 '을')에게 이용 허락함에 있어 필요한 제반 사항을 규정함을 목적으로 한다.",
                "2. 이용 허락의 범위",
                "갑은 을에게 다음과 같은 범위 내에서 저작물의 이용을 허락한다.",
                "- 이용 매체 : 온라인 웹사이트, 모바일 어플리케이션, 홍보용 인쇄물",
                "- 이용 기간 : 계약 체결일로부터 3년",
                "- 이용 지역 : 대한민국 및 전 세계",
                "3. 저작권료의 지급",
                "을은 갑에게 본 저작물의 이용 대가로 금 일천만원(￦10,000,000)을 계약 체결 후 14일 이내에 현금으로 지급한다.",
                "4. 저작인격권의 존중",
                "을은 저작물을 이용함에 있어 갑의 성명표시권을 준수하여야 하며, 저작물의 내용, 형식 및 제호의 동일성을 유지하여야 한다. 단, 부득이한 경우 갑의 사전 동의를 얻어 변경할 수 있다.",
                "5. 계약의 해지",
                "당사자 일방이 본 계약을 위반하는 경우, 상대방은 14일의 기간을 정하여 시정을 최고하고, 그 기간 내에 시정되지 아니하는 경우 본 계약을 해지할 수 있다."
            ]
        },
        {
            "title": "제3자 정보 제공 동의서",
            "content": [
                "1. 제공받는 자",
                "주식회사 데이터솔루션, (주)마케팅파트너스",
                "2. 제공받는 자의 이용 목적",
                "신규 서비스 개발 및 맞춤형 서비스 제공, 이벤트 및 광고성 정보 제공, 인구통계학적 특성에 따른 서비스 제공 및 광고 게재",
                "3. 제공하는 개인정보 항목",
                "성명, 생년월일, 성별, 휴대전화번호, 이메일 주소, 서비스 이용 기록, 구매 기록",
                "4. 보유 및 이용 기간",
                "제공받는 자의 이용 목적 달성 시까지 (단, 관계 법령에 정해진 규정에 따라 법정 기간 동안 보관)",
                "5. 동의 거부 권리 및 불이익",
                "귀하는 개인정보의 제3자 제공에 대한 동의를 거부할 권리가 있습니다. 다만, 동의를 거부할 경우 제휴사가 제공하는 혜택 및 맞춤형 서비스 제공이 제한될 수 있습니다.",
                "본인은 위와 같이 개인정보를 제3자에게 제공하는 것에 동의합니다."
            ]
        }
    ]

    # Define settings per doc to target 80-85% accuracy
    settings = [
        {"noise": 3000, "blur": 1.0, "angle": 0.5}, # Doc 1
        {"noise": 300, "blur": 0.1, "angle": 0.1}, # Doc 2
        {"noise": 0,   "blur": 0.1, "angle": 0.0}  # Doc 3
    ]

    accuracy_report = {}

    for i, doc in enumerate(docs):
        idx = i + 1
        print(f"\n[Processing Document #{idx}: {doc['title']}]...")
        
        # 1. Render Image
        img_filename = f"consent_form_{idx}.png"
        img_path = os.path.join(output_dir, img_filename)
        
        # Apply specific settings
        s = settings[i]
        img = render_dense_document(
            doc['title'], 
            doc['content'], 
            font_path, 
            img_path,
            noise_level=s["noise"],
            blur_radius=s["blur"],
            angle_range=s["angle"]
        )
        print(f"  -> Generated Image: {img_path}")

        # 2. Run OCR
        # Use PSM 3 (Auto) or 6 (Block) for full page
        ocr_text = pytesseract.image_to_string(img, lang="kor+eng", config="--psm 6")
        
        # 3. Save OCR Text
        txt_filename = f"consent_form_{idx}_ocr.txt"
        txt_path = os.path.join(output_dir, txt_filename)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(ocr_text)
        print(f"  -> Saved OCR Text: {txt_path}")

        # 4. Calculate Accuracy (vs Ground Truth)
        # Construct full GT text from content list
        gt_text = doc['title'] + "\n" + "\n".join(doc['content'])
        
        # Normalize for comparison
        c = ocr.cer(gt_text, ocr_text)
        acc = (1 - c) * 100.0
        
        print(f"  -> Real Accuracy: {acc:.2f}%")
        
        # Store in report
        accuracy_report[img_filename] = {
            "title": doc['title'],
            "ocr_text_file": txt_filename,
            "accuracy": round(acc, 2),
            "cer": round(c, 4)
        }

    # 5. Save Accuracy Report
    json_path = os.path.join(output_dir, "accuracy_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(accuracy_report, f, ensure_ascii=False, indent=2)
    print(f"\n[Done] Accuracy report saved to {json_path}")

if __name__ == "__main__":
    generate_artifacts()
