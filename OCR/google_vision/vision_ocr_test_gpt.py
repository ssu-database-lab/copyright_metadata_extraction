from google.cloud import vision
from typing import Tuple
import requests
import os

# ✅ Force IPv4 for gRPC (fixes WSL2 IPv6 issues)
os.environ["GRPC_DNS_RESOLVER"] = "native"

# ✅ Set up Google Cloud credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "conversational_AI_order_agent/semiotic-pager-466612-t0-c587b9296fb8.json"

def ocr_image(file_bytes: bytes) -> Tuple[str, str]:
    """
    Returns (detected_text, implied_language)
    """
    client = vision.ImageAnnotatorClient()
    image = vision.Image(content=file_bytes)

    # text_detection() handles printed + handwritten
    resp = client.text_detection(image=image)
    if resp.error.message:
        raise RuntimeError(resp.error.message)

    text = resp.text_annotations[0].description if resp.text_annotations else ""
    lang_hints = {anno.locale for anno in resp.text_annotations[1:]}
    detected_lang = ",".join(lang_hints) or "unknown"
    return text, detected_lang


# assumed Messenger gave you image URL
image_url = "/home/mbmk92/projects/TFG/KakaoTalk_20250719_033537332.jpg"
with open(image_url, "rb") as image_file:
    img_bytes = image_file.read()

text, lang = ocr_image(img_bytes)
print("OCR:", text[:100], "lang:", lang)
