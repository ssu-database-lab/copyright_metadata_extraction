"""
합성 개인정보(PII) 생성기 — 계약서 당사자용 (전부 가짜, 학습/평가 데이터 전용).

설계 (사용자 결정 2026-06):
- 갑(저작권자): 매니페스트의 실제 저작권자명 + 소속 유지 + **합성 담당자 연락처**
  (담당자명/부서/직급/연락처/이메일/주소, 기관이면 사업자등록번호).
- 을(이용자): **완전 합성** — 개인 또는 법인. 개인이면 주민등록번호, 법인이면
  사업자등록번호 포함 (사용자가 둘 다 포함 요청; 가짜 생성 허용).
- 결정론적: 원문인덱스로 seed → 재실행 시 동일 결과.

주민등록번호·사업자등록번호는 형식이 유효(체크섬 통과)한 가짜 값이다. 실제 인물/기관과
무관하며, 합성 학습 데이터임을 전제로 한다.
"""

from __future__ import annotations

# 기관(법인)으로 볼 수 있는 명칭 마커
_INSTITUTION_MARKERS = (
    "원", "관", "회", "사", "청", "부", "처", "구청", "시청", "도청", "대학교", "대학",
    "박물관", "진흥원", "재단", "위원회", "연구소", "연구원", "공사", "공단", "센터",
    "협회", "조합", "학회", "편집부", "출판", "서관", "방송", "신문",
)
_DEPARTMENTS = ["저작권관리팀", "콘텐츠사업팀", "법무팀", "디지털정보팀", "문화사업부",
                "아카이브팀", "홍보팀", "지식재산팀", "정보화사업팀", "기획조정실"]
_TITLES = ["담당", "주임", "대리", "과장", "차장", "팀장", "연구원", "선임연구원"]


def is_institution(*names: str) -> bool:
    return any(any(m in (n or "") for m in _INSTITUTION_MARKERS) for n in names)


def _fk(seed: int):
    from faker import Faker
    fk = Faker("ko_KR")
    fk.seed_instance(seed)
    return fk


def _mobile(fk) -> str:
    return f"010-{fk.numerify('####')}-{fk.numerify('####')}"


def _bizno(fk) -> str:
    """사업자등록번호 XXX-XX-XXXXX — 표준 체크섬을 만족하는 가짜 값."""
    d = [fk.random_int(0, 9) for _ in range(9)]
    w = [1, 3, 7, 1, 3, 7, 1, 3, 5]
    s = sum(a * b for a, b in zip(d, w)) + (d[8] * 5) // 10
    chk = (10 - (s % 10)) % 10
    n = "".join(map(str, d + [chk]))
    return f"{n[:3]}-{n[3:5]}-{n[5:]}"


def make_licensee(idx_seed: int) -> dict:
    """을(이용자) — 완전 합성. 약 55% 법인 / 45% 개인."""
    fk = _fk(idx_seed * 2 + 1)
    if fk.boolean(chance_of_getting_true=55):  # 법인
        return {
            "을_유형": "법인", "이용자명": fk.company(), "을_대표자": fk.name(),
            "을_사업자등록번호": _bizno(fk), "을_주민등록번호": "", "을_생년월일": "",
            "을_주소": fk.address().replace("\n", " "),
            "을_전화": fk.phone_number(), "을_휴대폰": _mobile(fk), "을_이메일": fk.email(),
        }
    return {  # 개인
        "을_유형": "개인", "이용자명": fk.name(), "을_대표자": "",
        "을_사업자등록번호": "", "을_주민등록번호": fk.ssn(),
        "을_생년월일": fk.date_of_birth(minimum_age=25, maximum_age=70).strftime("%Y-%m-%d"),
        "을_주소": fk.address().replace("\n", " "),
        "을_전화": "", "을_휴대폰": _mobile(fk), "을_이메일": fk.email(),
    }


def make_licensor_contact(idx_seed: int, owner_name: str, affiliation: str = "") -> dict:
    """갑(저작권자)의 합성 담당자 연락처. 실제 저작권자명/소속은 호출측에서 유지."""
    fk = _fk(idx_seed * 2)
    inst = is_institution(owner_name, affiliation)
    return {
        "갑_담당자": fk.name(),
        "갑_부서": fk.random_element(_DEPARTMENTS),
        "갑_직급": fk.random_element(_TITLES),
        "갑_연락처": _mobile(fk),
        "갑_이메일": fk.email(),
        "갑_주소": fk.address().replace("\n", " "),
        "갑_사업자등록번호": _bizno(fk) if inst else "",
    }


if __name__ == "__main__":
    # 스모크 테스트 + 사업자등록번호 체크섬 검증
    def valid_bizno(s: str) -> bool:
        n = [int(c) for c in s.replace("-", "")]
        if len(n) != 10:
            return False
        w = [1, 3, 7, 1, 3, 7, 1, 3, 5]
        chk = (10 - ((sum(a * b for a, b in zip(n[:9], w)) + (n[8] * 5) // 10) % 10)) % 10
        return chk == n[9]

    for idx in (88616, 66885, 112761):
        print(f"\n=== idx={idx} ===")
        lor = make_licensor_contact(idx, "한국문화예술교육진흥원")
        lee = make_licensee(idx)
        print("갑 담당자:", lor)
        print("을:", lee)
        bn = lee.get("을_사업자등록번호") or lor.get("갑_사업자등록번호")
        if bn:
            print(f"사업자등록번호 {bn} 체크섬 유효: {valid_bizno(bn)}")
