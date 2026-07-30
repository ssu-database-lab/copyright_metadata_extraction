# 공유마당 (gongu.copyright.or.kr) 다운로드 조사 보고서

조사일: 2026-07-20 · 방법: 4개 병렬 리서치 에이전트 + 실기기 라이브 검증 (curl, 익명 세션)
운영: 한국저작권위원회 (KCC). 서버렌더링 JSP — Playwright 불필요, plain HTTP로 전부 처리 가능.

## 한 줄 결론

**회원가입 없이 익명으로 목록 조회·상세 조회·원본 파일 다운로드가 전부 가능하다** (이미지·영상·어문 라이브 검증 완료).
회원가입(한국 휴대폰 본인인증 필요)은 검색 API 키·대량 DB 신청·AI 원천데이터 신청에만 필요.
KOGL(kogl.or.kr) 다운로더보다 **훨씬 단순** — SSO 로그인/팝업 처리 불필요, requests만으로 충분.

## 검증된 3단계 레시피 (2026-07-20 라이브 테스트)

```
1) 목록 → wrtSn 추출 (익명, 서버렌더링 HTML)
   GET /gongu/wrt/wrtCl/listWrtImage.do?menuNo=200018&pageIndex=1&sortSe=date&pageUnit=24
   → 200. grep -oE 'wrtSn=[0-9]+' 로 페이지당 24건. pageIndex 증가로 페이지네이션.

2) (선택) 상세 페이지에서 메타데이터
   GET /gongu/wrt/wrt/view.do?wrtSn={wrtSn}&menuNo={menuNo}
   → <dl><dt>라벨</dt><dd>값</dd> 구조: 저작물명/저작(권)자/이용조건/분류(장르)/해상도/카메라정보/원문파일명 등.
   라이선스는 <img src=".../img_licenseNN.png" alt="..."> 의 NN 또는 alt 텍스트로 파싱.

3) 원본 파일 다운로드 (익명!)
   GET /gongu/wrt/cmmn/wrtFileDownload.do?wrtSn={wrtSn}&fileSn=1
   → 200 + Content-Disposition: attachment. 사이트의 약관동의 체크박스·설문 팝업은
   클라이언트 JS 전용 — 서버는 강제하지 않음 (이미지·영상·텍스트 각각 검증).
```

### 코딩 시 필수 유의사항 (전부 실측)

| 항목 | 내용 |
|---|---|
| gzip | 서버가 JPEG/MP4까지 `content-encoding: gzip` 으로 보냄 → **`curl --compressed`** / requests는 자동 |
| 파일명 | Content-Disposition 의 한글이 **%-인코딩 UTF-8** (이미지) 또는 raw bytes(영상에서 관찰) → `urllib.parse.unquote(name)` 시도 후 실패 시 raw 처리 |
| Content-Length 없음 | chunked 전송 — 사전 크기 확인 불가, 스트리밍 저장 후 확인 (KOGL 영상과 동일한 상황) |
| 잘못된 fileSn | 404가 아니라 **HTTP 200 + 60바이트 HTML** `<script>alert('File Not Found[3]')...` → body 시작이 `<script>alert('File Not Found` 이거나 <200B 면 miss 판정; fileSn=1,2,3… 루프로 다중파일 저작물 수집 |
| 레이트리밋 | 미검출 (가벼운 프로브 기준). robots.txt 404, CAPTCHA 없음. 그래도 KOGL 때처럼 지터+보수적 동시성 권장 |
| 영상 | 동일 엔드포인트로 익명 다운로드 됨 (75MB MP4 검증, `ftyp mp42`). 스트리밍 iframe(streaming.copyright.or.kr:8080)은 미리보기용일 뿐 |

## 카테고리 목록 URL + 규모 (2026-07-20)

| 카테고리 | 목록 URL | menuNo | 총 건수 |
|---|---|---|---|
| 이미지 | `/gongu/wrt/wrtCl/listWrtImage.do?menuNo=200018` | 200018 | **1,389,252** |
| 영상 | `/gongu/wrt/wrtCl/listWrtVideo.do?menuNo=200026` | 200026 | **128,504** |
| 음악 | `/gongu/wrt/wrtCl/listWrtSound.do?menuNo=200020` | 200020 | (미조회) |
| 어문 | `/gongu/wrt/wrtCl/listWrtText.do?menuNo=200019` | 200019 | 만료(licenseCd=97)만 **52,110** |
| 기타(도형/SW) | `/gongu/wrt/wrtCl/listWrtEtc.do?menuNo=200023` | 200023 | 4,580 |
| 무료글꼴 | `/gongu/bbs/B0000018/list.do?menuNo=200195` (BBS) | 200195 | — |

공통 쿼리: `pageIndex`(1-base), `pageUnit`(12/24), `sortSe`(date/down/popular/recommand),
`wrtTy`, `depth2ClSn`, `licenseCd`, `searchSrcTrgetInttCd`(제공기관), `searchCrtDe`(창작연도), `searchWrd`.

## 필터 코드표

**licenseCd (라이선스):**
- `97` = **만료저작물** (퍼블릭 도메인 — 데이터셋 구축에 가장 안전)
- `98`=기증 자유이용, `99`=기증 이용허락 (콤보 `98,99`)
- CCL: `21`=BY, `22`=BY-ND, `23`=BY-SA, `24`=BY-NC, `25`=BY-NC-ND, `26`=BY-NC-SA (콤보 `20,21,...,27`)
- 공공누리: `01`~`04` = 제1~4유형 (콤보 `01,02,03,04`)

**wrtTy (1차 분류):** 이미지 `10004`=미술, `10006`=사진 · 기타 `10008`=도형, `10009`=컴퓨터프로그램

**depth2ClSn (2차 분류):**
- 이미지-미술: 10035 회화, 10036 캘리그라피, 10037 일러스트, 10038 조형, 10040 서예, 10042 만화
- 이미지-사진: 10045 풍경, 10046 식물, 10047 동물, 10048 사물, 10049 인물, 10050 광고, 10054 기타
- 영상: 10055 풍경 … 10061 공연, 10062 영화, 10063 다큐멘터리, 10064 뉴스, 10065 드라마, 10066 기타
- 음악: 10029 성악, 10030 대중음악, 10033 전통음악, 10128 효과음, 10139 국악 등

**파일유형 (category/d2WrtFileTyChk):** 01 텍스트, 02 이미지, 03 음원, 04 비디오, 05 기타/폰트

제공기관 코드(searchSrcTrgetInttCd)에는 국립중앙도서관(05), 문화체육관광부(07), 한국문화정보원(48),
국립현대미술관(19), 국립중앙박물관(38), 문화재청(33) 등 53개 — 조사 원문 참조.

## 공식 채널 (회원가입/신청 필요)

1. **data.go.kr Open API** — `한국저작권위원회_공유(만료)저작물 서비스_GW` (데이터셋 15158766)
   - `https://apis.data.go.kr/B552546/ShrWrtgService` — GET/XML, 6개 오퍼레이션 (이미지/멀티미디어/텍스트 × 목록/상세)
   - **만료저작물 ~16만 건만** 커버. 메타데이터 + **파일 URL 직접 반환** (limgpath 큰이미지, filepath 파일링크, 텍스트는 content 본문까지)
   - data.go.kr 활용신청 **자동승인** (한국 휴대폰 불필요 — 기존 data.go.kr 계정으로 가능), 개발계정 일 1,000건
2. **공유마당 자체 검색 API** — `GET /gongu/wrt/wrtApi/search.json?menuNo=..&pageUnit=3000&depth2ClSn=..&wrtFileTy=..&apiKey=KEY`
   - 전체 컬렉션 검색, 한 번에 3,000건. apiKey는 회원가입 후 **즉시 발급** (menuNo=200245). 스펙은 로그인 뒤에만 공개
   - 키 없으면 `{"msg":"apiKey값이 존재하지 않습니다"}` — 단, 목록 HTML 크롤링으로 완전 대체 가능(검증됨)
3. **대량 다운로드 / 대량 DB 신청** — `/gongu/useReqst/lqttDb2/forInsert.do?menuNo=200277` (로그인 필요, 기관·단체 대상 — 숭실대 해당). 수만 건 이상이면 이 공식 채널 권장
4. **인공지능 원천데이터 게시판** — menuNo=200311. **7.25M 건** AI 학습용 원천데이터(2022-2024 구축):
   민화, 한국적 웹툰, 태권도/택견 이미지, 궁궐·사찰 건축물, 국악 음원, K-POP 모션, 한국어 이미지+텍스트 멀티모달 등.
   데이터셋별 신청 버튼(로그인+신청서). **크롤링 없이 대규모 확보 가능한 지름길**

회원가입 = 저작권위원회 통합 One ID (oneid.copyright.or.kr): 만 14세+, **휴대폰 본인인증 또는 NICE i-PIN** 필수 (이메일만으로는 불가).

## 라이선스·약관 요점

- 만료(97): 퍼블릭 도메인 — 상업적 이용·재배포 제한 없음. **데이터셋 구축에 최우선**
- 기증(98/99): 국가 귀속, 원칙적 영리 이용 가능 (기증자가 비영리 조건 단 경우 제외; 99=이용허락은 담당자 승인 필요)
- 공공누리 1~4유형, CCL 6종: KOGL 데이터셋과 동일한 처리 (유형별 출처표시/상업/변경 조건을 인덱스에 기록)
- 이용약관(2014)에 크롤링 금지 조항 **없음**. 단 제12/17/20조에 "서비스로 얻은 정보의 무단 복제·재배포·상업이용 금지" 보일러플레이트 존재 — 저작물별 오픈 라이선스와 상충하는 구식 조항. 대량 수집 시 공식 대량 DB 채널 병행 신청이 깔끔한 방어책
- 다운로드 한도 문서화 없음. 정중한 속도(지터, 동시성 2-3)로

## 선행 사례

- **CarLifeContentBusinessDev/helper-scripts** (2026-04, Node/axios) — 최신·최적 기법: apiKey search.json(3,000건/호출) + wrtFileDownload.do 익명 다운로드, 300ms 지연, 재시도. 우리 레시피와 동일 결론
- leegeunhyeok/Python-GonguCrawler + jegalyongoh/Crawler (2018, BS4) — listWrt.do HTML 파싱, 패턴 여전히 유효
- 공유마당 소스 데이터셋: HF werty1248 1930년대 소설, AI-Hub 글자체(공유마당 폰트), 위키문헌 {{PD-공유마당}} 등

## 구현 계획 (우리 코드 재사용)

`api/module/dataset_builder/download_originals*.py`(KOGL)의 뼈대 재사용 — manifest 큐, skip-existing 재개,
지터, download_log 체크포인트, originals_index 빌드. 사이트 상호작용 계층만 교체:

- KOGL: Playwright SSO 로그인 + 팝업 폼 → **공유마당: requests 3줄** (목록 GET → wrtSn 정규식 → 다운로드 GET)
- 신규 모듈 제안: `api/module/dataset_builder/gongu_downloader.py`
  - CLI: `--category 이미지|영상|음악|어문 --license 97 --depth2 10045 --max N --out DIR`
  - 목록 페이지 크롤 → wrtSn 큐 → 상세 메타 파싱(dl/dt/dd) → fileSn 루프 다운로드 → gongu_index.xlsx
  - requests.Session + `Accept-Encoding: gzip` 자동, unquote 파일명, soft-404 판정, 지터 1-2.5s, 동시성 2-3

## 조사 원문

워크플로 전체 결과: 세션 tool-results/b2i6na4au.txt + tasks/wo06fjyy4.output (에이전트 5개: 사이트 구조,
Open API, 인증/약관, 선행 사례, 라이브 레시피 검증 — 총 121 tool call, 만료/기증/KOGL 라이선스 페이지·
data.go.kr·GitHub 근거 URL 포함)
