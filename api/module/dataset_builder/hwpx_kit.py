"""HWPX(한글 OWPML) 조각을 만들고 읽고 쓰는 최소 도구 — 계약서 생성기(generate_contracts_v5)용.

HWPX 는 zip 이다. 첫 항목 mimetype 은 무압축(STORED)이어야 하고, 나머지는 deflate 로 다시 쓴다.
입력 zip 의 ZipInfo 를 재사용하면 압축 정보가 섞여 "too many length or distance symbols" 가 나므로
파트마다 새 ZipInfo 를 만든다.

본문(section0.xml) 문단은 <hp:p> 로 만든다. 줄 배치 캐시(<hp:linesegarray>)는 넣지 않는다 —
한글이 열 때 다시 계산한다.
"""
from __future__ import annotations

import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence, Union
from xml.sax.saxutils import escape

Run = tuple[int, str]  # (charPrIDRef, 글자)


def esc(text: str) -> str:
    return escape(str(text), {'"': "&quot;"})


def read_parts(path: Union[str, Path]) -> dict[str, bytes]:
    with zipfile.ZipFile(path) as z:
        return {n: z.read(n) for n in z.namelist()}


def write_parts(parts: dict[str, bytes], path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    order = ["mimetype"] + [n for n in parts if n != "mimetype"]
    with zipfile.ZipFile(path, "w") as z:
        for name in order:
            info = zipfile.ZipInfo(name, date_time=(2026, 11, 3, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED if name == "mimetype" else zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            z.writestr(info, parts[name])


# ------------------------------------------------------------------ 문단
def para(runs: Union[str, Sequence[Run]], *, pr: int = 0, char: int = 0, page_break: bool = False) -> str:
    """문단 하나. runs 가 문자열이면 char 글자 모양 하나로 쓴다. 빈 문자열이면 빈 줄."""
    if isinstance(runs, str):
        runs = [(char, runs)]
    body = "".join(f'<hp:run charPrIDRef="{c}"><hp:t>{esc(t)}</hp:t></hp:run>' if t else f'<hp:run charPrIDRef="{c}"/>'
                   for c, t in runs) or f'<hp:run charPrIDRef="{char}"/>'
    return (f'<hp:p id="0" paraPrIDRef="{pr}" styleIDRef="0" pageBreak="{int(page_break)}" columnBreak="0" merged="0">'
            f"{body}</hp:p>")


def blank(char: int = 0, pr: int = 0) -> str:
    return para("", pr=pr, char=char)


# ------------------------------------------------------------------ 표
@dataclass
class Cell:
    paras: list[str]                 # para() 결과들
    col_span: int = 1
    row_span: int = 1
    border_fill: int = 4             # 4 = 실선 테두리 (원본 템플릿)
    v_align: str = "CENTER"


@dataclass
class Table:
    rows: list[list[Cell]]
    col_widths: list[int]            # HWPUNIT (1 mm ≈ 283.5)
    row_height: int = 1900
    table_id: int = 1900000000
    rows_heights: Optional[list[int]] = field(default=None)
    page_break: str = "CELL"       # CELL = 쪽에서 셀 단위로 나눔, NONE = 나누지 않고 통째로 다음 쪽

    def xml(self, anchor_char: int = 0, pr: int = 0) -> str:
        n_rows, n_cols = len(self.rows), len(self.col_widths)
        heights = self.rows_heights or [self.row_height] * n_rows
        width = sum(self.col_widths)
        out = [f'<hp:p id="0" paraPrIDRef="{pr}" styleIDRef="0" pageBreak="0" columnBreak="0" merged="0">'
               f'<hp:run charPrIDRef="{anchor_char}">'
               f'<hp:tbl id="{self.table_id}" zOrder="0" numberingType="TABLE" textWrap="TOP_AND_BOTTOM" '
               f'textFlow="BOTH_SIDES" lock="0" dropcapstyle="None" pageBreak="{self.page_break}" repeatHeader="1" '
               f'rowCnt="{n_rows}" colCnt="{n_cols}" cellSpacing="0" borderFillIDRef="4" noAdjust="0">'
               f'<hp:sz width="{width}" widthRelTo="ABSOLUTE" height="{sum(heights)}" heightRelTo="ABSOLUTE" protect="0"/>'
               '<hp:pos treatAsChar="0" affectLSpacing="0" flowWithText="1" allowOverlap="0" holdAnchorAndSO="0" '
               'vertRelTo="PARA" horzRelTo="COLUMN" vertAlign="TOP" horzAlign="LEFT" vertOffset="0" horzOffset="0"/>'
               '<hp:outMargin left="283" right="283" top="283" bottom="283"/>'
               '<hp:inMargin left="510" right="510" top="141" bottom="141"/>']
        occupied: set[tuple[int, int]] = set()
        for r, row in enumerate(self.rows):
            out.append("<hp:tr>")
            c = 0
            for cell in row:
                while (r, c) in occupied:
                    c += 1
                w = sum(self.col_widths[c:c + cell.col_span])
                h = sum(heights[r:r + cell.row_span])
                for dr in range(cell.row_span):
                    for dc in range(cell.col_span):
                        occupied.add((r + dr, c + dc))
                out.append(
                    f'<hp:tc name="" header="0" hasMargin="0" protect="0" editable="0" dirty="0" borderFillIDRef="{cell.border_fill}">'
                    f'<hp:subList id="" textDirection="HORIZONTAL" lineWrap="BREAK" vertAlign="{cell.v_align}" '
                    'linkListIDRef="0" linkListNextIDRef="0" textWidth="0" textHeight="0" hasTextRef="0" hasNumRef="0">'
                    + "".join(cell.paras) +
                    f'</hp:subList><hp:cellAddr colAddr="{c}" rowAddr="{r}"/>'
                    f'<hp:cellSpan colSpan="{cell.col_span}" rowSpan="{cell.row_span}"/>'
                    f'<hp:cellSz width="{w}" height="{h}"/>'
                    '<hp:cellMargin left="510" right="510" top="141" bottom="141"/></hp:tc>')
                c += cell.col_span
            out.append("</hp:tr>")
        out.append(f'</hp:tbl></hp:run><hp:run charPrIDRef="{anchor_char}"><hp:t/></hp:run></hp:p>')
        return "".join(out)


# ------------------------------------------------------------------ header.xml 손보기
def _bump_count(xml: str, tag: str, delta: int) -> str:
    return re.sub(rf'(<hh:{tag} itemCnt=")(\d+)(")', lambda m: f"{m.group(1)}{int(m.group(2)) + delta}{m.group(3)}", xml, count=1)


def clone_char_pr(header: str, src_id: int, new_id: int, *, height: Optional[int] = None, bold: Optional[bool] = None,
                  color: Optional[str] = None) -> str:
    """글자 모양 src_id 를 복제해 new_id 로 덧붙인다 (크기·굵게·색만 바꾼다)."""
    m = re.search(rf'<hh:charPr id="{src_id}" .*?</hh:charPr>', header, flags=re.S)
    if not m:
        raise ValueError(f"charPr {src_id} 없음")
    node = m.group(0).replace(f'<hh:charPr id="{src_id}"', f'<hh:charPr id="{new_id}"', 1)
    if height is not None:
        node = re.sub(r'height="\d+"', f'height="{height}"', node, count=1)
    if color is not None:
        node = re.sub(r'textColor="[^"]+"', f'textColor="{color}"', node, count=1)
    if bold is True and "<hh:bold/>" not in node:
        node = node.replace("<hh:underline", "<hh:bold/><hh:underline", 1)
    if bold is False:
        node = node.replace("<hh:bold/>", "")
    end = header.index("</hh:charProperties>")
    return _bump_count(header[:end] + node + header[end:], "charProperties", 1)


def add_border_fill(header: str, new_id: int, *, solid: bool = True, face: Optional[str] = None) -> str:
    """테두리(실선 0.12 mm) + 선택 배경색 face(#RRGGBB) 의 borderFill 을 덧붙인다."""
    line = 'type="SOLID" width="0.12 mm" color="#000000"' if solid else 'type="NONE" width="0.1 mm" color="#000000"'
    fill = (f'<hc:fillBrush><hc:winBrush faceColor="{face}" hatchColor="#999999" alpha="0"/></hc:fillBrush>'
            if face else "")
    node = (f'<hh:borderFill id="{new_id}" threeD="0" shadow="0" centerLine="NONE" breakCellSeparateLine="0">'
            '<hh:slash type="NONE" Crooked="0" isCounter="0"/><hh:backSlash type="NONE" Crooked="0" isCounter="0"/>'
            f'<hh:leftBorder {line}/><hh:rightBorder {line}/><hh:topBorder {line}/><hh:bottomBorder {line}/>'
            f'<hh:diagonal type="SOLID" width="0.1 mm" color="#000000"/>{fill}</hh:borderFill>')
    end = header.index("</hh:borderFills>")
    return _bump_count(header[:end] + node + header[end:], "borderFills", 1)


def set_char_color(header: str, ids: Iterable[int], color: str = "#000000") -> str:
    for i in ids:
        header = re.sub(rf'(<hh:charPr id="{i}" height="\d+" textColor=")[^"]+(")', rf"\g<1>{color}\g<2>", header, count=1)
    return header


def set_print_method(settings: str, method: int = 0) -> str:
    """settings.xml 의 인쇄 방식. 원본은 4(2쪽 모아 찍기)라 PDF 가 가로 한 장에 두 쪽씩 나온다 → 0(기본)."""
    return re.sub(r'(<config:config-item name="PrintMethod" type="short">)\d+(<)', rf"\g<1>{method}\g<2>", settings)


def section_shell(section_xml: str) -> tuple[str, str, str]:
    """section0.xml → (머리: 루트 여는 태그까지, 첫 문단(쪽 설정 secPr), 꼬리: 루트 닫는 태그)."""
    start = section_xml.index("<hp:p ")
    first_end = section_xml.index("</hp:p>", start) + len("</hp:p>")
    tail_start = section_xml.rindex("</hs:sec>")
    return section_xml[:start], section_xml[start:first_end], section_xml[tail_start:]
