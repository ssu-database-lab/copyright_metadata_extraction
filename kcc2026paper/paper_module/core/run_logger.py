"""학습 실행 전 과정을 txt로 기록하는 로거.

왜 필요한가:
- 훈련이 끝난 뒤 "그래프를 다른 방식으로 그리고 싶다" 싶을 때
  모델을 다시 돌리지 않고도 이 로그 파일에서 데이터를 뽑아 새 그래프를 만들 수 있도록.
- 하이퍼파라미터, 데이터셋 통계, 스텝별 loss, 평가 결과를 전부 남긴다.

포맷:
- `[섹션이름]` 헤더 + `key = value` 줄의 단순한 텍스트 (INI 유사).
- 같은 섹션 이름이 여러 번 나올 수 있다 (예: [train:0], [train:50] ...).
- `#` 으로 시작하면 주석.

사용 예:
    from paper_module.core.run_logger import open_log, log_section, close_log

    fp = open_log("data/out/runs/demo/run.txt")
    log_section(fp, "hparams", {"lr": 2e-5, "epochs": 5})
    log_section(fp, "dataset", {"train_size": 111381, "val_size": 27845})
    for step in range(1000):
        log_section(fp, f"train:{step}", {"loss": 0.5, "lr": 2e-5})
    log_section(fp, "eval:500", {"loss": 0.3, "f1": 0.72})
    log_section(fp, "final", {"best_step": 500, "best_f1": 0.72})
    close_log(fp)

읽어서 그래프로 만들기:
    from paper_module.core.run_logger import parse_log, log_to_history
    history = log_to_history("data/out/runs/demo/run.txt")
    # history → plot_training_curve 에 바로 넣을 수 있는 형태
"""
from datetime import datetime
from pathlib import Path


# =======================================================
# Writer 측 — 훈련 중에 한 줄씩 기록하는 함수들
# =======================================================

def open_log(path):
    """로그 파일을 연다 (기존 파일 있으면 덮어쓴다).

    Args:
        path: 파일 경로 문자열.

    Returns:
        파일 핸들 (이후 log_section / close_log 에 넘겨준다).
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fp = open(p, "w", encoding="utf-8")
    fp.write(f"# Run started: {datetime.now().isoformat(timespec='seconds')}\n")
    fp.write(f"# Log format: [section] blocks with key = value pairs\n")
    fp.write("\n")
    fp.flush()
    return fp


def log_section(fp, section_name, data):
    """한 섹션을 로그에 기록한다.

    Args:
        fp: open_log가 돌려준 파일 핸들.
        section_name: 섹션 이름 (예: "hparams", "train:100", "eval:500").
                      같은 이름이 여러 번 나와도 괜찮다.
        data: {key: value} 딕셔너리. value는 뭐든 str()로 변환되어 저장된다.
    """
    fp.write(f"[{section_name}]\n")
    for key, value in data.items():
        # 값에 줄바꿈이 들어있으면 로그가 깨지므로 공백으로 치환
        value_str = str(value).replace("\n", " ").replace("\r", " ")
        fp.write(f"{key} = {value_str}\n")
    fp.write("\n")
    fp.flush()


def close_log(fp):
    """로그 파일을 닫는다. 종료 시각을 남긴다."""
    fp.write(f"# Run ended: {datetime.now().isoformat(timespec='seconds')}\n")
    fp.close()


# =======================================================
# Reader 측 — 완료된 로그 파일에서 데이터를 뽑아내는 함수들
# =======================================================

def parse_log(path):
    """로그 파일을 읽어 섹션 단위 dict로 반환한다.

    Args:
        path: 로그 파일 경로 문자열.

    Returns:
        {section_name: [row_dict, row_dict, ...]}
        같은 이름이 여러 번 나왔을 경우 리스트에 쌓인다.
        예: {"train:0": [{"loss": "0.5", "lr": "2e-5"}],
             "train:100": [{"loss": "0.3", ...}], ...}
    """
    sections = {}
    current_section = None
    current_data = {}

    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        # 빈 줄이나 주석(#)은 건너뛴다
        if not line or line.startswith("#"):
            continue
        # 섹션 헤더: [name]
        if line.startswith("[") and line.endswith("]"):
            # 이전 섹션이 있으면 먼저 저장
            if current_section is not None:
                sections.setdefault(current_section, []).append(current_data)
            current_section = line[1:-1]
            current_data = {}
        # key = value 줄
        elif "=" in line:
            k, v = line.split("=", 1)
            current_data[k.strip()] = v.strip()

    # 파일 끝 — 마지막 섹션도 저장
    if current_section is not None:
        sections.setdefault(current_section, []).append(current_data)

    return sections


def log_to_history(path):
    """로그에서 학습 곡선용 history 리스트를 추출한다.

    Args:
        path: 로그 파일 경로.

    Returns:
        [{step, train_loss, val_loss, val_f1}, ...] 형태의 리스트.
        train/eval 기록이 있는 step 만 포함된다.
    """
    sections = parse_log(path)

    # 섹션 이름에 ":스텝번호" 로 저장되므로 파싱해서 step 별로 모은다
    train_data = {}   # step → train metrics
    eval_data = {}    # step → eval metrics

    for section_name, rows in sections.items():
        if ":" not in section_name:
            continue
        prefix, step_str = section_name.split(":", 1)
        try:
            step = int(step_str)
        except ValueError:
            continue
        # 각 섹션의 첫 번째 row만 사용 (한 스텝에 한 번만 기록한다고 가정)
        row = rows[0] if rows else {}
        if prefix == "train":
            train_data[step] = row
        elif prefix == "eval":
            eval_data[step] = row

    # train / eval 의 스텝을 합쳐서 시간순으로 history 구성
    all_steps = sorted(set(train_data.keys()) | set(eval_data.keys()))
    history = []
    for step in all_steps:
        entry = {"step": step}
        t = train_data.get(step, {})
        e = eval_data.get(step, {})
        # float 변환: 실패하면 None (예: 값이 str 그대로일 때)
        if "loss" in t:
            entry["train_loss"] = _to_float(t["loss"])
        if "loss" in e:
            entry["val_loss"] = _to_float(e["loss"])
        if "f1" in e:
            entry["val_f1"] = _to_float(e["f1"])
        history.append(entry)

    return history


def log_to_f1_dict(path, section_name="final"):
    """로그의 특정 섹션에서 라벨별 F1 점수를 추출한다.

    섹션에 `f1_per_label.라벨이름 = 값` 형태로 저장된 엔트리를 읽는다.
    예: [final]
        f1_per_label.name = 0.89
        f1_per_label.address = 0.82

    Args:
        path: 로그 파일 경로.
        section_name: F1 추출할 섹션 이름 (기본 "final").

    Returns:
        {라벨명: F1} 딕셔너리. (plot_f1_per_label에 바로 넣을 수 있음)
    """
    sections = parse_log(path)
    rows = sections.get(section_name, [])
    if not rows:
        return {}
    row = rows[0]

    f1_dict = {}
    prefix = "f1_per_label."
    for key, value in row.items():
        if key.startswith(prefix):
            label_name = key[len(prefix):]
            f = _to_float(value)
            if f is not None:
                f1_dict[label_name] = f
    return f1_dict


# =======================================================
# 내부 도우미
# =======================================================

def _to_float(s):
    """문자열을 float로 변환 실패 시 None 반환 (예외 먹힘)."""
    try:
        return float(s)
    except (ValueError, TypeError):
        return None
