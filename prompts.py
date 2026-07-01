from __future__ import annotations

import re
from typing import Any, Dict, List, Optional


def make_system_text() -> str:
    return (
        "너는 도로 주행 장면의 위험도를 평가하는 보조 시스템이다.\n"
        "입력으로 (1) 이미지, (2) 객체 탐지+거리 기반 증거(JSON)가 주어진다.\n"
        "반드시 이미지와 증거를 함께 참고하여 한국어로만 답하라.\n"
        "\n"
        "핵심 목표:\n"
        "- 최종적으로 [Risk Grade]와 [Risk] 한 줄의 품질을 높이기 위해,\n"
        "  먼저 주의 포인트와 장면 사실을 간단히 정리한 뒤 결론을 내려라.\n"
        "\n"
        "작성 지침:\n"
        "1) [Decision-Critical Objects]\n"
        "   - 주행에 영향을 줄 수 있는 '주의 대상'을 1~6개 bullet로 작성하라.\n"
        "   - 각 bullet은 (무엇/어디/왜 주의) 순서로 1문장으로 쓰고, 가능하면 근거를 덧붙여라.\n"
        "   - 정확한 거리/개수/수치를 맞추려 하지 말고, '근거리/원거리', '전방/측면', '진로 근접' 같은 상대적 표현을 사용하라.\n"
        "   - 증거(JSON)의 category_name/거리/신뢰도 등은 참고하되, 모호하면 단정하지 말고 '가능성/우려' 수준으로 말하라.\n"
        "\n"
        "2) [Factual Scene Description]\n"
        "   - 장면을 '사실'로만 2~5개 bullet로 요약하라.\n"
        "   - 추측/감정/과장 금지. 보이는 것과 증거로 뒷받침되는 것만 작성하라.\n"
        "   - 수치가 불확실하면 수치 대신 상대적 표현을 사용하라.\n"
        "\n"
        "3) [Risk Grade]\n"
        "   - 반드시 L/M/H 중 하나만 단독으로 출력하라.\n"
        "\n"
        "4) [Risk]\n"
        "   - 한국어 한 줄로, Risk Grade의 핵심 근거를 요약하라.\n"
        "   - 과도한 단정 대신 관측된 위험 요인을 중심으로 간결히 작성하라.\n"
        "\n"
        "출력 형식(헤더/순서/대괄호 표기 그대로, 추가 섹션 금지):\n"
        "[Decision-Critical Objects]\n"
        "- ...\n"
        "\n"
        "[Factual Scene Description]\n"
        "- ...\n"
        "\n"
        "[Risk Grade]\n"
        "L 또는 M 또는 H\n"
        "\n"
        "[Risk]\n"
        "한 줄 문장\n"
    )


def make_user_text(evidence_str: str) -> str:
    return (
        "아래는 객체 탐지/거리 기반 증거(JSON)이다. 이 증거와 이미지를 근거로 위험도를 판단하라.\n"
        "반드시 지정된 4개 섹션 헤더를 모두 포함해 출력하라.\n"
        "\n"
        "[Evidence JSON]\n"
        f"{evidence_str}\n"
    )


def build_target_text(grade: str, risk_line: str) -> str:
    risk_line = str(risk_line).strip()
    risk_line = risk_line.splitlines()[0].strip() if risk_line else ""
    return (
        "[Risk Grade]\n"
        f"{grade}\n\n"
        "[Risk]\n"
        f"{risk_line}\n"
    )


def extract_between(text: str, start_pat: str, end_pat: Optional[str]) -> str:
    if end_pat:
        m = re.search(start_pat + r"(.*?)" + end_pat, text, flags=re.DOTALL)
    else:
        m = re.search(start_pat + r"(.*)$", text, flags=re.DOTALL)
    return (m.group(1).strip() if m else "").strip()


def parse_sections(raw: str) -> Dict[str, Any]:
    dc = extract_between(raw, r"\[Decision-Critical Objects\]\s*", r"\[Factual Scene Description\]")
    fs = extract_between(raw, r"\[Factual Scene Description\]\s*", r"\[Risk Grade\]")
    rg = extract_between(raw, r"\[Risk Grade\]\s*", r"\[Risk\]")
    rk = extract_between(raw, r"\[Risk\]\s*", None)

    grade = ""
    for line in rg.splitlines():
        s = line.strip().upper()
        if s in {"L", "M", "H"}:
            grade = s
            break
    if grade == "":
        m = re.search(r"\b([LMH])\b", rg.upper())
        if m:
            grade = m.group(1)

    risk_line = rk.strip().splitlines()[0].strip() if rk.strip() else ""

    return {
        "decision_critical_objects": dc,
        "factual_scene_description": fs,
        "risk_grade_pred": grade,
        "risk_pred": risk_line,
    }


def build_infer_messages(evidence_str: str) -> List[Dict[str, Any]]:
    return [
        {"role": "system", "content": make_system_text()},
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": make_user_text(evidence_str)},
            ],
        },
    ]
