"""agents/_e_utils.py 회귀 테스트.

get_generator()는 세 노드가 모두 llm=None으로 그래프에 등록될 때(app.py 실제 실행 경로)
쓰인다. 노드 테스트는 전부 가짜 LLM을 직접 넘기기 때문에 이 경로가 가려져서,
import config 누락(NameError)이 pytest에서는 안 잡히고 실제 실행에서만 터진 적이 있다.
이 테스트는 get_generator()를 직접 불러서 그 경로를 확인한다.

ChatOpenAI를 실제로 생성하면 초기화 중 네트워크 호출(토크나이저 등)이 일어날 수 있어
pytest가 멈출 수 있다. langchain_openai.ChatOpenAI를 가짜로 바꿔 네트워크를 타지 않는다.
"""
import sys
import types

import pytest

import config
from agents._e_utils import get_generator


class _FakeChatOpenAI:
    def __init__(self, model):
        self.model = model


@pytest.fixture(autouse=True)
def fake_chat_openai(monkeypatch):
    fake_module = types.ModuleType("langchain_openai")
    fake_module.ChatOpenAI = _FakeChatOpenAI
    monkeypatch.setitem(sys.modules, "langchain_openai", fake_module)


def test_get_generator_raises_clear_error_when_model_unset(monkeypatch):
    monkeypatch.setattr(config, "GENERATOR_MODEL", "")
    with pytest.raises(RuntimeError, match="GENERATOR_MODEL"):
        get_generator()


def test_get_generator_reads_model_name_from_config(monkeypatch):
    monkeypatch.setattr(config, "GENERATOR_MODEL", "gpt-4.1-mini")
    llm = get_generator()
    assert llm.model == "gpt-4.1-mini"


# ---------------------------------------------------------------- 인용 표기 정리
# 실제 실행에서 LLM이 (DM-TQ-r0-01)처럼 소괄호로 인용한 사례가 있었다.
# final_check는 [ID] 형식으로 수치를 대조하므로 코드에서 형식을 맞춘다.
from agents._e_utils import normalize_cites, strip_unknown_cites  # noqa: E402


def test_paren_cite_becomes_bracket():
    assert normalize_cites("압축률이 보고됐다 (DM-TQ-r0-01).") == "압축률이 보고됐다 [DM-TQ-r0-01]."
    assert normalize_cites("(DM-TQ-r0-01, DM-TQ-r0-02)") == "[DM-TQ-r0-01, DM-TQ-r0-02]"


def test_adjacent_brackets_and_semicolons_merge_into_one():
    assert normalize_cites("지연 감소 [DM-IT-r0-01][DM-IT-r0-02]") == "지연 감소 [DM-IT-r0-01, DM-IT-r0-02]"
    assert normalize_cites("[DM-IT-r0-01] [DM-IT-r0-02; DM-IT-r0-01]") == "[DM-IT-r0-01, DM-IT-r0-02]"


def test_plain_parentheses_are_left_alone():
    text = "조건 A(128K 토큰)에서 측정했다 (단, 자가보고)."
    assert normalize_cites(text) == text


def test_strip_handles_paren_and_removes_unknown():
    out, removed = strip_unknown_cites("수치 (DM-TQ-r0-01, DM-TQ-r9-99) 확인.", {"DM-TQ-r0-01"})
    assert out == "수치 [DM-TQ-r0-01] 확인."
    assert removed == ["DM-TQ-r9-99"]
