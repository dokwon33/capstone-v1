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
