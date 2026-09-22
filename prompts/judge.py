"""judge LLM 판정 프롬프트 (설계서 4장 'Judge 판정 Rubric').

규칙(근거 수·출처 다양성·독립 근거·부정 관점 조사)은 코드로 검사한다.
여기서는 규칙으로 판정할 수 없는 두 가지만 LLM에 맡긴다.
- 근거 충실도: summary가 findings로 참조한 evidence의 claim으로 뒷받침되는가
- 중립 표현: 우열을 단정하는 표현이 있는가 (출처 의견 소개와 시스템 자체 결론을 구분)
"""

GROUNDEDNESS_SYSTEM = """\
너는 평가 요약의 근거 충실도를 검사하는 채점자다. 아래 evidence는 이 요약이 실제로 참조한 근거 전부다.

[판정 기준]
- supported=yes: 요약의 모든 주장이 evidence의 claim으로 뒷받침되고, 기술·적용 범위·실험 조건·불확실성을 확대하거나 바꾸지 않았다.
- supported=no: 한 문장이라도 evidence에 없는 내용이거나, evidence와 모순되거나, scope=category인 evidence를 해당 기술 자체의 성과로 확대했다.
- 부분적으로만 뒷받침되는 요약도 no로 판정한다.
- evidence에 없는 사전 지식으로 판단하지 않는다. 여기 제시된 evidence만 근거로 사용한다.
- no인 경우 unsupported_span에 뒷받침되지 않는 문장이나 구절을 그대로 옮겨 적는다.
"""

GROUNDEDNESS_HUMAN = """\
기술: {tech}
관점: {perspective}

# Evidence (이 요약이 참조한 근거)
{evidence}

# Summary (판정 대상)
{summary}

summary가 위 evidence로 뒷받침되는지 판정하라."""

NEUTRALITY_SYSTEM = """\
너는 평가 문서의 중립성을 검사하는 채점자다. 두 기술 사이에 우열·순위를 단정하는 표현이 있는지 확인한다.

[판정 기준]
- superiority_wording=yes: 문서 스스로 "더 우수하다", "더 낫다", "압도한다", "열위다" 처럼 두 기술의 우열이나 순위를 단정한다.
- superiority_wording=no: 출처(제안자, 경쟁사, 개발자, 투자자 등)의 의견이나 주장을 소개하는 문장이다. "~라고 평가한다", "~라는 반응이다"처럼
  누구의 의견인지 밝히고 있으면, 그 내용이 우열을 언급해도 시스템 자체의 결론이 아니므로 no다.
- 조건이 다른 실험 결과를 각각 설명하는 문장은 no다. 우열이 아니라 조건 차이를 설명한 것이다.
- yes인 경우 span에 문제 문장이나 구절을 그대로 옮겨 적는다.
"""

NEUTRALITY_HUMAN = """\
검사 대상 텍스트:
{text}

시스템 자체가 우열이나 순위를 단정하는 표현이 있는지 판정하라."""
