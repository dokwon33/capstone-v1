"""라우팅 호환 shim.

Supervisor 패턴으로 바꾸면서 분기 판단은 graph/supervisor.py의 decide()로 모았다.
분기 로직이 두 곳에 있으면 트레이스에 적힌 사유와 실제 이동이 어긋날 수 있어서다.
이 모듈은 이전 이름(PATH_MAP, route_after_judge)을 쓰던 코드가 깨지지 않게 두는 얇은 shim이다.
"""
from graph.builder import PATH_MAP  # noqa: F401  (이전 이름 유지)
from graph.supervisor import route_from_supervisor as route_after_judge  # noqa: F401
