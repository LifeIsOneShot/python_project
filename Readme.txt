1. racegame.py

화살표 방향키로 차를 움직일 수 있다.
차 img의 4개 모서리가 경기장 선에 닿으면 빨간 X를 표시한다.

2. racesimulation

코드 내에 지정된 경로에 따라 차가 이동한다.
이동 경로는 차량의 위치를 직접 측정해서 측정 오차가 있는것을 가정한 환경으로 Noise가 포함되어 있고 이 Noise를 칼만 필터링 알고리즘으로 감소시킨다.
racegame 과 마찬가지로 차 img 4개 모서리 경기장 선에 닿으면  빨간 X를 표시한다.