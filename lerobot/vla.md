## *VLA Imitaion Learning 특정 바둑돌을 줍고 특정 위치에 놓는 학습*

#### linux 구동환경
- jetson B01 4g / ubuntu18.04 / python 3.6 / memory swap 4g  / vnc viewer / usb flash boot set / boot jetback 4.5 / cuda jetpack 4.61 / usb 카메라 1개 / csi 카메라 1개
- lerobot은 python 3.10 지원 > jetson은 arm이라 anaconda 사용은 제한적, venv로 설치
- python 3.10, rerun 등은 직접 빌드하여 설치
- lerobot git clone 후 install requirements 설치 불가 > torchvision 버전이 미지원, 모든 라이브러리 직접 설치
- jetpack 4.61 : cuda는 python 3.6에서만 작동 > cuda 없이 실행 결정 (orin super 8g로 변경 예정)
- opencv 에서 csi 카메라 사용하려면 gstreamer 를 백엔드로 사용하도록 설치해야 함 > 직접 빌드


#### win 학습환경
- anaconda 환경, github과 동일한 세팅
- 구동환경에서 생성한 데이터셋을 허깅페이스 hub에 업로드하면,  hub에서 불러와서 학습
- 학습 완료된 모델을 다시 구동환경에 scp로 복사하여 사용
- isaac sim 설치


#### 로봇암 조립
- lerobot 중 so100 arm 선택 > 구버전이라 현버전(so101)과 3d 모델, 모터 등이 다름
- 3d 프린팅 : 국내 의뢰 가능 19만원 정도 / 알리에서 5~6만원 구매 가능
- STS3215 Servo 7.4V, 1/345 gear : 12개 알리 25만원 정도
- 각 모터 초기화 후 메뉴얼 따라 조립하면 됨, 단 모터 중앙 위치에 맞게 조립 필요
- 리더 암 모터 중 하나는 제거해도 되고 안해도 됨 > 하는 경우 매우 가동이 편한 대신 고정 안됨
- 유격이 적어 조립이 매우 힘듦, 모터 케이블 길이가 짧음 > so101에서 개선됨
- 스크립트 실행 시 초기에 4번 모터가 급격히 반대로 움직이는 현상이 발생 > 새로운 캘리브레이션에서 개선


#### 데이터셋 생성

https://huggingface.co/spaces/lerobot/visualize_dataset?path=%2Fdongseon%2Fso100_0510%2Fepisode_9%3Ft%3D6

![[dataset.jpg]](../image/dataset.jpg)

- 3인칭 카메라와 그리퍼 1인칭 카메라 세팅
- 프롬프트 { 검은 돌은 잡아 바둑판 좌표 (0,0) 에 놓아줘 }
- 돌이 너무 작아 그리퍼가 놓지지 않고 잡는 새밀한 작업이 어려움
- 줍는 위치, 색상 구분 및 바둑판 좌표를 모두 놓는 시나리오를 각 100씩만 해도 아주 많은 레코드 작업이 필요함
- 그립퍼 구분이 쉽게 형광 테이핑 및 끝에 사포 부착 후 데이터 생성
- 유튜버 조명이 필요함


#### isaac sim
- so-arm100 urdf 파일 사용
- 강화학습 or vla모델 결정 필요
- 부족한 데이터를 시뮬레이션 데이터로 생성하는 방안을 고려 중


#### phospho ai
- 양팔 구동
	- 리더암 모터에 기어 다시 장착 및 그립퍼로 교체
	- 오큘러스 퀘스트 컨트롤 가능하나 유료 앱 300불 정도


#### 모델 학습
- act https://arxiv.org/pdf/2304.13705  
	- flow matching 액션 청크
- pi0 
  - https://arxiv.org/html/2410.24164v1  
- diffusion_policy
  - https://arxiv.org/pdf/2303.04137  
- openVLA https://arxiv.org/pdf/2406.09246  rf-2 https://arxiv.org/pdf/2307.15818
	- 인터넷 데이터로 학습한 llm모델을 베이스로 두고 비전모델은 프로젝션 레이어로 연결
	- 액션 토큰 아웃
- smolvla > 최종 모델
	- 450m cpu 구동도 가능
	- llm + act
	- 비동기 추론 최고!

데이터셋 100개 학습 후 추론  
"Grasp a White stone and put it in the board on position (0,0)."

[![이미지 텍스트](https://img.youtube.com/vi/cgwXwE9i1xM/0.jpg)](https://www.youtube.com/watch?v=cgwXwE9i1xM)


 