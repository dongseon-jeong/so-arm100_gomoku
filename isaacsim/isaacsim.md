## 합성 데이터 생성

바둑판 주변 환경 및 돌모양 증강  
좌표별 100개 이상의 레코딩이 필요함  
모델 일반화가 가능하다면 주요 좌표만 생성해도 될 것으로 예상  


## 환경
isaacsim : win venv  
ros2 + moveit : win wsl 빌드
jetson ros2와 통신 오류로 sim2real을 아직 구현 안됨

## 액션 플랜

moveit
![[moveit.jpg]](../image/moveit.jpg)


isaacsim  
![[isaac_action.jpg]](../image/isaac_action.jpg)


## 환경 생성
바둑판, 돌 모델링  
조명, 배경 생성  
랜덤 환경 생성 및 합성 데이터 레코딩

## 강화학습
