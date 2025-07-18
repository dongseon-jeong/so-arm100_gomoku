## 합성 데이터 생성

바둑판 주변 환경 및 돌모양 증강  
좌표별 100개 이상의 레코딩이 필요함  
모델 일반화가 가능하다면 주요 좌표만 생성해도 될 것으로 예상  


## 환경
isaacsim : win venv  
ros2 + moveit : win wsl 빌드
jetson ros2와 통신 오류로 sim2real을 아직 구현 안됨

## 액션 플랜(참고 https://lycheeai-hub.com/project-so-arm101-x-isaac-sim-x-isaac-lab-tutorial-series)

moveit
![[moveit.jpg]](../image/moveit.jpg)


isaacsim  
![[isaac_action.jpg]](../image/isaac_action.jpg)


## isaacsim mcp(참고 https://github.com/omni-mcp/isaac-sim-mcp?tab=readme-ov-file)
돌 : 아이작심 mcp가 있어서, 돌은 랜덤으로 생성해달라고 할 예정


![[curs.jpg]](../image/curs.jpg)

![[curs2.jpg]](../image/curs2.jpg)



바닥&바둑판 : 마야나 블랜더로 할 예정, 텍스처는 실사 사용
조명&배경 : 랜더링하면서 조명 배치 예정


## 환경 생성
랜덤 환경 생성 및 합성 데이터 레코딩

## 강화학습
