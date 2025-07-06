
## *RL 구현

- matrix 입력 받아 다음 좌표 추출
	- agent : 두는 놈 
	- environment : 바둑판 matrix
	- state : 바둑판 matrix 전체
	- reward : 할인율은 어떻게?
	- action : discrete / 좌표에 두기
	- episode 는 5개 완성 시
	- exploration <> exploitation
	- value based >> policy based
	- temporal difference 로 value 업데이트
- 보상
	- 2개 만들면 +1
	- 3개 만들면 +2
	- 4개 만들면 +3
	- 5개 만들면 +100
	- 상대가 2개 만들면 -1
	- 상대가 3개 만들면 -2
	- 상대가 4개 만들면 -3
	- 상대가 5개 만들면 -100

상대나 본인이 5개 만들면 종료 / 30수 이상 종료
학습 시 각 색상의 돌의 보상을 함께 계산을 고려
모델은 dqn으로 구상

##### Q계산식
![[q_.jpg]](../image/q_.jpg)


##### Q table
상태 흑돌 (액션열 각 좌표들)
상태 백돌 (액션열 각 좌표들)

| step | agent | state | reward | epsilon | lr  | gamma | action_coor_00 | action_coor_01 | action_coor_02 | action_coor_03 | action_coor_04 | action_coor_05 |
| ---- | ----- | ----- | ------ | ------- | --- | ----- | -------------- | -------------- | -------------- | -------------- | -------------- | -------------- |
| 0    | white | alive | 0      | 1       | 0.1 | 0.99  | 0              | 0              | 0              | 0              | 0              | 0              |
| 1    | white | alive | 0      | 0.99    | 0.1 | 0.99  | 0              | 0              | 0              | 5              | 0              | 0              |
| 2    | white | die   | 0      | 0.99    | 0.1 | 0.99  | 0              | 0              | 0              | 0              | 0              | 0              |
|      |       |       |        |         |     |       |                |                |                |                |                |                |

#### library
gymnasium  
stable baseline3  


#### 환경구성
```
dqn으로 구현 시 이미 둔 자리를 다시 둘 경우 패널티로 게임이 종료됨
이미 둔 자리를 다시 두지 않기 위해 마이너스 보상을 적용하였지만, 탐색 시 매 스탭마다 랜덤으로 좌표가 생성되어 마이너스 보상으로 인한 학습이 진행되지 않음 > ppo로 변경하여 액션 마스크 적용
```

최대 턴 제한 : 30수  
현재 턴 저장  
상대편 모델의 연속된 돌 수는 마이너스 보상 적용  
스탭 : 조기 종료가 되지 않을 경우 아래와 같이 보상 적용
불필요한 자리에 두지 않기 위해 턴수가 길어질수록 마이너스 보상 적용
```
total_reward = agent_reward - opp_reward - self.current_turn*0.01
```

평가 :
5돌의 보상이 1.0일 경우 5개 돌을 만들지 않고 30턴 전까지 4개 구성을 최대한 많이 만들려는 행태를 보임 > 5의 보상을 100.0으로 올려서 학습 진행  
```
def _evaluate_board(self, player):
	chain, check_win = self._max_consecutive(player)
	reward = {2: 0.1, 3: 0.3, 4: 0.7, 5: 100.0}.get(chain, 0.0)
	return reward, check_win
```


#### 정책
기존 모델에 액션 마스크 적용 및 마스크 로짓 적용
```
class MaskedCategorical(Categorical):
    def __init__(self, logits, mask):
        logits = logits.clone()
        logits[~mask] = -1e8  # 매우 작은 값으로 설정해서 softmax에서 무시
        super().__init__(logits=logits)
```

#### 강화 학습
학습 시 인터벌 마다 상대 모델을 업데이트하여 상대를 이기기 위한 강화 학습을 진행  
인터벌 텀이 너무 크면 상대보다 메인 모델이 너무 잘해 학습이 잘안됨

```
# 학습 루프
for step in range(0, total_timesteps, update_interval):
    print(f"▶ Learning step {step} ~ {step + update_interval}")

    # 주인공 모델 학습
    main_model.learn(
        total_timesteps=update_interval,
        reset_num_timesteps=False,
        callback=WinnerLoggingCallback()
    )

    # 🆕 opponent 모델 업데이트
    print("🔁 Updating opponent model...")
    main_model.save("./rl_model/temp_main_model_ppo.zip")
    print('saved')
    updated_opponent_model = PPO.load("./rl_model/temp_main_model_ppo.zip")


    for env_idx in range(vec_env.num_envs):
        raw_env = vec_env.envs[env_idx].unwrapped
        raw_env.set_opponent_model(updated_opponent_model)
```

#### valid
```
[[0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 1 0 0 0 0 0 0 0 0]
 [0 0 0 0 2 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0]]

.............

[[0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 1 0 0 0 0 0 0]
 [0 0 0 2 0 2 0 0 0 0 0 0 0]
 [0 0 0 0 1 1 2 0 0 0 0 0 0]
 [0 0 0 0 2 1 0 0 0 0 0 0 0]
 [0 0 2 0 0 0 1 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0]]
.............

[[0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 1 0 0 0 0 0 0]
 [0 0 0 2 2 2 0 0 0 0 0 0 0]
 [0 2 2 1 1 1 2 0 0 0 0 0 0]
 [0 0 0 2 2 1 0 0 0 0 0 0 0]
 [0 0 2 1 1 1 1 0 0 0 0 0 0]
 [0 0 0 0 0 1 0 2 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0]]

[[0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 1 0 0 0 0 0 0]
 [0 0 0 2 2 2 0 0 0 0 0 0 0]
 [0 2 2 1 1 1 2 0 0 0 0 0 0]
 [0 0 0 2 2 1 0 0 0 0 0 0 0]
 [0 0 2 1 1 1 1 0 0 0 0 0 0]
 [0 0 0 0 0 1 0 2 0 0 0 0 0]
 [0 0 0 0 0 1 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0]]
```














## *참고 LLM 구현

llm 네트워크로 강화학습에 활용한 최근 논문(rl)
https://arxiv.org/pdf/2503.21683

대국 데이터가 있다면 vision 모델이 출력한 matrix를 llm에 프롬프트로 직접 넣는 방법도 있음
학습은 대국 데이터로 입력하고 출력은 다음 좌표를 출력하도록 학습

아래와 같은 데이터셋을 구축해야함

```prompt
다음 오목 메트릭스를 보고, 이기기 위한 다음 좌표를 예측하세요
사용자 :
{board matrix:[[0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,2,1,0,,,,,,,,2,0,0,0,0,0,0,0]]} 사용자가 검은 돌이고 agent가 흰돌이야, 흰돌이 다음에 둘 위치를
agent:
{6,-5}
```







