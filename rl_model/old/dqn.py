import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, List
import random
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
# from stable_baselines3.dqn.policies import DQNPolicy
from torch.distributions import Categorical
from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.distributions import CategoricalDistribution
import torch.nn.functional as F
import torch


class GomokuSelfPlayEnv(gym.Env):
    def __init__(self, opponent_model=None, board_size=15):
        super().__init__()
        self.board_size = board_size
        self.board = np.zeros((board_size, board_size), dtype=np.int8)
        self.action_space = spaces.Discrete(board_size * board_size)


        self.observation_space = spaces.Dict({
            "obs": spaces.Box(low=0, high=2, shape=(board_size * board_size,), dtype=np.int8),
            "action_mask": spaces.Box(low=0, high=1, shape=(board_size * board_size,), dtype=np.int8)
        })


        self.agent_player = 1  # 학습 대상 (흑돌)
        self.opponent_player = 2  # 동결된 정책 또는 이전 버전
        self.last_player = None  # 마지막 착수자
        self.opponent_model = None  # 나중에 할당

    def set_opponent_model(self, model):
        self.opponent_model = model

    def _get_action_mask(self):
        return np.array([1 if self._is_valid_action(i) else 0 for i in range(self.board_size ** 2)], dtype=np.int8)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        self.done = False
        self.last_player = None

        if random.random() < 0.5:
            center = self.board_size // 2
            self.board[center, center] = self.opponent_player
            self.last_player = self.opponent_player
        else:
            if self.opponent_model:
                obs = self.board.flatten()
                opp_action, _ = self.opponent_model.predict(obs, deterministic=True)
                if self._is_valid_action(opp_action):
                    self._place_stone(opp_action, self.opponent_player)
                    self.last_player = self.opponent_player

        obs = self.board.flatten()
        return {
            "obs": obs,
            "action_mask": self._get_action_mask()
        }, {}

    def valid_actions(self, obs: np.ndarray = None) -> List[int]:
        board = obs.reshape(self.board_size, self.board_size) if obs is not None else self.board
        return [i for i in range(self.board_size * self.board_size)
                if board[i // self.board_size, i % self.board_size] == 0]

    def _is_valid_action(self, action):
        row, col = divmod(action, self.board_size)
        return self.board[row, col] == 0

    def step(self, action):
        if not self._is_valid_action(action):
            return {
                "obs": self.board.flatten(),
                "action_mask": self._get_action_mask()
            }, -1.0, True, False, {"reason": "Invalid move"}

        self._place_stone(action, self.agent_player)
        self.last_player = self.agent_player
        reward, terminated = self._evaluate_board(self.agent_player)
        if terminated:
            return {
                "obs": self.board.flatten(),
                "action_mask": self._get_action_mask()
            }, reward, True, False, {}

        # 상대 턴
        obs = self.board.flatten()
        valid_actions = self.valid_actions(obs)
        opp_action = None

        if self.opponent_model and valid_actions:
            opp_action, _ = self.opponent_model.predict(obs, deterministic=True)
            if opp_action not in valid_actions:
                opp_action = random.choice(valid_actions)
        elif valid_actions:
            opp_action = random.choice(valid_actions)

        if opp_action is not None:
            self._place_stone(opp_action, self.opponent_player)
            self.last_player = self.opponent_player
            opp_reward, opp_terminated = self._evaluate_board(self.opponent_player)
            if opp_terminated:
                return {
                    "obs": self.board.flatten(),
                    "action_mask": self._get_action_mask()
                }, -opp_reward, True, False, {}

        return {
            "obs": self.board.flatten(),
            "action_mask": self._get_action_mask()
        }, reward, False, False, {}

    def _place_stone(self, action, player):
        row, col = divmod(action, self.board_size)
        self.board[row, col] = player

    def _evaluate_board(self, player):
        chain, check_win = self._max_consecutive(player)
        reward = {2: 0.1, 3: 0.3, 4: 0.7, 5: 1.0}.get(chain, 0.0)
        return reward, chain >= 5

    def _max_consecutive(self, player):
        for y in range(self.board_size):
            for x in range(self.board_size):
                if self.board[y][x] != player:
                    continue
                for dx, dy in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                    count = 0
                    for i in range(5):
                        nx, ny = x + dx * i, y + dy * i
                        if 0 <= nx < self.board_size and 0 <= ny < self.board_size and self.board[ny][nx] == player:
                            count += 1
                        else:
                            break

        if count == 5:
            check_win = True
        else:
            check_win = False
        return count, check_win
    
    def render(self):
        print(self.board)


class MaskedCategorical(Categorical):
    def __init__(self, logits=None, probs=None, mask=None):
        if mask is not None:
            logits = logits.clone()
            logits[~mask] = -1e8  # 매우 작은 값으로 마스킹
        super().__init__(logits=logits, probs=probs)



class MaskedActorCriticPolicy(MlpPolicy):
    def _build(self, lr_schedule):
        # 부모 클래스가 latent_pi, latent_vf, value_net, action_net 등을 설정
        super()._build(lr_schedule)

    def forward(self, obs, deterministic=False):
        # dict로 예상하지만 tensor가 들어오면 그대로 사용
        if isinstance(obs, dict):
            obs_tensor = obs["obs"]
            action_mask = obs["action_mask"].bool()
        else:
            # 예: obs가 tensor로 들어오면 mask를 따로 계산하거나, 오류처리
            obs_tensor = obs
            action_mask = torch.ones_like(obs_tensor).bool()  # 임시 마스크(모두 유효)


        # latent vector 생성
        features = self.extract_features(obs_tensor)
        latent_pi, latent_vf = self.mlp_extractor(features)

        distribution = self._get_action_dist_from_latent(latent_pi)
        masked_dist = MaskedCategorical(logits=distribution.distribution.logits, mask=action_mask)

        if deterministic:
            actions = masked_dist.probs.argmax(dim=1)
        else:
            actions = masked_dist.sample()

        log_prob = masked_dist.log_prob(actions)
        values = self.value_net(latent_vf)

        return actions, values, log_prob












env = DummyVecEnv([lambda: GomokuSelfPlayEnv()])

initial_opponent_model = PPO(
    policy=MaskedActorCriticPolicy,
    env=env,
    learning_rate=1e-4,
    verbose=1,
)
initial_opponent_model.learn(total_timesteps=100_000)










# 주인공 학습 환경 생성 (opponent 모델 포함)
vec_env = make_vec_env(
    GomokuSelfPlayEnv,
    n_envs=320,
    env_kwargs={"opponent_model": initial_opponent_model},
)

# 주인공 모델 (학습 대상)
main_model = PPO(
    MaskedActorCriticPolicy,
    vec_env,
    learning_rate=1e-4,
    n_steps=1024,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,         # exploration을 위해 추가
    vf_coef=0.5,
    max_grad_norm=0.5,
    verbose=1,
    # tensorboard_log="./tensorboard/"
)

# 총 학습 타임스텝 설정
total_timesteps = int(1e+6)
# 몇 스텝마다 opponent 모델을 업데이트할지 설정
update_interval = int(1e+5)

# 주기적인 학습 및 opponent 업데이트 루프
for step in range(0, total_timesteps, update_interval):
    print(f"▶ Learning step {step} ~ {step + update_interval}")

    # 현재 주인공 모델을 학습
    main_model.learn(total_timesteps=update_interval, reset_num_timesteps=False)

    # opponent 모델을 현재 주인공의 복사본으로 교체 (frozen copy)
    # opponent 모델 교체
    print("🔁 Updating opponent model...")
    main_model.save("temp_main_model.zip")
    updated_opponent_model = DQN.load("temp_main_model.zip")

    # 벡터 환경 내 opponent 모델 교체
    for env_idx in range(vec_env.num_envs):
        raw_env = vec_env.envs[env_idx].unwrapped
        raw_env.set_opponent_model(updated_opponent_model)


print("✅ 학습 완료")
main_model.save("gomoku_dqn_selfplay_final")



train_env = GomokuSelfPlayEnv()
obs, info = train_env.reset()
done = False

while not done:
    action, _states = main_model.predict(obs)
    obs, reward, done, truncated, info = train_env.step(action)
    train_env.render()

    if done:
        winner = train_env.last_player  # 누가 이겼는지
        chain, won = train_env._max_consecutive(winner)
        print(f"✅ 게임 종료! 승리자: {'Agent' if winner == 1 else 'Opponent'} /돌수 : {chain} / 승리 조건 만족 여부: {won}")
