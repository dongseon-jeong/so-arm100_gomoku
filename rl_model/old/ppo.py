import numpy as np
import random
from typing import Optional, Tuple, List

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.ppo.policies import ActorCriticPolicy
from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.distributions import CategoricalDistribution
from stable_baselines3.common.preprocessing import preprocess_obs
from stable_baselines3.common.torch_layers import MlpExtractor
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.preprocessing import get_flattened_obs_dim

import gymnasium as gym
from gymnasium import spaces

import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.distributions import Categorical
import torch as th


class GomokuSelfPlayEnv(gym.Env):
    def __init__(self, opponent_model=None, board_size=15):
        super().__init__()
        self.board_size = board_size
        self.board = np.zeros((board_size, board_size), dtype=np.int8)
        self.action_space = spaces.Discrete(board_size * board_size)


        self.observation_space = spaces.Dict({
            "obs": spaces.Box(low=0, high=2, shape=(board_size * board_size,), dtype=np.float32),
            "action_mask": spaces.Box(low=0, high=1, shape=(board_size * board_size,), dtype=np.int8)
        })

        self.max_turns = 60  # 수 제한
        self.current_turn = 0
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
        self.current_turn = 0
        # if random.random() < 0.5:
        #     center = self.board_size // 2
        #     self.board[center, center] = self.opponent_player
        #     self.last_player = self.opponent_player
        # else:
        #     if self.opponent_model:
        #         obs = self.board.flatten()
        #         opp_action, _ = self.opponent_model.predict(obs, deterministic=True)
        #         if self._is_valid_action(opp_action):
        #             self._place_stone(opp_action, self.opponent_player)
        #             self.last_player = self.opponent_player

        # obs = self.board.flatten()
        return self._get_obs(), {}

    def _get_obs(self):
        return {
            "obs": self.board.flatten().astype(np.float32),
            "action_mask": np.array(self.valid_actions_mask(), dtype=np.int8)
        }


    def valid_actions(self, obs: np.ndarray = None) -> List[int]:
        board = obs.reshape(self.board_size, self.board_size) if obs is not None else self.board
        return [i for i in range(self.board_size * self.board_size)
                if board[i // self.board_size, i % self.board_size] == 0]


    def valid_actions_mask(self):
        mask = np.zeros(self.board_size * self.board_size, dtype=np.int8)
        for idx in self.valid_actions():
            mask[idx] = 1
        return mask

    def _is_valid_action(self, action):
        row, col = divmod(action, self.board_size)
        return self.board[row, col] == 0

    def step(self, action):
        if not self._is_valid_action(action):
            return {
                "obs": self.board.flatten().astype(np.float32),
                "action_mask": np.array(self.valid_actions_mask(), dtype=np.int8)
            }, -1.0, True, False, {"reason": "Invalid move"}
        
        self.current_turn += 1
        
        self._place_stone(action, self.agent_player)
        self.last_player = self.agent_player
        reward, terminated = self._evaluate_board(self.agent_player)
        if terminated or self.current_turn >= self.max_turns:
            return {
                "obs": self.board.flatten().astype(np.float32),
                "action_mask": np.array(self.valid_actions_mask(), dtype=np.int8)
            }, reward, True, False, {}

        # 상대 턴
        obs_dict = {
            "obs": self.board.flatten().astype(np.float32),
            "action_mask": np.array(self.valid_actions_mask(), dtype=np.int8)
        }
        
        valid_actions = self.valid_actions(self.board.flatten())
        opp_action = None

        
        if self.opponent_model and valid_actions:

            opp_action, _ = self.opponent_model.predict(obs_dict, deterministic=True)

            if opp_action not in valid_actions:
                opp_action = random.choice(valid_actions)
        elif valid_actions:
            opp_action = random.choice(valid_actions)

        if opp_action is not None:
            self._place_stone(opp_action, self.opponent_player)
            self.last_player = self.opponent_player
            opp_reward, opp_terminated = self._evaluate_board(self.opponent_player)
            if opp_terminated or self.current_turn >= self.max_turns:
                return {
                    "obs": self.board.flatten().astype(np.float32),
                    "action_mask": np.array(self.valid_actions_mask(), dtype=np.int8)
                }, -opp_reward, True, False, {}

        return {
            "obs": self.board.flatten().astype(np.float32),
            "action_mask": np.array(self.valid_actions_mask(), dtype=np.int8)
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
    def __init__(self, logits, mask):
        logits = logits.clone()
        logits[~mask] = -1e8  # 매우 작은 값으로 설정해서 softmax에서 무시
        super().__init__(logits=logits)


class MaskedActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        # 반드시 먼저 부모 초기화 호출
        super().__init__(*args, **kwargs)
        self.features_extractor = self.make_features_extractor()
        self.features_dim = self.features_extractor.features_dim
        self.features_extractor_class = base_FlattenExtractor

    def _build_mlp_extractor(self) -> None:
        """
        Create the policy and value networks.
        Part of the layers can be shared.
        """
        # Note: If net_arch is None and some features extractor is used,
        #       net_arch here is an empty list and mlp_extractor does not
        #       really contain any layers (acts like an identity module).
        self.mlp_extractor = MlpExtractor(
            225,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            device=self.device,
        )

    def forward(self, obs_dict, deterministic=False):
        # obs = obs_dict["obs"].float().to(self.device)
        action_mask = obs_dict["action_mask"].bool().to(self.device)

        features = self.extract_features(obs_dict)  # obs는 Tensor
        
        latent_pi, latent_vf = self.mlp_extractor(features)

        logits = self.action_net(latent_pi)
        dist = MaskedCategorical(logits=logits, mask=action_mask)

        actions = torch.argmax(dist.probs, dim=-1) if deterministic else dist.sample()
        log_prob = dist.log_prob(actions)
        values = self.value_net(latent_vf)

        return actions, values, log_prob

    def _get_action_dist_from_latent(self, latent_pi):
        logits = self.action_net(latent_pi)
        return CategoricalDistribution(logits.shape[-1]), logits

    def predict(self, observation, state=None, episode_start=None, deterministic=False):
        
        try:

            # if isinstance(observation, dict):
            obs_tensor = {'obs': torch.FloatTensor(observation['obs']).to(self.device).unsqueeze(0),
                        'action_mask': torch.FloatTensor(observation['action_mask']).to(self.device).unsqueeze(0) }
            # else:
            #     raise TypeError("Expected observation to be a dict with 'obs' and 'action_mask'")
        except:
            print(observation)
        
        with torch.no_grad():
            actions, _, _ = self.forward(obs_tensor, deterministic=deterministic)
        

        return actions.cpu().numpy(), state

    def predict_values(self, obs):

        features = self.extract_features(obs, self.vf_features_extractor)
        latent_vf = self.mlp_extractor.forward_critic(features)
        return self.value_net(latent_vf)

    def extract_features(self, obs, features_extractor: Optional[BaseFeaturesExtractor] = None):
        
        # obs가 dict인 경우 'obs'만 추출
        if isinstance(obs, dict):
            obs = obs['obs']

        if self.share_features_extractor:
            return self.base_extract_features(obs, self.features_extractor)
        else:
            pi_features = self.base_extract_features(obs, self.pi_features_extractor)
            vf_features = self.base_extract_features(obs, self.vf_features_extractor)
            return pi_features, vf_features

    def base_extract_features(self, obs, features_extractor):
        # self.observation_space가 Dict이라면 'obs' 서브공간 사용
        if isinstance(self.observation_space, gym.spaces.Dict):
            obs_space = self.observation_space.spaces['obs']
        else:
            obs_space = self.observation_space

        preprocessed_obs = self.base_preprocess_obs(obs, obs_space)

        return features_extractor(preprocessed_obs)

    def base_preprocess_obs(self, obs, observation_space):
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        # obs가 Box일 경우 그대로 float 처리
        if isinstance(observation_space, gym.spaces.Box):
            return obs.float()

        # obs가 Discrete일 경우 one-hot 인코딩
        elif isinstance(observation_space, gym.spaces.Discrete):
            return F.one_hot(obs.long(), num_classes=observation_space.n).float()

        # obs가 MultiDiscrete일 경우 각 차원을 one-hot 후 concat
        elif isinstance(observation_space, gym.spaces.MultiDiscrete):
            nvec = observation_space.nvec
            return th.cat(
                [
                    F.one_hot(obs_.long(), num_classes=int(nvec[idx])).float()
                    for idx, obs_ in enumerate(th.split(obs.long(), 1, dim=1))
                ],
                dim=-1,
            ).view(obs.shape[0], sum(nvec))

        else:
            raise NotImplementedError(f"Unsupported observation type: {type(observation_space)}")

    def evaluate_actions(self, obs_dict, actions):
        obs = obs_dict["obs"].float().to(self.device)
        action_mask = obs_dict["action_mask"].bool().to(self.device)

        features = self.extract_features(obs_dict)
        latent_pi, latent_vf = self.mlp_extractor(features)

        logits = self.action_net(latent_pi)
        dist = MaskedCategorical(logits=logits, mask=action_mask)

        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        values = self.value_net(latent_vf)

        return values, log_prob, entropy


class base_FlattenExtractor(BaseFeaturesExtractor):
    """
    Feature extract that flatten the input.
    Used as a placeholder when feature extraction is not needed.

    :param observation_space: The observation space of the environment
    """

    def __init__(self, observation_space: gym.Space) -> None:
        super().__init__(observation_space, get_flattened_obs_dim(observation_space))
        self.flatten = nn.Flatten()

    def forward(self, observations: th.Tensor) -> th.Tensor:
        observations = observations['obs']
        return self.flatten(observations)




env = DummyVecEnv([lambda: GomokuSelfPlayEnv()])
# check_env(env)  # 환경 유효성 검사


initial_opponent_model = PPO(
    policy=MaskedActorCriticPolicy,
    env=env,
    learning_rate=1e-4,
    verbose=1,
    policy_kwargs=dict(
        net_arch=[dict(pi=[64, 64], vf=[64, 64])],
        activation_fn=nn.ReLU,
        share_features_extractor=True
    )
)
initial_opponent_model.learn(total_timesteps=100_000)




# 주인공 학습 환경 생성 (opponent 모델 포함)
vec_env = make_vec_env(
    GomokuSelfPlayEnv,
    n_envs=4,
    env_kwargs={"opponent_model": initial_opponent_model},
)

# 주인공 모델 (학습 대상)
main_model = PPO(
    MaskedActorCriticPolicy,
    vec_env,
    learning_rate=1e-4,
    n_steps=50,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,         # exploration을 위해 추가
    vf_coef=0.5,
    max_grad_norm=0.5,
    verbose=1,

    policy_kwargs=dict(
        net_arch=[dict(pi=[64, 64], vf=[64, 64])],
        activation_fn=nn.ReLU,
        share_features_extractor=True
    )

    # tensorboard_log="./tensorboard/"
)

# 총 학습 타임스텝 설정
total_timesteps = int(1000_000)
# 몇 스텝마다 opponent 모델을 업데이트할지 설정
update_interval = int(100_000)

# 주기적인 학습 및 opponent 업데이트 루프
for step in range(0, total_timesteps, update_interval):
    print(f"▶ Learning step {step} ~ {step + update_interval}")

    # 현재 주인공 모델을 학습
    main_model.learn(total_timesteps=update_interval, reset_num_timesteps=False)

    # opponent 모델을 현재 주인공의 복사본으로 교체 (frozen copy)
    # opponent 모델 교체
    print("🔁 Updating opponent model...")
    main_model.save("temp_main_model_ppo.zip")
    # updated_opponent_model = PPO(policy=MaskedActorCriticPolicy, env=env)
    # updated_opponent_model.set_parameters("temp_main_model_ppo.zip")
    updated_opponent_model = PPO.load("temp_main_model_ppo.zip")

    print("model loaded")
    # 벡터 환경 내 opponent 모델 교체
    for env_idx in range(vec_env.num_envs):
        raw_env = vec_env.envs[env_idx].unwrapped
        raw_env.set_opponent_model(updated_opponent_model)


print("✅ 학습 완료")
main_model.save("gomoku_ppo_selfplay_final")



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
