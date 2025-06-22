from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
import torch.nn as nn

from policy import MaskedActorCriticPolicy
from gym import GomokuSelfPlayEnv
from torch.utils.tensorboard import SummaryWriter
import gym
if not hasattr(gym, "__version__"):
    gym.__version__ = "0.26.2"  # ← 실제 설치된 버전에 맞게 적절히 수정



initial_opponent_t = 100000
total_timesteps = int(1000000)
update_interval = int(100000)
board_size = 13
max_turns = 40


class WinnerLoggingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.writer = None

    def _on_training_start(self):
        self.writer = SummaryWriter(log_dir="./tensorboard/")

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "winner" in info:
                if info["winner"] == "agent":
                    self.writer.add_scalar("game/winner", 1, self.num_timesteps)
                elif info["winner"] == "opponent":
                    self.writer.add_scalar("game/winner", -1, self.num_timesteps)
                else:
                    self.writer.add_scalar("game/winner", 0, self.num_timesteps)
        return True

# 초기 opponent 모델 학습
env = DummyVecEnv([lambda: GomokuSelfPlayEnv(board_size=board_size, max_turns = max_turns)])
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
initial_opponent_model.learn(total_timesteps=initial_opponent_t)

# 주인공 학습 환경 생성 (opponent 포함)
vec_env = make_vec_env(
    lambda: Monitor(GomokuSelfPlayEnv(board_size=board_size, 
                                      max_turns = max_turns, 
                                      opponent_model=initial_opponent_model), 
                                      info_keywords=("winner",)),
    n_envs=4,
)

# 주인공 모델 정의
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
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    verbose=1,
    tensorboard_log="./tensorboard/",
    policy_kwargs=dict(
        net_arch=[dict(pi=[64, 64], vf=[64, 64])],
        activation_fn=nn.ReLU,
        share_features_extractor=True
    )
)


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
    main_model.save("temp_main_model_ppo.zip")
    print('saved')
    updated_opponent_model = PPO.load("temp_main_model_ppo.zip")


    for env_idx in range(vec_env.num_envs):
        raw_env = vec_env.envs[env_idx].unwrapped
        raw_env.set_opponent_model(updated_opponent_model)

print("✅ 학습 완료")
main_model.save("gomoku_ppo_selfplay_final")

