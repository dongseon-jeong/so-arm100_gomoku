import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple

class GomokuEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, board_size: int = 15):
        super().__init__()
        self.board_size = board_size
        self.board = np.zeros((self.board_size, self.board_size), dtype=np.int8)

        # 0: empty, 1: black (agent), 2: white (opponent)
        self.action_space = spaces.Discrete(self.board_size * self.board_size)

        # Observation: board flattened (0, 1, 2 values)
        self.observation_space = spaces.Box(
            low=0, high=2, shape=(self.board_size * self.board_size,), dtype=np.int8
        )

        self.current_player = 1  # 1=agent (black), 2=opponent (white)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.board[:] = 0
        self.current_player = 1
        return self.board.flatten(), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        row, col = divmod(action, self.board_size)

        if self.board[row, col] != 0:
            return self.board.flatten(), -1.0, True, False, {"reason": "Invalid move"}

        # Agent move
        self.board[row, col] = 1
        reward, terminated = self._evaluate_board(1)
        if terminated:
            return self.board.flatten(), reward, True, False, {} # obs, reward, done, truncate, info

        # Opponent move (random policy for now)
        opponent_action = self._random_opponent_action()
        if opponent_action is not None:
            opp_row, opp_col = opponent_action
            self.board[opp_row, opp_col] = 2
            opp_reward, opp_terminated = self._evaluate_board(2)
            if opp_terminated:
                return self.board.flatten(), -opp_reward, True, False, {} # obs, reward, done, truncate, info

        return self.board.flatten(), reward, False, False, {}

    def _random_opponent_action(self):
        empty = np.argwhere(self.board == 0)
        if len(empty) == 0:
            return None
        return tuple(empty[np.random.choice(len(empty))])

    def _evaluate_board(self, player: int) -> Tuple[float, bool]:
        max_chain = self._max_consecutive(player)
        reward_table = {2: 0.1, 3: 0.3, 4: 0.7, 5: 1.0}
        reward = reward_table.get(max_chain, 0.0)
        terminated = max_chain >= 5

        if player == 2:
            return -reward, terminated
        return reward, terminated

    def _max_consecutive(self, player: int) -> int:
        max_chain = 0
        for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
            for r in range(self.board_size):
                for c in range(self.board_size):
                    chain = 0
                    for k in range(5):
                        nr, nc = r + dr * k, c + dc * k
                        if 0 <= nr < self.board_size and 0 <= nc < self.board_size:
                            if self.board[nr, nc] == player:
                                chain += 1
                            else:
                                break
                        else:
                            break
                    max_chain = max(max_chain, chain)
        return max_chain

    def render(self):
        print(self.board)


from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env

env = GomokuEnv()
check_env(env)  # 환경 유효성 검사

model = DQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)

obs, info = env.reset()
done = False
while not done:
    action, _states = model.predict(obs)
    obs, reward, done, truncated, info = env.step(action)
    env.render()
