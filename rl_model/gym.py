import numpy as np
import random
from typing import Optional, Tuple, List

import gymnasium as gym
from gymnasium import spaces


class GomokuSelfPlayEnv(gym.Env):
    def __init__(self, opponent_model=None, board_size=15, max_turns = 40):
        super().__init__()
        self.board_size = board_size
        self.board = np.zeros((board_size, board_size), dtype=np.int8)
        self.action_space = spaces.Discrete(board_size * board_size)


        self.observation_space = spaces.Dict({
            "obs": spaces.Box(low=0, high=2, shape=(board_size * board_size,), dtype=np.float32),
            "action_mask": spaces.Box(low=0, high=1, shape=(board_size * board_size,), dtype=np.int8)
        })

        self.max_turns = max_turns  # 수 제한
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
            return self._obs(), -1.0, True, False, {
                "reason": "Invalid move",
                "winner": "opponent"  # 잘못 둔 경우 상대 승
            }

        self.current_turn += 1
        self._place_stone(action, self.agent_player)
        self.last_player = self.agent_player

        agent_reward, agent_won = self._evaluate_board(self.agent_player)
        if agent_won:
            return self._obs(), agent_reward, True, False, {
                "winner": "agent"
            }

        if self.current_turn >= self.max_turns:
            return self._obs(), -0.5, True, False, {
                "reason": "max turn reached",
                "winner": "draw"
            }

        # ----- Opponent turn -----
        obs_dict = self._obs()
        valid_actions = self.valid_actions(self.board.flatten())
        opp_action = None

        if self.opponent_model and valid_actions:
            opp_action, _ = self.opponent_model.predict(obs_dict, deterministic=True)
            if opp_action not in valid_actions:
                opp_action = random.choice(valid_actions)
        elif valid_actions:
            opp_action = random.choice(valid_actions)

        opp_reward = 0.0  # 기본값 지정
        opp_won = False

        if opp_action is not None:
            self._place_stone(opp_action, self.opponent_player)
            self.last_player = self.opponent_player
            opp_reward, opp_won = self._evaluate_board(self.opponent_player)

            if opp_won:
                return self._obs(), -1.0, True, False, {
                    "winner": "opponent"
                }

            if self.current_turn >= self.max_turns:
                return self._obs(), -0.5, True, False, {
                    "reason": "max turn reached",
                    "winner": "draw"
                }

        total_reward = agent_reward - opp_reward - self.current_turn*0.01
        return self._obs(), total_reward, False, False, {}


    def _obs(self):
        return {
            "obs": self.board.flatten().astype(np.float32),
            "action_mask": np.array(self.valid_actions_mask(), dtype=np.int8)
        }



    def _place_stone(self, action, player):
        row, col = divmod(action, self.board_size)
        self.board[row, col] = player

    def _evaluate_board(self, player):
        chain, check_win = self._max_consecutive(player)
        reward = {2: 0.1, 3: 0.3, 4: 0.7, 5: 1.0}.get(chain, 0.0)
        return reward, check_win

    def _max_consecutive(self, player):
        max_count = 0
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
                    max_count = max(max_count, count)
                    if max_count == 5:
                        return max_count, True  # 조기 종료
        return max_count, False
    
    def render(self):
        print(self.board)