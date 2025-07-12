from stable_baselines3 import PPO
from gym import GomokuSelfPlayEnv


board_size = 13
max_turns = 50


# 추론 환경
main_model = PPO.load("rl_model/gomoku_ppo_selfplay_final.zip")
eval_env = GomokuSelfPlayEnv(board_size=board_size, max_turns = max_turns)
updated_opponent_model = PPO.load("rl_model/temp_main_model_ppo.zip")
eval_env.set_opponent_model(updated_opponent_model)

obs, info = eval_env.reset()
done = False

while not done:
    action, _ = main_model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = eval_env.step(action)
    eval_env.render()

# ✅ 승리자 확인
winner = info.get("winner", None)
chain, won = eval_env._max_consecutive(eval_env.last_player)

if winner:
    print(f"✅ 게임 종료! 승리자: {winner} / 연속돌수: {chain} / 승리 조건 만족 여부: {won}")
else:
    print(f"❗ 게임 종료! winner 정보 없음 / 연속돌수: {chain} / 승리 조건 만족 여부: {won}")
