import gym
from stable_baselines3 import PPO

env = gym.make('CartPole-v1')
env.reset()
models_dir = "models/PPO"
logdir = "logs"


model = PPO(
    'CnnPolicy', 
    env, 
    verbose=1, 
    tensorboard_log=logdir, 
    learning_rate=0.00003
)

TIMESTEPS = 1_000
iters = 0
for i in range(100):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO")
    model.save(f"{models_dir}/{TIMESTEPS*i}")

env.close()