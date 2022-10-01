import gym
from stable_baselines3 import PPO

env = gym.make('CartPole-v1')
env.reset()
models_dir = "models/PPO"
logdir = "logs"

model_path = f"{models_dir}/90000.zip"
model = PPO.load(model_path, env=env)


observation = env.reset()
for _ in range(1000):
   env.render()
   action, _states = model.predict(observation)  # User-defined policy function
   observation, reward, done, info = env.step(action)

   if done:
      observation = env.reset()
env.close()