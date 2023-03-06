import gym
from gym_examples.envs import GridWorldEnv
from gym.wrappers import FlattenObservation
# import gym_examples.wrappers as wrappers
import numpy as np

def main():
    env = gym.make('gym_examples/GridWorld-v0')
    # env = gym.make('gym_examples/GridWorld-v0', render_mode="human") # enable visualization
    env.action_space.seed(42)
    observation, info = env.reset(seed=42)
    last_obs = None
    observations = []
    for i in range(50000):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        curr_state = np.ravel_multi_index(observation['agent'], (env.size, env.size))
        if last_obs is not None:
            observations.append([
                last_obs[0], # s
                last_obs[1], # a
                last_obs[2], # r
                curr_state   # sp
            ])
        last_obs = [
            curr_state,
            action,
            reward
        ]
        # print(i, observation, reward)
        # env.render() # popup window visualization
        if terminated or truncated:
            observation, info = env.reset()
    o = np.array(observations)
    np.savetxt('grid_simple_random.csv', o.astype(int), header='s,a,r,sp', fmt='%i')
        

main()