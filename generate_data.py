import gym
from gym_examples.envs import GridWorldEnv
from gym.wrappers import FlattenObservation
# import gym_examples.wrappers as wrappers
import numpy as np
import random

VIZ = False
VERBOSE = False
N_OBS = 50000
CORRECT = False
is_correct = 'correct' if CORRECT else 'incorrect'

def main():
    if VIZ:
        env = gym.make('gym_examples/GridWorld-v0', render_mode="human") # enable visualization
    else:
        env = gym.make('gym_examples/GridWorld-v0')
    env.action_space.seed(42)
    observation, info = env.reset(seed=42)
    last_obs = None
    observations = []
    for i in range(N_OBS):
        if CORRECT:
            action = random.choice((2,3)) # correct: only go left and down
        else:
            action = env.action_space.sample() # buggy
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
        if VERBOSE:
            print(i, observation, reward)
        if VIZ:
            env.render() # popup window visualization
        if terminated or truncated:
            observation, info = env.reset()
    o = np.array(observations)
    np.savetxt(f'grid_simple_{is_correct}_{N_OBS}.csv', o.astype(int), header='s,a,r,sp', fmt='%i')
        

main()