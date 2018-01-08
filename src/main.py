import gym
import numpy as np

from util import *

from wolp_agent import *


def run(episodes=250, collecting_data=True):

    experiment = ('InvertedPendulum-v1')
    env = gym.make(experiment)

    steps = env.spec.timestep_limit

    max_actions = 1e2
    agent = WolpertingerAgent(env, k_nearest_neighbors=int(0.1 * max_actions),
                              max_actions=max_actions)

    file_name = "data_" + str(episodes) + '_' + agent.get_name()

    rewards = []
    for i in range(episodes):
        observation = env.reset()

        total_reward = 0
        print('Episode ', i, '/', episodes - 1, 'started...', end='')
        for t in range(steps):

            env.render()

            action = agent.act(observation)

            prev_observation = observation
            observation, reward, done, info = env.step(action)

            episode = {'obs': prev_observation,
                       'action': action,
                       'reward': reward,
                       'obs2': observation,
                       'done': done,
                       't': t}

            agent.observe(episode)

            total_reward += reward
            if done or (t == steps - 1):
                t += 1
                rewards.append(total_reward)
                print('Reward:', total_reward, 'Steps:', t)
                break

    # end of episodes
    print('Average rewards:', np.average(rewards))


if __name__ == '__main__':
    run()
