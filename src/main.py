#!/usr/bin/python3
import gym

import numpy as np

from wolp_agent import *
from ddpg.agent import DDPGAgent
import util.data
from util.timer import Timer


def run(episodes=2500,
        render=False,
        experiment='InvertedPendulum-v1',
        max_actions=1e3,
        knn=0.1):

    env = gym.make(experiment)

    print(env.observation_space)
    print(env.action_space)

    steps = env.spec.timestep_limit

    # agent = DDPGAgent(env)
    agent = WolpertingerAgent(env, max_actions=max_actions, k_ratio=knn)

    timer = Timer()

    data = util.data.Data()
    data.set_agent(agent.get_name(), int(max_actions), agent.k_nearest_neighbors, 3)
    data.set_experiment(experiment, agent.low.tolist(), agent.high.tolist(), episodes)

    agent.add_data_fetch(data)

    full_epoch_timer = Timer()
    reward_sum = 0

    for ep in range(episodes):

        timer.reset()
        observation = env.reset()

        total_reward = 0
        print('Episode ', ep, '/', episodes - 1, 'started...', end='')
        for t in range(steps):

            if render:
                env.render()

            action = agent.act(observation)

            data.set_action(action.tolist())

            data.set_state(observation.tolist())

            prev_observation = observation
            observation, reward, done, info = env.step(action)

            data.set_reward(reward)

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
                reward_sum += total_reward
                time_passed = timer.get_time()
                print('Reward:{} Steps:{} t:{} ({}/step) Cur avg={}'.format(total_reward, t,
                                                                            time_passed, round(
                                                                                time_passed / t),
                                                                            round(reward_sum / (ep + 1))))

                data.finish_and_store_episode()

                break
    # end of episodes
    time = full_epoch_timer.get_time()
    print('Run {} episodes in {} seconds and got {} average reward'.format(
        episodes, time / 1000, reward_sum / episodes))

    data.save()


if __name__ == '__main__':
    run()
