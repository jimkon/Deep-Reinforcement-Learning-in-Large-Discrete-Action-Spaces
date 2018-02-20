#!/usr/bin/python3
import gym

import numpy as np

from util import *

from wolp_agent import *
from ddpg.agent import DDPGAgent
import util.data_old as d_old
import util.data as d_new
from util.timer import Timer

AUTO_SAVE_AFTER_EPISODES = 500


def run(episodes=5000,
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

    file_name = "data_" + str(episodes) + '_' + agent.get_name()
    print(file_name)
    data_fetcher = d_old.Data(file_name)

    data_fetcher.add_arrays(['experiment', 'max_actions', 'action_space',
                             'rewards', 'count', 'actions', 'done'])
    data_fetcher.add_arrays(['state_' + str(i) for i in range(agent.observation_space_size)])

    agent.add_data_fetch(data_fetcher)

    data_fetcher.add_to_array('experiment', experiment)
    data_fetcher.add_to_array('max_actions', max_actions)
    data_fetcher.add_to_array('action_space', agent.get_action_space())

    timer = Timer()

    data_new = d_new.Data()
    data_new.set_agent(agent.get_name(), int(max_actions), agent.k_nearest_neighbors, 3)
    data_new.set_experiment(experiment, agent.low.tolist(), agent.high.tolist(), episodes)
    data_new.print_data()

    full_epoch_timer = Timer()
    reward_sum = 0

    for ep in range(episodes):
        timer.reset()
        observation = env.reset()

        total_reward = 0
        print('Episode ', ep, '/', episodes - 1, 'started...', end='')
        for t in range(steps):

            data_fetcher.reset_timers()

            if render:
                env.render()

            action = agent.act(observation)

            data_fetcher.add_to_array('actions', action)  # -------
            data_new.set_action(action.tolist())

            data_new.set_state(observation.tolist())
            for i in range(agent.observation_space_size):
                data_fetcher.add_to_array('state_' + str(i), observation[i])
            prev_observation = observation
            observation, reward, done, info = env.step(action)

            data_new.set_reward(reward)

            episode = {'obs': prev_observation,
                       'action': action,
                       'reward': reward,
                       'obs2': observation,
                       'done': done,
                       't': t}

            agent.observe(episode)

            total_reward += reward
            data_fetcher.add_to_array('done', 1 if done else 0)

            if done or (t == steps - 1):

                t += 1
                data_fetcher.add_to_array('rewards', total_reward)  # ------
                reward_sum += total_reward
                time_passed = timer.get_time()
                print('Reward:{} Steps:{} t:{} ({}/step) Cur avg={}'.format(total_reward, t,
                                                                            time_passed, round(
                                                                                time_passed / t),
                                                                            round(reward_sum / (ep + 1))))

                if ep % AUTO_SAVE_AFTER_EPISODES == AUTO_SAVE_AFTER_EPISODES - 1:
                    data_fetcher.temp_save()
                    data_new.finish_and_store_episode()
                else:
                    data_new.end_of_episode()

                break
    # end of episodes
    time = full_epoch_timer.get_time()
    print('Run {} episodes in {} seconds and got {} average reward'.format(
        episodes, time / 1000, reward_sum / episodes))

    data_fetcher.save()


if __name__ == '__main__':
    run()
