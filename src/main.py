import gym

import numpy as np

from util import *

from wolp_agent import *
from ddpg.agent import DDPGAgent
from util.data import Data
from util.data import Timer

AUTO_SAVE_AFTER_EPISODES = 500


def run(episodes=2500,
        collecting_data=False,
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
    data_fetcher = Data(file_name)

    data_fetcher.add_arrays(['experiment', 'max_actions', 'action_space',
                             'rewards', 'count', 'actions', 'done'])
    data_fetcher.add_arrays(['state_' + str(i) for i in range(agent.observation_space_size)])

    data_fetcher.add_timers(['render', 'act', 'step', 'saving'], 'run_')
    data_fetcher.add_timer('t_run_observe', one_hot=False)
    agent.add_data_fetch(data_fetcher)

    data_fetcher.add_to_array('experiment', experiment)
    data_fetcher.add_to_array('max_actions', max_actions)
    data_fetcher.add_to_array('action_space', agent.get_action_space())

    timer = Timer()
    full_epoch_timer = Timer()
    reward_sum = 0

    for ep in range(episodes):
        timer.reset()
        observation = env.reset()

        total_reward = 0
        print('Episode ', ep, '/', episodes - 1, 'started...', end='')
        for t in range(steps):

            data_fetcher.reset_timers()

            if not collecting_data:
                env.render()

            data_fetcher.sample_timer('render')  # ------

            action = agent.act(observation)

            data_fetcher.add_to_array('actions', action)  # -------

            data_fetcher.sample_timer('act')  # ------

            for i in range(agent.observation_space_size):
                data_fetcher.add_to_array('state_' + str(i), observation[i])
            prev_observation = observation
            observation, reward, done, info = env.step(action)

            episode = {'obs': prev_observation,
                       'action': action,
                       'reward': reward,
                       'obs2': observation,
                       'done': done,
                       't': t}

            data_fetcher.sample_timer('step')  # ------
            data_fetcher.add_to_array('count', 1)

            # print('\n' + str(episode['obs']))
            data_fetcher.start_timer('observe')
            agent.observe(episode)
            data_fetcher.sample_timer('observe')  # ------

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

                data_fetcher.sample_timer('saving')  # ------
                break
    # end of episodes
    time = full_epoch_timer.get_time()
    print('Run {} episodes in {} seconds and got {} average reward'.format(
        episodes, time / 1000, reward_sum / episodes))

    # printing average times for running and training steps
    data_fetcher.print_times(groups=['run_'])
    data_fetcher.print_times(groups=['agent_'], total_time_field='count')

    #
    data_fetcher.save()


if __name__ == '__main__':
    run()
