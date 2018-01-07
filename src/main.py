import gym

import numpy as np

from util import *

from wolp_agent import *
from ddpg.agent import DDPGAgent
from util.data import Data
from util.data import Timer

time_now = -1


def run(episodes=2500, collecting_data=True, experiment='InvertedPendulum-v1'):

    env = gym.make(experiment)

    print(env.observation_space)
    print(env.action_space)

    steps = env.spec.timestep_limit

    # agent = DDPGAgent(env)
    max_actions = 1e2
    agent = WolpertingerAgent(env, max_actions=max_actions)

    # file_name = "results/data_" + agent.get_name() + str(episodes) + ".txt"
    file_name = "data_" + str(episodes) + '_' + agent.get_name()
    print(file_name)
    result_fetcher = Data(file_name)

    result_fetcher.add_arrays(['experiment', 'max_actions', 'action_space',
                               'rewards', 'count', 'actions', 'done'])
    result_fetcher.add_arrays(['state_' + str(i) for i in range(agent.observation_space_size)])

    result_fetcher.add_timers(['render', 'act', 'step', 'saving'], 'run_')
    result_fetcher.add_timer('t_run_observe', one_hot=False)
    agent.add_data_fetch(result_fetcher)

    result_fetcher.add_to_array('experiment', experiment)
    result_fetcher.add_to_array('max_actions', max_actions)
    result_fetcher.add_to_array('action_space', agent.get_action_space())

    timer = Timer()

    for i in range(episodes):
        timer.reset()
        observation = env.reset()
        # for i in range(agent.observation_space_size):
        #     result_fetcher.add_to_array('state_' + str(i), observation[i])

        total_reward = 0
        print('Episode ', i, '/', episodes - 1, 'started...', end='')
        for t in range(steps):

            result_fetcher.reset_timers()

            if not collecting_data:
                env.render()

            result_fetcher.sample_timer('render')  # ------

            action = agent.act(observation)

            result_fetcher.add_to_array('actions', action)  # -------

            result_fetcher.sample_timer('act')  # ------

            for i in range(agent.observation_space_size):
                result_fetcher.add_to_array('state_' + str(i), observation[i])
            prev_observation = observation
            observation, reward, done, info = env.step(action)

            episode = {'obs': prev_observation,
                       'action': action,
                       'reward': reward,
                       'obs2': observation,
                       'done': done,
                       't': t}

            result_fetcher.sample_timer('step')  # ------
            result_fetcher.add_to_array('count', 1)

            # print('\n' + str(episode['obs']))
            result_fetcher.start_timer('observe')
            agent.observe(episode)
            result_fetcher.sample_timer('observe')  # ------

            total_reward += reward
            result_fetcher.add_to_array('done', 1 if done else 0)
            if done or (t == steps - 1):
                t += 1
                result_fetcher.add_to_array('rewards', total_reward)  # ------

                time_passed = timer.get_time()
                print('Reward:', total_reward, 'Steps:', t, 't:',
                      time_passed, '({}/step)'.format(round(time_passed / t)))

                if not collecting_data:
                    # save_episode(episode_history)
                    pass
                else:
                    pass
                    # if i % 100 == 0:
                    # result_fetcher.async_save()
                result_fetcher.sample_timer('saving')  # ------
                break
    # end of episodes

    result_fetcher.async_save()
    # result_fetcher.print_data()

    result_fetcher.print_times(groups=['run_'])
    result_fetcher.print_times(groups=['agent_'], total_time_field='count')


if __name__ == '__main__':
    run()
