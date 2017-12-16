import gym
import numpy as np

import os
import sys
import inspect
# Use this if you want to include modules from a subfolder
cmd_subfolder = os.path.realpath(os.path.abspath(os.path.join(
    os.path.split(inspect.getfile(inspect.currentframe()))[0], "util")))
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)


from util.timer import *
from util.data import *

from agent import *
# import util.performance_data.timer as timer
time_now = -1


def main():
    # eps = [10000, 5000, 5001, 2000, 2001, 2002]

    eps = [205]

    for i in eps:
        run(episodes=i,
            collecting_data=True)


def run(episodes=[10000], collecting_data=True):

    experiment = ('CartPole-v1',
                  'InvertedPendulum-v1',
                  'LunarLanderContinuous-v2')[1]
    env = gym.make(experiment)

    print(env.observation_space)
    print(env.action_space)

    steps = env.spec.timestep_limit

    agent = DDPGAgent(env)
    # agent = WolpertingerAgent(env, k_nearest_neighbors=1, max_actions=1e3, data_fetch=result_fetcher)
    # agent.load_expierience()
    # exit()
    # agent = DiscreteRandomAgent(env)


    # file_name = "results/data_" + agent.get_name() + str(episodes) + ".txt"
    file_name = "results/data_" + agent.get_name() + str(episodes)
    result_fetcher = Fulldata(file_name)
    result_fetcher.add_arrays(['rewards', 'count'])
    result_fetcher.add_timers(['render', 'act', 'step', 'saving'], 'run_')
    result_fetcher.add_timer('run_observe', one_hot=False)
    agent.add_data_fetch(result_fetcher)


    timer = Timer()

    for i in range(episodes):
        timer.reset()
        observation = env.reset()
        total_reward = 0
        print('Episode ', i, '/', episodes - 1, 'started', end='... ')
        for t in range(steps):

            result_fetcher.reset_timers()

            if not collecting_data:
                env.render()
            result_fetcher.sample_timer('render')  # ------

            action = agent.act(observation)
            result_fetcher.sample_timer('act')  # ------

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
                    if i % 100 == 0:
                        result_fetcher.async_save()
                result_fetcher.sample_timer('saving')  # ------
                break
    # end of episodes
    if collecting_data:
        result_fetcher.async_save()

    result_fetcher.print_times(groups=['run_'])
    result_fetcher.print_times(groups=['agent_'], total_time_field='count')



def save_episode(episode, overwrite=True):
    from pathlib import Path
    import datetime
    from os import makedirs

    string = str(episode).replace('},', '},\n')

    if overwrite:
        file = open('results/last_episode', 'w')
        file.write(string)
        file.close()
    else:
        now = datetime.datetime.now()

        dir_name = "results/%s-%s-%s" % (now.day, now.month, now.year)
        file = Path(dir_name)
        if not file.is_dir():
            makedirs(dir_name)

        counter = 0
        while True:
            file_name = dir_name + '/episode_%d.txt' % (counter)
            file = Path(file_name)
            if file.is_file():
                print(file_name + " exists")
                counter += 1
            else:
                file = open(file_name, 'w')
                file.write(string)
                file.close()
                break


if __name__ == '__main__':
    main()
