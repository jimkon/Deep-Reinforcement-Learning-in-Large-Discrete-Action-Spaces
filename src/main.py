import gym
import numpy as np

from agent import *
from util import *
# import util.performance_data.timer as timer
time_now = -1


def main():
    # eps = [10000, 5000, 5001, 2000, 2001, 2002]
    eps = [20]
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
    # agent = WolpertingerAgent(env, k_nearest_neighbors=1, max_actions=1e3)
    # agent.load_expierience()
    # exit()
    # agent = DiscreteRandomAgent(env)

    episode_history = []
    reward_history = []
    file_name = "results/reward_history_" + agent.get_name() + str(episodes) + ".txt"
    timer = Timer()
    episode_timings = Time_stats("Episode times",
                                 ['render', 'act', 'step', 'observe', 'saving'])
    for i in range(episodes):
        timer.reset()
        observation = env.reset()
        total_reward = 0
        print('Episode ', i, '/', episodes - 1, 'started', end='... ')
        for t in range(steps):

            episode_timings.reset_timers()

            if not collecting_data:
                env.render()
            episode_timings.add_time('render')

            action = agent.act(observation)
            episode_timings.add_time('act')

            prev_observation = observation
            observation, reward, done, info = env.step(action)
            episode = {'obs': prev_observation,
                       'action': action,
                       'reward': reward,
                       'obs2': observation,
                       'done': done,
                       't': t}

            episode_timings.add_time('step')

            if collecting_data:
                episode_history.append(episode)
                episode_timings.add_time('saving')

            # print('\n' + str(episode['obs']))

            agent.observe(episode)
            episode_timings.add_time('observe')

            total_reward += reward
            if done or (t == steps - 1):
                t += 1
                if not collecting_data:
                    # save_episode(episode_history)
                    pass
                else:
                    reward_history.append(total_reward)
                    if i % 100 == 0:
                        np.savetxt(file_name, np.array(reward_history), newline='\n')
                        save_episode(episode_history)

                episode_timings.add_time('saving')

                episode_timings.increase_count(n=t)

                time_passed = timer.get_time()
                print('Reward:', total_reward, 'Steps:', t, 't:',
                      time_passed, '({}/step)'.format(round(time_passed / t)))

                break
    # end of episodes
    if collecting_data:
        np.savetxt(file_name, np.array(reward_history), newline='\n')
        save_episode(episode_history)

    agent.train_timings.set_count(episode_timings.get_count())
    agent.get_train_timings().print_stats()
    episode_timings.print_stats()


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
