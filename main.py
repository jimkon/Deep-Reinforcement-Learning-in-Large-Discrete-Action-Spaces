import gym
import numpy as np
from agent import *
from timer import *
time_now = -1

# do i need exploration noise ?


def main():
    # eps = [10000, 5000, 5001, 2000, 2001, 2002]
    eps = [2000]
    for i in eps:
        run(episodes=i,
            collecting_data=False)


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
    # agent = DiscreteRandomAgent(env)

    episode_history = []
    reward_history = []

    timings = {'render': 0,
               'act': 0,
               'step': 0,
               'observe': 0,
               'saving': 0,
               'steps': 0,
               'total': 0}

    for i in range(episodes):
        timer = Timer()
        observation = env.reset()
        total_reward = 0
        print('Episode ', i, '/', episodes - 1, 'started', end='... ')
        for t in range(steps):

            timer.dt()  # reset dt timer
            if not collecting_data:
                env.render()
            timings['render'] += timer.dt()

            action = agent.act(observation)
            timings['act'] += timer.dt()

            prev_observation = observation
            observation, reward, done, info = env.step(action)
            episode = {'obs': prev_observation,
                       'action': action,
                       'reward': reward,
                       'obs2': observation,
                       'done': done,
                       't': t}

            timings['step'] += timer.dt()

            # if not collecting_data:
            #     episode_history.append(episode)

            # print('\n' + str(episode['obs']))

            agent.observe(episode)
            timings['observe'] += timer.dt()

            total_reward += reward
            if done or (t == steps - 1):
                t += 1
                if not collecting_data:
                    # save_episode(episode_history)
                    pass
                else:
                    reward_history.append(total_reward)
                    np.savetxt("results/reward_history_" + str(episodes) +
                               ".txt", np.array(reward_history), newline='\n')
                timings['saving'] += timer.dt()
                timings['steps'] += t
                time_passed = timer.get_time()
                timings['total'] += time_passed
                print('Reward:', total_reward, 'Steps:', t, 't:',
                      time_passed, '({}/step)'.format(round(time_passed / t)))

                break
    # end of episodes
    print('key\t\tabs\t\tavg/step\t%\\total\n-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-\nSteps:',
          timings['steps'])
    keys = list(timings.keys())
    keys.sort()
    for key in keys:
        if key is 'steps':
            continue
        temp = timings[key]
        print('{}\t\t{}\t\t{:6.2f}\t\t{:6.2f}'.format(key, temp,
                                                      temp / timings['steps'], 100 * temp / timings['total']))


def save_episode(episode, overwrite=True):
    from pathlib import Path
    import datetime
    from os import makedirs

    string = str(episode).replace('},', '\n').replace('{', '')

    if overwrite:
        file = open('results/clipboard.txt', 'w')
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
