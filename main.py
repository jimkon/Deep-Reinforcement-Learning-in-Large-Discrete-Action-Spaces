import gym
import numpy as np
from agent import *
from time import time
# from ou_noise import OUNoise
time_now = -1

# do i need exploration noise ?


def main():
    eps = [10000, 5000, 5001, 2000, 2001, 2002]
    for i in eps:
        run(episodes=i)


def run(episodes=10000):
    collecting_data = True

    experiment = ('CartPole-v1',
                  'InvertedPendulum-v1',
                  'LunarLander-v2')[1]
    env = gym.make(experiment)

    print(env.observation_space)
    print(env.action_space)

    steps = env.spec.timestep_limit
    # agent = DDPGAgent(env)
    agent = WolpertingerAgent(env, k_nearest_neighbors=10, max_actions=1e3)
    # agent = DiscreteRandomAgent(env)
    # exploration_noise = OUNoise(agent.action_space_size)

    episode_history = []
    reward_history = []
    for i in range(episodes):
        delta_t(reset=True)
        observation = env.reset()
        total_reward = 0
        print('Episode ', i, '/', episodes - 1, 'started', end='... ')
        for t in range(steps):

            if not collecting_data:
                env.render()

            action = agent.act(observation)  # + exploration_noise.noise()

            prev_observation = observation
            observation, reward, done, info = env.step(action)
            episode = {'obs': prev_observation,
                       'action': action,
                       'reward': reward,
                       'obs2': observation,
                       'done': done,
                       't': t}

            episode_history.append(episode)

            # print('\n' + str(episode['obs']))

            agent.observe(episode)

            total_reward += reward
            if done or (t == steps - 1):
                reward_history.append(total_reward)
                time_passed = delta_t()
                print('Reward:', total_reward, 'Steps:', t, 't=',
                      time_passed, '({}/step)'.format(round(time_passed / t)))

                if not collecting_data:
                    save_episode(episode_history)
                # exploration_noise.reset()  # reinitializing random noise for action exploration
                break

        np.savetxt("results/reward_history_" + str(episodes) +
                   ".txt", np.array(reward_history), newline='\n')


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


def delta_t(reset=False):

    global time_now
    if reset:
        time_now = -1

    if time_now == -1:
        time_now = int(round(time() * 1000))
    return int(round(time() * 1000)) - time_now


if __name__ == '__main__':
    main()
