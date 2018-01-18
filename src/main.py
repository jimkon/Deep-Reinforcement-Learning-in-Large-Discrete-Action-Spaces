import gym

import numpy as np

from util import *

from wolp_agent import *

<<<<<<< HEAD

def run(episodes=250, collecting_data=True):

    experiment = ('InvertedPendulum-v1')
=======

def run(episodes=30,
        collecting_data=False,
        experiment='InvertedPendulum-v1',
        max_actions=1e3,
        knn=0.1):

>>>>>>> master
    env = gym.make(experiment)

    steps = env.spec.timestep_limit

<<<<<<< HEAD
    max_actions = 1e2
    agent = WolpertingerAgent(env, k_nearest_neighbors=int(0.1 * max_actions),
                              max_actions=max_actions)
=======
    # agent = DDPGAgent(env)
    agent = WolpertingerAgent(env, max_actions=max_actions, k_ratio=knn)
>>>>>>> master

    file_name = "data_" + str(episodes) + '_' + agent.get_name()
<<<<<<< HEAD

    rewards = []
    for i in range(episodes):
=======
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
    full_epoch_timer = Timer()
    reward_sum = 0

    for ep in range(episodes):
        timer.reset()
>>>>>>> master
        observation = env.reset()

        total_reward = 0
        print('Episode ', ep, '/', episodes - 1, 'started...', end='')
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
<<<<<<< HEAD
            if done or (t == steps - 1):
                t += 1
                rewards.append(total_reward)
                print('Reward:', total_reward, 'Steps:', t)
                break

    # end of episodes
    print('Average rewards:', np.average(rewards))
=======
            result_fetcher.add_to_array('done', 1 if done else 0)

            if done or (t == steps - 1):
                t += 1
                result_fetcher.add_to_array('rewards', total_reward)  # ------
                reward_sum += total_reward
                time_passed = timer.get_time()
                print('Reward:{} Steps:{} t:{} ({}/step) Cur avg={}'.format(total_reward, t,
                                                                            time_passed, round(
                                                                                time_passed / t),
                                                                            round(reward_sum / (ep + 1))))

                if ep % 500 == 0:
                    result_fetcher.temp_save()

                result_fetcher.sample_timer('saving')  # ------
                break
    # end of episodes
    time = full_epoch_timer.get_time()
    print('Run {} episodes in {} seconds and got {} average reward'.format(
        episodes, time / 1000, reward_sum / episodes))
    result_fetcher.save()
    # result_fetcher.print_data()

    # result_fetcher.print_times(groups=['run_'])
    # result_fetcher.print_times(groups=['agent_'], total_time_field='count')
>>>>>>> master


if __name__ == '__main__':
    run()
