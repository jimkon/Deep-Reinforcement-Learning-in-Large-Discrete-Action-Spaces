#!/usr/bin/python3
import data_old
import data
import numpy as np
import sys
sys.path.insert(
    0, "/home/jim/Desktop/dip/Deep-Reinforcement-Learning-in-Large-Discrete-Action-Spaces/src/")
import action_space


def get_all_pkl_files(directory):
    from os import listdir
    from os.path import isfile, join, dirname, realpath, splitext

    mypath = directory
    # mypath = DIRECTORY
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    files = []
    for f in onlyfiles:
        if splitext(f)[1] in '.zip':
            files.append(f)
    return files


def update_pickle_file(file_name, eps=0, k=0, v=0):
    d_old = data_old.Data(file_name)
    d_old.load()
    print(file_name, 'loaded')
    # d_old.print_fields()

    d_new = data.Data()
    d_new.set_agent('Wolp',
                    int(d_old.get_data('max_actions')[0]),
                    k,
                    v)
    d_new.set_experiment(d_old.get_data('experiment')[0],
                         [-3],
                         [3],
                         eps)

    space = action_space.Space([-3], [3], int(d_old.get_data('max_actions')[0]))
    # print(space.get_space())
    # d_new.print_data()

    done = d_old.get_data('done')
    actors_result = d_old.get_data('actors_result')
    actions = d_old.get_data('actions')
    state_0 = d_old.get_data('state_0').tolist()
    state_1 = d_old.get_data('state_1').tolist()
    state_2 = d_old.get_data('state_2').tolist()
    state_3 = d_old.get_data('state_3').tolist()
    rewards = d_old.get_data('rewards').tolist()
    ep = 0
    temp = 0
    l = len(done)
    for i in range(l):
        d_new.set_action(space.import_point(actions[i]).tolist())
        d_new.set_actors_action(space.import_point(actors_result[i]).tolist())
        d_new.set_ndn_action(space.import_point(
            space.search_point(actors_result[i], 1)[0]).tolist())
        state = [state_0[i], state_1[i], state_2[i], state_3[i]]
        d_new.set_state(state)
        d_new.set_reward(1)
        if done[i] > 0:
            # print(ep, i - temp, 'progress', i / l)
            temp = i

            ep += 1
            # if ep % 200 == 199:
            #     d_new.finish_and_store_episode()
            # else:
            d_new.end_of_episode()

    d_new.save()


if __name__ == "__main__":
    # update_pickle_file('data_10001_Wolp3_1000k1000_InvertedPendulum-v1',
    #                    eps=10001, k=1000, v=3)

    # update_pickle_file('data_10000_Wolp3_100k10_InvertedPendulum-v1',
    #                    eps=10000, k=10, v=4)
    # update_pickle_file('data_10000_Wolp3_10000k1000_InvertedPendulum-v1',
    #                    eps=10000, k=1000, v=4)
    # update_pickle_file('data_10000_Wolp3_1000k100_InvertedPendulum-v1',
    #                    eps=10000, k=100, v=3)

    # print(np.ones(10).tolist())
    # exit()
    # folder = "results/obj/"
    # files = get_all_pkl_files(folder)
    # count = 0
    # for f in files:
    #     # if f == 'data_10000_agen4_exp1000k10#0.json.zip' or f == 'data_10000_Wolp3_Inv10k10#0.json.zip':
    #     #     continue
    #     f = 'data_10001_Wolp3_Inv1000k1000#0.json.zip'
    #     print(f)
    #     # continue
    #     d = data.load(folder + f)
    #     d.data['agent']['version'] = 3
    #     for episode in d.data['simulation']['episodes']:
    #         print(episode['rewards'])
    #         reward = int(episode['rewards'][0][0])
    #         # print(reward)
    #
    #         episode['rewards'] = np.ones(reward).tolist()
    #         # print(episode['rewards'])
    #
    #     d.save()
    # exit()
    #     # update_pickle_file(f, eps=count)
    #     count += 1
    #     exit()
