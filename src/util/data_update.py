#!/usr/bin/python3
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

    folder = "results/obj/"
    files = get_all_pkl_files(folder)

    f = 'data_10000_Wolp3_Inv100k10#0.json.zip'

    d = data.load(folder + f)

    for episode in d.data['simulation']['episodes']:

        episode['rewards'] = 0
        episode['actions'] = 0
        episode['ndn_actions'] = 0
        episode['actors_actions'] = 0
    # episode['states'] = 0

    d.save('saved/')
