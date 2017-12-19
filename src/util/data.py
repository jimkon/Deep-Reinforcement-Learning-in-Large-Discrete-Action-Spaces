import numpy as np
import threading
import pickle
from timer import *


def save_dictionary(dict, path):
    with open('results/obj/' + path + '.pkl', 'wb') as f:
        pickle.dump(dict, f, 0)
    # f = open(path, 'w+')
    # f.write(str(dict).replace(", \'", ',\n\'').replace('{', '{\n'))
    # f.close()
    # with open(path, 'w+') as f:
    #     f.write(str(dict).replace(", \'", ',\n\'').replace('{', '{\n'))


class Fulldata:

    def __init__(self, name='default_name'):
        self.name = name
        self.data = {}
        self.timers = {}

    def _add(self, field_name, timer, timer_one_hot=True):
        self.data[field_name] = np.array([])
        if timer:
            self.timers[field_name] = Timer(timer_one_hot)

    def add_array(self, field_name):
        self._add(field_name, False)
        # self.data[field_name] = np.array([])

    def add_arrays(self, fields, prefix=''):
        for f in fields:
            self.add_array(prefix + f)

    def add_to_array(self, field_name, value, abs_name=False):
        if abs_name:
            self.data[field_name] = np.append(self.data[field_name], value)
        else:
            fields = self.get_keys(field_name)
            for f in fields:
                self.data[f] = np.append(self.data[f], value)

    def add_timer(self, field_name, one_hot=True):
        self._add(field_name, True, one_hot)
        # self.add_array(name)
        # self.timers[name] = Timer()

    def add_timers(self, names, prefix='', one_hot=True):
        for f in names:
            self.add_timer(prefix + f, one_hot)

    def start_timer(self, field_name):
        fields = self.get_keys(field_name)
        for f in fields:
            self.timers[f].reset()

    def sample_timer(self, field_name, abs_name=False):
        if abs_name:
            self.data[field_name] = np.append(
                self.data[field_name], self.timers[field_name].get_time())
        else:
            fields = self.get_keys(field_name)
            timer_keys = self.timers.keys()
            for f in fields:
                if f in timer_keys:
                    self.data[f] = np.append(self.data[f], self.timers[f].get_time())

        self.reset_timers_one_hot()

    def reset_timers(self):
        for t in self.timers:
            self.timers[t].reset()

    def reset_timers_one_hot(self):
        for t in self.timers:
            self.timers[t].reset_one_hot()

    def set_data(self, field_name, data):
        self.data[field_name] = data

    def get_data(self, field_name):
        return self.data[field_name]

    def print_data(self, field_name=''):
        keys = list(self.get_keys(field_name))
        keys.sort()
        for key in keys:
            print(key, self.data[key].shape, self.data[key])

    def print_fields(self):
        for k in self.get_keys():
            print(k)

    def load(self, path=None):
        if path is None:
            path = self.name
        with open('results/obj/' + path + '.pkl', 'rb') as f:
            self.data = pickle.load(f)
        # from numpy import array
        # if path is None:
        #     path = self.name
        # with open(self.name, 'r') as f:
        #     self.data = eval(f.read())
            # print(self.data)

    def async_save(self):
        thread = save_fulldata(self)
        thread.start()

    def print_times(self, other_keys=None, groups=None, total_time_field=None):
        final_keys = []
        if (other_keys is None) and (groups is None):
            final_keys = self.timers.keys()
        else:
            if other_keys is not None:
                final_keys.extend(other_keys)

            if groups is not None:
                timers = self.timers.keys()
                for g in groups:
                    for t in timers:
                        if g in t:
                            final_keys.append(t)

        if (final_keys is None) or (len(final_keys) == 0):
            print("No items found to be printed")
            return

        times = {}
        total_time = 0
        samples = []

        for key in final_keys:
            times[key] = np.sum(self.get_data(key))
            total_time += times[key]

            samples.append(len(self.get_data(key)))

        count = max(samples)
        if total_time_field is not None:
            count = np.sum(self.get_data(total_time_field))

        print('\n\nName: {}\tCount: {} Group:{}'.format(self.name, count, groups))
        print('key\t\tabs\t\tavg/unit\t% of total')
        print('-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-')

        keys = list(final_keys)
        keys.sort()
        max_key_len = 5
        for key in keys:
            if max_key_len < len(key):
                max_key_len = len(key)

        for key in keys:
            temp = times[key]
            avg = temp / count
            total = 100 * temp / total_time
            print('{}{}\t\t{}\t\t{:6.2f}\t\t{:6.2f}'.format(
                key, '.' * (max_key_len - len(key)), temp, avg, total))

        print('Total{}\t\t{}\t\t{:6.2f}\t\t 100.0'.format(
            '.' * (max_key_len - 5), total_time, total_time / count))

    def get_keys(self, key=''):
        res = []
        for k in self.data.keys():
            # if k.find(key) >= 0:
            if key in k:
                res.append(k)

        return res

    def get_empty_clone(self):
        res = Fulldata(self.name + '_clone')
        res.add_arrays(self.get_keys())
        return res


class save_fulldata(threading.Thread):
    def __init__(self, fd):
        threading.Thread.__init__(self)
        self.dict = fd.data
        self.path = fd.name

    def run(self):
        save_dictionary(self.dict, self.path)


if __name__ == '__main__':
    n = 2511
    fd = Agent_data(
        name='data_Wolp_betaDDPGAgent' + str(n))

    fd.load()

    print(fd.find_episode(1))
