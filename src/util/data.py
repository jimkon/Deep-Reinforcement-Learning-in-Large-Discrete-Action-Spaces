import numpy as np
import threading
from timer import *


def save_dictionary(dict, path):
    f = open(path, 'w+')
    f.write(str(dict).replace(", \'", ',\n\'').replace('{', '{\n'))
    f.close()
    # with open(path, 'w+') as f:
    #     f.write(str(dict).replace(", \'", ',\n\'').replace('{', '{\n'))


class Fulldata:

    def __init__(self, name='default_name'):
        self.name = name
        self.data = {}
        self.timers = {}

    def _add(self, field_name, timer, timer_one_hot=False):
        self.data[field_name] = np.array([])
        if timer:
            self.timers[field_name] = Timer(timer_one_hot)

    def add_array(self, field_name):
        self._add(field_name, False)
        # self.data[field_name] = np.array([])

    def add_arrays(self, fields, prefix=''):
        for f in fields:
            self.add_array(prefix + f)

    def add_to_array(self, field_name, value):
        fields = self.get_keys(field_name)
        for f in fields:
            self.data[f] = np.append(self.data[f], value)

    def add_timer(self, field_name, one_hot=False):
        self._add(field_name, True, one_hot)
        # self.add_array(name)
        # self.timers[name] = Timer()

    def add_timers(self, names, prefix='', one_hot=False):
        for f in names:
            self.add_timer(prefix + f, one_hot)

    def start_timer(self, name):
        self.timers[name].reset()

    def sample_timer(self, field_name):
        fields = self.get_keys(field_name)
        for f in fields:
            self.data[f] = np.append(self.data[f], self.timers[f].get_time())

        self.reset_timers_one_hot()

    def reset_timers(self):
        for t in self.timers:
            self.timers[t].reset()

    def reset_timers_one_hot(self):
        for t in self.timers:
            self.timers[t].reset_one_hot()

    def get_data(self, field_name):
        return self.data[field_name]

    def print_data(self, field_name=''):
        keys = list(self.get_keys(field_name))
        keys.sort()
        for key in keys:
            print(key, self.data[key].shape, self.data[key])

    def load(self, path=None):
        from numpy import array
        if path is None:
            path = self.name
        with open(self.name, 'r') as f:
            self.data = eval(f.read())
            # print(self.data)

    def async_save(self):
        thread = save_fulldata(self)
        thread.start()

    def print_times(self, other_keys=None, groups=None):
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

        # if len(res) == 1:
        #     return res[0]
        return res


class save_fulldata(threading.Thread):
    def __init__(self, fd):
        threading.Thread.__init__(self)
        self.dict = fd.data
        self.path = fd.name

    def run(self):
        save_dictionary(self.dict, self.path)


if __name__ == '__main__':
    import time

    fd = Fulldata(
        name='/home/jim/Desktop/dip/Deep-Reinforcement-Learning-in-Large-Discrete-Action-Spaces/src/util/default_name')
    fd.add_array('array_1')
    fd.add_timers(['timer_1', 'timer_2'], one_hot=True)
    for i in range(10):
        fd.add_to_array('array_1', i)
    fd.add_array('array_2')
    for i in range(10):
        fd.add_to_array('array_2', i)
    fd.add_array('array_3')
    for i in range(20):
        fd.add_to_array('array_3', i)

    fd.add_to_array('array', 420)
    time.sleep(0.5)

    fd.add_timers(['timer_3', 'timer_4', 'ti'])
    time.sleep(0.7)
    fd.sample_timer('timer')
    #
    # fd.async_save()
    #

    fd.print_data()
    # fd.load()
    # print(fd.get_keys())
    fd.print_times(other_keys=['array_1'], groups=['timer'])
