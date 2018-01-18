import numpy as np
import pickle
from timer import *
import os


class Data:

    def __init__(self, name='default_name'):
        self.name = name
        self.data = {}
        self.timers = {}
        self.temp_saves = 0

    def _add(self, field_name, timer, timer_one_hot=True):
        if field_name in self.data.keys():
            return
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

    def reset_fields(self):
        for key in self.data.keys():
            self.data[key] = np.array([])

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

    def merge(self, data):
        for key in self.data.keys():
            if key not in data.data.keys():
                continue

            final_data = np.concatenate((self.get_data(key), data.get_data(key)))
            self.set_data(key, final_data)

    def load(self, path=None):
        if path is None:
            path = self.name
        with open('results/obj/' + path + '.pkl', 'rb') as f:
            self.data = pickle.load(f)

    def temp_save(self):
        self.save(path='temp/{}_temp{}'.format(self.name, self.temp_saves), final_save=False)
        self.temp_saves += 1
        self.reset_fields()

    def save(self, path=None, final_save=True):
        if path == None:
            path = self.name

        if final_save and self.temp_saves > 0:
            clone_data = self.get_empty_clone()
            for i in range(self.temp_saves):
                temp_file = 'temp/{}_temp{}'.format(self.name, i)
                temp_data = Data()
                temp_data.load(path=temp_file)
                clone_data.merge(temp_data)
                os.remove('results/obj/' + temp_file + '.pkl')
            clone_data.merge(self)
            self.data = clone_data.data

        with open('results/obj/' + path + '.pkl', 'wb') as f:
            pickle.dump(self.data, f, 0)

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
            print("data.Data.print_times: No items found to be printed")
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
        res = Data(self.name + '_clone')
        res.add_arrays(self.get_keys())
        return res
