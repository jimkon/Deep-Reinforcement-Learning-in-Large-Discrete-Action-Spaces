from time import time


class Timer:

    def __init__(self, one_hot=True):
        self.reset()
        self.one_hot = one_hot

    def reset(self):
        self.now = Timer._get_current_milis()

    def reset_one_hot(self):
        if self.one_hot:
            self.reset()

    def get_time(self, reset=False):
        return Timer._get_current_milis() - self.now

    @staticmethod
    def _get_current_milis():
        return int(round(time() * 1000))


class Time_stats:

    def __init__(self, name, fields, one_active=True):
        self.name = name
        self.count = 0
        self.one_active = one_active
        self.values = {}
        self.timers = {}
        for str in fields:
            self.values[str] = 0
            self.timers[str] = Timer()

    def start(self, field):
        self.timers[field].reset()

    def add_time(self, field):
        self.values[field] += self.timers[field].get_time()
        if self.one_active:
            self.reset_timers()

    def increase_count(self, n=1):
        self.count += n

    def set_count(self, n):
        self.count = n

    def get_count(self):
        return self.count

    def reset_timers(self):
        for key in self.timers.keys():
            self.start(key)

    def reset_values(self):
        for key in self.values.keys():
            self.values[key] = 0

    def get_total(self):
        total = 0
        for key in self.values.keys():
            total += self.values[key]
        return total

    def print_stats(self):
        print('\nName: {}\tCount: {}'.format(self.name, self.count))
        print('key\t\tabs\t\tavg/unit\t% of total')
        print('-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-')

        keys = list(self.values.keys())
        keys.sort()
        total_time = max(self.get_total(), 1)
        count = max(self.count, 1)
        for key in keys:
            temp = self.values[key]
            avg = temp / count
            total = 100 * temp / total_time
            print('{}\t\t{}\t\t{:6.2f}\t\t{:6.2f}'.format(
                key, temp, avg, total))

        total_time = self.get_total()
        print('Total\t\t{}\t\t{:6.2f}\t\t 100.0'.format(total_time, total_time / count))
