from time import time


class Timer():

    def __init__(self):
        self.reset()

    def reset(self):
        self.now = Timer._get_current_milis()
        self.dt_now = self.now

    def get_time(self):
        return Timer._get_current_milis() - self.now

    def dt(self):
        res = Timer._get_current_milis() - self.dt_now
        self.dt_now = Timer._get_current_milis()
        return res

    @staticmethod
    def _get_current_milis():
        return int(round(time() * 1000))


if __name__ == '__main__':
    t = Timer()
    print(t.get_time())
    temp = 0
    while(t.get_time() <= 1000):
        temp += t.dt()
        continue
    print(t.get_time(), 'dt', temp)
