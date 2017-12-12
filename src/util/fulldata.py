import numpy as np
import threading
from timer import *
from data_graph import *


def save_dictionary(dict, path):
    with open(path, 'w') as f:
        f.write(str(dict).replace(", \'", ',\n\'').replace('{', '{\n'))


class Fulldata:

    def __init__(self, name='default_name'):
        self.name = name
        self.data = {}

    def add_array(self, field_name):
        self.data[field_name] = np.array([])

    def add_to(self, field_name, value):
        self.data[field_name] = np.append(self.data[field_name], value)

    def get_data(self, field_name):
        return self.data[field_name]

    def print_data(self):
        for key in self.data.keys():
            print(key, self.data[key].shape, self.data[key])

    def load(self, path=None):
        from numpy import array
        if path is None:
            path = self.name
        with open(self.name, 'r') as f:
            self.data = eval(f.read())

    def async_save(self):
        thread = save_fulldata(self)
        thread.start()


class save_fulldata(threading.Thread):
    def __init__(self, fd):
        threading.Thread.__init__(self)
        self.dict = fd.data
        self.path = fd.name

    def run(self):
        save_dictionary(self.dict, self.path)


if __name__ == '__main__':
    fd = Fulldata()
    # fd.add_array('array_1')
    # for i in range(10):
    #     fd.add_to('array_1', i)
    # fd.add_array('array_2')
    # for i in range(10):
    #     fd.add_to('array_2', i)
    # fd.add_array('array_3')
    # for i in range(20):
    #     fd.add_to('array_3', i)
    # fd.print_data()
    #
    # fd.async_save()
    fd.load()
    fd.print_data()
