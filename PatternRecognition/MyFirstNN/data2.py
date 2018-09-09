import numpy as np
import random
import time


class Dataset:
    def __init__(self,
                 size=3,
                 length=1000,
                 batch_size=3):

        self.size = size
        self.length = length
        self.batch_size = batch_size
        self.train_data = None
        self.test_data = None
        self.current_indx = 0
        self.current_epoch = 0
        self.total_batch_per_epoch = 0
        self.test_data_count = 0

        self.generate()
        self.shuffle_data(self.train_data)

    def generate(self):
        random.seed(time.time)
        data = np.array(np.random.random((self.length, self.size)) >= 0.5, dtype=np.int32)
        input_data = data.copy()

        label_data, _ = np.split(data, indices_or_sections=[1, ], axis=-1)
        label_data = np.array(label_data, dtype=np.int32)
        raw_data = [input_data, label_data]

        self.split_training_testing(raw_data)

    def shuffle_data(self, data_pair):
        combind_data = list(zip(*data_pair))
        random.shuffle(combind_data)
        return list(zip(*combind_data))

    def split_training_testing(self, raw_data):
        self.test_data_count = int(self.length * 0.1)
        test_data = list(zip(*raw_data))[:self.test_data_count]
        train_data = list(zip(*raw_data))[:-self.test_data_count]

        self.train_data = list(zip(*train_data))
        self.test_data = list(zip(*test_data))

        self.total_batch_per_epoch = len(self.train_data) // self.batch_size

    def get_next(self):
        if self.current_indx+1 >= self.total_batch_per_epoch:
            self.current_indx = 0
            self.current_epoch += 1
            self.shuffle_data(self.train_data)

        batch_input, batch_label = list(zip(*list(zip(*self.train_data))[self.current_indx:self.current_indx+self.batch_size]))
        self.current_indx += 1

        batch_data = [batch_input, batch_label]
        return batch_data
