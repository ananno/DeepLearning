import numpy as np
import random
import time

class Dataset:
    def __init__(self,
                 total_data=1000, # train+test = 1000 data
                 input_size=3, # how many inputs
                 batch_size=3, # training data per training
                 test_train_ratio=0.1): # 10 train-dataset = 1 test-dataset

        self.batch_size        = batch_size
        self.total_data        = total_data
        self.test_train_ratio  = 0.1
        self.input_size        = input_size
        self.train_data        = None
        self.test_data         = None

        self.current_index   = 0
        self.test_data_count = 0
        self.current_epoch   = 0
        self.total_batch_per_epoch = 0

        # Ready the datasets
        self.generate()
        self.shuffle_data(self.train_data)

    def generate(self):

        # generate inputs
        np.random.seed(int(time.time()))
        data = np.array(np.random.random((self.total_data, self.input_size)) >= 0.5, dtype=np.int32)

        # copy data to input_data
        input_data = data.copy()

        # set label data for the inputs
        label_data, _ = np.split(data, indices_or_sections=[1, ], axis=-1)

        # set the dataset that is about to split into
        # test and train dataset
        raw_data   = [input_data, label_data]

        self.split_training_testing(raw_data)

    def shuffle_data(self, data_pair):
        combind_data = list(zip(*data_pair))
        random.shuffle(combind_data)
        return list(zip(*combind_data))

    def split_training_testing(self, raw_data):
        # set the number of testing dataset
        self.test_data_count = int(self.total_data * self.test_train_ratio)

        # set test and train dataset
        test_data  = list(zip(*raw_data))[:self.test_data_count]
        train_data = list(zip(*raw_data))[:-self.test_data_count]
        self.train_data = list(zip(*train_data))
        self.test_data  = list(zip(*test_data))
        self.total_batch_per_epoch = len(self.train_data) // self.batch_size


    def get_next(self):
        if (self.current_index + 1) >= self.total_batch_per_epoch:
            self.current_index = 0
            self.current_epoch += 1
            self.shuffle_data(self.train_data)

        batch_input, batch_label = list(zip(*list(zip(*self.train_data))[self.current_index:self.current_index+self.batch_size]))
        self.current_index += 1
        batch_data = [batch_input, batch_label]
        return batch_data

