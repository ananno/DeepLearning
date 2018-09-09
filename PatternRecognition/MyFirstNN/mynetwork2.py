import numpy as np
import time
from .data2 import Dataset


class Network:
    def __init__(self):
        print("\n\t\tRECOGNITION OF A CERTAIN PATTERN")
        print("\n\t\t--------------------------------")
        # generates dataset
        self.dataset = Dataset()

        shape = (self.dataset.batch_size, 1)

        # random weights of 3x1 matrix, with values in the range -1 to 1
        self.synaptic_weights = np.random.random(shape)

        # Print initial weights
        print("Initializing random weights...")

    # Sigmoid function
    @staticmethod
    def sigmoid_the_normalizer(x):
        return 1 / (1 + np.exp(-x))

    # The derivative of the Sigmoid function.
    # This is the gradient of the Sigmoid curve.
    # It indicates how confident we are about the existing weight.
    @staticmethod
    def sigmoid_derivative(x):
        return x * (1 - x)

    # Training
    def train(self, no_of_training=10000):
        print("Starting Training is in progress...")
        training_started = time.time()
        # training loop
        for iteration in range(no_of_training):
            # get batch data
            input_data, input_label = self.dataset.get_next()
            input_data = np.array(input_data)
            input_label = np.array(input_label)

            # Output matrix
            output = self.activation(input_data)

            # error matrix
            loss = output - input_label

            # optimization gradient
            grad = np.dot(input_data.transpose(), loss * self.sigmoid_derivative(output))

            # update weights
            self.synaptic_weights += grad

            print("Step %s : Loss: %s" % (iteration, np.average(np.sum(loss, axis=-1))))

        # print out relevant info.
        print("Training has been completed.")
        print("After training", no_of_training, 'times in',
              str(round(time.time() - training_started, 2)) + 's.')

    # activation function
    def activation(self, inputs):
        dot_product = np.dot(inputs, self.synaptic_weights)
        return self.sigmoid_the_normalizer(dot_product)

    def accuracy(self, network_output, expected_output):
        correct = 0
        for i, j in zip(network_output, expected_output):
            if round(i[0], 2) == j[0]:
                correct += 1

        result = (correct / float(len(network_output))) * 100
        return round(result, 2)

    def test(self):
        test_data = self.dataset.test_data
        true_count = 0
        for i, (test_input, test_label) in enumerate(zip(*test_data)):
            print("Step %s", i)
            # print(test_input)

            print("Prediction...")
            output = self.activation([test_input])
            output = self.accuracy(output, [test_label])
            print(output)

            print("Expected...")
            print(test_label)

            if output == test_label:
                true_count += 1

            print("Accuracy: ", true_count/(i+1))

        print("\nFinal Accuracy: ", true_count/self.dataset.test_data_count)
