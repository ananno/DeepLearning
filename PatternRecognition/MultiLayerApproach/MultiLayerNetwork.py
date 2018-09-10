import numpy as np
import time
from .dataset import Dataset

class NeuronLayer:
    def __init__(self,
                 number_of_neurons,
                 inputs_connected_to_each_neuron):

        # Dimension of the weight matrix
        shape = (
            inputs_connected_to_each_neuron,
            number_of_neurons
        )
        np.random.seed(int(time.time()))
        self.synaptic_weights = np.random.random(shape)
        self.output = None
        self.loss   = None
        self.grad   = None
        self.cost   = None

class Network:
    def __init__(self, layer1, layer2):

        # Define layers for this network
        self.layer1 = layer1
        self.layer2 = layer2

        # generates dataset
        self.dataset = Dataset()


    # Sigmoid function
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # The derivative of the Sigmoid function.
    # This is the gradient of the Sigmoid curve.
    # It indicates how confident we are about the existing weight.
    @staticmethod
    def sigmoid_derivative(x):
        return x * (1 - x)



    # Training
    def train(self, no_of_training=10000):
        print("Training is being initialized...")
        training_started = time.time()
        # training loop
        for iteration in range(no_of_training):
            # get batch data
            input_data, input_label = self.dataset.get_next()
            input_data = np.array(input_data)
            input_label = np.array(input_label)

            # Output matrix
            self.layer1.output, self.layer2.output = self.activation(input_data)

            # error matrix
            self.layer2.loss = input_label - self.layer2.output
            self.layer2.grad = self.layer2.loss * self.sigmoid(self.layer2.output)

            self.layer1.loss = self.layer2.grad.dot(self.layer2.synaptic_weights.T)
            self.layer1.grad = self.layer1.loss * self.sigmoid_derivative(self.layer1.output)

            self.layer1.cost = input_data.T.dot(self.layer1.grad)
            self.layer2.cost = self.layer1.output.T.dot(self.layer2.grad)

            self.layer1.synaptic_weights += self.layer1.cost
            self.layer2.synaptic_weights += self.layer2.cost

            print("Step %s : L1_loss: %s, L2_loss: %s" % (
                iteration,
                round(np.average(np.sum(self.layer1.loss, axis=-1)), 8),
                round( np.average(np.sum(self.layer2.loss, axis=-1)), 8)
            ))

    # activation function
    def activation(self, inputs):
        layer1_output = self.sigmoid(
            np.dot(inputs, self.layer1.synaptic_weights)
        )
        layer2_output = self.sigmoid(
            np.dot(layer1_output, self.layer2.synaptic_weights)
        )
        return layer1_output, layer2_output

    # Think
    def think(self, inputs):
        result = self.sigmoid(
            np.dot(inputs, self.layer2.synaptic_weights)
        )
        return round(result[0][0], 2)


    def test(self):
        test_data = self.dataset.test_data
        true_count = 0
        for i, (test_input, test_label) in enumerate(zip(*test_data)):
            output = self.think([test_input])
            if output ==  test_label[0]:
                true_count += 1

            print("Step: %s, Predicted: %s, Expected: %s, Accuracy: %s" % (
                i, output, test_label[0], round(true_count/(i+1), 2)
            ))

        print("\n\tFINAL ACCURACY OVER ALL TEST:", true_count/self.dataset.test_data_count, "\n")
