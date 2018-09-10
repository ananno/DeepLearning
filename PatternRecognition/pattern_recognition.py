from MyFirstNN.MyNetwork import Network
from MyFirstNN import data
import sys

if __name__ == "__main__":

    try:
    	no_of_training = int(sys.argv[1])
    except:
    	no_of_training = 10000

    # intialization of the neural network.
    nn = Network()

    # train network
    nn.train(data.training_set_inputs, data.training_set_outputs, no_of_training=no_of_training)

    # test the neural network against some new data
    nn.test(data.test_set_inputs, data.test_set_outputs)
