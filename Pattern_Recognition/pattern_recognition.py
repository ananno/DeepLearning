from MyFirstNN.MyNetwork import Network
from MyFirstNN import data


if __name__ == "__main__":
	
	# intialization of the neural network.
	nn = Network()

	# train network
	nn.train(data.training_set_inputs, data.training_set_outputs, no_of_training=100000)

	# test the neural network against some new data
	nn.test(data.test_set_inputs, data.test_set_outputs)
