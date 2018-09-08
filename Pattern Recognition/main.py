from MyFirstNN.MyNetwork import Network
from MyFirstNN import data


if __name__ == "__main__":
	
	#Intialise a single neuron neural network.
	nn = Network()

	# train model
	nn.train(data.training_set_inputs, data.training_set_outputs, no_of_training=100000)

	# test the neural network model against some new data
	nn.test(data.test_set_inputs, data.test_set_outputs)
	