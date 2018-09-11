from MultiLayerApproach.MultiLayerNetwork import Network, NeuronLayer
import sys
import numpy as np
import time

if __name__ == "__main__":

    # set training number from command
    try:
        no_of_training = int(sys.argv[1])
    except:
        no_of_training = 1000

    np.random.seed(int(time.time()))
    
    # First layer - hidden layer: having 3 neurons,
    # each connected to 3 inputs
    layer1 = NeuronLayer(
        number_of_neurons=3,
        inputs_connected_to_each_neuron=3
    )

    # Second layer - output layer:
    # 3 neurons are connected to the output neuron
    layer2 = NeuronLayer(
        number_of_neurons=1,
        inputs_connected_to_each_neuron=3
    )

    # intialize the neural network
    neuralNet = Network(layer1=layer1, layer2=layer2)

    # train the neural network
    neuralNet.train(no_of_training=no_of_training)

    # test the neural network
    neuralNet.test()
