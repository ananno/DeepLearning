from MyFirstNN.mynetwork2 import Network

# intialization of the neural network.
nn = Network()

# train network
nn.train(no_of_training=100000)

# test the neural network against some new data
nn.test()
