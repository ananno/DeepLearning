import numpy as np
import time

class Network():
	def __init__(self):
		print ("\n\t\tRECOGNITION OF A CERTAIN PATTERN")
		print ("\n\t\t--------------------------------")		
		# generates same random numbers everytime
		np.random.seed(1)
		
		# random weights of 3x1 matrix, with values in the range -1 to 1
		self.synaptic_weights = 2 * np.random.random((3, 1)) - 1

		# Print initial weights
		print ("\n\t[+] Initial random weights:", ' '.join( str(round(i[0], 6)) for i in self.synaptic_weights))

	# Sigmoid function
	def sigmoid_the_normalizer(self, x):
		return 1 / (1 + np.exp(-x))

	# The derivative of the Sigmoid function.
	# This is the gradient of the Sigmoid curve.
	# It indicates how confident we are about the existing weight.
	def sigmoid_derivative(self, x):
		return x * (1 - x)

	# Training
	def train(self, training_set_inputs, training_set_outputs, no_of_training=10000):
		print("\n\t[+] Training is in progress...")
		training_started = time.time()
		# training loop
		for iteration in range(no_of_training):
			# Output matrix
			output = self.activation(training_set_inputs)

			# error matrix
			error = training_set_outputs - output

			# cost for each training
			cost = np.dot(training_set_inputs.T, error * self.sigmoid_derivative(output))
			
			# update weights
			self.synaptic_weights += cost

		# print out relevant info.
		print("\n\t[+] Training has been completed.")
		print("\n\t[+] After training", no_of_training, 'times in', str(round(time.time()-training_started, 2))+'s.')
		print ("\n\t[+] New weights:", ' '.join( str(round(i[0], 6)) for i in self.synaptic_weights ))


	# activation function
	def activation(self, inputs):
		dot_product = np.dot(inputs, self.synaptic_weights)
		return self.sigmoid_the_normalizer(dot_product)

	def accuracy(self, network_output, expected_output):
		correct = 0
		for i, j in zip(network_output, expected_output):
			if round(i[0], 2) == j[0]:
				correct += 1

		result  = (correct / float(len(network_output))) * 100
		return round( result, 2)

	def test(self, test_inputs, test_outputs):
		print ("\n\t[+] Testing model against:", ', '.join( str(i) for i in test_inputs ))
		outputs = self.activation(test_inputs)
		print("\n\t[+] Accuracy:", self.accuracy(outputs, test_outputs),"%\n")
