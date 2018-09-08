import numpy as np

# training data
training_set_inputs = np.array([
	[0, 0, 1], 
	[1, 1, 1], 
	[1, 0, 1], 
])
training_set_outputs = np.array([
	[0, 1, 1]
]).T # .T i.e. transpose makes the row matrix -> column matrix



# test data
test_set_inputs  = np.array([
	[1, 0, 1],
	[1, 1, 0],
	[0, 0, 0]
])

test_set_outputs = np.array([
	[1, 1, 0]
]).T # .T i.e. transpose makes the row matrix -> column matrix