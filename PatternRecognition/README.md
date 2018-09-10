# Pattern Recognition
> Goal of this single neural network is to **recognize a certain pattern** 
> from some sample data in this case 3 input data

### Guess the pattern with the following input and output

|   Input   | Output |
| --------- |:------:|
|   0 0 1   |    0   |
|   1 1 1   |    1   |
|   1 0 1   |    1   |

> Well, we can see the output is only related to the first input no matter what the other two inputs. Right?
> So, our goal is to ***construct***, ***train*** and ***test*** our neural network to recognize this silly 
> pattern and make some real prediction over unknown data

### Here's our *single neuron neural network*'s schematic representation

![image](https://github.com/iraihankabir/DeepLearning/blob/master/PatternRecognition/img/schemetic_network.png)


## Working principles

* Set some random weights
* Calculate output: ***Σ*** ***W*** *i* ***X*** *i*
* Normalize the output with ***Sigmoid function*** to be between 0 to 1
* Calculate error i.e. `difference between expected output and neural network's output`
* Calculate the Error Weighted Derivative i.e. cost of the training
  `cost = error.input.sigmoidCurveGradientOfNetworkOutput(network_output)`
  where `sigmoidCurveGradientOfNetworkOutput = network_output * (1-network_output)`
* Then adjust the weights with the cost
  `new_weights = weights + cost`
> This is a single training case. This training is iterated many times to adjust the inital weights.
* Test the network with some new data


### Sigmoid function

![image](https://github.com/iraihankabir/DeepLearning/blob/master/PatternRecognition/img/sigmoid_function.png)


# Test this project

`Install numpy if you don't have it installed with the following command`
```bash
pip install numpy
```
* copy the folder MyFirstNN and pattern_recognition.py
* open pattern_recognition.py and change the `no_of_training`'s value
* run the pattern_recognition.py file

```bash
.\pattern_recognition.py
```
***or***
```bash
python pattern_recognition.py
```





***After training the network 10000 times***

![image](https://github.com/iraihankabir/DeepLearning/blob/master/PatternRecognition/img/training-1.png)





***After training the network 100000 times***

![image](https://github.com/iraihankabir/DeepLearning/blob/master/PatternRecognition/img/training-2.png)
