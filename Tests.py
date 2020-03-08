'''
    Tests.py

    Used for testing the neural network.
    For now, results are manually validated by setting
    the debug flag to True.

    This setup is based on and validated by 
    'A Step by Step Backpropagation Example' by Matt Mazur @
    https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
'''

from Brain import Brain

# Input: r, g, b background color, Output: desired text color
data = [
    {'input': [.05, .1], 'output': [.01, .99]}
]

test_values = {
    'num_inner_layer_nodes': 2,
    'biases': [[.35, .35], [.6, .6]],
    'weights': [[[.15, .20], [.25, .3]], [[.4, .45], [.5, .55]]]
}

learning_rate = .5

brain = Brain()
brain.train(data, iterations=1, learning_rate=learning_rate,
            test_values=test_values)
