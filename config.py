# How many layers in the 'black box' (usually 1)
NUM_HIDDEN_LAYERS = 1

NODE_TYPES = dict(INPUT='INPUT', OUTPUT='OUTPUT', INNER='BLACK BOX')

# How heavy should each step affect the weights and biases
LEARNING_RATE = 0.3

# How should past outcomes effects the weights and biases
MOMENTUM = 0.1

# What value to start at when generating random weights/biases
STARTING_MIN = 0

# What value to end at when generating random weights/biases
STARTING_MAX = 100

# Values to use for examples
SAMPLE_ITERATIONS = 50000
