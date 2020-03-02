import sys
import random
import math
from config import NUM_HIDDEN_LAYERS, NODE_TYPES, LEARNING_RATE, MOMENTUM, STARTING_MIN, STARTING_MAX


def random_value():
    '''
        Return a random value represented
        as a decimal between -1 to 1.
    '''
    return random.randint(STARTING_MIN, STARTING_MAX)/100.0


def activation_function(x):
    '''
        Run the activaiton function on the
        computed node value.

        Note: currently only the sigmoid function is supported
    '''
    return 1/(1 + math.exp(-x))


def nodes_in_inner_layer(num_of_input_nodes, num_of_output_nodes):
    '''
        Calculate the number of nodes that should be in
        each hidden layer.
        This value is the same for all the hidden 
        layers of the black box.
    '''
    return round((2/3) * num_of_input_nodes) + num_of_output_nodes


def count_nodes(training_data):
    '''
        Returns the number of input values and output values
        as a tuple.

        @param training_data: list of data to train the network with
    '''
    try:
        first_set = training_data[0]
    except IndexError:
        sys.exit("No training data was supplied, quitting...")

    input_data = first_set.get('input', None)
    output_data = first_set.get('output', None)

    if (not input_data) or (not output_data):
        sys.exit("Invalid data set format: {0}".format(first_set))

    return (len(input_data), len(output_data))


def calculate_error(actual_value, computed_value):
    '''
        Calculate the error between expected and computed
        values.
    '''
    difference = actual_value - computed_value
    error = (difference ** 2) / 2
    return difference, error


class NeuralNode:
    '''
        Class representing a node in any layer of the
        neural network.
    '''

    def __init__(self, layer, identifier, node_type):
        self.layer = layer
        self.identifier = identifier
        self.node_type = node_type
        self.bias = random_value()
        self.weights = []

    def __str__(self):
        return 'Layer {0}, Node {1} - Type: {2}, Bias: {3}'.format(self.layer, self.identifier, self.node_type, self.bias)


class Weight:
    '''
        Class representing the weight between two nodes.
    '''

    def __init__(self, starting_node_index, ending_node_index):
        self.starting_node_index = starting_node_index
        self.ending_node_index = ending_node_index
        self.value = random_value()

    def __str__(self):
        return 'Weight for {0}-{1}: {2}'.format(self.starting_node_index, self.ending_node_index, self.value)


class Brain:
    '''
        Class representing the 'brain' or computation
        engine of the neural network. All major functions
        are called on this class.
    '''

    def __init__(self):
        # 2-D array where each layer is a list of nodes
        self.inner_layers = []
        # array of output nodes
        self.output_nodes = []
        # number of input values
        self.num_input_nodes = 0
        # number of output values
        self.num_output_nodes = 0
        # number of nodes in the hidden layer
        self.num_inner_layer_nodes = 0
        # whether or not the network is trained
        self.is_trained = False
        # DEBUG
        self.iteration_index = 0

    def run(self, data_set):
        '''
            Run the given data_set through the neural network.
        '''
        if (self.is_trained):
            computed_values = self.train_data_set({'input': data_set})
            return computed_values[1]
        else:
            print("Please train the neural network first...")

    def train(self, training_data, **kwargs):
        '''
            Train the neural network with a set of training data.

            @param training_data: input and outputs to train the network
                                  ( i.e. [{ 'input': [], 'output': [] }])
            @param **kwargs: optional parameters to configure the network
                                iterations: int (number of iterations)
        '''
        # Should reset the neural network if trying to re-train
        if (self.is_trained):
            self.reset()

        iterations = kwargs.get('iterations', 1)

        # Calculate the number of nodes in the hidden layer
        self.num_input_nodes, self.num_output_nodes = count_nodes(
            training_data)
        self.num_inner_layer_nodes = nodes_in_inner_layer(
            num_input_nodes, num_output_nodes)

        self.initialize_hidden_layers()
        self.initialize_output_layer()

        for i in range(1, iterations + 1):
            # DEBUG
            self.iteration_index = i
            if (self.iteration_index % 1000 == 0):
                print(
                    "\n===============ITERATION {0}===============".format(i))
            for data_set in training_data:
                computed_values = self.train_data_set(data_set)
                self.propagate_error(data_set, computed_values)

        # Mark this network as trained
        self.is_trained = True

    def reset(self):
        '''
            Reset the initial values of the brain.
        '''
        if (self.is_trained):
            self.inner_layers = []
            self.output_nodes = []
            self.num_input_nodes = 0
            self.num_output_nodes = 0
            self.num_inner_layer_nodes = 0
            self.is_trained = False
            # DEBUG
            self.iteration_index = 0

    def initialize_hidden_layers(self):
        '''
            Build the inner layers of the black box
            (or hidden layers) with weights and biases.
        '''
        for i in range(0, NUM_HIDDEN_LAYERS):
            # initialize empty inner layer array
            self.inner_layers.append([])

        for inner_layer_index in range(0, NUM_HIDDEN_LAYERS):
            for inner_node_index in range(0, self.num_inner_layer_nodes):
                # Create the inner layer node
                node = NeuralNode(inner_layer_index + 1,
                                  inner_node_index, NODE_TYPES['INNER'])
                nodes_per_layer = if inner_layer_index == 0 else self.num_inner_layer_nodes
                for input_node_index in range(0, nodes_per_layer):
                    # Create initial weights to the inner layer node
                    node.weights.append(
                        Weight(input_node_index, inner_node_index))
            # Append the node to the current layer
            self.inner_layers[inner_layer_index].append(node)

    def initialize_output_layer(self):
        '''
            Initialize the output layer of the neural
            network with weights and biases.
        '''
        for output_node_index in range(0, self.num_output_nodes):
            # Create the output node
            node = NeuralNode(NUM_HIDDEN_LAYERS + 1,
                              output_node_index, NODE_TYPES['OUTPUT'])
            # Create initial weights to the output layer node
            for inner_node_index in range(0, self.num_inner_layer_nodes):
                node.weights.append(
                    Weight(inner_node_index, output_node_index))
            # Append the node to the list of output nodes
            self.output_nodes.append(node)

    def train_data_set(self, data_set):
        '''
            Perform a forward pass to compute output values
            based on current weights and biases

            @param data_set: current input and output data
                             ( i.e. { 'input': [], 'output': [] } )
        '''
        previous_layer_values = []
        current_layer_values = []
        calculated_values_per_layer = []

        # print(
        #     "---------------Run input set: {0}---------------".format(data_set))
        input_data = data_set.get('input', None)
        if (not input_data):
            sys.exit("Invalid data set format: {0}".format(data_set))
        input_data = input_data.values() if type(input_data) != list else input_data

        previous_layer_values = input_data.copy()

        # Loop through each inner layer of the system
        for inner_layer_index in range(0, NUM_HIDDEN_LAYERS):
            # print("Beginning layer {0}".format(inner_layer_index + 1))
            for node in self.inner_layers[inner_layer_index]:
                current_layer_values.append(
                    self.calculate_node_value(node, previous_layer_values))

            # save values from the current layer
            previous_layer_values = current_layer_values.copy()
            calculated_values_per_layer.append(current_layer_values.copy())
            current_layer_values = []

        # Loop through each node of the output layer
        # print("Beginning layer {0}".format(NUM_HIDDEN_LAYERS + 1))
        for node in self.output_nodes:
            current_layer_values.append(
                self.calculate_node_value(node, previous_layer_values))

        calculated_values_per_layer.append(current_layer_values.copy())
        return calculated_values_per_layer

    def propagate_error(self, data_set, computed_values):
        output_data = data_set.get('output', None)
        if (not output_data):
            sys.exit("Invalid data set format: {0}".format(data_set))
        output_data = output_data.values() if type(
            output_data) != list else output_data

        input_data = data_set.get('input', None)
        if (not input_data):
            sys.exit("Invalid data set format: {0}".format(data_set))
        input_data = input_data.values() if type(input_data) != list else input_data

        value_set_index = len(computed_values) - 1
        # print("Calculate error for the output layer {0}".format(
        #     value_set_index + 1))
        layer_value_set = computed_values[value_set_index]
        prev_layer_value_set = computed_values[value_set_index - 1]

        # Calculate errors for each layer, and the total error
        total_error = 0
        errors = []
        for node_value_index in range(0, len(layer_value_set)):
            # print("	-->	node: {0}".format(node_value_index))
            difference, error = calculate_error(
                output_data[node_value_index], layer_value_set[node_value_index])
            errors.append([difference, error])
            total_error = total_error + error

        # Calculate the delta for each node value
        # print("TOTAL Error = {0}".format(total_error))
        for node_value_index in range(0, len(errors)):
            # print("	-->	node: {0}".format(node_value_index))
            # First take the negative difference between the computed and expected values
            first_multiplier = -errors[node_value_index][0]
            # Next multiply the current node value by (1 - itself)
            computed_node_value = layer_value_set[node_value_index]
            second_multiplier = computed_node_value * \
                (1 - computed_node_value)
            # Finally compute the change in the current node value with respect to the weight
            # of each of the nodes in the previous layer
            propagated_hidden_layer_values = []
            for prev_node_value_index in range(0, len(prev_layer_value_set)):
                third_multiplier = prev_layer_value_set[prev_node_value_index]
                delta_value = first_multiplier * second_multiplier * third_multiplier
                # print("                 Result for current node {0}, prev node {1}: {2}"
                #       .format(node_value_index, prev_node_value_index, delta_value))
                weight_ref = self.output_nodes[node_value_index].weights[prev_node_value_index]
                # print("                   {0}".format(weight_ref))

                propagated_hidden_layer_values.append(first_multiplier *
                                                      second_multiplier * weight_ref.value)

                weight_ref.value = weight_ref.value - LEARNING_RATE * delta_value
                # print("                   NEW {0}".format(weight_ref))

            E_total_over_out_h = 0
            for propagated_value in propagated_hidden_layer_values:
                E_total_over_out_h = E_total_over_out_h + propagated_value

            for prev_node_value_index in range(0, len(prev_layer_value_set)):
                prev_node_value = prev_layer_value_set[prev_node_value_index]
                out_h_over_net_h = prev_node_value * (1 - prev_node_value)
                for input_value_index in range(0, len(input_data)):
                    E_total_over_weight = E_total_over_out_h * \
                        out_h_over_net_h * input_data[input_value_index]
                    weight_ref = self.inner_layers[value_set_index -
                                                   1][prev_node_value_index].weights[input_value_index].value
                    weight_ref = weight_ref - LEARNING_RATE * E_total_over_weight

    def calculate_node_value(self, node, previous_layer_values):
        # Calculate the node value
        calculated_node_value = 0
        # print("START NODE: {0}".format(node.identifier))
        # print("		bias for node {0}: {1}".format(node.identifier, node.bias))
        # print("		weights for node {0}:".format(node.identifier))
        for weight in node.weights:
            input_value = previous_layer_values[weight.starting_node_index]
            # print("			input value: {0}, weight: {1}".format(
            #     input_value, weight.value))
            calculated_node_value = calculated_node_value + \
                (input_value * weight.value)
        calculated_node_value = calculated_node_value + node.bias
        # print("		node value for node {0}: {1}".format(
        #     node.identifier, calculated_node_value))
        # Run the node value through the activation function
        calculated_node_value = activation_function(calculated_node_value)
        # print("		sigmoid the node value: {0}".format(calculated_node_value))
        return calculated_node_value
