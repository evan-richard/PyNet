"""
    Brain.py

    Used to prototype a neural network
    implementation in Python that will be 
    later translated to C code for performance.
"""

import sys
import random
import math
from config import (
    NUM_HIDDEN_LAYERS,
    NODE_TYPES,
    LEARNING_RATE,
    MOMENTUM,
    STARTING_MIN,
    STARTING_MAX,
    DEBUG,
)


def random_value():
    """
        Return a random value represented
        as a decimal between -1 to 1.
    """
    return random.randint(STARTING_MIN, STARTING_MAX) / 100.0


def activation_function(x):
    """
        Run the activaiton function on the
        computed node value.

        Note: currently only the sigmoid function is supported
    """
    return 1 / (1 + math.exp(-x))


def nodes_in_inner_layer(num_of_input_nodes, num_of_output_nodes):
    """
        Calculate the number of nodes that should be in
        each hidden layer.
        This value is the same for all the hidden 
        layers of the black box.
    """
    return round((2 / 3) * num_of_input_nodes) + num_of_output_nodes


def count_nodes(training_data):
    """
        Returns the number of input values and output values
        as a tuple.

        @param training_data: list of data to train the network with
    """
    try:
        first_set = training_data[0]
    except IndexError:
        sys.exit("No training data was supplied, quitting...")

    input_data = first_set.get("input", None)
    output_data = first_set.get("output", None)

    if (not input_data) or (not output_data):
        sys.exit("Invalid data set format: {0}".format(first_set))

    return (len(input_data), len(output_data))


def calculate_error(actual_value, computed_value):
    """
        Calculate the error between expected and computed
        values.
    """
    difference = actual_value - computed_value
    error = (difference ** 2) / 2
    return error


def calculate_node_value(node, previous_layer_values):
    """
        This is the forward pass. Calculate the total net input
        for the current node, then send it through an activation
        function.

        @param node: current node
        @param previous_layer_values: these will either be the input values,
            or the values computed from the previous layer
        @return: the computed value of the current node from the forward pass
    """
    calculated_node_value = 0

    if DEBUG:
        print("START NODE: {0}".format(node.identifier))
        print("		bias for node {0}: {1}".format(node.identifier, node.bias))
        print("		weights for node {0}:".format(node.identifier))

    for weight in node.weights:
        input_value = previous_layer_values[weight.starting_node_index]

        if DEBUG:
            print("			input value: {0}, weight: {1}".format(input_value, weight.value))

        calculated_node_value = calculated_node_value + (input_value * weight.value)

    calculated_node_value = calculated_node_value + node.bias

    if DEBUG:
        print(
            "		node value for node {0}: {1}".format(
                node.identifier, calculated_node_value
            )
        )

    # Run the node value through the activation function
    calculated_node_value = activation_function(calculated_node_value)

    if DEBUG:
        print("		sigmoid the node value: {0}".format(calculated_node_value))

    return calculated_node_value


class NeuralNode:
    """
        Class representing a node in any layer of the
        neural network.
    """

    def __init__(self, layer, identifier, node_type, test_biases):
        self.layer = layer
        self.identifier = identifier
        self.node_type = node_type
        self.bias = (
            test_biases[layer - 1][identifier] if test_biases else random_value()
        )
        self.weights = []

    def __str__(self):
        return "Layer {0}, Node {1} - Type: {2}, Bias: {3}".format(
            self.layer, self.identifier, self.node_type, self.bias
        )


class Weight:
    """
        Class representing the weight between two nodes.
    """

    def __init__(self, starting_node_index, ending_node_index, test_weights=None):
        self.starting_node_index = starting_node_index
        self.ending_node_index = ending_node_index
        self.value = (
            test_weights[ending_node_index][starting_node_index]
            if test_weights
            else random_value()
        )

    def __str__(self):
        return "Weight for {0}-{1}: {2}".format(
            self.starting_node_index, self.ending_node_index, self.value
        )


class Brain:
    """
        Class representing the 'brain' or computation
        engine of the neural network. All major functions
        are called on this class.
    """

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
        # learning rate of the neural network
        self.learning_rate = LEARNING_RATE
        # DEBUG
        self.iteration_index = 0
        self.test_values = None

    def run(self, data_set):
        """
            Run the given data_set through the neural network.
        """
        if self.is_trained:
            computed_values = self.train_data_set({"input": data_set})
            return computed_values[1]
        else:
            print("Please train the neural network first...")

    def train(self, training_data, **kwargs):
        """
            Train the neural network with a set of training data.

            @param training_data: input and outputs to train the network
                                  ( i.e. [{ 'input': [], 'output': [] }])
            @param **kwargs: optional parameters to configure the network
                                iterations: int (number of iterations)
        """
        # Should reset the neural network if trying to re-train
        if self.is_trained:
            self.reset()

        iterations = kwargs.get("iterations", 1)
        self.learning_rate = kwargs.get("learning_rate", LEARNING_RATE)
        self.test_values = kwargs.get("test_values", None)

        # Calculate the number of nodes in the hidden layer
        self.num_input_nodes, self.num_output_nodes = count_nodes(training_data)
        self.num_inner_layer_nodes = (
            self.test_values["num_inner_layer_nodes"]
            if self.test_values
            else nodes_in_inner_layer(self.num_input_nodes, self.num_output_nodes)
        )

        self.initialize_hidden_layers()
        self.initialize_output_layer()

        for i in range(1, iterations + 1):
            # DEBUG
            self.iteration_index = i
            # if self.iteration_index % 1000 == 0:
            #     print("\n===============ITERATION {0}===============".format(i))
            for data_set in training_data:
                computed_values = self.train_data_set(data_set)
                self.propagate_error(data_set, computed_values)

        # Mark this network as trained
        self.is_trained = True

    def reset(self):
        """
            Reset the initial values of the brain.
        """
        if self.is_trained:
            self.inner_layers = []
            self.output_nodes = []
            self.num_input_nodes = 0
            self.num_output_nodes = 0
            self.num_inner_layer_nodes = 0
            self.is_trained = False
            self.learning_rate = LEARNING_RATE
            # DEBUG
            self.iteration_index = 0

    def initialize_hidden_layers(self):
        """
            Build the inner layers of the black box
            (or hidden layers) with weights and biases.
        """
        for i in range(0, NUM_HIDDEN_LAYERS):
            # initialize empty inner layer array
            self.inner_layers.append([])

        for inner_layer_index in range(0, NUM_HIDDEN_LAYERS):
            for inner_node_index in range(0, self.num_inner_layer_nodes):
                # Create the inner layer node
                node = NeuralNode(
                    inner_layer_index + 1,
                    inner_node_index,
                    NODE_TYPES["INNER"],
                    self.test_values["biases"] if self.test_values else None,
                )
                nodes_per_layer = (
                    self.num_input_nodes
                    if inner_layer_index == 0
                    else self.num_inner_layer_nodes
                )
                for input_node_index in range(0, nodes_per_layer):
                    # Create initial weights to the inner layer node
                    node.weights.append(
                        Weight(
                            input_node_index,
                            inner_node_index,
                            self.test_values["weights"][inner_layer_index]
                            if self.test_values
                            else None,
                        )
                    )
                # Append the node to the current layer
                self.inner_layers[inner_layer_index].append(node)

    def initialize_output_layer(self):
        """
            Initialize the output layer of the neural
            network with weights and biases.
        """
        for output_node_index in range(0, self.num_output_nodes):
            # Create the output node
            node = NeuralNode(
                NUM_HIDDEN_LAYERS + 1,
                output_node_index,
                NODE_TYPES["OUTPUT"],
                self.test_values["biases"] if self.test_values else None,
            )
            # Create initial weights to the output layer node
            for inner_node_index in range(0, self.num_inner_layer_nodes):
                node.weights.append(
                    Weight(
                        inner_node_index,
                        output_node_index,
                        self.test_values["weights"][NUM_HIDDEN_LAYERS]
                        if self.test_values
                        else None,
                    )
                )
            # Append the node to the list of output nodes
            self.output_nodes.append(node)

    def train_data_set(self, data_set):
        """
            Run the data set through the computation model

            @param data_set: current input and output data
                ( i.e. { 'input': [], 'output': [] } )
            @return: a 2D array representing the computed values of each
                node in each layer
        """
        previous_layer_values = []
        current_layer_values = []
        calculated_values_per_layer = []

        if DEBUG:
            print("---------------Run input set: {0}---------------".format(data_set))

        input_data = data_set.get("input", None)
        if not input_data:
            sys.exit("Invalid data set format: {0}".format(data_set))
        input_data = (
            list(input_data.values()) if type(input_data) != list else input_data
        )

        previous_layer_values = input_data.copy()

        # Loop through each inner layer of the system
        for inner_layer_index in range(0, NUM_HIDDEN_LAYERS):
            if DEBUG:
                print("Beginning layer {0}".format(inner_layer_index + 1))

            for node in self.inner_layers[inner_layer_index]:
                current_layer_values.append(
                    calculate_node_value(node, previous_layer_values)
                )

            # save values from the current layer
            previous_layer_values = current_layer_values.copy()
            calculated_values_per_layer.append(current_layer_values.copy())
            current_layer_values = []

        if DEBUG:
            print("Beginning layer {0}".format(NUM_HIDDEN_LAYERS + 1))

        # Loop through each node of the output layer
        for node in self.output_nodes:
            current_layer_values.append(
                calculate_node_value(node, previous_layer_values)
            )

        calculated_values_per_layer.append(current_layer_values.copy())
        return calculated_values_per_layer

    def propagate_error(self, data_set, computed_values):
        """
            Need to back-propagate the errors through the system and
            adjust the weights appropriately.
            (Currently this only works for a network with 1 hidden layer)
        """
        total_error = 0

        # Retrieve the input data from the data set as a list of values
        input_data = data_set.get("input", None)
        if not input_data:
            sys.exit("Invalid data set format: {0}".format(data_set))
        input_data = (
            list(input_data.values()) if type(input_data) != list else input_data
        )

        # Retrieve the output data from the data set as a list of values
        output_data = data_set.get("output", None)
        if not output_data:
            sys.exit("Invalid data set format: {0}".format(data_set))
        output_data = (
            list(output_data.values()) if type(output_data) != list else output_data
        )

        # We want to loop backwards through the computed values to
        # propagate the errors
        value_set_index = len(computed_values) - 1
        layer_value_set = computed_values[value_set_index]
        prev_layer_value_set = computed_values[value_set_index - 1]

        # Calculate the delta for each node value
        for node_value_index in range(0, len(layer_value_set)):
            if DEBUG:
                print("-->	node: {0}".format(node_value_index))

            actual_value = output_data[node_value_index]
            computed_value = layer_value_set[node_value_index]

            error = calculate_error(actual_value, computed_value)
            total_error = total_error + error

            E_total_over_out_h = self.propagate_output_layer(
                node_value_index, actual_value, computed_value, prev_layer_value_set
            )

            self.propagate_hidden_layers(
                value_set_index, prev_layer_value_set, input_data, E_total_over_out_h
            )

        # if DEBUG or (self.iteration_index % 1000 == 0):
        #     print("TOTAL ERROR: {0}".format(total_error))

    def propagate_output_layer(
        self, node_value_index, actual_value, computed_value, prev_layer_value_set
    ):
        """
            This is the first step in backward propagation.
            Reassign a new weight to the output nodes based on
            the partial derivative of the total error WRT the current weight.

            @return: the sum of the partial derivatives of the output layer
        """
        total = 0

        # First take the negative difference between the computed and expected values
        first_multiplier = -(actual_value - computed_value)
        # Next multiply the current node value by (1 - itself)
        second_multiplier = computed_value * (1 - computed_value)
        # Finally compute the change in the current node value WRT the weight
        # of each of the nodes in the previous layer
        for prev_node_value_index in range(0, len(prev_layer_value_set)):
            third_multiplier = prev_layer_value_set[prev_node_value_index]
            if DEBUG:
                print(
                    "                 Output values: {0}, {1}, {2}".format(
                        first_multiplier, second_multiplier, third_multiplier
                    )
                )

            delta_value = first_multiplier * second_multiplier * third_multiplier

            if DEBUG:
                print(
                    "                 Result for current node {0}, prev node {1}: {2}".format(
                        node_value_index, prev_node_value_index, delta_value
                    )
                )

            weight_ref = self.output_nodes[node_value_index].weights[
                prev_node_value_index
            ]

            if DEBUG:
                print("                   {0}".format(weight_ref))

            total = total + (first_multiplier * second_multiplier * weight_ref.value)

            weight_ref.value = weight_ref.value - self.learning_rate * delta_value

            if DEBUG:
                print("                   NEW {0}".format(weight_ref))

        return total

    def propagate_hidden_layers(
        self, value_set_index, prev_layer_value_set, input_data, E_total_over_out_h
    ):
        """
            Propagate the error through the hidden layers of the
            system and adjust weights accordingly
        """
        for prev_node_value_index in range(0, len(prev_layer_value_set)):
            prev_node_value = prev_layer_value_set[prev_node_value_index]
            out_h_over_net_h = prev_node_value * (1 - prev_node_value)
            for input_value_index in range(0, len(input_data)):
                E_total_over_weight = (
                    E_total_over_out_h
                    * out_h_over_net_h
                    * input_data[input_value_index]
                )
                weight_ref = self.inner_layers[value_set_index - 1][
                    prev_node_value_index
                ].weights[input_value_index]

                if DEBUG:
                    print("                   {0}".format(weight_ref))

                weight_ref.value = (
                    weight_ref.value - self.learning_rate * E_total_over_weight
                )

                if DEBUG:
                    print("                   NEW {0}".format(weight_ref))
