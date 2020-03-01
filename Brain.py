import sys
import random
import math
from config import NUM_HIDDEN_LAYERS, NODE_TYPES, LEARNING_RATE, MOMENTUM, STARTING_MIN, STARTING_MAX


def random_value():
    return random.randint(STARTING_MIN, STARTING_MAX)/100.0


def activation_function(x):
    return 1/(1 + math.exp(-x))


class NeuralNode:

    def __init__(self, layer, identifier, node_type):
        self.layer = layer
        self.identifier = identifier
        self.node_type = node_type
        self.bias = random_value()
        self.weights = []

    def assign_bias(self):
        pass

    def __str__(self):
        return 'Layer {0}, Node {1} - Type: {2}, Bias: {3}'.format(self.layer, self.identifier, self.node_type, self.bias)


class Weight:

    def __init__(self, starting_node_index, ending_node_index):
        self.starting_node_index = starting_node_index
        self.ending_node_index = ending_node_index
        self.value = random_value()

    def __str__(self):
        return 'Weight for {0}-{1}: {2}'.format(self.starting_node_index, self.ending_node_index, self.value)


class Brain:

    def __init__(self):
        self.inner_layers = []
        self.output_nodes = []
        # DEBUG
        self.iteration_index = 0
        # DEBUG
        self.iterations = 0

    def train(self, training_data, **kwargs):
        iterations = kwargs.get('iterations', 1)
        # DEBUG
        self.iterations = iterations

        # Initialize the 'black box'
        for i in range(0, NUM_HIDDEN_LAYERS):
            # Add a layer of nodes (as a list)
            self.inner_layers.append([])

        # Calculate the number of nodes in the hidden layer
        num_input_nodes, num_output_nodes = self.count_nodes(training_data)
        nodes_per_layer = self.nodes_in_inner_layer(
            num_input_nodes, num_output_nodes)
        # print("Running training data with {0} nodes in the inner layer.".format(
        #     nodes_per_layer))

        # Build the first layer of the black box (usually the only layer)
        for inner_node_index in range(0, nodes_per_layer):
            # Create the inner layer node
            node = NeuralNode(1, inner_node_index, NODE_TYPES['INNER'])
            # Create initial weights to the first inner layer node
            for input_node_index in range(0, num_input_nodes):
                node.weights.append(Weight(input_node_index, inner_node_index))
            # Append the node to the current layer
            self.inner_layers[0].append(node)

        # Build each additional inner layer of the black box
        # (if any) and add weights to the inner nodes
        for inner_layer_index in range(1, NUM_HIDDEN_LAYERS):
            for inner_node_index in range(0, nodes_per_layer):
                # Create the inner layer node
                node = NeuralNode(inner_layer_index + 1,
                                  inner_node_index, NODE_TYPES['INNER'])
                # Create initial weights to the inner layer node
                for input_node_index in range(0, nodes_per_layer):
                    node.weights.append(
                        Weight(input_node_index, inner_node_index))
            # Append the node to the current layer
            self.inner_layers[inner_layer_index].append(node)

        # Build our output layer of brain
        for output_node_index in range(0, num_output_nodes):
            # Create the output node
            node = NeuralNode(NUM_HIDDEN_LAYERS + 1,
                              output_node_index, NODE_TYPES['OUTPUT'])
            # Create initial weights to the output layer node
            for inner_node_index in range(0, nodes_per_layer):
                node.weights.append(
                    Weight(inner_node_index, output_node_index))
            # Append the node to the list of output nodes
            self.output_nodes.append(node)

        for i in range(1, iterations + 1):
            # DEBUG
            self.iteration_index = i
            print("\n===============ITERATION {0}===============".format(i))
            for j in range(0, len(training_data)):
                computed_values = self.train_data_set(training_data[j], j)
                if (self.iteration_index % 10000 == 0):
                    print("All computed values: {0}".format(computed_values))
                self.propagate_error(training_data[j], computed_values)

    def train_data_set(self, data_set, training_data_index):
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
            difference, error = self.calculate_error(
                output_data[node_value_index], layer_value_set[node_value_index])
            errors.append([difference, error])
            total_error = total_error + error

        # Calculate the delta for each node value
        if (self.iteration_index == self.iterations):
            print("TOTAL Error = {0}".format(total_error))
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

    def calculate_error(self, node_value, computed_value):
        difference = node_value - computed_value
        error = (difference ** 2) / 2
        # print("		error: {0}".format(error))
        return difference, error

    def nodes_in_inner_layer(self, num_of_input_nodes, num_of_output_nodes):
        return round((2/3) * num_of_input_nodes) + num_of_output_nodes

    def count_nodes(self, training_data):
        try:
            first_set = training_data[0]
        except IndexError:
            sys.exit("No training data was supplied, quitting...")

        input_data = first_set.get('input', None)
        output_data = first_set.get('output', None)

        if (not input_data) or (not output_data):
            sys.exit("Invalid data set format: {0}".format(first_set))

        return (len(input_data), len(output_data))
