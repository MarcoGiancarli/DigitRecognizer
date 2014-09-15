__author__ = 'MarcoGiancarli, m.a.giancarli@gmail.com'


import random
from math import exp

def sigmoid(t):
    return 1 / (1 + exp(-t))


class NeuralNetwork:
    layers = []

    # Inputs should be a list of lists containing the input values in each row
    inputs = []

    # Outputs at some index i should be a list of integers where each integer corresponds
    # to the index of the output node which contains the correct label
    outputs = []

    # Step size variable
    alpha = 0

    def __init__(self, layer_sizes, init_style, alpha, labels=None):
        self.alpha = alpha
        prev_num_nodes = 0
        for num_nodes in layer_sizes:
            if len(self.layers) == 0:
                self.layers.append(Layer(num_nodes, NodeType.Input, init_style, prev_num_nodes))
            elif 1 + len(self.layers) < len(layer_sizes):  # add 1 because we haven't added the current layer yet
                self.layers.append(Layer(num_nodes, NodeType.Inner, init_style, prev_num_nodes))
            elif 1 + len(self.layers) == len(layer_sizes):
                if labels is not None and num_nodes != len(labels):
                    pass
                    #TODO throw exception here
                if labels is None:
                    # if no labels specified for the output nodes, use their indices
                    labels = [str(node_number) for node_number in range(num_nodes)]
                self.layers.append(Layer(num_nodes, NodeType.Output, init_style, prev_num_nodes, labels))
            else:
                pass
                #TODO add an exception here
            prev_num_nodes = num_nodes

    ''' This method is used for supervised training on a whole data set. '''
    def train(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs
        # same size as outputs, all 0s
        output_vectors = [[0 for dummy in self.layers[-1].nodes] for dummy in outputs]
        for vector, output in output_vectors, outputs:
            vector[output] += 1
        # iterate through data set and feed forward
        for input, y in inputs, output_vectors:
            _, A, Z = self.predict(input)
            D = [[] for dummy in len(A)]  # same size as A
            D[-1] = [a-y for a in A[-1]]
            for a, d, d_prev, layer in reversed(A[1:-1]), reversed(D[1:-1]), reversed(D[1:]), reversed(self.layers[:-1]):
                d = [sum(node.weights)*d_prev_i for node, d_prev_i in layer.nodes, d_prev]
                for di, ai in d, a:
                    di *= ai * (1-ai)
            gradient = []
            for d, a in D, A:
                gradient.append([])
                for di in d:
                    gradient[-1].append([])
                    for ai in a:
                        gradient[-1][-1].append(di * ai / len(inputs))
            for d in D:
                gradient.append([di/len(inputs) for di in d])
            #TODO add regularization
            self.gradient_descent(gradient)

    ''' This method is used for adding new rows to the data set and training again '''
    def retrain(self, inputs, outputs):
        # add input and output to current set and call train
        self.inputs.append(inputs)
        self.outputs.append(outputs)
        self.train(self.inputs, self.outputs)

    def predict(self, example):
        if len(example) != len(self.layers[0]):
            #TODO throw exception
            pass
        A = []
        Z = []
        A.append(example)
        Z.append([])
        for layer in self.layers[1:]:
            a, z = layer.feed_forward(A[-1])
            A.append(a)
            Z.append(z)
        prediction = self.layers[-1].nodes[A[-1].index(max(A[-1]))].label
        return prediction, A, Z  # return values for back prop

    def gradient_descent(self, gradient):
        for layer, g in self.layers[1:], gradient[1:]:
            for node, gi in layer, g:
                for weight, gij in node, gi:
                    weight = weight - self.alpha * gij


class Layer:
    nodes = []

    def __init__(self, size, node_type, init_style, prev_num_nodes, labels=None):
        if node_type is NodeType.Output and labels is not None:
            # if the node type is output, add the labels
            for node_number in range(size):
                self.nodes.append(Node(node_type, init_style, prev_num_nodes, labels[node_number]))
        else:
            # other node types don't get a label
            for node_number in range(size):
                self.nodes.append(Node(node_type, init_style, prev_num_nodes))
            #TODO do stuff

    def feed_forward(self, prev_values):
        a = []
        z = []
        for node in self.nodes:
            ai, zi = node.feed_forward(prev_values)
            a.append(ai)
            z.append(zi)
        return a, z


class Node:
    NODE_TYPE = ''

    # use theta in a linear combination with the previous row to get feed forward value
    weights = []

    def __init__(self, node_type, init_style, prev_num_nodes, label=None):
        self.NODE_TYPE = node_type
        if node_type is NodeType.Output:
            self.label = str(label)
        if init_style is NodeInitStyle.Ones:
            weights = [1.0 for dummy in range(prev_num_nodes+1)]
        if init_style is NodeInitStyle.Zeros:
            weights = [0.0 for dummy in range(prev_num_nodes+1)]
        if init_style is NodeInitStyle.Random:
            weights = [random.randrange(-1, 1) for dummy in range(prev_num_nodes+1)]

    def feed_forward(self, prev_values):
        prev_a = prev_values.clone()
        prev_a.push(1)
        if len(prev_a) != len(self.weights):
            #TODO throw an exception
            pass
        zi = sum([weight*value for weight, value in self.weights, prev_a])
        ai = sigmoid(zi)
        return ai, zi


class NodeType:
    Input = 'Input'
    Inner = 'Inner'
    Output = 'Output'


class NodeInitStyle:
    Zeros = 'Zeros'
    Ones = 'Ones'
    Random = 'Random'