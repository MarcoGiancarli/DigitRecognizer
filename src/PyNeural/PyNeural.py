__author__ = 'MarcoGiancarli, m.a.giancarli@gmail.com'


import random
import math
import numpy as np

random.seed(911*1000)
# It would be like 9-11 times 100.
# 9-11 times 100? Jesus, that's...
# Yes, 91,100.

# def sigmoid(t):
#     if t > 30.0:
#         return 1.0
#     if t < -30.0:
#         return 0.0
#     return 1.0 / (1.0 + math.expm1(-t))

# trying out tanh instead of normal sigmoid as the activation function; should be easier to compute
def sigmoid(x):
    return math.tanh(x)

# derivative of our sigmoid function
def dsigmoid(y):
    return 1.0 - y**2

def make_weight(init_style):
    if init_style is NodeInitStyle.Random:
        return random.randrange(-10000.0, 10000.0)/10000.0
    elif init_style is NodeInitStyle.Zeros:
        return 0.0
    elif init_style is NodeInitStyle.Ones:
        return 1.0

def output_vector_to_scalar(vector):
    # get the index of the max in the vector
    return vector.index(max(vector))

def output_scalar_to_vector(scalar, num_outputs):
    # same size as outputs, all 0s
    vector = [0] * num_outputs
    # add 1 to the correct index
    vector[scalar] += 1
    return vector


#TODO: add methods to save state and return aspects (such as size of network, weights, etc.)
#TODO: the distribution for 100 untrained predictions turned out to be [50, 27, 8, 4, 3, 6, 1, 0 , 1]. Fix predict!!!
#TODO: learning curves?
class NeuralNetwork:
    def __init__(self, layer_sizes, init_style, alpha, labels=None):
        self.alpha = alpha

        if labels is None:
            self.labels = range(layer_sizes[-1])
        else:
            self.labels = labels
        if len(labels) != layer_sizes[-1]:
            #TODO: throw exception here
            print 'Fucked up because the number of labels is wrong'
            exit(1)

        if init_style not in [NodeInitStyle.Random, NodeInitStyle.Ones, NodeInitStyle.Zeros]:
            #TODO: throw exception here
            print 'Fucked up because the init style is not chosen from the NodeInitStyle class'
            exit(1)

        # theta represents the weights for each node. we skip the first layer because it has no weights.
        self.theta = [[]]
        l = 1
        while l < len(layer_sizes):
            # append a matrix which represents the initial weights for layer l
            self.theta.append(np.mat(
                    [[make_weight(init_style) for prev_node in range(layer_sizes[l-1])]  # add weight for each prev node
                                              for node in range(layer_sizes[l])]  # do this for each current node
            ))
            l += 1


        # self.layers = []
        # # Inputs should be a list of lists containing the input values in each row
        # self.inputs = []
        # # Outputs at some index i should be a list of integers where each integer corresponds
        # # to the index of the output node which contains the correct label
        # self.outputs = []
        # # Step size variable
        # self.alpha = alpha
        # prev_num_nodes = 0
        # for num_nodes in layer_sizes:
        #     if len(self.layers) == 0:
        #         self.layers.append(Layer(num_nodes, NodeType.Input, init_style, prev_num_nodes))
        #     elif 1 + len(self.layers) < len(layer_sizes):  # add 1 because we haven't added the current layer yet
        #         self.layers.append(Layer(num_nodes, NodeType.Inner, init_style, prev_num_nodes))
        #     elif 1 + len(self.layers) == len(layer_sizes):
        #         if labels is not None and num_nodes != len(labels):
        #             #TODO throw exception here
        #             print 'Fucked up because the number of labels is wrong'
        #             exit()
        #         if labels is None:
        #             # if no labels specified for the output nodes, use their indices
        #             labels = [str(node_number) for node_number in range(num_nodes)]
        #         self.layers.append(Layer(num_nodes, NodeType.Output, init_style, prev_num_nodes, labels))
        #     else:
        #         #TODO add an exception here
        #         print 'Fucked up on network init'
        #         exit()
        #     prev_num_nodes = num_nodes

    def feed_forward(self, input_matrix):
        A = [].append(input_matrix)
        Z = [[]]  # z_1 doesn't exist
        l = 1
        while l < len(self.theta):
            Z[l] = self.theta[l] * A[l-1]
            temp = [[sigmoid(Zij) for Zij in Zi] for Zi in Z[l]]
            A[l] = np.mat([temp, [1]*len(A[l-1]) ])
            l += 1
        return A, Z

    ''' This method is used for supervised training on a whole data set. '''
    def train(self, inputs, outputs):


        # self.inputs = inputs
        # self.outputs = outputs
        # # same size as outputs, all 0s
        # output_vectors = [[0 for dummy in self.layers[-1].nodes] for dummy in outputs]
        # for vector, output in zip(output_vectors, outputs):
        #     vector[output] += 1
        # # iterate through data set and feed forward
        # for input, y in zip(inputs, output_vectors):
        #     _, A, Z = self.predict(input)
        #     D = [[] for dummy in range(len(A))]  # same size as A
        #     D[-1] = [ai-yi for ai, yi in zip(A[-1], y)]
        #     for a, d, d_prev, layer in zip(reversed(A[1:-1]), reversed(D[1:-1]),
        #                                    reversed(D[2:]), reversed(self.layers[1:-1])):
        #         d = [sum(node.weights)*d_prev_i for node, d_prev_i in zip(layer.nodes, d_prev)]
        #         for di, ai in zip(d, a):
        #             di *= ai * (1-ai)
        #     gradient = []
        #     for d, a in zip(D, A):
        #         gradient.append([])
        #         for di in d:
        #             gradient[-1].append([])
        #             for ai in a:
        #                 gradient[-1][-1].append(di * ai / len(inputs))
        #     for d in D:
        #         gradient.append([di/len(inputs) for di in d])
        #     #TODO add regularization
        #     self.gradient_descent(gradient)

    def predict(self, example):


        # if len(example) != len(self.layers[0].nodes):
        #     #TODO throw exception
        #     print 'Fucked up on predict'
        #     exit()
        # A = []
        # Z = []
        # A.append(list(example))  # a_1 == x
        # Z.append([])  # no values for z_1
        # prev_a = list(example)
        # for layer in self.layers[1:]:
        #     a, z = layer.feed_forward(prev_a)
        #     A.append(a)
        #     Z.append(z)
        #     prev_a = a
        # prediction = self.layers[-1].nodes[A[-1].index(max(A[-1]))].label
        # print len(A[0]), len(Z[0])
        # return prediction, A, Z  # return values for back prop

    def gradient_descent(self, gradient):


        # print len(gradient[2]), len(gradient[2][0]), len(self.layers[2].nodes), len(self.layers[2].nodes[0].weights)
        # print len(gradient[2]), len(gradient[2][0]), len(self.layers[1].nodes), len(self.layers[1].nodes[0].weights)
        # print '---->'
        # for layer, g in zip(self.layers[1:], gradient[1:]):
        #     for node, gi in zip(layer.nodes, g):
        #         new_weights = [weight - self.alpha * gij for weight, gij in zip(node.weights, gi)]
        #         node.set_weights(new_weights)
        # print len(gradient[2]), len(gradient[2][0]), len(self.layers[2].nodes), len(self.layers[2].nodes[0].weights)
        # print len(gradient[2]), len(gradient[2][0]), len(self.layers[1].nodes), len(self.layers[1].nodes[0].weights)


# class Layer:
#     def __init__(self, size, node_type, init_style, prev_num_nodes, labels=None):
#         self.nodes = []
#         if node_type is NodeType.Output and labels is not None:
#             # if the node type is output, add the labels
#             for node_number in range(size):
#                 self.nodes.append(Node(node_type, init_style, prev_num_nodes, labels[node_number]))
#         else:
#             # other node types don't get a label
#             for node_number in range(size):
#                 self.nodes.append(Node(node_type, init_style, prev_num_nodes))
#         print node_type, prev_num_nodes, size  #TODO remove
#
#     def feed_forward(self, prev_a):
#         a = []
#         z = []
#         for node in self.nodes:
#             ai, zi = node.feed_forward(prev_a)
#             a.append(ai)
#             z.append(zi)
#         return a, z


# class Node:
#     def __init__(self, node_type, init_style, prev_num_nodes, label=None):
#         self.NODE_TYPE = node_type
#         if node_type is NodeType.Output:
#             self.label = str(label)
#         # alpha values
#         self.weights = []
#         if node_type is not NodeType.Input:
#             self.weights = []
#             if init_style is NodeInitStyle.Ones:
#                 for dummy in range(prev_num_nodes+1):
#                     self.weights.append(1)
#             elif init_style is NodeInitStyle.Zeros:
#                 for dummy in range(prev_num_nodes+1):
#                     self.weights.append(0)
#             elif init_style is NodeInitStyle.Random:
#                 for dummy in range(prev_num_nodes+1):
#                     self.weights.append(random.randrange(-10000.0, 10000.0)/10000.0)
#             else:
#                 #TODO throw exception here
#                 print 'Fucked up on init node weights'
#                 exit()
#
#     def feed_forward(self, prev_values):
#         prev_a = [1.0] + list(prev_values)
#         if len(prev_a) != len(self.weights):
#             #TODO throw an exception
#             print 'Fucked up on node feed_forward'
#             exit(1)
#         zi = sum([weight*value for weight, value in zip(self.weights, prev_a)])
#         ai = sigmoid(zi)
#         return ai, zi
#
#     def set_weights(self, new_weights):
#         self.weights = list(new_weights)


class NodeType:
    Input = 'Input'
    Inner = 'Inner'
    Output = 'Output'


class NodeInitStyle:
    Zeros = 'Zeros'
    Ones = 'Ones'
    Random = 'Random'