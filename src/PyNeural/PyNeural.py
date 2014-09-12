__author__ = 'MarcoGiancarli, m.a.giancarli@gmail.com'


class NeuralNetwork:
    layers = []

    # Inputs should be a list of lists containing the input values in each row
    inputs = []

    # Outputs at some index i should be a list of integers where each integer corresponds
    # to the index of the output node which contains the correct label
    outputs = []

    def __init__(self, layer_sizes, init_style, labels=None):
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

    ''' This method is used internally for training on a complete data set. '''
    def train(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs
        # iterate through data set and train neural network
        #TODO iterate & train

    ''' This method is used internally for adding new rows to the data set and training again '''
    def retrain(self, inputs, outputs):
        # add input and output to current set and call train
        self.inputs.append(inputs)
        self.outputs.append(outputs)
        self.train(self.inputs, self.outputs)


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


class Node:
    NODE_TYPE = ''

    # use theta in a linear combination with the previous row to get feed forward value
    weights = []

    def __init__(self, node_type, init_style, prev_num_nodes, label=None):
        self.NODE_TYPE = node_type
        if node_type is NodeType.Output:
            self.label = str(label)
        #TODO add init for theta here

    def feed_forward(self, prev_layer_values):
        if len(prev_layer_values) != len(self.weights):
            #TODO throw an exception
            pass
        val = sum(zip(self.weights, prev_layer_values))
        return val


class NodeType:
    Input = 'Input'
    Inner = 'Inner'
    Output = 'Output'


class NodeInitStyle:
    Zeros = 'Zeros'
    Ones = 'Ones'
    Random = 'Random'