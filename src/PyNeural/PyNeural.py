__author__ = 'MarcoGiancarli, m.a.giancarli@gmail.com'


import random
import math
import numpy as np

# Use a consistent random seed so that all tests are also consistent.
# random.seed(911*100)
# It would be like 9-11 times 100.
# 9-11 times 100? Jesus, that's...
# Yes, 91,100.


# trying out tanh instead of normal sigmoid as the activation function; should be easier to compute. Emulates sigmoid
def sigmoid(x):
    return (math.tanh(x) + 1) / 2
sigmoid = np.vectorize(sigmoid, otypes=[np.float])


# derivative of our sigmoid function
def d_sigmoid(x):
    return sigmoid(x) * (1-sigmoid(x))
d_sigmoid = np.vectorize(d_sigmoid, otypes=[np.float])


def make_weight(init_style):
    if init_style is NodeInitStyle.Random:
        return random.randrange(-10000.0, 10000.0)/10000.0
    elif init_style is NodeInitStyle.Zeros:
        return 0.0
    elif init_style is NodeInitStyle.Ones:
        return 1.0


def output_vector_to_scalar(vector):
    # get the index of the max in the vector
    return vector.tolist().index(max(vector))


def output_scalar_to_vector(scalar, num_outputs):
    # same size as outputs, all 0s
    vector = [0] * num_outputs
    # add 1 to the correct index
    vector[scalar] += 1
    return vector


#TODO: add methods to save state and return aspects (such as size of network, weights, etc.)
#TODO: learning curves?
#TODO: regularization
class NeuralNetwork:
    def __init__(self, layer_sizes, init_style, alpha, labels=None, reg_constant=0):
        self.alpha = alpha
        self.regularization_constant = reg_constant

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
        self.theta = [None] * len(layer_sizes)
        for l in range(1, len(layer_sizes)):
            # append a matrix which represents the initial weights for layer l
            self.theta[l] = np.mat(
                    # add a weight for each prev node (and one additional weight)
                    [[make_weight(init_style) for prev_node in range(layer_sizes[l-1]+1)]
                                              # do the above for each current node
                                              for node in range(layer_sizes[l])]
            )

    ''' Feed forward and return lists of matrices A and Z for all training samples,
        where A[l][i][j] is A at layer l, training sample i, and node j. '''
    def feed_forward(self, input_matrix):
        A = [None]*len(self.theta)
        Z = [None]*len(self.theta)
        A[0] = input_matrix.getT()  # 1 x n
        Z[0] = None  # z_1 doesn't exist
        for l in range(1, len(self.theta)):
            # add a constant (1) to the set of weights that correspond with each node
            num_ones_to_add = A[l-1].shape[1]
            A_with_ones = np.vstack((np.ones(num_ones_to_add), A[l-1]))
            Z[l] = self.theta[l] * A_with_ones
            # temp = [[sigmoid(Z_ij) for Z_ij in Z_i] for Z_i in Z[l].tolist()] TODO remove
            A[l] = sigmoid(Z[l])

        return A, Z

    ''' Back propagate for all training samples. '''
    def back_prop(self, input, output):
        # note: outputs is a list of the indices of the guessed output node

        A, Z = self.feed_forward(np.mat(input))

        # let y be a matrix where y[i] is the output vector for training sample i
        y = np.mat(output_scalar_to_vector(output, self.theta[-1].shape[0]))

        # let delta be a list of matrices where delta[l][i][j] is delta
        # at layer l, training sample i, and node j
        delta = [None] * len(self.theta)  # the delta is None for the input layer, others we assign later
        delta[-1] = np.multiply(A[-1] - y.getT(), d_sigmoid(Z[-1]))

        for l in reversed(range(1, len(self.theta)-1)):  # note: no error on input layer, we have the output layer
            theta_t_delta = self.theta[l+1].getT() * delta[l+1]
            delta[l] = np.multiply(np.mat(theta_t_delta.tolist()[1:]), d_sigmoid(Z[l]))

        # Calculate the partial derivatives for all theta values using delta
        D = [None]*len(self.theta)  # make a list of size L, where L is the number of layers
        for l in range(1, len(self.theta)):
            D[l] = A[l-1] * delta[l].getT()

        return D, delta

    ''' This method is used for supervised training on a data set. '''
    def train(self, inputs, outputs, test_inputs=None, test_outputs=None, iteration_cap=5000):
        m = len(outputs)
        for iteration in range(iteration_cap):
            print 'Training iteration:', str(iteration + 1)

            d, b = self.back_prop(inputs[0], outputs[0])
            gradient = [np.mat(d_l) for d_l in d]
            bias_gradient = [np.mat(b_l) for b_l in b]
            for input, output in zip(inputs[1:], outputs[1:]):
                d, b = self.back_prop(input, output)
                for l in range(1, len(self.theta)):
                    gradient[l] = np.add(gradient[l], d[l])
                    bias_gradient[l] = np.add(bias_gradient[l], b[l])

            # add regularization to the gradient matrices


            gradient_with_bias = [None]*len(self.theta)
            for l in range(1, len(self.theta)):
                gradient_with_bias[l] = np.hstack((bias_gradient[l], gradient[l].getT()))

            # divide by m now because we couldn't while in the back_prop method
            gradient_with_bias = [g / m for g in gradient_with_bias[1:]]

            self.gradient_descent(gradient_with_bias)

            #TODO: use a small test set to test the error after each iteration.
            if test_inputs is not None and test_outputs is not None:
                #TODO: throw mad exceptions for shit
                num_tests = len(test_outputs)
                num_correct = 0
                for test_input, test_output in zip(test_inputs, test_outputs):
                    prediction = int(self.predict(test_input))
                    if prediction == test_output:
                        num_correct += 1
                test_error = 1 - (float(num_correct) / float(num_tests))
                print 'Test at iteration', str(iteration+1) + ': ', str(num_correct), '/', str(num_tests), \
                        '-- Error:', str(test_error)

            if test_error < 0.04:
                break

    ''' This method calls feed_forward and returns just the prediction labels for all samples. '''
    def predict(self, input):
        A, _ = self.feed_forward(np.mat(input))
        prediction = output_vector_to_scalar(A[-1])
        # if self.labels is not None:
        #     if len(self.labels) != len(A[-1]):
        #         # TODO: throw exception
        #         print 'Fucked up because the number of labels didnt match the number of outputs'
        #         exit(1)
        #     return self.labels[prediction]
        # else:
        #     return prediction
        return prediction

    def gradient_descent(self, gradient):
        for l in range(1, len(self.theta)):
            # gradient doesnt have a None value at index 0, but theta does
            self.theta[l] = np.add(self.theta[l], (-1.0 * self.alpha) * gradient[l-1])

        # TODO: maybe remove this feature? could help fine tuning at the end of training
        self.alpha = self.alpha * 0.999


class NodeInitStyle:
    Zeros = 'Zeros'
    Ones = 'Ones'
    Random = 'Random'