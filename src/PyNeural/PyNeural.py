__author__ = 'MarcoGiancarli, m.a.giancarli@gmail.com'


# import random
# import math
import numpy as np


# Using tanh instead of normal sigmoid because it's easier to compute. Emulates sigmoid.
# def sigmoid(x):
#     return (math.tanh(x) + 1) / 2
# sigmoid = np.vectorize(sigmoid, otypes=[np.float])
def sigmoid(x):
    return (np.tanh(x) + 1) / 2


# derivative of our sigmoid function
# def d_sigmoid(x):
#     return sigmoid(x) * (1-sigmoid(x))
# d_sigmoid = np.vectorize(d_sigmoid, otypes=[np.float])
def d_sigmoid(x):
    return (np.tanh(x)+1) * (1-np.tanh(x)) / 4


def output_vector_to_scalar(vector):
    # get the index of the max in the vector
    m,i = max((v,i) for i,v in enumerate(vector.tolist()))
    return i


def output_scalar_to_vector(scalar, num_outputs):
    # same size as outputs, all 0s
    vector = [0] * num_outputs
    # add 1 to the correct index
    vector[scalar] += 1
    return vector


#TODO: add methods to save state
#TODO: learning curves?
#TODO: regularization
class NeuralNetwork:
    def __init__(self, layer_sizes, alpha, labels=None, reg_constant=0):
        self.alpha = alpha
        self.regularization_constant = reg_constant

        if labels is None:
            self.labels = range(layer_sizes[-1])
        elif len(labels) != layer_sizes[-1]:
            #TODO: throw exception here
            print 'Fucked up because the size of layer does not match the size of the outputs. (' + \
                  str(len(labels)) + ' != ' + str(layer_sizes[-1]) + ')'
            exit(1)
        else:
            self.labels = labels

        # theta represents the weights for each node. we skip the first layer because it has no weights.
        self.theta = [None] * len(layer_sizes)
        for l in range(1, len(layer_sizes)):
            # append a matrix which represents the initial weights for layer l
            # for every node in layer l, add a weight for each node in layer l-1 plus one bias weight
            # beta = 0.7 * (layer_sizes[l-1] / layer_sizes[l])
            self.theta[l] = np.random.random((layer_sizes[l], layer_sizes[l-1]+1)) * 2 - 1
            # norm = math.sqrt(np.multiply(self.theta[l], self.theta[l]).sum())
            # self.theta[l] = self.theta[l] * (beta / norm)

    ''' Feed forward and return lists of matrices A and Z for one set of inputs. '''
    def feed_forward(self, input_vector):
        A = [None]*len(self.theta)
        Z = [None]*len(self.theta)
        A[0] = input_vector.T  # 1 x n
        Z[0] = None  # z_1 doesn't exist
        for l in range(1, len(self.theta)):
            # add a constant (1) to the set of weights that correspond with each node
            A_with_ones = np.concatenate((np.array([1]), A[l-1]))
            Z[l] = np.dot(self.theta[l], A_with_ones)
            A[l] = sigmoid(Z[l])

        return A, Z

    ''' Back propagate for one training sample. '''
    def back_prop(self, input_vector, output_vector):
        A, Z = self.feed_forward(input_vector)

        # let delta be a list of matrices where delta[l][i][j] is delta
        # at layer l, training sample i, and node j
        delta = [None] * len(self.theta)  # the delta is None for the input layer, others we assign later
        delta[-1] = np.multiply(A[-1] - output_vector.T, d_sigmoid(Z[-1]))

        for l in reversed(range(1, len(self.theta)-1)):  # note: no error on input layer, we have the output layer
            theta_t_delta = np.dot(self.theta[l+1].T, delta[l+1])
            delta[l] = np.multiply(theta_t_delta[1:], d_sigmoid(Z[l]))

        # Calculate the partial derivatives for all theta values using delta
        D = [None]*len(self.theta)  # make a list of size L, where L is the number of layers
        for l in range(1, len(self.theta)):
            D[l] = np.dot(np.atleast_2d(A[l-1]).T, np.atleast_2d(delta[l]))

        return D, delta

    ''' This method is used for supervised training on a data set. '''
    def train(self, inputs, outputs, test_inputs=None, test_outputs=None, epoch_cap=10000, error_goal=0):
        # create these first so that we don't have to do it every epoch
        input_vectors = [np.array(x) for x in inputs]
        output_vectors = [np.array(output_scalar_to_vector(y, self.theta[-1].shape[0])) for y in outputs]
        test_input_vectors = [np.array(x) for x in test_inputs]
        test_output_vectors = [np.array(output_scalar_to_vector(y, self.theta[-1].shape[0])) for y in test_outputs]

        m = len(outputs)
        for iteration in range(epoch_cap):
            for input_vector, output_vector in zip(input_vectors, output_vectors):
                gradient, bias = self.back_prop(input_vector, output_vector)
                gradient_with_bias = [None]*len(self.theta)

                for l in range(1, len(self.theta)):
                    gradient_with_bias[l] = np.vstack((bias[l], gradient[l]))
                    gradient_with_bias[l] = gradient_with_bias[l].T

                gradient_with_bias = [g / m for g in gradient_with_bias[1:]]
                self.gradient_descent(gradient_with_bias)

            # test the updated system against the validation set
            if test_inputs is not None and test_outputs is not None:
                num_tests = len(test_output_vectors)
                num_correct = 0
                for test_input, test_output in zip(test_input_vectors, test_outputs):
                    prediction = self.predict(test_input)
                    if prediction == test_output:
                        num_correct += 1
                test_accuracy = float(num_correct) / float(num_tests)
                print 'Test at epoch', str(iteration+1) + ': ', str(num_correct), '/', str(num_tests), \
                    '-- Accuracy:', str(test_accuracy)

                if num_correct / num_tests > 1 - error_goal:
                    break

    ''' This method calls feed_forward and returns just the prediction labels for all samples. '''
    def predict(self, input):
        A, _ = self.feed_forward(np.array(input))
        return np.argmax(A[-1])

    def gradient_descent(self, gradient):
        for l in range(1, len(self.theta)):
            # gradient doesnt have a None value at index 0, but theta does
            self.theta[l] = np.add(self.theta[l], (-1.0 * self.alpha) * gradient[l-1])