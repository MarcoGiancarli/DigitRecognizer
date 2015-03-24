__author__ = 'MarcoGiancarli, m.a.giancarli@gmail.com'


from csv import reader
from csv import writer
from csv import QUOTE_NONE
import numpy as np
from PyNeural.PyNeural import NeuralNetwork


print 'Loading training set...'

training_x_raw = []
training_y_raw = []
training_x = []
training_y = []

with open('res/datasets/train.csv', ) as training_file:
    training_data = reader(training_file, delimiter=',')
    skipped_titles = False
    # i = 0               ###
    for line in training_data:
        # if i > 1000:    ###
        #     break       ###
        # i += 1          ###
        if not skipped_titles:
            skipped_titles = True
            continue
        fields = list(line)
        training_y_raw = fields[0]
        training_x_raw = fields[1:]
        # remove the damn labels
        training_y.append(int(training_y_raw))
        training_x.append([int(val) for val in training_x_raw])

x_array = np.array(training_x)
# normalize the training set
training_x = ((x_array - np.average(x_array)) / np.std(x_array)).tolist()

print 'Training set loaded. Samples:', len(training_x)
print 'Training network...'

layer_sizes = [784,300,10]
alpha = 0.02
network = NeuralNetwork(layer_sizes, alpha)

network.train(training_x[:-2000], training_y[:-2000], test_inputs=training_x[-2000:],
        test_outputs=training_y[-2000:], epoch_cap=150, error_goal=0.018, dropconnect_chance=0.15)

print 'Network trained.'

num_correct = 0
num_tests = 0
for x, y in zip(training_x[-2000:], training_y[-2000:]):
    prediction = network.predict(x)
    num_tests += 1
    if int(prediction) == y:
        num_correct += 1
print str(num_correct), '/', str(num_tests)

# clear junk
network.momentum = None
network.dropconnect_matrices = None
training_x = None
training_y = None
training_data = None
training_x_raw = None
training_y_raw = None

print 'Loading test data...'

test_x_raw = []
test_x = []
test_y = []

with open('gen/nn_benchmark5.csv', 'wb') as output_file:
    w = writer(output_file, delimiter=',', quoting=QUOTE_NONE)
    w.writerow(['ImageId','Label'])

    with open('res/datasets/test.csv', ) as test_file:
        test_data = reader(test_file, delimiter=',')
        skipped_titles = False
        num_predictions = 0
        for line in test_data:
            if not skipped_titles:
                skipped_titles = True
                continue
            fields = list(line)
            test_x_raw = fields
            # remove the damn labels
            test_x.append([int(val) for val in test_x_raw])
            num_predictions += 1
            if num_predictions % 100 == 0:
                x_array = np.array(test_x)
                # normalize the test set
                test_x = ((x_array - np.average(x_array)) / np.std(x_array)).tolist()
                for i in range(100):
                    w.writerow([num_predictions-99+i, network.predict(test_x[i])])
                test_x = []
                x_array = []


print 'Predicted labels and stored as "nn_benchmark.csv".'