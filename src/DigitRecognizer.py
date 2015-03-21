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
        # if i > 700:     ###
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

layer_sizes = [784,117,10]
alpha = 0.02
network = NeuralNetwork(layer_sizes, alpha, reg_constant=1)

network.train(training_x[:-2000], training_y[:-2000], test_inputs=training_x[-2000:],
        test_outputs=training_y[-2000:], epoch_cap=200, error_goal=0.0199, dropout_chance=0.2)

print 'Network trained.'

num_correct = 0
num_tests = 0
for x, y in zip(training_x[-2000:], training_y[-2000:]):
    prediction = network.predict(x)
    # print str(prediction) + ' -- correct number: ' + str(y)
    num_tests += 1
    if int(prediction) == y:
        num_correct += 1
print str(num_correct), '/', str(num_tests)


print 'Loading test data...'

test_x_raw = []
test_x = []
test_y = []

with open('gen/nn_benchmark.csv', 'wb') as output_file:
    w = writer(output_file, delimiter=',', quoting=QUOTE_NONE)
    with open('res/datasets/test.csv', ) as test_file:
        test_data = reader(test_file, delimiter=',')
        skipped_titles = False
        num_in_batch = 0
        for line in test_data:
            if not skipped_titles:
                skipped_titles = True
                continue
            fields = list(line)
            test_x_raw = fields
            # remove the damn labels
            test_x.append([int(val) for val in test_x_raw])
            num_in_batch += 1
            if num_in_batch == 1000:
                x_array = np.array(test_x)
                # normalize the test set
                test_x = ((x_array - np.average(x_array)) / np.std(x_array)).tolist()
                for i in range(1000):
                    w.writerow([network.predict(test_x[i])])


print 'Predicted labels and stored as "nn_benchmark.csv".'