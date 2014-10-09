__author__ = 'MarcoGiancarli, m.a.giancarli@gmail.com'


from csv import reader
from csv import writer
from csv import QUOTE_NONE
from PyNeural.PyNeural import NeuralNetwork
from PyNeural.PyNeural import NodeInitStyle

print 'Loading training set...'

training_x_raw = []
training_y_raw = []
training_x = []
training_y = []

with open('res/datasets/train.csv', ) as training_file:
    training_data = reader(training_file, delimiter=',')
    skipped_titles = False
    # i = 0              # remove me
    for line in training_data:
    #     i += 1         # remove me
    #     if i > 12000:  # remove me
    #         break      # remove me
        if not skipped_titles:
            skipped_titles = True
            continue
        fields = list(line)
        training_y_raw = fields[0]
        training_x_raw = fields[1:]
        # remove the damn labels
        training_y.append(int(training_y_raw))
        training_x.append([int(val) for val in training_x_raw])

# normalize the data so that the range is from -1 to 1
for i in range(len(training_x)):
    for j in range(len(training_x[i])):
        training_x[i][j] = (training_x[i][j] / 127.5) - 1.0

print 'Training set loaded. Samples:', len(training_x)
print 'Training network...'

#TODO: test with a range of alpha values to get an appropriate step size.
network = NeuralNetwork([784,28,10], NodeInitStyle.Random, 5, reg_constant=1)

#TODO: make some tests to see if network initialized correctly
network.train(training_x[:-2000], training_y[:-2000], test_inputs=training_x[-2000:], test_outputs=training_y[-2000:])

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

with open('res/datasets/test.csv', ) as test_file:
    test_data = reader(test_file, delimiter=',')
    skipped_titles = False
    for line in test_data:
        if not skipped_titles:
            skipped_titles = True
            continue
        fields = list(line)
        test_x_raw = fields
        # remove the damn labels
        test_x.append([int(val) for val in test_x_raw])

print 'Loaded test data.'
print 'Predicting test data...'

# normalize the data so that the range is from -1 to 1
for i in range(len(test_x)):
    for j in range(len(test_x[i])):
        test_x[i][j] = (test_x[i][j] / 127.5) - 1.0

predictions = []
for input in test_x:
    predictions.append(network.predict(test_x))

with open('gen/nn_benchmark.csv', 'wb') as output_file:
    w = writer(output_file, delimiter=',', quoting=QUOTE_NONE)
    w.writerow(['ImageId', 'Label'])
    for test_num, label in zip(range(1, len(predictions)+1), predictions):
        w.writerow([test_num, label])

print 'Predicted labels and stored as "nn_benchmark.csv".'