__author__ = 'MarcoGiancarli'


from csv import reader
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
    i = 0             # remove me
    for line in training_data:
        i += 1        # remove me
        if i > 200:   # remove me
            break     # remove me
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
# print 'Loading test set...'
#
# with open('res/datasets/test.csv', ) as test_file:
#    test_data = reader(test_file, delimiter=',')
#    test_x = [row[1:] for row in test_data]
#    test_y = [row[0] for row in test_data]
#
# print 'Test set loaded.'
print 'Training network...'

#TODO: test with a range of alpha values to get an appropriate step size.
network = NeuralNetwork([784,140,10], NodeInitStyle.Random, .1, labels=[str(i) for i in range(10)], reg_constant=1)

#TODO: make some tests to see if network initialized correctly
network.train(training_x[:-100], training_y[:-100])

print 'Network trained.'

num_correct = 0
num_tests = 0
for x, y in zip(training_x[-100:], training_y[-100:]):
    prediction = network.predict(x)
    print str(prediction) + ' -- correct number: ' + str(y)
    num_tests += 1
    if int(prediction) == y:
        num_correct += 1

print str(num_correct), ' / ', str(num_tests)