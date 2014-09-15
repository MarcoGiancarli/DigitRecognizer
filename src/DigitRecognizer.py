__author__ = 'MarcoGiancarli'


from csv import reader

with open('res/datasets/train.csv', ) as training_file:
    training_data = reader(training_file, delimiter=',')

# separate the label (y) from the inputs (x)
training_x = [row[1:] for row in training_data]
training_y = [row[0] for row in training_data]

