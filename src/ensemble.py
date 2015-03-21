__author__ = 'MarcoGiancarli, m.a.giancarli@gmail.com'


# This program is to be used to combine the results of an ensemble of trained
# networks. It reads the results from the benchmark csv files and write the mode
# into a file named 'ensemble_benchmark.csv'.

from csv import reader
from csv import writer
from csv import QUOTE_NONE

files_in_ensemble = [
    'nn_benchmark.csv'
    'nn_benchmark1.csv'
    'nn_benchmark2.csv'
    'nn_benchmark3.csv'
    'nn_benchmark4.csv'
]

input_files = [open(file_name,'r') for file_name in files_in_ensemble]

with open('gen/ensemble_benchmark.csv', 'wb') as output_file:
    w = writer(output_file, delimiter=',', quoting=QUOTE_NONE)

    for dummy in range(28000):
        current_predictions = ()
        prediction_counts = [0] * 10
        for input_file in input_files:
            current_predictions += input_file.readline()
        for prediction in current_predictions:
            prediction_counts[int(prediction)] += 1
        _, averaged_prediction = max(prediction_counts)
        print 'Predictions: '+current_predictions+' -- Average:'+averaged_prediction
        w.writerow(averaged_prediction)