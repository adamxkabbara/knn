import numpy as np
import math
import operator
import random

def euclideanDistance(instance1, instance2):
  return np.sqrt(np.sum((instance1 - instance2)**2))

## getNieghbors calculates the distance of each point in the training_set
# to the test_point.
# training_set: a set of points
# test_point: a point
# retrun value: all the k nearest neighbors of test_point
def findNeighbors(training_set, test_point, k):
  # Store the distances
  distances = []

  # Compute the distances of all data points 
  for index in range(len(training_set)):
    dist = euclideanDistance(training_set[index], test_point)
    distances.append((training_set[index], dist))

  # sort the distances
  distances.sort(key=operator.itemgetter(1))

  # Get the first k data points
  neighbors = []
  for index in range(k):
    neighbors.append(distances[index][0])

  return neighbors
# END getNeigbors

## findPredictions return at prediction based on the max number of votes in
#  k neighbors.
# neighbors: set of neighbors
def findPredictions(neighbors):
  # contains a list of labels and there count
  votes = {}
  
  for index in range(len(neighbors)):
    label = neighbors[index][-1]
    if label in votes:
      votes[label] += 1
    else:
      votes[label] = 1
  return getMaxVote(votes)

def getMaxVote(labels):
  max_votes = 0

  for label, count in labels.items():
    if count > max_votes:
      max_votes = count

  max_label = []
  for label, count in labels.items():
    if count == max_votes:
      max_label.append(label)

  return max_label[random.randint(0,len(max_label) - 1)]

def getAccuracy(predictions, train_set):
  wrong_predection = 0

  for index in range(len(predictions)):
    if (predictions[index] != train_set[index][-1]):
      wrong_predection += 1

  return (wrong_predection / float(len(predictions))) * 100

def training_error(data_set):
  for k in [1, 5, 9, 15]:
    predictions =[]
    for index in range(len(data_set)):
      neighbors = findNeighbors(data_set, data_set[index], k)
      predictions.append(findPredictions(neighbors))

    print('k=', k, '|',getAccuracy(predictions, data_set), '%')

def validation(training_set, data_set):
  for k in [1, 5, 9, 15]:
    predictions =[]
    for index in range(len(data_set)):
      neighbors = findNeighbors(training_set, data_set[index], k)
      predictions.append(findPredictions(neighbors))

    print('k=', k, '|',getAccuracy(predictions, data_set), '%')

def projection(projecting_set, instance_set):
  label_index = len(instance_set[0]) - 1
  projection = np.empty([len(instance_set), label_index])
  labels = np.empty([len(instance_set), 1])

  index = 0
  for row in instance_set:
    labels[index] = instance_set[index][label_index]
    projection[index] = np.delete(row, label_index)
    index+=1
  
  projection = np.matmul(projection, projecting_set)
  final = np.empty([len(projection), len(projection[0]) + 1])

  index = 0
  for row in instance_set:
    final[index] = np.append(projection[index], labels[index])
    index+=1

  return final

## Read in training and test files
training_set = np.loadtxt('train.txt', delimiter=' ', dtype=int)
validation_set = np.loadtxt('validate.txt', delimiter=' ', dtype=int)
test_set = np.loadtxt('test.txt', delimiter=' ', dtype=int)
projection_set = np.loadtxt('projection.txt', delimiter=' ', dtype=float)

print('Data: train')
print('----------------')
training_error(training_set)

print('Data: validation')
print('----------------')
validation(training_set, validation_set)

print('Data: test')
print('----------------')
validation(training_set, test_set)

pro_training = projection(projection_set, training_set)
print('projection: train')
print('----------------')
training_error(pro_training)

pro_validtion = projection(projection_set, validation_set)
print('projection: validation')
print('----------------')
validation(pro_training, pro_validtion)

pro_test = projection(projection_set, test_set)
print('projection: test')
print('----------------')
validation(pro_training, pro_test)