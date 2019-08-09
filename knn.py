import numpy as np
import math
import operator

def euclideanDistance(instance1, instance2, length):
  dist = 0
  for i in range(length):
    dist = dist + pow(instance1[i] - instance2[i], 2)

  return math.sqrt(dist)

## getNieghbors calculates the distance of each point in the training_set
# to the test_point.
# training_set: a set of points
# test_point: a point
# retrun value: all the k nearest neighbors of test_point
def getNeighbors(training_set, test_point, k):
  # find the dimension of a data point
  dimension = len(training_set[0]) - 1
  # Store the distances
  distances = []

  # Compute the distances of all data points 
  for index in range(len(training_set)):
    dist = euclideanDistance(training_set[index], test_point, dimension)
    distances.append((train_set[index], dist))

  # sort the distances
  distances.sort(key=operator.itemgetter(1))

  # Get the first k data points
  neighbors = []
  for index in range(k):
    neighbors.append(distances[index][0])

  return neighbors
# END getNeigbors

## getPredictions return at prediction based on the max number of votes in
#  k neighbors
# neighbors: set of neighbors
def getPredictions(neighbors):
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
  max_label = ''
  max_votes = 0

  for label, count in labels.items():
    if count > max_votes:
      max_label = label
      max_votes = count

  return max_label

def getAccuracy(predictions, train_set):
  wrong_predection = 0

  for index in range(len(predictions)):
    if (predictions[index] != train_set[index][-1]):
      wrong_predection += 1

  return (wrong_predection / float(len(predictions))) * 100.0

## Read in training and test files
train_file = 'pa1train.txt'
train_set = np.loadtxt(train_file, delimiter=' ', dtype=int)

test_file = 'pa1test.txt'
test_set = np.loadtxt(test_file, delimiter=' ')
# END reading training and test files

predictions =[]

for index in range(len(train_set)):
  neighbors = getNeighbors(train_set, train_set[index], 3)
  predictions.append(getPredictions(neighbors))

print(getAccuracy(predictions, train_set))