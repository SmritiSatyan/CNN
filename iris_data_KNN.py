import csv
import random
import operator
import math

iris_data_path = "C:\\Users\\satyanarayana.smriti\\Desktop\\SMRITI_Experiments\\iris_data.csv" 

def loadDataset(filename, split, trainingSet=[] , testSet=[]):
	with open(filename, 'r') as csvfile:
	    lines = csv.reader(csvfile)
	    dataset = list(lines)
	    for x in range(len(dataset)-1):
	        for y in range(4):
	            dataset[x][y] = float(dataset[x][y])
	        if random.random() < split:
	            trainingSet.append(dataset[x])
	        else:
	            testSet.append(dataset[x])


def euclideanDistance(instance1, instance2, length):
	distance = 0
	for x in range(length):
		distance += pow((instance1[x] - instance2[x]), 2)
	return math.sqrt(distance)

def getNeighbors(trainingSet, testInstance, k):
	distances = []
	length = len(testInstance)-1
	for x in range(len(trainingSet)):
		dist = euclideanDistance(testInstance, trainingSet[x], length)
		distances.append((trainingSet[x], dist))
	distances.sort(key=operator.itemgetter(1))
	neighbors = []
	for x in range(k):
		neighbors.append(distances[x][0])
	return neighbors

def getResponse(neighbors):
	classVotes = {}
	for x in range(len(neighbors)):
		response = neighbors[x][-1]
		if response in classVotes:
			classVotes[response] += 1
		else:
			classVotes[response] = 1
	sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
	return sortedVotes[0][0]

def getAccuracy(testSet, predictions):
	correct = 0
	for x in range(len(testSet)):
		if testSet[x][-1] == predictions[x]:
			correct += 1
	return (correct/float(len(testSet))) * 100.0
	
def main():
	# prepare data
	trainingSet=[]
	testSet=[]
	split = 0.67
	loadDataset(iris_data_path, split, trainingSet, testSet)
	print('Train set: ' + repr(len(trainingSet)))
	print('Test set: ' + repr(len(testSet)))
	# generate predictions
	predictions=[]
	k = 3
	for x in range(len(testSet)):
		neighbors = getNeighbors(trainingSet, testSet[x], k)
		result = getResponse(neighbors)
		predictions.append(result)
		print('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))
	accuracy = getAccuracy(testSet, predictions)
	print('Accuracy: ' + repr(accuracy) + '%')
	
main()







#from collections import Counter
#import math
#
#def knn(data, query, k, distance_fn, choice_fn):
#    neighbor_distances_and_indices = []
#    
#    # 3. For each example in the data
#    for index, example in enumerate(data):
#        # 3.1 Calculate the distance between the query example and the current
#        # example from the data.
#        distance = distance_fn(example[:-1], query)
#        
#        # 3.2 Add the distance and the index of the example to an ordered collection
#        neighbor_distances_and_indices.append((distance, index))
#    
#    # 4. Sort the ordered collection of distances and indices from
#    # smallest to largest (in ascending order) by the distances
#    sorted_neighbor_distances_and_indices = sorted(neighbor_distances_and_indices)
#    
#    # 5. Pick the first K entries from the sorted collection
#    k_nearest_distances_and_indices = sorted_neighbor_distances_and_indices[:k]
#    
#    # 6. Get the labels of the selected K entries
#    k_nearest_labels = [data[i][1] for distance, i in k_nearest_distances_and_indices]
#    print(k_nearest_distances_and_indices , choice_fn(k_nearest_labels))
#    # 7. If regression (choice_fn = mean), return the average of the K labels
#    # 8. If classification (choice_fn = mode), return the mode of the K labels
#    return k_nearest_distances_and_indices , choice_fn(k_nearest_labels)
#
#def mean(labels):
#    return sum(labels) / len(labels)
#
#def mode(labels):
#    return Counter(labels).most_common(1)[0][0]
#
#def euclidean_distance(point1, point2):
#    sum_squared_distance = 0
#    for i in range(len(point1)):
#        sum_squared_distance += math.pow(point1[i] - point2[i], 2)
#    return math.sqrt(sum_squared_distance)
#
#def main():
#    '''
#    # Regression Data
#    # 
#    # Column 0: height (inches)
#    # Column 1: weight (pounds)
#    '''
#    reg_data = [
#       [65.75, 112.99],
#       [71.52, 136.49],
#       [69.40, 153.03],
#       [68.22, 142.34],
#       [67.79, 144.30],
#       [68.70, 123.30],
#       [69.80, 141.49],
#       [70.01, 136.46],
#       [67.90, 112.37],
#       [66.49, 127.45],
#    ]
#    
#    # Question:
#    # Given the data we have, what's the best-guess at someone's weight if they are 60 inches tall?
#    reg_query = [60]
#    reg_k_nearest_neighbors, reg_prediction = knn(
#        reg_data, reg_query, k=3, distance_fn=euclidean_distance, choice_fn=mean)
#    print("Function reg kNN called, and returns")
#    print(knn(reg_data, reg_query, k=3, distance_fn=euclidean_distance, choice_fn=mean))
#    
#    
#    '''
#    # Classification Data
#    # 
#    # Column 0: age
#    # Column 1: likes pineapple
#    '''
#    clf_data = [
#       [22, 1],
#       [23, 1],
#       [21, 1],
#       [18, 1],
#       [19, 1],
#       [25, 0],
#       [27, 0],
#       [29, 0],
#       [31, 0],
#       [45, 0],
#    ]
#    # Question:
#    # Given the data we have, does a 33 year old like pineapples on their pizza?
#    clf_query = [33]
#    clf_k_nearest_neighbors, clf_prediction = knn(
#        clf_data, clf_query, k=3, distance_fn=euclidean_distance, choice_fn=mode)
#    print("Function CLF called, and returns")
#    print(knn(clf_data, clf_query, k=3, distance_fn=euclidean_distance, choice_fn=mode))
#
#
#if __name__ == '__main__':
#    main()
#    
#    
#    
#    
##-----------------------------------------------------------------------------------------------------------------------------    
##-----------------------------------------------------------------------------------------------------------------------------
#    
## Example of kNN implemented from Scratch in Python
#
#
#import csv   
#iris_data_path = "C:\\Users\\satyanarayana.smriti\\Desktop\\SMRITI_Experiments\\iris_data.csv" 
#with open(iris_data_path, 'r') as csvfile:
#	lines = csv.reader(csvfile)
#	for row in lines:
#		print(', '.join(row))
#        
#

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        