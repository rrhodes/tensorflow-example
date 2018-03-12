import numpy as np
import pandas as pd
import tensorflow as tf

def retrieveData():
    trainingData = pd.read_csv("training-data.csv", header=None).as_matrix()
    testData = pd.read_csv("test-data.csv", header=None).as_matrix()

    return trainingData, testData

def separateFeaturesAndCategories(trainingData, testData):
    trainingFeatures = formatFeatures(trainingData[:, :-1])
    trainingCategories = trainingData[:, -1:]
    testFeatures = formatFeatures(testData[:, :-1])
    testCategories = testData[:, -1:]

    return trainingFeatures, trainingCategories, testFeatures, testCategories

def formatFeatures(trainingFeatures):
    formattedFeatures = dict()
    numColumns = trainingFeatures.shape[1]

    for i in range(0, numColumns):
        formattedFeatures[str(i + 1)] = trainingFeatures[:, i]

    return formattedFeatures

def defineFeatureColumns(formattedFeatures):
    featureColumns = []

    for key in formattedFeatures.keys():
        featureColumns.append(tf.feature_column.numeric_column(key=key))

    return featureColumns

def instantiateEstimator(featureColumns):
    classifier = tf.estimator.DNNClassifier(
        feature_columns = featureColumns,
        hidden_units = [20, 30, 20],
        n_classes = 10
    )

    return classifier

def train(features, labels, batchSize):
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    return dataset.shuffle(1000).repeat().batch(batchSize)

def evaluate(features, labels, batchSize):
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels)).batch(batchSize)

    return dataset

def trainClassifier(classifier, trainingFeatures, trainingCategories):
    classifier.train(
        input_fn = lambda:train(trainingFeatures, trainingCategories, 100),
        steps = 50
    )

def evalClassifier(classifier, testFeatures, testCategories):
    eval = classifier.evaluate(
        input_fn = lambda:evaluate(testFeatures, testCategories, 100)
    )

    return eval

def main():
    trainingData, testData = retrieveData()

    trainingFeatures, trainingCategories, testFeatures, testCategories = \
        separateFeaturesAndCategories(trainingData, testData)

    featureColumns = defineFeatureColumns(trainingFeatures)

    classifier = instantiateEstimator(featureColumns)

    trainClassifier(classifier, trainingFeatures, trainingCategories)

    eval = evalClassifier(classifier, testFeatures, testCategories)

    print('Test accuracy: {accuracy:0.3f}\n'.format(**eval))

if __name__ == "__main__":
    main()
