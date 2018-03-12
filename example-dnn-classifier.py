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

def formatFeatures(features):
    formattedFeatures = dict()
    numColumns = features.shape[1]

    for i in range(0, numColumns):
        formattedFeatures[str(i)] = features[:, i]

    return formattedFeatures

def defineFeatureColumns(features):
    featureColumns = []

    for key in features.keys():
        featureColumns.append(tf.feature_column.numeric_column(key=key))

    return featureColumns

def instantiateClassifier(featureColumns):
    classifier = tf.estimator.DNNClassifier(
        feature_columns = featureColumns,
        hidden_units = [20, 30, 20],
        n_classes = 10
    )

    return classifier

def train(features, labels, batchSize):
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    return dataset.shuffle(1000).repeat().batch(batchSize)

def test(features, labels, batchSize):
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels)).batch(batchSize)

    return dataset

def trainClassifier(classifier, trainingFeatures, trainingCategories):
    classifier.train(
        input_fn = lambda:train(trainingFeatures, trainingCategories, 100),
        steps = 50
    )

def testClassifier(classifier, testFeatures, testCategories):
    accuracy = classifier.evaluate(
        input_fn = lambda:test(testFeatures, testCategories, 100)
    )

    return accuracy

def main():
    trainingData, testData = retrieveData()

    trainingFeatures, trainingCategories, testFeatures, testCategories = \
        separateFeaturesAndCategories(trainingData, testData)

    featureColumns = defineFeatureColumns(trainingFeatures)

    classifier = instantiateClassifier(featureColumns)

    trainClassifier(classifier, trainingFeatures, trainingCategories)

    accuracy = testClassifier(classifier, testFeatures, testCategories)

    print('Test accuracy: {accuracy:0.3f}\n'.format(**accuracy))

if __name__ == "__main__":
    main()
