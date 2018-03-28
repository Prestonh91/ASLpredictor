import warnings
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

# warnings.filterwarnings('ignore')


def loadHand():
    data = pd.read_csv("compiled.txt", header=None)
    targets = data[0]
    data = data.drop([34, 0], axis=1)

    return train_test_split(data, targets, test_size=0.3)


def small_grid():
    train, test, train_t, test_t = loadHand()
    test_t = test_t.as_matrix()

    for i in range(19, 26, 1):
        for j in range(19, 26, 1):
            classifier = MLPClassifier(hidden_layer_sizes=(i, j),
                                       max_iter=400,
                                       activation='logistic')
            classifier.fit(train, train_t)
            predictions = classifier.predict(test)

            correct = 0
            for k in range(len(predictions)):
                if predictions[k] == test_t[k]:
                    correct += 1

            print("This is for (", i, " ", j, ")")
            print("You got", correct, "out of", len(predictions), "total datapoints")
            print(correct / len(predictions) * 100)


def main():
    #small_grid()
    train, test, train_t, test_t = loadHand()
    test_t = test_t.as_matrix()

    classifier = MLPClassifier(hidden_layer_sizes=(25, 25),
                               max_iter=500,
                               activation='logistic',
                               learning_rate='invscaling')
    classifier.fit(train, train_t)
    prediction = classifier.predict(test)

    correct = 0
    for i in range(len(prediction)):
        if test_t[i] == prediction[i]:
            correct += 1

    print("You got", correct, "out of", len(prediction), "total datapoints")
    print(correct / len(prediction) * 100)

    train, test, train_t, test_t = loadHand()
    test_t = test_t.as_matrix()
    prediction = classifier.predict(test)
    correct = 0
    for i in range(len(prediction)):
        if test_t[i] == prediction[i]:
            correct += 1

    print("You got", correct, "out of", len(prediction), "total datapoints1")
    print(correct / len(prediction) * 100)

    print(classifier.get_params())


if __name__ == "__main__":
    main()