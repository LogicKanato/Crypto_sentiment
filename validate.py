from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score

### Outputs errors of single layer NN
### neurons is a list of the numbers of neurons
### ie) neurons = [5,10,20,50,100,150,200,250,300,350,400]
def single_layer(neurons, x_train, y_train):
    accuracies = []
    for i, num_neurons in enumerate(neurons):
        clf = MLPClassifier(hidden_layer_sizes=(num_neurons),max_iter = 1000)
        accuracies.append(1 - cross_val_score(clf, x_train, y_train, cv=5).mean())

    return accuracies