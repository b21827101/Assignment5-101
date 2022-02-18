import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-0.005*x))

def sigmoid_derivative(x):
    return 0.005 * x * (1 - x )

def read_and_divide_into_train_and_test(csv_file):
    file = pd.read_csv(csv_file)
    a, b= file.shape
    file.replace('?', 1, inplace=True)
    Exclude=["Class", "Code_number"]
    X=file.drop(Exclude, axis=1)
    y=file["Class"]
    d=int(0.8 * a)
    training_inputs, test_inputs, training_labels,test_labels=X[:d], X[d:], y[:d], y[d:]
    file.replace('?', 1, inplace=True)
    del file["Code_number"]
    del file["Class"]
    file.Bare_Nuclei = file.Bare_Nuclei.astype(np.int64)
    display = plt.figure(figsize=(20,15))
    plt.matshow(file.corr(), display.number)
    plt.xticks(range(len(file.columns)), file.columns, fontsize=16, rotation=90)
    plt.yticks(range(len(file.columns)), file.columns, fontsize=16, rotation=0)
    cb=plt.colorbar()
    cb.ax.tick_params(labelsize=15)
    plt.show()
    return training_inputs, training_labels, test_inputs, test_labels

accuracy=0
def run_on_test_set(test_inputs, test_labels, weights):
    tp = 0
    test_inputs=np.array(test_inputs).astype(int)
    test_outputs = sigmoid(np.dot(test_inputs, weights))
    test_predictions=[]

    for i in test_outputs:
        if i > 0.5:
            test_predictions.append([1])
        else:
            test_predictions.append([0])
            
    test_predictions=np.array(test_predictions)
    for predicted_val, label in zip(test_predictions, test_labels):
        if predicted_val == label:
            tp += 1
    global accuracy
    accuracy = tp / len(test_predictions)
    return accuracy

def plot_loss_accuracy(accuracy_array, loss_array):
    accuracy_array=np.asarray(accuracy_array)
    loss_array=np.asarray(loss_array)
    plt.plot(accuracy_array,color="blue")
    plt.plot(loss_array,color="green")
    plt.legend(["Accuracy", "Loss"],loc="upper right")
    plt.xlabel("Epoch")
    plt.tight_layout()
    plt.show()

def main():
    csv_file = './breast-cancer-wisconsin.csv'
    iteration_count = 2500
    np.random.seed(1)
    weights = 2 * np.random.random((9, 1)) - 1
    accuracy_array = []
    loss_array = []
    training_inputs, training_labels, test_inputs, test_labels = read_and_divide_into_train_and_test(csv_file)

    for iteration in range(iteration_count):
        training_inputs=np.array(training_inputs).astype(int)
        outputs = np.dot(training_inputs, weights)
        outputs = sigmoid(outputs)
        training_labels=np.array(training_labels).astype(int)
        training_labels=training_labels.reshape((-1,1))
        loss = training_labels - outputs
        tunings = loss * sigmoid_derivative(outputs)
        weights+= np.dot(training_inputs.transpose(),tunings)
        run_on_test_set(test_inputs, test_labels, weights)
        x=np.mean(loss)
        loss_array.append(x)
        accuracy_array.append([accuracy])
    plot_loss_accuracy(accuracy_array, loss_array)

if __name__ == '__main__':
    main()
