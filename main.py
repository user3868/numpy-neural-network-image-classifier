import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def initialize_parameters(input_size, hidden_size, output_size):
    W1 = np.random.randn(input_size, hidden_size) * 0.01
    b1 = np.zeros((1, hidden_size))
    W2 = np.random.randn(hidden_size, output_size) * 0.01
    b2 = np.zeros((1, output_size))
    return W1, b1, W2, b2


def forward_propagation(X, W1, b1, W2, b2):
    Z1 = np.dot(X, W1) + b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = sigmoid(Z2)
    return A2, A1


def compute_cost(Y, Y_hat):
    m = Y.shape[0]
    return -1 / m * np.sum(Y * np.log(Y_hat) + (1 - Y) * np.log(1 - Y_hat))


def compute_cost_lambda(Y, Y_hat, W1, W2, lambda_=0.0):
    m = Y.shape[0]
    cost = -1 / m * np.sum(Y * np.log(Y_hat) + (1 - Y) * np.log(1 - Y_hat))
    regularization = (lambda_ / (2 * m)) * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
    total_cost = cost + regularization
    return total_cost


def backward_propagation(X, Y, A1, A2, W2):
    m = X.shape[0]
    dZ2 = A2 - Y
    dW2 = np.dot(A1.T, dZ2) / m
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m
    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * A1 * (1 - A1)
    dW1 = np.dot(X.T, dZ1) / m
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m
    return dW1, db1, dW2, db2


def update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate):
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    return W1, b1, W2, b2


def predict(X, W1, b1, W2, b2):
    A2, _ = forward_propagation(X, W1, b1, W2, b2)
    predictions = A2 > 0.5
    return predictions


def load_image(file_path):
    with Image.open(file_path) as img:
        img = img.resize((28, 28))
        img_array = np.array(img)
        return img_array


def load_images_from_folder(folder_path):
    images = []
    labels = []
    for file_name in os.listdir(folder_path):
        if file_name.startswith("1_"):
            label = 1
        else:
            label = 0
        file_path = os.path.join(folder_path, file_name)
        img_array = load_image(file_path)
        images.append(img_array.flatten())
        labels.append(label)
    return np.array(images), np.array(labels)


def split_dataset(X, Y, train_ratio=0.7, m=None):
    if m is not None:
        X, Y = X[:m], Y[:m]
    m = X.shape[0]
    train_size = (int)(m * train_ratio)
    indices = np.random.permutation(m)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    return X[train_indices], Y[train_indices], X[val_indices], Y[val_indices]


def _test_single_image(file_path, W1, b1, W2, b2):
    img_array = load_image(file_path)
    img_array = img_array / 255.0
    img_array = img_array.reshape(1, -1)
    prediction = predict(img_array, W1, b1, W2, b2)
    return '1' if prediction[0, 0] else '0'


def plot_cost_vs_m(m_value, cost_train, cost_val):
    plt.plot(m_value, cost_train, label='Training Cost')
    plt.plot(m_value, cost_val, label='Validation Cost')
    plt.xlabel('m')
    plt.ylabel('Cost')
    plt.title('Cost vs. Number of Training Examples')
    plt.legend()
    plt.show()


def train_and_evaluate(X_train, Y_train, X_val, Y_val, num_iterations, learning_rate):
    input_size = X_train.shape[1]
    hidden_size = 100
    output_size = 1
    W1, b1, W2, b2 = initialize_parameters(input_size, hidden_size, output_size)

    for i in range(num_iterations):
        A2_train, A1_train = forward_propagation(X_train, W1, b1, W2, b2)
        cost_train = compute_cost(Y_train, A2_train)
        A2_val, A1_val = forward_propagation(X_val, W1, b1, W2, b2)
        cost_val = compute_cost(Y_val, A2_val)

        dW1, db1, dW2, db2 = backward_propagation(X_train, Y_train, A1_train, A2_train, W2)
        W1, b1, W2, b2 = update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate)
    return cost_train, cost_val


def evaluate_cost_vs_m(X, Y, train_ratio, num_iterations, learning_rate, m_values):
    costs_train = []
    costs_val = []
    for m in m_values:
        X_train, Y_train, X_val, Y_val = split_dataset(X, Y, train_ratio, m)
        cost_train, cost_val = train_and_evaluate(X_train, Y_train, X_val, Y_val, num_iterations, learning_rate)
        costs_train.append(cost_train)
        costs_val.append(cost_val)
    return costs_train, costs_val


def compute_accuracy(X, Y, W1, b1, W2, b2):
    predictions, _ = forward_propagation(X, W1, b1, W2, b2)
    predicted_labels = predictions > 0.5
    accuracy = np.mean(predicted_labels == Y)
    return accuracy


def evaluate_accuracy_vs_learning_rate(X, Y, train_ratio, num_iterations, learning_rates):
    accuracies_train = []
    accuracies_val = []
    X_train, Y_train, X_val, Y_val = split_dataset(X, Y, train_ratio)
    for learning_rate in learning_rates:
        input_size = X_train.shape[1]
        hidden_size = 100
        output_size = 1
        W1, b1, W2, b2 = initialize_parameters(input_size, hidden_size, output_size)
        for _ in range(num_iterations):
            A2_train, A1_train = forward_propagation(X_train, W1, b1, W2, b2)
            dW1, db1, dW2, db2 = backward_propagation(X_train, Y_train, A1_train, A2_train, W2)
            W1, b1, W2, b2 = update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate)
        accuracies_train.append(compute_accuracy(X_train, Y_train, W1, b1, W2, b2))
        accuracies_val.append(compute_accuracy(X_val, Y_val, W1, b1, W2, b2))
    return accuracies_train, accuracies_val


def plot_accuracy_vs_learning_rate(learning_rates, accuracies_train, accuracies_val):
    plt.plot(learning_rates, accuracies_train, label='Training Accuracy')
    plt.plot(learning_rates, accuracies_val, label='Validation Accuracy')
    plt.xlabel('Learning Rate')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. Learning Rate')
    plt.xscale('log')
    plt.legend()
    plt.show()


def plot_cost_vs_num_iterations(costs_train, costs_val, num_iterations):
    plt.plot(range(num_iterations), costs_train, label='Training Cost')
    plt.plot(range(num_iterations), costs_val, label='Validation Cost')
    plt.xlabel('Number of Iterations')
    plt.ylabel('Cost')
    plt.title('Cost vs. Number of Iterations')
    plt.legend()
    plt.show()


def _test_one_image(X, Y, train_ratio, num_iterations, learning_rate, test_image_path):
    costs_train = []
    costs_val = []
    X_train, Y_train, X_val, Y_val = split_dataset(X, Y, train_ratio)
    input_size = X_train.shape[1]
    hidden_size = 100
    output_size = 1
    W1, b1, W2, b2 = initialize_parameters(input_size, hidden_size, output_size)
    for i in range(num_iterations):
        A2_train, A1_train = forward_propagation(X_train, W1, b1, W2, b2)
        cost_train = compute_cost(Y_train, A2_train)
        A2_val, A1_val = forward_propagation(X_val, W1, b1, W2, b2)
        cost_val = compute_cost(Y_val, A2_val)
        dW1, db1, dW2, db2 = backward_propagation(X_train, Y_train, A1_train, A2_train, W2)
        W1, b1, W2, b2 = update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate)
        costs_train.append(cost_train)
        costs_val.append(cost_val)
        if i % 10 == 0:
            print(f"Iteration {i}: Training Cost {cost_train},Validation Cost: {cost_val}")

    plot_cost_vs_num_iterations(costs_train, costs_val, num_iterations)
    prediction = _test_single_image(test_image_path, W1, b1, W2, b2)
    print('Predicted Class: ', prediction)


def _test_all_image(X, Y, train_ratio, num_iterations, learning_rate, folder_path):
    X_train, Y_train, X_val, Y_val = split_dataset(X, Y, train_ratio)
    input_size = X_train.shape[1]
    hidden_size = 100
    output_size = 1
    W1, b1, W2, b2 = initialize_parameters(input_size, hidden_size, output_size)
    for i in range(num_iterations):
        A2_train, A1_train = forward_propagation(X_train, W1, b1, W2, b2)
        A2_val, A1_val = forward_propagation(X_val, W1, b1, W2, b2)
        dW1, db1, dW2, db2 = backward_propagation(X_train, Y_train, A1_train, A2_train, W2)
        W1, b1, W2, b2 = update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate)

    image_pred_0 = 0
    image_pred_1 = 0

    for filename in os.listdir(folder_path):
        prediction = _test_single_image(os.path.join(folder_path, filename), W1, b1, W2, b2)
        if prediction == '1':
            image_pred_1 += 1
            # print('location of Class 1: ', os.path.join(folder_path,filename))

        else:
            image_pred_0 += 1
            # print('location of Class 0: ', os.path.join(folder_path,filename))
    # print(f"0: {image_pred_0},1: {image_pred_1}")
    return image_pred_0, image_pred_1


def main():
    folder_path = 'labelImage'
    X, Y = load_images_from_folder(folder_path)
    X = X / 255.0
    Y = Y.reshape(-1, 1)
    num_iterations = 1800
    learning_rate = 0.1
    train_ratio = 0.7

    # graph: accuracy_vs_learning_rate
    learning_rates = [0.001, 0.01, 0.1, 1]
    accuracies_train, accuracies_val = evaluate_accuracy_vs_learning_rate(X, Y, train_ratio, num_iterations,
                                                                          learning_rates)
    plot_accuracy_vs_learning_rate(learning_rates, accuracies_train, accuracies_val)

    # graph: cost_vs_m
    m_values = list(range(2, 180, 10))
    costs_train, costs_val = evaluate_cost_vs_m(X, Y, train_ratio, num_iterations, learning_rate, m_values)
    plot_cost_vs_m(m_values, costs_train, costs_val)

    # test one image
    test_image_path = 'label_1/1_(100,800)_Screenshot_20231215-153648.png'
    _test_one_image(X, Y, train_ratio, num_iterations, learning_rate, test_image_path)

    test_folder_path = 'label_1'
    image_pred_0, image_pred_1 = _test_all_image(X, Y, train_ratio, num_iterations, learning_rate, test_folder_path)
    TP = image_pred_1
    FP = image_pred_0
    test_folder_path = 'label_0'
    image_pred_0, image_pred_1 = _test_all_image(X, Y, train_ratio, num_iterations, learning_rate, test_folder_path)
    TN = image_pred_0
    FN = image_pred_1
    Precision = TP / (TP + FP)
    Recall = TP / (TP + FN)
    print(f'\nPrecision:{Precision},Recall: {Recall}\n')


if __name__ == "__main__":
    main()
