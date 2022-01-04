import numpy as np
import matplotlib.pyplot as plt
import functools

def gensmallm(x_list: list, y_list: list, m: int):
    """
    gensmallm generates a random sample of size m along side its labels.

    :param x_list: a list of numpy arrays, one array for each one of the labels
    :param y_list: a list of the corresponding labels, in the same order as x_list
    :param m: the size of the sample
    :return: a tuple (X, y) where X contains the examples and y contains the labels
    """
    assert len(x_list) == len(y_list), 'The length of x_list and y_list should be equal'

    x = np.vstack(x_list)
    y = np.concatenate([y_list[j] * np.ones(x_list[j].shape[0]) for j in range(len(y_list))])

    indices = np.arange(x.shape[0])
    np.random.shuffle(indices)

    rearranged_x = x[indices]
    rearranged_y = y[indices]

    return rearranged_x[:m], rearranged_y[:m]


def bayeslearn(x_train: np.array, y_train: np.array):
    """

    :param x_train: 2D numpy array of size (m, d) containing the the training set. The training samples should be binarized
    :param y_train: numpy array of size (m, 1) containing the labels of the training set
    :return: a triple (allpos, ppos, pneg) the estimated conditional probabilities to use in the Bayes predictor
    """
    y_pos_count = sum([1 for y in y_train if y == 1])
    y_neg_count = len(y_train) - y_pos_count
    allpos = y_pos_count / len(y_train)

    pixel_y_pos_count = np.zeros((len(x_train[0]), 1))
    pixel_y_neg_count = np.zeros((len(x_train[0]), 1))

    # Count all the y's foreach coordinate - ignoring nan entries
    for y, x in zip(y_train, x_train):
        for pixel_index, pixel in enumerate(x):
            if y == 1:
                if pixel == 1:
                    pixel_y_pos_count[pixel_index] += 1
            if y == -1:
                if pixel == 1:
                    pixel_y_neg_count[pixel_index] += 1

    # Calculate conditional probabilities 
    ppos = (pixel_y_pos_count / len(x_train)) / allpos
    pneg = (pixel_y_neg_count / len(x_train)) / (1-allpos)
    
    return allpos, ppos, pneg


def bayespredict(allpos: float, ppos: np.array, pneg: np.array, x_test: np.array):
    """

    :param allpos: scalar between 0 and 1, indicating the fraction of positive labels in the training sample
    :param ppos: numpy array of size (d, 1) containing the empirical plug-in estimate of the positive conditional probabilities
    :param pneg: numpy array of size (d, 1) containing the empirical plug-in estimate of the negative conditional probabilities
    :param x_test: numpy array of size (n, d) containing the test samples
    :return: numpy array of size (n, 1) containing the predicted labels of the test samples
    """

    y_bar = np.zeros((len(x_test), 1))

    p_x1 = np.log(ppos/pneg)
    p_x1[np.isnan(p_x1)] = 0
    p_x0 = np.log((1-ppos)/(1-pneg))
    p_x0[np.isnan(p_x0)] = 0

    for x_index, x in enumerate(x_test):
        accumulator = np.log(allpos / (1 - allpos))
        for pixel_index in range(len(x)):
            # P[x(i)=1|Y]
            if x[pixel_index] == 1 and p_x1[pixel_index, 0] != 0:
                accumulator += p_x1[pixel_index, 0]
            if x[pixel_index] == 0 and p_x0[pixel_index, 0] != 0:
                accumulator += p_x0[pixel_index, 0]
            
        y_bar[x_index] = 1 if accumulator >= 0 else -1

    return y_bar


def simple_test():
    # load sample data from question 2, digits 3 and 5 (this is just an example code, don't forget the other part of
    # the question)
    data = np.load('mnist_all.npz')

    train3 = data['train0']
    train5 = data['train1']

    test3 = data['test0']
    test5 = data['test1']

    m = 1000
    n = 50
    d = train3.shape[1]

    x_train, y_train = gensmallm([train3, train5], [-1, 1], m)

    x_test, y_test = gensmallm([test3, test5], [-1, 1], n)

    # threshold the images (binarization)
    threshold = 128
    x_train = np.where(x_train > threshold, 1, 0)
    x_test = np.where(x_test > threshold, 1, 0)

    # run naive bayes algorithm
    allpos, ppos, pneg = bayeslearn(x_train, y_train)

    assert isinstance(ppos, np.ndarray) \
           and isinstance(pneg, np.ndarray), "ppos and pneg should be numpy arrays"

    assert 0 <= allpos <= 1, "allpos should be a float between 0 and 1"

    y_predict = bayespredict(allpos, ppos, pneg, x_test)

    assert isinstance(y_predict, np.ndarray), "The output of the function bayespredict should be numpy arrays"
    assert y_predict.shape == (n, 1), f"The output of bayespredict should be of size ({n}, 1)"

    print(f"Prediction error = {np.mean(y_test.reshape((len(y_test), 1)) != y_predict)}")


def solve_binary_classification(sample_sizes, num1, num2):
    # load sample data
    data = np.load('mnist_all.npz')

    train1 = data['train' + str(num1)]
    train2 = data['train' + str(num2)]

    test1 = data['test' + str(num1)]
    test2 = data['test' + str(num2)]

    n = 50
    err = np.zeros((len(sample_sizes), 1))
    for i in range(len(sample_sizes)):
        x_train, y_train = gensmallm([train1, train2], [-1, 1], int(sample_sizes[i]))
        x_test, y_test = gensmallm([test1, test2], [-1, 1], n)

        # threshold the images (binarization)
        threshold = 128
        x_train = np.where(x_train > threshold, 1, 0)
        x_test = np.where(x_test > threshold, 1, 0)

        # run naive bayes algorithm
        allpos, ppos, pneg = bayeslearn(x_train, y_train)
        y_predict = bayespredict(allpos, ppos, pneg, x_test)
        err[i] = np.mean(y_test.reshape((len(y_test), 1)) != y_predict)

    return err


def q2_a():
    sample_sizes = np.linspace(1000, 10000, 10)
    err_diff01 = solve_binary_classification(sample_sizes, 0, 1)
    err_diff35 = solve_binary_classification(sample_sizes, 3, 5)
    plt.figure(1)
    plt.plot(sample_sizes, err_diff01)
    plt.plot(sample_sizes, err_diff35)
    plt.legend(["classify 0,1", "classify 3,5"])
    plt.xlabel("Sample Size")
    plt.ylabel("Error")
    plt.show()

if __name__ == '__main__':
    # before submitting, make sure that the function simple_test runs without errors
    # simple_test()
    q2_a()
    # here you may add any code that uses the above functions to solve question 2
