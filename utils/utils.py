import numpy as np


def r_squared(y, y_hat):

    n_data = len(y)
    y_mean = np.mean(y)

    residual_sum_squares = 0
    total_sum_squares = 0
    for i in range(n_data):
        residual_sum_squares += (y[i] - y_hat[i])**2
        total_sum_squares += (y[i] - y_mean)**2

    # R Squares
    r_squared = 1 - residual_sum_squares / total_sum_squares

    return r_squared