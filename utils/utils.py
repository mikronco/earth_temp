import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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


class BaselineRain():
    def __init__(self):
        super().__init__()

    def predict(self, x):
        return x
    

def normalize_multivariate_data(data, scaling_values=None):
    normed_data = np.zeros(data.shape, dtype=data.dtype)
    scale_cols = ["mean", "std"]
    if scaling_values is None:
        scaling_values = pd.DataFrame(np.zeros((data.shape[-1], len(scale_cols)), dtype=np.float32),
                                      columns=scale_cols)
    for i in range(data.shape[-1]):
        scaling_values.loc[i, ["mean", "std"]] = [data[:, :, :, i].mean(), data[:, :, :, i].std()]
        normed_data[:, :, :, i] = (data[:, :, :, i] - scaling_values.loc[i, "mean"]) / scaling_values.loc[i, "std"]
    return normed_data, scaling_values


def plot_conv_layers_out(example, X_test, conv_graph):
    conv_outs = conv_graph([X_test[example:example+1]])
    fig = plt.figure(figsize=(10, 10), facecolor='white')
    num_conv_1 = conv_outs[0].shape[-1]
    ax_0 = fig.add_axes([0, 0.65, 0.35, 0.35], xticks=[], yticks=[])
    refl = ax_0.pcolormesh(X_test[example, :, :, 0], cmap="gist_ncar")
    plt.colorbar(refl, ax=ax_0)
    ax_0.quiver(X_test[example, :, :, 1], 
                X_test[example, :,:,2])
    ax_0.set_title("Radar Reflectivity and 10 m Winds", fontsize=12)
    ax_0.set_xlabel(f"Probability of Strong Rotation: {conv_outs[-1][0, 0]:0.3f}", fontsize=12)
    refl_contours = [40, 60]
    refl_array = X_test[example, :, :, 0]
    for i in range(num_conv_1):
        ax_box = [0.4, i/ num_conv_1, 1/num_conv_1, 1/num_conv_1]
        ax = fig.add_axes(ax_box, xticks=[], yticks=[])
        ax.pcolormesh(np.arange(0, 34, 2), np.arange(0, 34, 2), conv_outs[0][0, :, :, i], vmin=0, vmax=3, cmap="Reds")
        ax.contour(np.arange(0, 32), np.arange(0, 32), refl_array, refl_contours, 
                   vmin=0, vmax=80, cmap="Blues")
    ax.set_title("Conv. Layer 1")
    num_conv_2 = conv_outs[1].shape[-1]    
    for i in range(num_conv_2):
        ax_box = [0.6, i/ num_conv_2, 1/num_conv_2, 1/num_conv_2]
        ax = fig.add_axes(ax_box, xticks=[], yticks=[])
        ax.pcolormesh(np.arange(0, 36, 4), np.arange(0, 36,4), conv_outs[1][0, :, :, i], vmin=0, vmax=3, cmap="Reds")
        ax.contour(np.arange(0, 32), np.arange(0, 32), refl_array, refl_contours, 
                   vmin=0, vmax=80, cmap="Blues")
    ax.set_title("Conv. Layer 2")