import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools

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


def generator_timeseries(data, lookback, delay, min_index, max_index,
              shuffle=False, batch_size=128, step=6):
    if max_index is None:
        max_index = len(data) - delay - 1
    i = min_index + lookback
    while 1:
        if shuffle:
            rows = np.random.randint(
                min_index + lookback, max_index, size=batch_size)
        else:
            if i + batch_size >= max_index:
                i = min_index + lookback
            rows = np.arange(i, min(i + batch_size, max_index))
            i += len(rows)
        samples = np.zeros((len(rows),
                           lookback // step,
                           data.shape[-1]))
        targets = np.zeros((len(rows),))
        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay][1]
        yield samples, targets



en_lat_bottom = -5
en_lat_top = 5
en_lon_left = 360 - 170
en_lon_right = 360 - 120


def get_area_mean(tas, lat_bottom, lat_top, lon_left, lon_right):
    """The array of mean temperatures in a region at all time points."""
    return tas.loc[:, lat_bottom:lat_top, lon_left:lon_right].mean(
        dim=('lat', 'lon'))


def get_enso_mean(tas):
    """The array of mean temperatures in the El Nino 3.4 region.
    At all time point.
    """
    return get_area_mean(
        tas, en_lat_bottom, en_lat_top, en_lon_left, en_lon_right)


class FeatureExtractor(object):

    def __init__(self):
        pass

    def transform(self, X_ds):
        """Compute the El Nino mean at time t - (12 - X_ds.n_lookahead).
        Corresponding the month to be predicted.
        """
        # This is the range for which features should be provided. Strip
        # the burn-in from the beginning.
        valid_range = np.arange(X_ds.n_burn_in, len(X_ds['time']))
        enso = get_enso_mean(X_ds['tas'])
        # Roll the input series back so it corresponds to the month to be
        # predicted
        enso_rolled = np.roll(enso, 12 - X_ds.n_lookahead)
        # Strip burn in.
        enso_valid = enso_rolled[valid_range]
        # Reshape into a matrix of one column
        X_array = enso_valid.reshape((-1, 1))
        return X_array



class PreProc():

    def __init__(self):
        pass

    def scale(self, im, nR, nC):
        nR0 = im.shape[-3]     # source number of rows 
        nC0 = im.shape[-2]  # source number of columns 
        return [[ im[:,int(nR0 * r / nR), int(nC0 * c / nC),:]  
             for c in range(nC)] for r in range(nR)]
    
    def normalize(self, img):
        out = np.empty_like(img)
        for i in range(img.shape[-1]):
            out[:,:,:,i] = (img[:,:,:,i] - img[:,:,:,i].min())/(img[:,:,:,i].max()-img[:,:,:,i].min())
        
        return out

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues, ylabel='True label', xlabel='Predicted label', filename=None):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    np.set_printoptions(precision=2, suppress=True)
    
    if normalize:
        cm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + np.finfo(np.float).eps)
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap, vmin=0, vmax=1)
    plt.title(title, fontsize=20)
    # plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontsize=20)
    plt.yticks(tick_marks, classes, fontsize=20)
    
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",
                 fontsize=12)

    plt.tight_layout()
    plt.ylabel(ylabel, fontsize=20)
    plt.xlabel(xlabel, fontsize=20)