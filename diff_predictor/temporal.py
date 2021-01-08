import sys
import csv
from multipledispatch import dispatch
from os import listdir, getcwd, chdir
from os.path import isfile, join
import pandas as pd
import numpy as np


modulename = 'core'
if modulename not in sys.modules:
    import core

try:
    import tensorflow as tf
except ImportError:
    raise ImportError('Tensorflow package is not installed. please install\
                      either the cpu or gpu version before running')

# Test for gpu availability:
try:
    with tf.device('GPU'):
        a = tf.constant([1.0, 2.0, 3.0, 4.0], shape=[2,2])
        b = tf.constant([1.0, 2.0, 3.0, 4.0], shape=[2,2])
        c = tf.matmul(a, b)
        print('Detected you are succesfully\
              running tensorflow on a GPU:')
        PHYSICAL_DEVICES = tf.config.list_physical_devices('GPU')
        print(physical_devices)
except Exception:
    print('GPU calculation not detected. Running Tensorflow on CPU:')
    PHYSICAL_DEVICES = tf.config.list_physical_devices('GPU')
    print(physical_devices)


def zero_df(df, col, frames=(0, 651)):
    '''
    Zeros a single dataframe column so that the first value will be
    located at the start of the track.
    '''
    try:
        shift_val = df.iloc[frames[0]:frames[1]][col].reset_index().dropna().index[0]
    except:
        shift_val = frames[0] - frames[1] - 1
    return df.iloc[frames[0]:frames[1]][col].reset_index().shift(-shift_val, fill_value=np.nan)[col]


def get_zeroed_tracks(df, col, frames=650):
    '''
    Creates an array of all the tracks for a single column in a file
    in which the value is zeroed to frame = 0
    '''
    lower = 0
    upper = frames+1
    value = []
    while (upper <= len(df)):
        value.append(list(zero_df(df, col=col, frames=[lower, upper])))
        lower = upper
        upper = lower + frames + 1
    return value


def get_xy_data(df, target, feat=None, frames=650):
    '''
    Creates x and y array datasets in the correct shape for LSTM
    based off input track_df data.

    Parameters
    ----------
    df : pd.dataframe
        track dataframe to turn into x and y array datasets
    target : string
        the target column that will be used for prediction
    feat : [string] : None
        optional features columns to add. Will output in its own
        numpy array
    frames : int : 650
        number of frames trajectory is tracked for
    Returns
    -------
    tuple(result) : tuple([np.ndarray],..)
        either a 2-length or 3-length of data arrays in order of
        ([datax], [datay], [datafeat]) where:
        datax : np.ndarray
            x-data for prediction
        datay : np.ndarray
            y-data (target) for prediction
        datafeat : np.ndarray
            feature-data for prediction

    Note: This function takes a large amount of memory. Make sure
    you have at least 32GB memory when running.
    '''
    n_tracks = int((len(df))/(frames + 1))
    frame = get_zeroed_tracks(df, 'Frame', frames=frames)
    X = get_zeroed_tracks(df, 'X', frames=frames)
    Y = get_zeroed_tracks(df, 'Y', frames=frames)
    MSDs = get_zeroed_tracks(df, 'MSDs', frames=frames)
    trgt = df[target]
    datax = []
    datay = []
    if feat is not None:
        datafeat = []
        trackfeat = []
    for j in range(n_tracks):
        trackx = []
        tracky = []
        for i in range(frames + 1):
            trackx.append([int(frame[j][i]), X[j][i], Y[j][i], MSDs[j][i]])
        datax.append(trackx)
        del(trackx)
        tracky.append(trgt[(frames + 1) * (j + 1) - 1])
        datay.append(tracky)
        del(tracky)
        if feat is not None:
            trackfeat.append(list(df.loc[(frames + 1) * (j + 1) - 1, feat]))
            datafeat.append(trackfeat)
            del(trackfeat)
            trackfeat = []
    del(df, frame, X, Y, MSDs, trgt)
    datax = np.array(datax)
    datax = datax.reshape(n_tracks, frames + 1, 4)
    datay = np.array(datay)
    datay = datay.reshape(n_tracks, 1)
    result = [datax, datay]
    if feat is not None:
        datafeat = np.array(datafeat)
        datafeat = datafeat.reshape(n_tracks, len(feat))
        result += [datafeat]
        del(datafeat)
    del(datax, datay)
    return tuple(result)


@dispatch(np.ndarray, np.ndarray, datafeat=np.ndarray, track=int)
def get_track(datax, datay, datafeat = np.ndarray(0), track = 0):
    '''
    Get data from a specific track in a numpy array.

    Parameters
    ----------
    datax : numpy.ndarray
        Dataframe to extract track from
    datay : numpy.ndarray
        track number
    frames : int : 650
        number a frames a track runs for
    Returns
    -------
    A pandas dataframe of the specified track
    '''
    result = [datax[track], datay[track]]
    if datafeat.size is not 0:
        result += [datafeat[track]]
    return tuple(result)


@dispatch(pd.DataFrame, track=int, frames=int)
def get_track(df, track, frames=650):
    '''
    Get data from a specific track in pandas dataframe.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe to extract track from
    track : int
        track number
    frames : int : 650
        number a frames a track runs for
    Returns
    -------
    A pandas dataframe of the specified track
    '''
    return df.loc[(frames + 1) * (track):(frames + 1)*(track + 1) - 1]


def get_feat(df : pd.DataFrame, track : int, frames, feat):
    '''
    Get feature data based on the track# and the feature.

    Parameters
    ----------
    df : pd.DataFrame
        dataframe to extract features from
    track : int
        track# to get data from.
    feat : string
        feature to look at
    Returns
    -------
    A 650 row pandas dataframe of the feature selected
    '''
    return df.loc[(frames+1)*(track+1)-1, feat]


def remove_track(datax, datay, datafeat=np.ndarray(0), track = 0):
    '''
    Remove a specified track # from xy array data.

    Parameters
    ----------
    datax : np.array()
        x data to remove track from
    datay : np.array()
        y data to remove track from
    datafeat : np.ndarray : np.ndarray(0)
        feature data to remove track from. This is optional.
    track : int : 0
        track to remove. Default is the first track in the input data

    Returns
    -------
    A tuple containing the transformed data (datax, datay, datafeat)
    '''
    assert (isinstance(datax, np.ndarray) and isinstance(datax, np.ndarray)),\
           "Input data needs to be numpy.ndarray. Call get_xy_data() first"
    result = [np.delete(datax, track, axis=0), np.delete(datay, track, axis=0)]
    if datafeat.size is not 0:
        result += [np.delete(datafeat, track, axis=0)]
    return tuple(result)


def balance_data(datax, datay, datafeat=np.ndarray(0), **kwargs):
    if 'random_state' not in kwargs:
        random_state = 1
    else:
        random_state = kwargs['random_state']
    if 'seed' not in kwargs:
        seed = 1234
    else:
        seed = kwargs['seed']
    np.random.seed(seed)
    assert (isinstance(datax, np.ndarray) and isinstance(datax, np.ndarray)),\
           "Input data needs to be numpy.ndarray. Call get_xy_data() first"
    target_locs = []
    bal_data_locs = []
    targets = np.unique(datay)
    if targets.dtype is np.dtype('float64'):
        targets = targets.astype('int')
    for name in targets:
        target_locs.append(np.where(np.any(datay==name, axis=1))[0])
    print(f"Ratio before data balance " +
          f"({':'.join([str(i) for i in targets])}) = " +
          f"{':'.join([str(len(i)) for i in target_locs])}")
    min_length = min([len(i) for i in target_locs])
    for i in range(len(targets)):
        bal_data_locs = np.append(bal_data_locs,
                                  np.random.choice(target_locs[i],
                                                   size=min_length,
                                                   replace=False)).astype('int')
    bal_datax = datax[bal_data_locs]
    bal_datay = datay[bal_data_locs]
    result = [bal_datax, bal_datay]
    if datafeat.size is not 0:
        result += [datafeat[bal_data_locs]]
    target_locs = []
    for name in targets:
        target_locs.append(np.where(np.any(bal_datay==name, axis=1))[0])
    print(f"Ratio after balance " +
          f"({':'.join([str(i) for i in targets])}) = " +
          f"{':'.join([str(len(i)) for i in target_locs])}")
    return result


def split_data(datax, datay, datafeat=np.ndarray(0), split=0.8, test_val_split=1.0, seed=1234):
    '''
    Will split numpy array based data into training, testing, and (optionally) eval sets.

    Parameters
    ----------
    datax : numpy.ndarray
        Numpy array of x data to be split
    datay : numpy.ndarray
        Numpy array of y data to be split
    datafeat : numpy.ndarray : np.ndarray(0)
        Optional numpy array of feature data to be split
    split : int : 0.8
        Split ratio of size training set to size testing set.
        Note if an evaluation split is wanted, it will take the
        data from the tetsting set.
        ex. 0.8 == 80% of data splitting to training data.
    test_val_split : int :1.0
        Optional split if evaluation set is wanted. Set to 1.0 if
        only a testing set is desired.
        ex. 0.9 == 90% of the testing/evaluation split splitting to testing
    seed : int 1234
        Optional seed used for sudo random splitting
    Returns
    -------
    A list of up to three tuples containing the split data:
        [(X_train, ytrain, feat_train),
         (X_test, y_test, feat_test),
         (X_eval, y_eval, feat_eval)]
    '''
    np.random.seed(seed)
    result = []
    train_index = np.random.choice(np.arange(0, len(datax)), int(len(datax)*split), replace=False)
    test_val_index = np.setdiff1d(np.arange(0, len(datax)), train_index)
    datax = np.nan_to_num(datax, copy=True, nan=-1.0, posinf=-1.0, neginf=-1.0)
    datay = np.nan_to_num(datay, copy=True, nan=-1.0, posinf=-1.0, neginf=-1.0)
    X_train = datax[train_index]
    y_train = datay[train_index]
    if datafeat.size is not 0:
        datafeat = np.nan_to_num(datafeat, copy=True, nan=-1.0, posinf=-1.0, neginf=-1.0)
        feat_train = datafeat[train_index]
        train_set = (X_train, y_train, feat_train)
    else:
        train_set = (X_train, y_train)
    result += [tuple(train_set)]
    if test_val_split == 1.0:
        X_test = datax[test_val_index]
        y_test = datay[test_val_index]
        if datafeat.size is not 0:
            datafeat = np.nan_to_num(datafeat, copy=True, nan=-1.0, posinf=-1.0, neginf=-1.0)
            feat_test = datafeat[test_val_index]
            test_set = (X_test, y_test, feat_test)
        else:
            test_set = (X_test, y_test)
        result += [tuple(test_set)]
    else:
        datax = datax[test_val_index]
        datay = datay[test_val_index]
        test_index = np.random.choice(np.arange(0, len(datax)),
                                      int(len(datax)*test_val_split),
                                      replace=False)
        X_test = datax[test_index]
        y_test = datay[test_index]
        eval_index = np.setdiff1d(np.arange(0, len(datax)), test_index)
        X_eval = datax[eval_index]
        y_eval = datay[eval_index]
        if datafeat.size is not 0:
            datafeat = datafeat[test_val_index]
            feat_test = datafeat[test_index]
            feat_eval = datafeat[eval_index]
            test_set = (X_test, y_test, feat_test)
            eval_set = (X_test, y_test, feat_test)
        else:
            test_set = (X_test, y_test)
            eval_set = (X_test, y_test)
        result += [tuple(test_set)] + [tuple(eval_set)]
    return result


def numpy_one_hot_encode(mat, encoder=None):
    '''
    One hot encoding for numpy data array

    Parameters
    ----------
    mat : numpy.ndarray
        numpy array to encode
    encoder : numpy.ndarray : None
        Key to what to encode classes as
    Returns
    -------
    mat : numpy.ndarray
        encoded numpy array
    encoder : ndarray
        encoder for numpy array to be used to decode
    '''
    if encoder is None:
        encoder = np.unique(mat)
    mat = np.array(encoder == mat).astype(int)
    return mat, encoder


def numpy_decode(mat, encoder):
    '''
    Decoder for numpy one hot encoding array. This method
    requires an encoder taken from numpy_one_hot_encode()

    Parameters
    ----------
    mat : numpy.ndarray
        numpy array to decode
    encoder : numpy.ndarray
        encoder take when one hot encoding to decode
    '''
    return np.array([i[i!=0] for i in mat * encoder])
