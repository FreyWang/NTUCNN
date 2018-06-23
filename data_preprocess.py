import numpy as np
import random
import os
from sklearn.decomposition import PCA
from scipy.interpolate import interp2d
import matplotlib.pyplot as plt
import time
import h5py

PROJ_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(PROJ_ROOT_DIR, 'data', 'data_32frame')
data_name = {
    # all data and its index
    0: ('y_train', 'y_test'),
    1: ('X_train', 'X_test'),  # original data
    2: ('normX_train', 'normX_test'),  # person diff, scatter xyz
    3: ('timediff_Xtrain', 'timediff_Xtest'),  # time diff, zero padding in the last frame
    4: ('jointdiff_Xtrain', 'jointdiff_Xtest'),  # joint diff adjacent
    5: ('diff_pair_train', 'diff_pair_test'),  # joint diff in every pair, then PCA to 224
    6: ('diff_pair_train_noPCA', 'diff_pair_test_noPCA'),  # joint diff in every pair, then sample to 224
    7: ('diff_train_noSample', 'diff_test_noSample'),   # joint diff in every pair, no sample, no PCA, (32*300)
    8: ('FFT2_train', 'FFT2_test'),  # 2DFFT data
    9: ('FFT2_train_middle', 'FFT2_test_middle'),  # 1/3~2/3 part of FFT image
}
data_shape = {
    # Given name, return shape
    'y_train': (39889,),
    'y_test': (16390,),
    'X_train': (39889, 32, 150),
    'X_test': (16390, 32, 150),
    'normX_train': (39889, 3, 32, 25),
    'normX_test': (16390, 3, 32, 25),
    'timediff_Xtrain': (39889, 3, 32, 25),
    'timediff_Xtest': (16390, 3, 32, 25),
    'jointdiff_Xtrain': (39889, 3, 32, 24),
    'jointdiff_Xtest': (16390, 3, 32, 24),
    'diff_pair_train': (39889, 3, 32, 224),
    'diff_pair_test': (16390, 3, 32, 224),
    'diff_pair_train_noPCA': (39889, 3, 32, 224),
    'diff_pair_test_noPCA': (16390, 3, 32, 224),
    'diff_train_noSample': (39889, 3, 32, 300),
    'diff_test_noSample': (16390, 3, 32, 300),
    'FFT2_train': (39889, 3, 224, 224),
    'FFT2_test': (16390, 3, 224, 224),
    'FFT2_train_middle': (39889, 3, 75, 75),
    'FFT2_test_middle': (16390, 3, 75, 75),

}
imageDim = 224

def input(index):
    """
    :param index: number of data_name
    :return:  read data
    """
    if isinstance(index, int):
        X_train = np.memmap(os.path.join(IMAGE_DIR, data_name[index][0] + '.dat'),
              dtype='float32', mode='r', shape=data_shape[data_name[index][0]])
        X_test = np.memmap(os.path.join(IMAGE_DIR, data_name[index][1] + '.dat'),
              dtype='float32', mode='r', shape=data_shape[data_name[index][1]])
    elif isinstance(index, list):
        X_train = []
        X_test = []
        for i in index:
            X_train.append(np.memmap(os.path.join(IMAGE_DIR, data_name[i][0] + '.dat'),
              dtype='float32', mode='r', shape=data_shape[data_name[i][0]]))
            X_test.append(np.memmap(os.path.join(IMAGE_DIR, data_name[i][1] + '.dat'),
              dtype='float32', mode='r', shape=data_shape[data_name[i][1]]))
    else:
        print('type error')
    return X_train, X_test


def output(index, train, test):
    """
    write data to disk
    :param index: get data name
    :param train:  train_data
    :param test:  test_data
    :return:
    """
    assert isinstance(index, int), 'parameter wrong'
    fp = np.memmap(os.path.join(IMAGE_DIR, data_name[index][0] + '.dat'),
                   dtype='float32', mode='w+', shape=data_shape[data_name[index][0]])
    fp[:] = train[:]
    del fp

    fp = np.memmap(os.path.join(IMAGE_DIR, data_name[index][1] + '.dat'),
                   dtype='float32', mode='w+', shape=data_shape[data_name[index][1]])
    fp[:] = test[:]
    del fp


def scatter_cord(X):
    """
    scatter 3 corrdinate x,y,z from an action data
    :param X: shape=[time,joints]=[32,75]
    :return: [time,joints,corrdinate] = [32.25,3]
    """
    new = []
    x_ind = [i for i in range(0, X.shape[1], 3)]
    y_ind = [i for i in range(1, X.shape[1], 3)]
    z_ind = [i for i in range(2, X.shape[1], 3)]
    new.append(np.array([X[:, i] for i in x_ind]).T)
    new.append(np.array([X[:, i] for i in y_ind]).T)
    new.append(np.array([X[:, i] for i in z_ind]).T)
    return np.array(new)


def scatter_all(X):
    """
    first person1 coordinate subtract person2, then scatter to three channel x,y,z
    input: [number,time,joints]=[*,32,150]
    :return: scatter to difference bettween two person wiht 3 coordinates x, y, z ,that is ,(person1 - person2)
             shape =[number, channel, time, joint]=[number, 3, 32, 25]
    """
    new_X = []
    number, time, joint = X.shape
    for i in range(number):
        data = X[i]
        person1 = data[:, 0:(joint // 2)]
        person2 = data[:, (joint // 2):joint]
        person_diff = person1 - person2
        person_diff = scatter_cord(person_diff)
        new_X.append(person_diff)
    return np.array(new_X)


def image():
    """
    use person1 - person2, then scatter coordinate to x, y ,z
    :return: 32*25*3 data
    """

    X_train, X_test = input(1)
    normX_train = scatter_all(X_train)
    print(normX_train.shape)
    normX_test = scatter_all(X_test)
    print(normX_test.shape)
    output(2, normX_train, normX_test)


def time_diff(X):
    """

    :param X:2D tensor,shape=[time, joint]
    :return: time_diff data,shape=[time-1, joint]
    """
    time_diff = []
    for i in range(X.shape[0]-1):
        time_diff.append(X[i+1] - X[i])
    return np.array(time_diff)


def time_diff_3D(X):
    """
    first calculate time different, then scatter to 3 coordinate x,y,z
    :param X: 3D tensor,shape = [num,time,joint]
    :return: scattered time diff data ,shape = [num,31,25,3]
    """
    time_diff_image = []
    for i in range(X.shape[0]):
        time_diff_image.append(time_diff(X[i]))
    return scatter_all(np.array(time_diff_image))


def time_diff_image():
    # normX_train = np.memmap(os.path.join(IMAGE_DIR, 'normX_train.dat'),
    #                            dtype='float32', mode='r', shape=(39889, 3, 32, 25))
    # normX_test = np.memmmap(os.path.join(IMAGE_DIR, 'normX_test.dat'),
    #                         dtype='float32', mode='r', shape=(16390, 3, 32, 25))
    X_train, X_test = input(1)
    timediff_Xtrain = time_diff_3D(X_train)
    print(timediff_Xtrain.shape)
    timediff_Xtest = time_diff_3D(X_test)
    print(timediff_Xtest.shape)
    output(3, timediff_Xtrain, timediff_Xtest)

def joint_diff(X):
    """
    X[:, joint+1]-X[:, joint]
    :param X:2D tensor,shape=[time,joint]
    :return:joint diff data, shape =[time,joint-1]
    """
    joint_diff = []
    for i in range(X.shape[1]-1):
        joint_diff.append(X[:, i+1] - X[:, i])
    return np.array(joint_diff).T


def joint_diff_4D(X):
    """

    :param X:4D tensor,shape=[num,channel,time,joint]=[num,3,32,25]
    :return:[num, 3, 32, 24]
    """
    joint_diff_image = []
    for i in range(X.shape[0]):
        joint_diff_channel = []
        for j in range(X.shape[1]):
            joint_diff_channel.append(joint_diff(X[i][j]))
        joint_diff_image.append(np.array(joint_diff_channel))
    return np.array(joint_diff_image)


def joint_diff_image():
    """
    :return: diff data between adjacent joints
    """
    normX_train, normX_test = input(2)
    jointdiff_Xtrain = joint_diff_4D(normX_train)
    print(jointdiff_Xtrain.shape)
    jointdiff_Xtest = joint_diff_4D(normX_test)
    print(jointdiff_Xtest.shape)
    output(4, jointdiff_Xtrain, jointdiff_Xtest)


def diff_pair(X):
    """

    :param X:3D tensor , one channel, shape = (num, time , joint) =(num, 32, 25)
    :return: diff between every two joint ,shape = (num, 32, 300)
    """
    num, time, joint = X.shape
    diffX = []
    for i in range(num):
        temp =[]
        for j in range(joint):
            for k in range(j+1, joint):
                temp.append(X[i][:, k] - X[i][:, j])
        temp = np.array(temp).T
        diffX.append(temp)
    diffX = np.array(diffX)
    return diffX


def PCA_data(X_data, k):
    #reduce dimension of feature
    #row: feature dim
    #col: sample
    data = X_data.reshape((-1,X_data.shape[2]))
    pca = PCA(n_components=k)
    newData = pca.fit_transform(data)
    newData = newData.reshape(X_data.shape[0], X_data.shape[1], k)
    return newData


def pca_diff(train, test):
    """
    first concatenate , then do pca
    :param train:3D tensor for one channel,shape = (num, time,joint)
    :param test:
    :return:
    """

    data = np.concatenate((train, test), axis=0)
    diff_data = diff_pair(data)
    #pca_data = PCA_data(diff_data, imageDim)  # low dim = 224
    pca_train = diff_data[0:train.shape[0]]  # uncomment last line and diff_data->pca_data to do PCA
    pca_test = diff_data[train.shape[0]:data.shape[0]]
    return pca_train, pca_test


def diff_pair_image():
    """
    first do difference in evert pairs, then pca to 224 dim
    :return: shape = (num, 3, 32, 224)
    """
    normX_train, normX_test = input(2)
    coor_xtrain = normX_train[:, 0, :, :]
    coor_xtest = normX_test[:, 0, :, :]
    coor_ytrain = normX_train[:, 1, :, :]
    coor_ytest = normX_test[:, 1, :, :]
    coor_ztrain = normX_train[:, 2, :, :]
    coor_ztest = normX_test[:, 2, :, :]
    pca_xtrain, pca_xtest = pca_diff(coor_xtrain, coor_xtest)  # shape = (num, 32, 224)
    pca_ytrain, pca_ytest = pca_diff(coor_ytrain, coor_ytest)
    pca_ztrain, pca_ztest = pca_diff(coor_ztrain, coor_ztest)
    diff_pair_train = np.zeros((normX_train.shape[0], normX_train.shape[1],
                                normX_train.shape[2], pca_xtrain.shape[2]))
    diff_pair_train[:, 0, :, :] = pca_xtrain
    diff_pair_train[:, 1, :, :] = pca_ytrain
    diff_pair_train[:, 2, :, :] = pca_ztrain
    diff_pair_test = np.zeros((normX_test.shape[0], normX_test.shape[1],
                                normX_test.shape[2], pca_xtest.shape[2]))
    diff_pair_test[:, 0, :, :] = pca_xtest
    diff_pair_test[:, 1, :, :] = pca_ytest
    diff_pair_test[:, 2, :, :] = pca_ztest
    print(diff_pair_train.shape)
    print(diff_pair_test.shape)
    # sample_num = 224#choose sample to caculate projection matrix
    # sample_ind = random.sample(range(0,diff_pair_train.shape[3]), sample_num)
    # diff_pair_train = diff_pair_train[:,:,:, sample_ind]
    # diff_pair_test = diff_pair_test[:,:,:,sample_ind]
    # output(6, diff_pair_train, diff_pair_test)
    output(7, diff_pair_train, diff_pair_test)

def interp2(image,newshape):
    w, h = image.shape
    x = np.linspace(0, 1, w)
    y = np.linspace(0, 1, h)
    xx, yy = np.meshgrid(x, y)
    f = interp2d(xx, yy, image, kind='linear')
    xnew = np.linspace(0, 1, newshape[0])
    ynew = np.linspace(0, 1, newshape[1])
    newImage = f(xnew, ynew)
    return newImage

def FFT2(X):
    """

    :param X: 2D tensor
    :return: crop 224*224 from the middle of FFT image
    """
    w, h = X.shape
    assert (w >= imageDim) & (h >= imageDim), '({},{}) image is too small'.format(w, h)
    FX = np.fft.fft2(X)
    FX = np.abs(np.fft.fftshift(FX))
    return FX[w//2-imageDim//2:w//2+imageDim//2,
           h//2-imageDim//2:h//2+imageDim//2]

def interp_FFT(X):
    """
    first interploation ,then FFT
    :param X:4D diff pair data ,shape = (*, 3,32, 300)
    :return: FFT data, sample = (*, 3, 224, 224)
    """
    interpDim = 300
    new = np.zeros((X.shape[0], X.shape[1], imageDim, imageDim), dtype='float32')
    count = 0
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            start = time.time()
            new[i][j] = FFT2(interp2(X[i][j], (interpDim, interpDim)))
            end = time.time()
            count += 1
            t = os.system('clear')  # store 0
            print('Processed {}/({}*{}, time used = {}s)'.
                  format(count, X.shape[0], X.shape[1], end - start))
    return new

def FFT_image():
    X_train, X_test = input(7)  # (*, 3, 32, 300)
    FFT2_train = interp_FFT(X_train)
    FFT2_test = interp_FFT(X_test)
    output(8, FFT2_train, FFT2_test)

def readFromMatlab(filename, variableNameInMatlab):
    """
    read big data from matlab-v7.3:
    matlab:
    X = rand([100,100])
    save('X.mat', 'X', '-v7.3')
    python:
    dict_data = h5py.File('X.mat')
    data = np.array(dict_data['X'])

    :param filename: 'X.mat' ,     type = string
    :param variableNameInMatlab: 'X',    type = string
    :return:
    """
    dict_data = h5py.File(filename, 'r')
    data = np.array(dict_data[variableNameInMatlab])
    dict_data.close()
    return data

def diff_matrix(data):
    """
    newMatrix[T, i,j] = data[T,i] - data[T, j]
    :param data: shape = (T, joint) = (32, 25)
    :return:  shape = (T, joint, joint)
    """
    newMatrix = []
    T, joint = data.shape
    for i in range(T):
        diff = np.zeros((joint, joint))
        for j in range(joint):
            for k in range(joint):
                diff[j][k] = data[i][j] - data[i][k]
        newMatrix.append(diff)
    newMatrix = np.array(newMatrix)
    assert newMatrix.shape == (T, joint, joint), 'wrong shape!'
    return newMatrix




if __name__ == '__main__':
    pass





