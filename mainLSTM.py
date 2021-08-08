import xlrd
from STS.modelcnn_cross_LSTM import *
from data import *
import cv2
import scipy.io as sio
import scipy.ndimage as ndimage
import numpy as np
import random

def print_label(directory_name, len, img):

    for i in range(len):
        cv2.imwrite(directory_name+str(i)+'.png', img[i,:,:])

def read_directory(directory_name, size):
    array_of_img = np.empty(size, dtype='float32') # this if for store all of the image data

    # this loop is for read each image in this foder,directory_name is the foder name with images.
    i=1
    for filename in os.listdir(directory_name):
        img = sio.loadmat(directory_name + "/" + filename)
        temp = img['com']
        temp1 = ndimage.interpolation.zoom(temp,size[1:]/np.array(np.shape(temp)),order=1,mode='nearest')
        temp1[temp1 <= 0] = 0
        array_of_img[(i - 1), :, :, :, :] = temp1
        #print(img)
        i = i+1
    return array_of_img
def read_directory_label(directory_name, size):
    array_of_img = np.empty(size, dtype='float32')  # this if for store all of the image data

    # this loop is for read each image in this foder,directory_name is the foder name with images.
    i = 1
    for filename in os.listdir(directory_name):
        img = sio.loadmat(directory_name + "/" + filename)
        temp = img['labelCT3d']
        temp1 = ndimage.interpolation.zoom(temp, size[1:-1] / np.array(np.shape(temp)), order=1, mode='nearest')
        temp1[temp1 <= 0] = 0
        array_of_img[(i - 1), :, :, :, :] = np.expand_dims(temp1,3)
        # print(img)
        i = i + 1
    return array_of_img

data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')


crosslist = np.reshape(range(0,220),(11,20))


shift = sio.loadmat('./shift.mat')
X = shift['X_shift']
Y = shift['Y_shift']
list_shift = sio.loadmat('./list_shift.mat')
listori = list_shift['list_shift']


shift = sio.loadmat('./mir.mat')
X = np.concatenate((X,shift['X_mir']),axis=0)
Y = np.concatenate((Y,shift['Y_mir']),axis=0)

list_mir = sio.loadmat('./list_mir.mat')
listori = (list(listori)+list(list_mir['list_mir']))

X = np.transpose(X, (0, 4, 2, 3, 1))


pathrep = '.'
xrand = sio.loadmat(pathrep +'/'+ 'xrand.mat')

x = xrand['xrand']
X_rand = X[x[0,:],:,:,:,:]
Y_rand = Y[x[0,:],:]
X=X_rand
Y=Y_rand

listrand=[0 for x in range(0,np.size(X,0))]
for i in range(0,np.size(X,0)):
    listrand[i]=listori[x[0,i]]
sio.savemat(pathrep +'/'+ 'list.mat', {'listrand':listrand})
listrand = sio.loadmat(pathrep +'/'+ 'list.mat')
listrand=listrand['listrand']

for irep in range(4,7):
    pathrep = './FeatureRecursion'+ str(irep)
    if not os.path.exists(pathrep):
        os.mkdir(pathrep)
    for i in range(1,12):

        model = cofcrossLSTM_time()
        model_checkpoint = ModelCheckpoint('unet_membrane.hdf5', monitor='loss', verbose=1, save_best_only=True)
        x_train = X
        x_train = np.delete(x_train, crosslist[i-1,:], 0)
        y_train = Y
        y_train = np.delete(y_train, crosslist[i - 1, :], 0)

        model.fit(x_train, y_train, batch_size=1, epochs=10, shuffle=True, verbose=1, validation_data=None)
        x_test = X[crosslist[i-1,:]]
        y_test = Y[crosslist[i-1,:]]

        list_test = listrand[crosslist[i-1,:]]
        results_label = model.predict(x_test, verbose=1)
        cross = 'cross' + str(i)
        ind = np.argmax(results_label,1)
        results_label_f = np.zeros(results_label.shape)
        for itemp in range(0,20):
            results_label_f[itemp,ind[itemp]]=1

        acc = np.sum(np.multiply(results_label_f, y_test)) / 20
        sio.savemat(pathrep +'/'+ cross+'.mat', {'results_label':results_label,'acc':acc,'list_test':list_test})
        results = model.evaluate(x_test, y_test)

        f = open(pathrep +'/'+ cross + '+result.txt', "w")
        for line in results:
            f.write(str(line) + '\n')
        f.close()