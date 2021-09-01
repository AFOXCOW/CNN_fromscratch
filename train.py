import h5py, cv2, os, time
import numpy as np
import net_grad_clip as mycnn

animalname = ["goldfish","frog","koala","jellyfish","penguin","dog","yak","house","bucket", "instrument","nail","fence","cauliflower","bell" "peper","mushroom","orange","lemon","banana","coffee","beach","unknown"]
DATANAME = 'data_21.hdf5'

def readDATASET(DATANAME):
    print('read the data:{}\n'.format(DATANAME))
    hf = h5py.File(DATANAME,'r')
    data = [hf.get('data')[()], hf.get('label')[()]]
    hf.close()
    return data


data = readDATASET(DATANAME)
random_index = np.arange(500, dtype=np.int16)
np.random.shuffle(random_index)

div_num = int(500*0.8)
traindata = data[0][:, random_index[:div_num]]
testdata = data[0][:, random_index[div_num:]]

label_index = np.arange(21, dtype=np.int16)

trainlabel = np.repeat(label_index, div_num)
testlabel = np.repeat(label_index, 500-div_num)

traindata = traindata.reshape(21*div_num, 64, 64, 3)
testdata = testdata.reshape(21*(500-div_num), 64, 64, 3)

trainmean = np.mean(traindata, axis=(1, 2)).reshape((21*div_num, 1, 1, 3))
testmean = np.mean(testdata, axis=(1, 2)).reshape((21*(500-div_num), 1, 1, 3))

trainstd = np.std(traindata, axis=(1, 2)).reshape((21*div_num, 1, 1, 3))
teststd = np.std(testdata, axis=(1, 2)).reshape((21*(500-div_num), 1, 1, 3))

traindata = (traindata - trainmean)/trainstd
testdata = (testdata - testmean)/teststd

trainlabel = np.eye(21)[trainlabel]
testlabel = np.eye(21)[testlabel]

l2 = 0
lr = 1e-3

#网络结构
conv1 = mycnn.Conv2D(3, 32, (3, 3))
relu1 = mycnn.Relu()
pool1 = mycnn.MaxPooling2D()
conv2 = mycnn.Conv2D(32, 16, (3, 3))
relu2 = mycnn.Relu()
pool2 = mycnn.MaxPooling2D()
conv3 = mycnn.Conv2D(16, 8, (3, 3))
relu3 = mycnn.Relu()
flat = mycnn.Flatten()
dense1 = mycnn.Dense(2048, 100)
relu4 = mycnn.Relu()
dense2 = mycnn.Dense(100, 21)
softmax = mycnn.Softmax()


criterion = mycnn.CrossEntropy()

layers = [conv1, relu1, pool1, conv2, relu2, pool2,conv3,relu3, flat, dense1,relu4,dense2, softmax]

Convnet = mycnn.Net(layers)
#可以选择从预训练的weights中进行初始化
#Convnet.load_weights("BestWeights.npy") 
model_name = "backup.npy"
Convnet.train(traindata, trainlabel, testdata, testlabel, epochs=5, batch_size = 40, decay=0.99, criterion=criterion, lr=lr,shuffle=True, model_name=model_name,clip=None, momentum=0.9)

