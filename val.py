import h5py, cv2, os, time, pprint
import numpy as np
import net_grad_clip as mycnn

animalname = ["goldfish","frog","koala","jellyfish","penguin","dog","yak","house","bucket", "instrument","nail","fence","cauliflower","bell" "peper","mushroom","orange","lemon","banana","coffee","beach","unknown"]
#model_weights = 'model_sgd_without_decay_momentum_clip.npy'
model_weights = 'BestWeights.npy'
DATADIR = 'test/test_images'
LABELNAME = 'test/test_annotation'

with open(LABELNAME, 'rt') as f:
    con = f.read().strip()

arr = con.split('\n')
labels = list(map(lambda x: x.split('\t')[1], arr))
labels = list(map(lambda x: int(x)-1, labels))

def getImageData(path):
    return cv2.imread(path)

def getDATADIR(DATADIR):
    imglist = os.listdir(DATADIR)
    length = len(imglist)
    data = np.zeros((length, 64, 64, 3))
    for index, img in enumerate(imglist):
        imgpath = "{}\\{}.JPEG".format(DATADIR, index+1)
        data[index] = getImageData(imgpath)

    datamean = np.mean(data, axis=(1, 2)).reshape((length, 1, 1, 3))
    datastd = np.std(data, axis=(1, 2)).reshape((length, 1, 1, 3))

    data = (data - datamean)/datastd

    return data

def readDATASET(DATANAME):
    print('read the data:{}\n'.format(DATANAME))
    hf = h5py.File(DATANAME,'r')
    data = [hf.get('data')[()], hf.get('label')[()]]
    hf.close()
    return data

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

pprint.pprint(Convnet.layers)

Convnet.load_weights(model_weights)

data = getDATADIR(DATADIR)
# acc = Convnet.predict(data, labels, animalname)
acc = Convnet.predict(data, labels, animalname)
print("top1 acc:{}".format(acc[0]))
print("top3 acc:{}".format(acc[1]))
