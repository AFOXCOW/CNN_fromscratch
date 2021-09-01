import keras, h5py, cv2, os
import numpy as np

DATADIR = 'dataset'
DATANAME = 'data_21.hdf5'

def getImageData(path):
    return cv2.imread(path)

def getData(DATADIR):
    li = np.arange(500)
    data = np.zeros((21, 500, 64, 64, 3))
    label = np.zeros(21*500)
    for i in range(1, 21):
        print("handling {}\n".format(animalname[i-1]))
        imgdir = "{}\\{}\\images".format(DATADIR, i)
        imglist = os.listdir(imgdir)
        np.random.shuffle(imglist)
        label[500*(i-1):500*i] = i-1
        for index, imgname in enumerate(imglist[:500]):
            img = "{}\\{}".format(imgdir, imgname)
            data[i-1, index] = getImageData(img)

    return [data, label]
def writeDATASET(DATANAME, data):
    print('write the data:{}\n'.format(DATANAME))
    hf = h5py.File(DATANAME, 'w')
    hf.create_dataset('data', data=data[0])
    hf.create_dataset('label', data=data[1])
    hf.close()

def readDATASET(DATANAME):
    print('read the data:{}\n'.format(DATANAME))
    hf = h5py.File(DATANAME,'r')
    data = [hf.get('data')[()], hf.get('label')[()]]
    hf.close()
    return data

data = getData(DATADIR)
writeDATASET(DATANAME, data)