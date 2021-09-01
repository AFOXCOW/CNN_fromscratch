import h5py, cv2, os
import numpy as np
import matplotlib.pyplot as plt
from progressbar import *

DATADIR = 'dataset'
DATANAME = 'data_21.hdf5'
animalnamepath = 'animalname.txt'

def getImageData(path):
    return cv2.imread(path)

def getDATADIR(DATADIR):
    imglist = os.listdir(DATADIR)
    length = len(imglist)
    data = np.zeros((length, 64, 64, 3))
    for index, img in enumerate(imglist):
        imgpath = "{}\\{}".format(DATADIR, img)
        data[index] = getImageData(imgpath)

    datamean = np.mean(data, axis=(1, 2)).reshape((length, 1, 1, 3))
    datastd = np.std(data, axis=(1, 2)).reshape((length, 1, 1, 3))

    data = (data - datamean)/datastd

    return data


def getData(DATADIR):
    li = np.arange(500)
    data = np.zeros((20, 500, 64, 64, 3))
    label = np.zeros(20*500)
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


class Conv2D(object):
    """docstring for Conv2D"""
    def __init__(self, channel, filter_size, kernel, strides=(1, 1), padding='same', kernel_regularizer = 0):
        super(Conv2D, self).__init__()
        self.name = 'conv'
        self.input_data = None
        self.output_data = None

        f_in = np.prod(kernel+(channel,))
        f_out = np.prod(kernel+(filter_size,))
        limit = np.sqrt(6.0/(f_in+f_out))
        self.weights = np.random.uniform(low=-limit, high=limit, size=kernel+(  channel, filter_size))
        self.bias = np.zeros(filter_size)

        self.channel = channel
        self.kernel = kernel
        self.filter_size = filter_size
        self.strides = strides
        self.padding = padding
        self.kernel_regularizer = kernel_regularizer
        self.weights_grad = None
        self.bias_grad = None
        self.data_grad = None

        self.weights_grad_li = None
        self.bias_grad_li = None
        self.data_grad_li = None

        self.weights_v = None
        self.bias_v = None

    def forward(self, input_data):
        self.data = input_data
        self.shape = input_data.shape
        if len(input_data.shape) == 2:
            H, W, CHANNEL = input_data.shape + (1,)
        elif len(input_data.shape) == 3:
            H, W, CHANNEL = input_data.shape
        else:
            raise Exception('Wrong input data shape!')

        if self.padding == 'same':
            kernel = self.kernel
            
            #step = kernel[0]

            self.output_shape = (self.shape[0], self.shape[1], self.filter_size)
            self.output_data = np.zeros(self.output_shape)
            self.pad_data = np.pad(self.data, ((1, 1), (1, 1), (0, 0)), 'constant', constant_values=0)
            
            step = self.output_shape[0]
            # reg = np.sum(0.5*self.kernel_regularizer*(self.weights**2), axis=(0, 1, 2))

            # Version 1
            # for i in range(self.kernel[0]):
            #     for j in range(self.kernel[1]):
            #         for k in range(self.output_shape[2]):
            #             mask = self.pad_data[i:i+step, j:j+step, :]
            #             w = self.weights[i, j, :, k]
            #             w = w.reshape(1, 1, self.channel)

            #             self.output_data[:, :, k] = self.output_data[:, :, k] + np.sum(mask*w, axis=2)

            # self.output_data = self.output_data + reg + self.bias

            # Version 2
            for i in range(self.kernel[0]):
                for j in range(self.kernel[1]):
                    mask = self.pad_data[i:i+step, j:j+step, :]
                    mask = np.expand_dims(mask, -1)
                    w = self.weights[i, j]
                    #np.save("w",w)
                    self.output_data = self.output_data + np.sum(mask*w, axis=2)

            # self.output_data = self.output_data + reg + self.bias
            self.output_data = self.output_data + self.bias

            # Original
            # for i in range(self.output_shape[0]):
            #     for j in range(self.output_shape[1]):
            #         for k in range(self.output_shape[2]):
            #             mask = self.pad_data[i:i+step, j:j+step, :]

            #             w = self.weights[:, :, :, k]
            #             bias = self.bias[k]
            #             #print(np.sum(mask*w))
            #             self.output_data[i, j, k] = np.sum(mask*w + 0.5*self.kernel_regularizer*(w**2)) + bias
            return self.output_data

    def backward(self, input_grad):
        x_step = input_grad.shape[0]
        w_step = self.kernel[0]
        
        self.weights_grad = np.zeros(self.weights.shape)
        self.bias_grad = np.zeros(self.bias.shape)
        self.data_grad = np.zeros(self.shape)

        if self.weights_grad_li is None:
            self.weights_grad_li = []

        if self.bias_grad_li is None:
            self.bias_grad_li = []

        if self.data_grad_li is None:
            self.data_grad_li = []

        pad_grad = np.pad(input_grad, ((1, 1), (1, 1), (0, 0)), 'constant', constant_values=0)
        flip_w = np.flipud(np.fliplr(self.weights))
        
        self.bias_grad = np.sum(input_grad, axis=(0, 1))

        # for k in range(self.filter_size):
            
        #     for i in range(self.kernel[0]):
        #         for j in range(self.kernel[0]):
        #             mask = self.pad_data[i:i+x_step, j:j+x_step]
        #             w = input_grad[:, :, k]

        #             for ch in range(self.channel):
        #                 self.weights_grad[i, j, ch, k] = np.sum(mask[:, :, ch]*w[:, :, ch])
            
        #     for i in range(self.shape[0]):
        #         for j in range(self.shape[1]):
        #             mask = pad_grad[i:i+w_step, j:j+w_step]
        #             w = flip_w[:, :, :, k]

        #             for ch in range(self.channel):
        #                 self.data_grad[i, j, ch, k] = np.sum(mask[:, :, ch]*w[:, :, ch])
        
        #Origianl Version
        # for i in range(self.kernel[0]):
        #     for j in range(self.kernel[1]):

        #         for ch in range(self.channel):
        #             #mask: h*w*filter_size
        #             #w:h*w*filter_size
        #             mask = self.pad_data[i:i+x_step, j:j+x_step, ch]
        #             mask = np.dstack([mask]*self.filter_size) + self.kernel_regularizer*self.weights[i, j, ch]
        #             w = input_grad

        #             self.weights_grad[i, j, ch] += np.sum(mask*w, axis=(0, 1))

        
        # 
        # for i in range(self.shape[0]):
        #     for j in range(self.shape[1]):
        #         for ch in range(self.channel):
        #             mask = pad_grad[i:i+w_step, j:j+w_step]
        #             w = flip_w[:, :, ch, :]

        #             self.data_grad[i, j, ch] = np.sum(mask*w)
        
        # Version 1
        for i in range(self.kernel[0]):
            for j in range(self.kernel[1]):

                for ch in range(self.channel):
                    #mask: h*w*filter_size
                    #w:h*w*filter_size
                    mask = self.pad_data[i:i+x_step, j:j+x_step, ch]
                    # mask = np.dstack([mask]*self.filter_size) + self.kernel_regularizer*self.weights[i, j, ch]
                    mask = np.dstack([mask]*self.filter_size)
                    w = input_grad

                    self.weights_grad[i, j, ch] = np.sum(mask*w, axis=(0, 1))

                    mask_data = pad_grad[i:i+x_step, j:j+x_step]
                    w_data = flip_w[i, j, ch]

                    self.data_grad[:, :, ch] = self.data_grad[:, :, ch] + np.sum(mask_data*w_data, axis=2)

        # Version 2
        # for i in range(self.kernel[0]):
        #     for j in range(self.kernel[1]):
        #         #mask: h*w*filter_size
        #         #w:h*w*filter_size
        #         mask = self.pad_data[i:i+x_step, j:j+x_step]
        #         mask = np.stack([mask]*self.filter_size, axis=2) + self.kernel_regularizer*self.weights[i, j]
        #         w = input_grad

        #         self.weights_grad[i, j] += np.sum(mask*w, axis=(0, 1))

        #         mask_data = pad_grad[i:i+x_step, j:j+x_step]
        #         w_data = flip_w[i, j]

        #         self.data_grad = self.data_grad + np.sum(mask*w, axis=2)
        self.weights_grad_li.append(self.weights_grad)
        self.bias_grad_li.append(self.bias_grad)

        return [self.data_grad, self.weights_grad, self.bias_grad]

    def set_zero(self):
        self.weights_grad_li = None
        self.bias_grad_li = None  
    
    def set_v_zero(self):
        self.weights_v = None
        self.bias_v = None

class MaxPooling2D(object):
    """docstring for MaxPooling2D"""
    def __init__(self):
        self.name = 'pooling'
        self.input_data = None
        self.output_data = None
        self.pos = None

    def forward(self, input_data):
        self.data = input_data
        self.shape = input_data.shape
        self.pos = np.zeros(self.shape)

        output_shape = (int(self.shape[0]/2), int(self.shape[1]/2), self.shape[2])
        self.output_data = np.zeros(output_shape)

        # Origianl Version
        # for i in range(output_shape[0]):
        #     for j in range(output_shape[1]):
        #         for ch in range(output_shape[2]):
        #             mask = self.data[2*i:2*(i+1), 2*j:2*(j+1), ch]
        #             max_data = mask.max()
        #             max_index = mask.argmax()
        #             row = int(max_index/2)
        #             col = max_index%2

        #             self.pos[2*i+row, 2*j+col, ch] = 1
        #             self.output_data[i, j, ch] = max_data
        
        # Version 1
        for i in range(output_shape[0]):
            for j in range(output_shape[1]):
                mask = self.data[2*i:2*(i+1), 2*j:2*(j+1)]
                max_data = mask.max(axis=(0, 1))
                max_index = mask.reshape(4, output_shape[2]).argmax(axis=0)
                row = (max_index/2).astype(np.int)
                col = max_index%2

                self.pos[2*i+row, 2*j+col, np.arange(output_shape[2])] = 1
                self.output_data[i, j, np.arange(output_shape[2])] = max_data
        return self.output_data

    def backward(self, input_grad):

        grad = input_grad.repeat(2, axis=0).repeat(2, axis=1)
        self.data_grad = grad*self.pos
        return [self.data_grad]

class Relu(object):
    """docstring for Relu"""
    def __init__(self):
        self.name = 'relu'
        self.data = None

    def forward(self, input_data):
        self.data = input_data
        self.shape = input_data.shape
        self.output_data = np.copy(input_data)
        self.output_data[self.output_data<0] = 0
        return self.output_data

    def backward(self, input_grad):
        #修改 2019/01/02
        #self.data_grad = np.ones(self.shape)
        self.data_grad = np.copy(input_grad)
        self.data_grad[self.data<0] = 0

        return [self.data_grad]

class Flatten(object):
    """docstring for Flatten"""
    def __init__(self):
        self.name = 'flatten'
        self.data = None
        self.shape = None

    def forward(self, input_data):
        #test
        # self.data = np.transpose(input_data, (2, 0, 1))
        # self.shape = self.data.shape

        self.data = input_data
        self.shape = input_data.shape
        
        self.output_data = np.ravel(self.data)#先深度再行列

        return self.output_data

    def backward(self, input_grad):
        #test
        # self.data_grad = input_grad.reshape(self.shape)
        # self.data_grad = np.transpose(self.data_grad, (1, 2, 0))

        self.data_grad = input_grad.reshape(self.shape)
        return [self.data_grad]

class Dense(object):
    """docstring for Dense"""
    def __init__(self, input_size, output_size, kernel_regularizer = 0):
        self.name = 'dense'
        self.data = None
        self.shape = None
        self.output_data = None

        f_in = input_size
        f_out = output_size
        limit = np.sqrt(6.0/(f_in+f_out))
        self.weights = np.random.uniform(low=-limit, high=limit, size=(output_size, input_size))
        self.bias = np.zeros(output_size)

        self.input_size = input_size
        self.output_size = output_size
        self.kernel_regularizer = kernel_regularizer

        self.weights_grad = None
        self.bias_grad = None
        self.data_grad = None

        self.weights_grad_li = None
        self.bias_grad_li = None
        self.data_grad_li = None

        self.weights_v = None
        self.bias_v = None

    def forward(self, input_data):
        self.data = input_data
        self.shape = input_data.shape

        # reg = self.weights**2
        # reg = reg.sum(axis=1)
        # #2019/01/02 增加代码 改为列向量
        # reg = np.expand_dims(reg, -1)
        
        self.output_data = np.zeros(self.output_size)
        
        #self.output_data = np.dot(self.weights, np.expand_dims(self.data, -1)) + np.expand_dims(self.bias, -1) + 0.5*self.kernel_regularizer*reg
        self.output_data = np.dot(self.weights, np.expand_dims(self.data, -1)) + np.expand_dims(self.bias, -1)
        # self.output_data = np.dot(self.weights.T, self.data) + self.bias + 0.5*self.kernel_regularizer*reg
        # 2019/01/02
        self.output_data = np.ravel(self.output_data)
        
        return self.output_data

    def backward(self, input_grad):
        self.weights_grad = np.zeros(self.weights.shape)
        self.bias_grad = np.zeros(self.bias.shape)
        self.data_grad = np.zeros(self.shape)

        if self.weights_grad_li is None:
            self.weights_grad_li = []

        if self.bias_grad_li is None:
            self.bias_grad_li = []

        if self.data_grad_li is None:
            self.data_grad_li = []

        #self.bias_grad += np.copy(input_grad)
        self.bias_grad = input_grad

        y_grad = input_grad.reshape((input_grad.shape[0], 1))
        x_grad = self.data.reshape((1, self.shape[0]))

        input_grad_T = np.expand_dims(input_grad, -1)
        # reg_grad = self.kernel_regularizer*self.weights*input_grad_T
        # self.weights_grad += np.dot(y_grad, x_grad) + reg_grad

        self.weights_grad = np.dot(y_grad, x_grad)
        self.data_grad = np.dot(input_grad, self.weights)

        self.weights_grad_li.append(self.weights_grad)
        self.bias_grad_li.append(self.bias_grad)

        return [self.data_grad, self.weights_grad, self.bias_grad]

    def set_zero(self):
        self.weights_grad_li = None
        self.bias_grad_li = None

    def set_v_zero(self):
        self.weights_v = None
        self.bias_v = None

class Softmax(object):
    """docstring for Softmax"""
    def __init__(self):
        self.name = 'softmax'
        self.data = None
        self.shape = None

    def forward(self, input_data):
        self.data = input_data
        self.shape = input_data.shape
        max_value = input_data.max()
        self.output_data = np.exp(input_data-max_value)/np.sum(np.exp(input_data-max_value))#这里softmax符号

        return self.output_data

    def backward(self, input_grad):
        self.data_grad = np.zeros(self.shape)
        min_index = np.argmin(input_grad)
        # self.data_grad_index = input_grad
        # min_index = input_grad
        # y = np.diag(self.output_data)
        #改过 2019/1/1
        #原版本 y = np.eyes(self.shape)， 错误

        # f1 = self.output_data.reshape((self.shape[0], 1))
        # f2 = self.output_data.reshape((1, self.shape[0]))
        # softmax_grad = y - np.dot(f1, f2)

        # self.data_grad = np.dot(input_grad, softmax_grad)

        self.data_grad = np.copy(self.output_data)
        self.data_grad[min_index] = self.data_grad[min_index] - 1

        return [self.data_grad]

class CrossEntropy(object):
    """docstring for CrossEntropy"""
    def __init__(self):
        self.name = 'crossentropy'
        self.y = None
    
    #2019/01/02 forward 函数更改，添加y_label
    def forward(self, input_data, y_label):
        self.y = y_label
        self.data = input_data
        self.shape = input_data.shape

        index = np.argmax(y_label)
        

        self.output_data = -np.log(input_data[index]+np.e**(-9))

        # self.output_data = -1*self.y*np.log(np.e**(-7)+input_data)#原始版本是直接添负号而不是-1*形式
        # self.output_data[self.output_data < 0] = 0
        # self.output_data = np.sum(self.output_data)

        return self.output_data
    # def backward(self, y_label):
    #     self.data_grad = np.zeros(self.shape)
    #     index = np.argmax(y_label)
    #     self.data_grad[index] = -1
    #     # self.data_grad_index=index = np.argmax(y_label)

    #     return [self.data_grad]
        


class Net(object):
    """docstring for Net"""
    def __init__(self, layers):
        self.layers = []
        for layer in layers:
            #self.layers.append(layers)#不加s
            self.layers.append(layer)

        self.length = len(self.layers)
    
    #forward 函数改动-2019/01/02:添加 y_label
    def forward(self, x):
        for index, layer in enumerate(self.layers):
            if index == 0:
                # start=time.time()
                data = layer.forward(x)
                # end=time.time()
            # elif index == self.length - 1:
            #     # start=time.time()
            #     data = layer.forward(data, y_label)
            #     # end=time.time()
            else:
                # start=time.time()
                data = layer.forward(data)
                # end=time.time()
            # print(layer.name)
            # print(end-start)
        return data

    def backward(self, y):
        
        for index, layer in enumerate(self.layers):
            
            i = self.length - index - 1
            if index == 0:
                grad = self.layers[i].backward(y)#少了self 20190102
            else:
                grad = self.layers[i].backward(grad[0])
        
        return grad
    
    def set_zero(self):
        w_name = ['conv', 'dense']
        for layer in self.layers:
            if layer.name in w_name:
                layer.set_zero()


    def set_v_zero(self):
        w_name = ['conv', 'dense']
        for layer in self.layers:
            if layer.name in w_name:
                layer.set_v_zero()

    def SGD(self, lr=1e-3, momentum=0, batch_size=64, clip = None):
        w_name = ['conv', 'dense']

        for layer in self.layers:
            if layer.name in w_name:
                if layer.weights_v is None:
                    layer.weights_v = np.sum(layer.weights_grad_li, axis=0)/batch_size
                else:
                    layer.weights_v = momentum*layer.weights_v + np.sum(layer.weights_grad_li, axis=0)
                
                if layer.bias_v is None:
                    layer.bias_v = np.sum(layer.bias_grad_li)/batch_size
                else:
                    layer.bias_v = momentum*layer.bias_v + np.sum(layer.bias_grad_li)/batch_size
                   
                #layer.weights_v[layer.weights_v>1e2] = 1e2
                if clip is not None:
                    layer.weights_v[layer.weights_v>clip]=clip
                    layer.weights_v[layer.weights_v<-clip]=-clip
                    layer.bias_v[layer.bias_v>clip]=clip
                    layer.bias_v[layer.bias_v<-clip]=-clip
                #print("max grad:{}".format(np.max(layer.bias_grad)))
                layer.weights = layer.weights - lr*layer.weights_v
                layer.bias = layer.bias - lr*layer.bias_v

        self.set_zero()

    def train(self, x, y_label, x_val, y_val, epochs=100, batch_size=64, lr=1e-3, decay=0.9, momentum=0.9,clip=None, criterion=None, model_name='model.npy', shuffle=False):
        x_len = x.shape[0]
        print('batch size:{}'.format(batch_size))
        best_acc = 0

        index_li = np.arange(x.shape[0])
        
        cur_lr = lr
        for epoch in range(epochs):
            print("Epoch:{}".format(epoch))
            
            self.set_v_zero()
            pbar = ProgressBar().start() 

            if shuffle == True:
                np.random.shuffle(index_li)
                
            for i, index in enumerate(index_li):
                
                #np.save("i",i)
                pbar.update(int((i / (x.shape[0] - 1)) * 100))#进度条
                #forward 函数改动-2019/01/02:添加 y_label
                #self.forward(x)
                #self.backward(y_label)
                # start=time.time()
                output = self.forward(x[index])
                loss = criterion.forward(output, y_label[index])
                # end = time.time()
                # print("forward time : ")
                # print(end-start)
                # start=time.time()
                self.backward(-y_label[index])
                # end = time.time()
                # print("backward time : ")
                # print(end-start)
                if (i+1)%batch_size == 0:
                    if decay != 0:
                        cur_lr = decay*cur_lr
                    self.SGD(lr=cur_lr, momentum=momentum, clip=clip, batch_size=batch_size)


            pbar.finish()  

            acc = self.validate(x_val, y_val)
            print("\nacc:{}\n".format(acc))
            if best_acc > acc[0]:
                self.save_weights(model_name)
            else:
                best_acc = acc[0]

    def validate(self, x, y):
        length = x.shape[0]
        top1_count = 0
        top3_count = 0
        pbar = ProgressBar().start() 
        for i in range(length):
            pbar.update(int((i / (length - 1)) * 100))#进度条
            output = self.forward(x[i])
            pred_1 = np.argsort(output)[-1]
            pred_3 = np.argsort(output)[-3:]
            label = np.argsort(y[i])[-1]

            if label in pred_3:
                top3_count = top3_count + 1
            if label == pred_1:
                top1_count = top1_count + 1
        pbar.finish()  
        return (top1_count/length, top3_count/length)
    
    def get_weights(self):
        w = []
        w_name = ['conv', 'dense']
        for layer in self.layers:
            if layer.name in w_name:
                w.append([layer.weights, layer.bias])

        return w

    def set_weights(self, params):
        i = 0
        w_name = ['conv', 'dense']
        for layer in self.layers:
            if layer.name in w_name:
                layer.weights = params[i][0]
                layer.bias = params[i][1]
                i = i + 1

    def load_weights(self, model):
        i = 0
        w_name = ['conv', 'dense']
        params = np.load(model)
        for layer in self.layers:
            if layer.name in w_name:
                layer.weights = params[i][0]
                layer.bias = params[i][1]
                i = i + 1

    def save_weights(self, model_name):
        params = self.get_weights()
        np.save(model_name, params)


    def predict(self, x, labels, info):
        # pred_li = []
        print('--------------------------------------------------------------------------------------------------------------------------------')
        length = x.shape[0]
        top1_count = 0
        top3_count = 0
        for index in range(x.shape[0]):
            img = "{}.JPEG".format(index+1)
            output = self.forward(x[index])
            pred = np.argsort(output)[-3:]
            
            # pred_li.append(info[pred])
            label = labels[index]

            if label in pred:
                top3_count = top3_count + 1
            if label == pred[-1]:
                top1_count = top1_count + 1
            print('\t{:15}\ttruth:{:15}\tpred:{:15}:{:.2f}\t{:15}:{:.2f}\t{:15}:{:.2f}'.format(img, info[label], info[pred[-1]], output[pred[-1]], info[pred[-2]], output[pred[-2]], info[pred[-3]], output[pred[-3]]))
        return (top1_count/length, top3_count/length)



'''
#convolutional layer test
input_data = np.ones((4, 4, 3))#length height channel
for i in range(0,4):
    for j in range(0,4):
        for k in range(0,3):
            input_data[i][j][k]=i+j

conv = Conv2D(3, 2, (3, 3))

conv.weights = np.ones((3, 3, 3, 2)) #size  filternumber
for i in range(0,3):
    for j in range(0,3):
        for k in range(0,3):
            conv.weights[i][j][k][0]=i+j
for i in range(0,3):
    for j in range(0,3):
        for k in range(0,3):
            conv.weights[i][j][k][1]=4-i-j

conv.bias = np.ones(8)

output = conv.forward(input_data)
'''
#print(output[:,:,0])

'''
#maxpooling layer test
a=np.array([[[1.],[2.],[3.],[4.]],[[2.],[3.],[4.],[5.]],[[3.],[4.],[5.],[6.]],[[4.],[5.],[6],[7]]])
input_data = a
print(input_data[:,:,0])
pool=MaxPooling2D()

output=pool.forward(input_data)
print(output[:,:,0])



#relu layer test
a=np.array([[[-1.],[2.],[3.],[4.]],[[-2.],[3.],[4.],[5.]],[[3.],[-4.],[5.],[6.]],[[4.],[-5.],[6],[7]]])
input_data = a
print(input_data[:,:,0])
relu=Relu()
output=relu.forward(input_data)
print(output[:,:,0])


#flatten layer test
a=np.array([[[1.],[2.],[3.],[4.]],[[2.],[3.],[4.],[5.]],[[3.],[4.],[5.],[6.]],[[4.],[5.],[6],[7]]])
input_data = a
flat=Flatten()
output=flat.forward(input_data)



#dense layer test

dense_input_data=output

dens=Dense(16,2)
dens.weights=np.ones((2,16))

dens.bias=np.ones(2)

dens_output=dens.forward(dense_input_data)

#softmax layer test 

softm=Softmax()
soft_out=softm.forward(dens_output)


#loss layer test
y_label=np.array([1,0])
loss = CrossEntropy()
loss.y=y_label
ls=loss.forward(soft_out)
print(ls)
'''

'''
conv1 = Conv2D(3, 32, (3, 3))
relu1 = Relu()
pool1 = MaxPooling2D()

flat = Flatten()
dense = Dense(32768, 21)
softmax = Softmax()
loss = CrossEntropy()
y_label=np.zeros(21)
y_label[0]=1
loss.y=y_label

layers = [conv1, relu1, pool1, flat, dense, softmax, loss]

Convnet = Net(layers)
output=Convnet.forward(input_data)
'''
