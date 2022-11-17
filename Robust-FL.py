import keras
import numpy as np
from tensorflow.keras import models
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Reshape
import collections
from sklearn.cluster import KMeans

local_epoch = 3
client_num = 30
global_epoch = 100
f = 16
alpha = 0.8


def load_MNIST(PATH):

    train_images = np.load(PATH + '/x_train.npy')

    train_labels = np.load(PATH + '/y_train.npy')

    test_images = np.load(PATH + '/x_test.npy')

    test_labels = np.load(PATH + '/y_test.npy')

    return train_images, train_labels, test_images, test_labels

(train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()#images是一个28*28的矩阵，labels是一个数字（不是onehot）

def getById(id, train_images, train_labels):
    resImg = []
    resLab = []
    for i in range(len(train_labels)):
        if train_labels[i] == id:
            resImg.append(train_images[i])
            resLab.append(id)
    return resImg,resLab

def NonIID(degree, dataSize, clientNum):   #Non-IID data distribution
    '''
    :param degree: non-iid degree ()
    :param dataSize: the data each client holds
    :param clientNum: the total number of clients
    :return:
    '''
    imagesGroup = []
    labelsGroup = []
    resImg = []
    resLab = []
    for i in range(10):
        tempimg,templab = getById(i,train_images,train_labels)
        imagesGroup.append(tempimg)
        labelsGroup.append(templab)
    dominated = int(degree*dataSize)
    non_dominated = (dataSize-dominated)//9
    last_non_dominated = dataSize - dominated - non_dominated*8
    flag = [0]*10
    dominated_label = 0
    for i in range(clientNum):
        tempimg = []
        templab = []
        tempimg.extend(imagesGroup[dominated_label][flag[dominated_label]:flag[dominated_label]+dominated])
        templab.extend(labelsGroup[dominated_label][flag[dominated_label]:flag[dominated_label] + dominated])
        flag[dominated_label]=flag[dominated_label]+dominated
        for j in range(1,9):
            non_dominated_label = (dominated_label+j)%10
            tempimg.extend(imagesGroup[non_dominated_label][flag[non_dominated_label]:flag[non_dominated_label]+non_dominated])
            templab.extend(labelsGroup[non_dominated_label][flag[non_dominated_label]:flag[non_dominated_label] + non_dominated])
            flag[non_dominated_label] = flag[non_dominated_label] +non_dominated
        non_dominated_label = (dominated_label + 9) % 10
        tempimg.extend(imagesGroup[non_dominated_label][flag[non_dominated_label]:flag[non_dominated_label] + last_non_dominated])
        templab.extend(labelsGroup[non_dominated_label][flag[non_dominated_label]:flag[non_dominated_label] + last_non_dominated])
        flag[non_dominated_label] = flag[non_dominated_label] + last_non_dominated
        dominated_label = (dominated_label+1)%10
        index_ = [k for k in range(len(tempimg))]
        np.random.shuffle(index_)
        tempimg = np.array(tempimg)[index_]
        templab = np.array(templab)[index_]
        resImg.append(tempimg)
        resLab.append(templab)

    return resImg,resLab

N_image,N_label = NonIID(0.95,300,30)

def LabelFlip(label):   #label flipping attack
    return np.array([9-i for i in label])

def Model():
    model = models.Sequential()
    model.add(Reshape((28, 28, 1), input_shape=(28, 28)))
    model.add(Conv2D(32, (5, 5), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    return model

class Client:
    def __init__(self,trainImages,trainLabels):
        self.trainImages = trainImages
        self.trainLabels = trainLabels
        self.trainSetNum = len(trainImages)
        self.model = Model()
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])     #配置模型的一些参数
        self.alpha = 0
        self.distToMean =1000
        self.byzantine=False
    def broadcast(self,globalModelWeight):  #broadcast the global model to client
        self.model.set_weights(globalModelWeight)
    def train(self):    #local training
        self.model.fit(self.trainImages, self.trainLabels, batch_size=64, epochs=local_epoch)

def flat(nums): #flatten model weights to a 1d
    res = []
    for i in nums:
        if isinstance(i,collections.Iterable):
            res.extend(flat(i))
        else:
            res.append(i)
    return res


def EuclideanDistance(modelweight1,modelweight2): #calculating the Euclidean Distance between two model weights
    list1=np.array(modelweight1)
    list2=np.array(modelweight2)
    list3=np.square(list1-list2)
    sum=[]
    for l in list3:
        for c in l:
            sum.append(np.sum(np.sum(c)))
    return np.sqrt(np.sum(sum))


def ExpSmoo(clients,epoch):
    global s1, s2
    if epoch % 10 == 0 and epoch != 0:  #fine-tuning the predicted model every 10 iterations
        globalModel.fit(train_images[300:320],train_labels[300:320],batch_size=20, epochs=15)
        s1 = globalModel.get_weights()
        s2 = globalModel.get_weights()

    normal_index=[]
    dis = [EuclideanDistance(clients[i].model.get_weights(), (2*np.array(s1)-np.array(s2)+alpha/(1-alpha)*(np.array(s1)-np.array(s2)))) for i in range(len(clients))]
    zipped_dis = list(enumerate(dis))
    print("distances between local models and predicted model: ",zipped_dis)
    zipped_dis.sort(key=lambda x: x[1])
    print("sorting the distances: ", zipped_dis)

    km = KMeans(n_clusters=2)
    km.fit([[dis[i]] for i in range(len(clients))])
    fault_value = np.mean(km.cluster_centers_)
    print("断层点位置：",fault_value)
    for i in range(len(clients)):
        if zipped_dis[i][1]<fault_value:
            normal_index.append(zipped_dis[i][0])
        else:
            break
    normal_index.sort()
    print("Final indexs for aggregation: ",normal_index,(len(normal_index)))
    s1 = alpha * np.mean([clients[i].model.get_weights() for i in normal_index], axis=0) + (1 - alpha) * np.array(s1)
    s2 = alpha * np.array(s1) + (1 - alpha) * np.array(s2)
    return np.mean([clients[i].model.get_weights() for i in normal_index], axis=0)


def Sampling(model_weight,randomIndex):
    flatted_modelweight = flat(model_weight)
    return [flatted_modelweight[i] for i in randomIndex]

InitialModel = Model()
InitialModel.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
InitialModel.fit(train_images[300:320],train_labels[300:320],batch_size=5, epochs=20)   #initializing correct bias model
InitialModel_weight = InitialModel.get_weights()
s1 = InitialModel_weight
s2 = InitialModel_weight

if __name__ == '__main__':
    globalModel = Model()
    globalModel.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    globalModel.summary()
    globalModel_weight = InitialModel_weight
    clients=[]
    for i in range(0,client_num):
        clients.append(Client(N_image[i],N_label[i]))
        if i>=client_num-f:
            clients[i].byzantine = True

    for c in clients:
        c.broadcast(globalModel_weight)
    for epoch in range(global_epoch):
        for i in range(client_num):
            print("training on client " + str(i) + "(global epoch " + str(epoch + 1) + "/" + str(global_epoch) + ")")
            clients[i].train()
        for i in range(client_num):
            if clients[i].byzantine:
                clients[i].model.set_weights(np.array(clients[i].model.get_weights()) * (-0.8)) #sign flipping attak
        globalModel_weight = ExpSmoo(clients,epoch)
        globalModel.set_weights(globalModel_weight)
        test_loss, test_acc = globalModel.evaluate(test_images, test_labels)
        print(test_acc)
        for c in clients:
            c.broadcast(globalModel_weight)