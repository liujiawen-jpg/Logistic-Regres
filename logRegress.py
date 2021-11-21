import numpy as np
import matplotlib.pyplot as plt
import random
def loadDataset():
    dataMat =[]
    labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat

def sigmoid(inX):
    return 1.0/(1+np.exp(-inX))

def gradAscent(dataMatIn, classLabels):
    dataMatrix = np.mat(dataMatIn)
    labelMat = np.mat(classLabels).transpose()
    m,n = np.shape(dataMatrix)
    alpha = 0.001
    maxCycles = 500
    weights = np.ones((n,1))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix*weights)
        error = (labelMat - h)
        weights = weights+ alpha* dataMatrix.transpose()*error
    return weights
def plotBestFit(weights):
    dataMat, labelMat = loadDataset()
    dataArr = np.array(dataMat)
    n = np.shape(dataArr[0])
    data1 = dataArr[np.array(labelMat)==1]
    data0 = dataArr[np.array(labelMat) == 0]
    xcord1 = data1[:,1].tolist()
    ycord1 =data1[:,2].tolist()
    xcord0 = data0[:,1].tolist()
    ycord0 =data0[:,2].tolist()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord0, ycord0, s=30, c='green')
    x = np.arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    plt.xlabel('X1');
    plt.ylabel('X2');
    plt.show()

def stocGradAscent0(dataMatrix, classLabels):
    m,n = np.shape(dataMatrix)
    alpha = 0.01 #学习率
    weights = np.ones(n)
    for i in range(200):
        for i in range(m):
            h = sigmoid(np.sum(dataMatrix[i]*weights))
            error = classLabels[i] - h
            weights = weights + alpha * error * dataMatrix[i]
    return weights

def stocGradAscent1(dataMatrix, classLabels, Epoch=200):
    m,n = np.shape(dataMatrix)
    weights = np.ones(n)
    for i in range(Epoch):
        dataIndex = [x for x in range(m)]
        for j in range(m):
            alpha = 4/(1.0+j+i)+0.01 #防止学习率系数为0
            randInx = int(np.random.uniform(0,len(dataIndex)))
            h = sigmoid(np.sum(dataMatrix[randInx]*weights))
            error = classLabels[randInx]-h
            weights = weights + alpha*error*dataMatrix[randInx]
            del(dataIndex[randInx])
    return weights

def classify(inX, weights):
    prob = sigmoid(np.sum(inX*weights))
    if prob >0.5:
        return 1.0
    return 0
def colicTest():
    frTrain = open('horseColicTraining.txt')
    frTest = open('horseColicTest.txt')
    trainingSet = []
    trainingLabel = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabel.append(float(currLine[21]))
    trainWeights = stocGradAscent1(np.array(trainingSet), trainingLabel, 500)
    errorCount = 0
    numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec +=1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if(int(classify(lineArr,trainWeights))) != int([21]):
            errorCount +=1
    errorRate = float(errorCount)/numTestVec
    print("错误率是: ", errorRate)
    return errorRate
def multiTest(epoch):
    errorSum = 0.0
    for k in range(epoch):
        errorSum += colicTest()
    print("%d 次测试，平均错误率为%f" % (epoch, errorSum/float(epoch)))

def dogvsCatTest(trainingSet, trainingLabel, testSet, testLabel):
    trainWeights = stocGradAscent1(np.array(trainingSet), trainingLabel, 500)
    errorCount = 0
    numTestVec = 0.0
    for i in range(len(testSet)):
        numTestVec +=1.0
        if(int(classify(np.array(testSet[i]),trainWeights))) != testLabel[i]:
            errorCount +=1
    errorRate = float(errorCount)/numTestVec
    print("错误率是: ", errorRate)
    return errorRate
