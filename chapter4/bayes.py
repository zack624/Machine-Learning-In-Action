# -*- coding: utf-8 -*-

from numpy import *

# 加载测试数据
def loadDataSet():
    postingList = [['my','dog','has','flea','problems','help','please'],
            ['maybe','not','take','him','to','dog','park','stupid'],
            ['my','dalmation','is','so','cute','I','love','him'],
            ['stop','posting','stupid','worthless','garbage'],
            ['mr','licks','ate','my','steak','how','to','stop','him'],
            ['quit','buying','worthless','dog','food','stupid']
            ]
    classVec = [0,1,0,1,0,1]
    return postingList,classVec

# 创建词汇表
def createVocabList(dataSet):
    vocabList = set([])
    for document in dataSet:
        vocabList = vocabList | set(document)
    return list(vocabList)

# 句子转换成向量：同词汇表长度，单词出现则为1，否则为0
def setOfWords2Vec(vocabList,inWords):
    retList = [0]*len(vocabList)
    for word in inWords:
        if word in vocabList:
            retList[vocabList.index(word)] = 1
        else:
            print "the word: %s is not in my vocaburary." % word
    return retList

# 词袋模型
def bagOfWords2VecMN(vocabList,inWords):
    retList = [0]*len(vocabList)
    for word in inWords:
        if word in vocabList:
            retList[vocabList.index(word)] += 1 # 统计各词出现的次数,且排除部分高频词汇
    return retList

# 求各类别下各词的条件概率的对数ln[p(Wj|Ci)]及类别1的概率p(C1)
def trainNB0(trainMatrix,trainCategory): # trainMatrix指各句子对应的词汇表向量的矩阵
    docNum = len(trainMatrix)
    vocabLen = len(trainMatrix[0])
    pAbusive = sum(trainCategory)/float(len(trainCategory)) # 类别1的概率
    p0Mat = ones(vocabLen); p1Mat = ones(vocabLen) # 各值为1的矩阵
    p0Num = 2.0; p1Num = 2.0 # 应用Laplace Smoothing
    for i in range(docNum):
        if trainCategory[i] == 0:
            p0Mat += trainMatrix[i]
            p0Num += sum(trainMatrix[i])
        else:
            p1Mat += trainMatrix[i]
            p1Num += sum(trainMatrix[i])
    p0Rate = log(p0Mat/p0Num) # numpy.log()可以矩阵各值求自然对数，math.log()不行
    p1Rate = log(p1Mat/p1Num)
    return p0Rate,p1Rate,pAbusive

# 应用贝叶斯公式比较ln[p(Wj|Ci)*p(Ci)]=ln[p(Wj|Ci)]+ln[(p(Ci)]
def classifyNB(vec,p0Mat,p1Mat,pAbusive):
    p1 = sum(vec*p1Mat) + log(pAbusive)
    p0 = sum(vec*p0Mat) + log(1.0 - pAbusive)
    if p0 > p1:
        return 0
    else:
        return 1

# 测试
def testingNB():
    postList,classVec = loadDataSet()
    vocabList = createVocabList(postList)
    mat = []
    for i in postList:
        mat.append(setOfWords2Vec(vocabList,i))
    p0Mat,p1Mat,pA = trainNB0(array(mat),array(classVec))
    word1 = ['garbage','stupid']
    vec1 = array(setOfWords2Vec(vocabList,word1))
    print word1,'classify as: ',classifyNB(vec1,p0Mat,p1Mat,pA)
    word2 = ['love','my','dalmation']
    vec2 = array(setOfWords2Vec(vocabList,word2))
    print word2,'classify as: ',classifyNB(vec2,p0Mat,p1Mat,pA)


def textParse(bigString):
    import re
    listOfTokens = re.split(r'\W*',bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]


def spamTest():
    docList = []; classList = []; fullText = []
    for i in range(1,26):
        spamWordList = textParse(open('email/spam/%d.txt' % i).read())
        docList.append(spamWordList)
        fullText.extend(spamWordList)
        classList.append(1)
        hamWordList = textParse(open('email/ham/%d.txt' % i).read())
        docList.append(hamWordList)
        fullText.extend(hamWordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    trainSet = range(50); testSet = []
    for i in range(10):
        randIndex = int(random.uniform(0,len(trainSet)))
        testSet.append(trainSet[randIndex])
        del(trainSet[randIndex])
    trainMat = []; trainClasses = []
    for j in trainSet:
        trainMat.append(setOfWords2Vec(vocabList,docList[j]))
        trainClasses.append(classList[j])
    p0Vec,p1Vec,pA = trainNB0(array(trainMat),array(trainClasses))
    errorCount = 0
    for k in testSet:
        retClass = classifyNB(array(setOfWords2Vec(vocabList,docList[k])),p0Vec,p1Vec,pA)
        if retClass != classList[k]:
            errorCount += 1.0
            print "classification error: ",k,",class: ",classList[k],",",docList[k],"\n"
    print "the error rate is %f." % (float(errorCount)/len(testSet))


def calcMostFreq(vocabList,fullText):
    import operator
    freqDict = {}
    for text in vocabList:
        freqDict[text] = fullText.count(text)
    sortedList = sorted(freqDict.iteritems(),key=operator.itemgetter(1),reverse=True)
    return sortedList[:30]


def localWords(feed1,feed0):
    import feedparser
    docList = []; classList = []; fullText = []
    minLen = min(len(feed1['entries']),len(feed0['entries']))
    for i in range(minLen):
        wordList = textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.append(wordList)
        classList.append(1)
        wordList = textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.append(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    top30Freq = calcMostFreq(vocabList,fullText)
    for pairW in top30Freq:
        if pairW[0] in vocabList:
            vocabList.remove(pairW[0])
    trainingSet = range(2*minLen); testSet = []
    for i in range(20):
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat = []; trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(bagOfWords2VecMN(vocabList,docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam = trainNB0(array(trainMat),array(trainClasses))
    errorCount = 0
    for i in testSet:
        wordVector = bagOfWords2VecMN(vocabList,docList[i])
        if classifyNB(array(wordVector),p0V,p1V,pSpam) != classList[i]:
            errorCount += 1
    print 'the error rate is: ',float(errorCount)/len(testSet)
    return vocabList,p0V,p1V

