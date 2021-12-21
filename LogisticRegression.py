import numpy as np
import os
import math
import sys
import re

def readStopWords(path):
    readStopWords = open('./stopwords.txt', 'r', encoding = "ISO-8859-1")
    words = readStopWords.read().strip().lower().split()
    for w in words:
        stopWords.add(w)

def readFiles(path, isSpam,removeStopWords):   
    files = os.listdir(path)
    for file in files:
        f = open(os.path.join(path, file), 'r', encoding = "ISO-8859-1")
        text = f.read()
        # Replace special characters and numbers
        text = re.sub('[^a-zA-Z]', ' ', text)
        words = text.strip().lower().split()
        rowToTrain = {}
        rowToTrain["0"] = 1.0 
        for w in words:
            if removeStopWords=='no' and  w in stopWords:
                continue
            vocabulary.add(w)
            if rowToTrain.get(w) is None:
                rowToTrain[w] = 0
            rowToTrain[w] += 1
        attributes.append(rowToTrain)
        if isSpam:
            target.append(0)
        else:
            target.append(1)

def calculateProbability(inputindex):
    global oldParameters
    wx = oldParameters["0"]
    for word in inputindex:
        if word in oldParameters:
            wx += oldParameters[word] * inputindex[word]
    if wx >= 500:
        return 1
    #Sigmoid Function Calculator
    exponentialPowerWX = math.exp(wx)
    prob = exponentialPowerWX / (1 + exponentialPowerWX)
    return prob

def logisticRegModel():
    global oldParameters, updatedParameters
    updatedParameters = oldParameters.copy()
    targetlength = len(target)
    for i in range(iter):
        for i in oldParameters:
            sum = 0.0
            for idx in range(targetlength):
                targetIndex = target[idx]
                attributeIndex = attributes[idx]
                
                if i in attributeIndex:
                    prob = calculateProbability(attributeIndex)
                    sum += attributeIndex[i] * (targetIndex - prob)
                    
            updatedParameters[i] = oldParameters[i] + learningRate * sum - lamda * learningRate * oldParameters[i]
        oldParameters = updatedParameters.copy()

def testModel(path, isSpam,removeStopWords):
    classifiedCorrrectly = 0
    totalFiles = 0
    files = os.listdir(path)
    for file in files:
        totalFiles += 1
        f = open(os.path.join(path, file), 'r', encoding = "ISO-8859-1")
        text = f.read()
        # Replace special characters and numbers
        text = re.sub('[^a-zA-Z]', ' ', text)
        words = text.strip().lower().split()
        rowToTest = {}
        for w in words:
            if w in oldParameters:
                if removeStopWords=='no' and  w in stopWords:
                    continue
                if rowToTest.get(w) is None:
                    rowToTest[w] = 0
                rowToTest[w] += 1
        wx = oldParameters["0"]
        for w in rowToTest:
            wx += oldParameters[w] * rowToTest[w]
        if isSpam:
            if wx <= 0:
                classifiedCorrrectly += 1
        else:
            if wx > 0:
                classifiedCorrrectly += 1
    return classifiedCorrrectly, totalFiles

if __name__ == '__main__':
    #Files In directories
    spamTrainPath = 'train/spam'
    hamTrainPath = 'train/ham'
    spamTestPath = 'test/spam'
    hamTestPath = 'test/ham'
    path=r'./stopwords.txt'
    lamda = float(sys.argv[1])
    iter = int(sys.argv[2])
    removeStopWords = sys.argv[3]
    #inputs and target
    attributes = []
    target = []
    oldParameters = {}
    updatedParameters = {} 
    #Store disticnt words
    vocabulary = set() 
    stopWords = set()
    #Setting default learning rate
    learningRate = 0.01
    #Reading Stop Words
    if removeStopWords=='no':
        readStopWords(path)
    #Read Files
    readFiles(spamTrainPath, True,removeStopWords)
    readFiles(hamTrainPath, False,removeStopWords)
    oldParameters["0"] = 0.0
    for eachWord in vocabulary:
        oldParameters[eachWord] = 0.0 
    #Train Model
    logisticRegModel()
    #Test Model
    spamCorrect, totalSpam = testModel(spamTestPath, True,removeStopWords)
    hamCorrect, totalHam = testModel(hamTestPath, False,removeStopWords)
    totalClassifiedCorrect = spamCorrect + hamCorrect
    totalFiles = totalSpam + totalHam
    print ("Accuracy:", round(totalClassifiedCorrect/totalFiles, 6)) 