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
    global spamFrequency, hamFrequency
    files = os.listdir(path)
    
    for file in files:
        f = open(os.path.join(path, file), 'r', encoding = "ISO-8859-1")
        text = f.read()

        # Replace special characters and numbers
        text = re.sub('[^a-zA-Z]', ' ', text)
        words = text.strip().lower().split()
        
        for w in words:
            if removeStopWords=='no' and  w in stopWords:
                continue
            vocabulary.add(w)
            if isSpam:
                spamFrequency =spamFrequency + 1
                cnt = spamDictionary.get(w)
                if cnt is None:
                    cnt = 0
                cnt = cnt + 1
                spamDictionary.update({w: cnt}) 
            else:
                hamFrequency =hamFrequency + 1
                cnt = hamDictionary.get(w)
                if cnt is None:
                    cnt = 0
                cnt = cnt + 1
                hamDictionary.update({w: cnt})

def trainModel(path, isSpam,removeStopWords):
    classfiedCorrectly = 0
    totalFiles = 0
    
    files = os.listdir(path)
    for f in files:
        totalFiles = totalFiles+ 1
        
        # Calculate prior probabilities
        spamProb = math.log2(123/463)
        hamProb = math.log2(340/463)
        
        f = open(os.path.join(path, f), 'r', encoding = "ISO-8859-1")
        fileData = f.read()
        
        # Replace special characters and numbers
        fileData = re.sub('[^a-zA-Z]', ' ', fileData)
        words = fileData.strip().lower().split()
        
        for word in words: 
            #Spam
            if removeStopWords=='no' and  word in stopWords:
                continue
            if spamDictionary.get(word) is None:
                spamProb =spamProb + math.log2(1 / (spamFrequency + len(vocabulary)))
            else:
                spamProb = spamProb + math.log2((spamDictionary.get(word) + 1) / (spamFrequency + len(vocabulary)))
            # Ham
            if hamDictionary.get(word) is None:
                hamProb =hamProb +  math.log2(1 / (hamFrequency + len(vocabulary)))
            else:
                hamProb = hamProb+ math.log2((hamDictionary.get(word) + 1) / (hamFrequency + len(vocabulary)))
        if isSpam:
            if spamProb > hamProb:
                classfiedCorrectly =classfiedCorrectly + 1
        else:
            if hamProb > spamProb:
                classfiedCorrectly = classfiedCorrectly + 1

    return classfiedCorrectly, totalFiles

if __name__ == '__main__':
    # Files in Directories
    spamTrainPath = 'train/spam'
    hamTrainPath = 'train/ham'
    spamTestPath = 'test/spam'
    hamTestPath = 'test/ham'
    path=r'./stopwords.txt'
    removeStopWords = sys.argv[1]
    #Spam & ham directories 
    spamDictionary = {}
    hamDictionary = {} 
    #Store distinct words
    vocabulary = set() 
    stopWords = set()
    # Frequency
    spamFrequency = 0
    hamFrequency = 0
    #Read stop words
    readStopWords(path)
    #Read files
    readFiles(spamTrainPath, True,removeStopWords)
    readFiles(hamTrainPath, False,removeStopWords)
    #Train model
    spamCorrect, totalSpam = trainModel(spamTestPath, True,removeStopWords)
    hamCorrect, totalHam = trainModel(hamTestPath, False,removeStopWords)
    #Accuracy Calculation
    totalCorrectClassified = spamCorrect + hamCorrect
    totalFiles = totalSpam + totalHam
    print ("Accuracy:",round(totalCorrectClassified/totalFiles, 6)) 