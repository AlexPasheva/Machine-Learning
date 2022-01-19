#############################################################################
### Търсене и извличане на информация. Приложение на дълбоко машинно обучение
### Стоян Михов
### Зимен семестър 2020/2021
#############################################################################
###
### Домашно задание 3
###
#############################################################################
import numpy as np
import torch

import sys

import pickle
import math


corpusFileName = 'corpusFunctions'
modelFileName = 'modelLSTM'
trainDataFileName = 'trainData'
testDataFileName = 'testData'
char2idFileName = 'char2id'

device = torch.device("cuda:0")
#device = torch.device("cpu")

batchSize = 32
char_emb_size = 32

hid_size = 128
lstm_layers = 2
dropout = 0.5

epochs = 3
learning_rate = 0.001

defaultTemperature = 0.4

#############################################################################
### Търсене и извличане на информация. Приложение на дълбоко машинно обучение
### Стоян Михов
### Зимен семестър 2020/2021
##########################################################################
###
### Домашно задание 3
###
#############################################################################


corpusSplitString = ';)\n'
maxProgramLength = 10000
symbolCountThreshold = 100


def splitSentCorpus(fullSentCorpus, testFraction=0.1):
    random.seed(42)
    random.shuffle(fullSentCorpus)
    testCount = int(len(fullSentCorpus) * testFraction)
    testSentCorpus = fullSentCorpus[:testCount]
    trainSentCorpus = fullSentCorpus[testCount:]
    return testSentCorpus, trainSentCorpus


def getAlphabet(corpus):
    symbols = {}
    for s in corpus:
        for c in s:
            if c in symbols:
                symbols[c] += 1
            else:
                symbols[c] = 1
    return symbols


def prepareData(corpusFileName, startChar, endChar, unkChar, padChar):
    file = open(corpusFileName, 'r')
    poems = file.read().split(corpusSplitString)
    symbols = getAlphabet(poems)

    assert startChar not in symbols and endChar not in symbols and unkChar not in symbols and padChar not in symbols
    charset = [startChar, endChar, unkChar, padChar] + \
        [c for c in sorted(symbols) if symbols[c] > symbolCountThreshold]
    char2id = {c: i for i, c in enumerate(charset)}

    corpus = []
    for i, s in enumerate(poems):
        if len(s) > 0:
            corpus.append([startChar] + [s[i]
                          for i in range(min(len(s), maxProgramLength))] + [endChar])

    testCorpus, trainCorpus = splitSentCorpus(corpus, testFraction=0.01)
    print('Corpus loading completed.')
    return testCorpus, trainCorpus, char2id



def generateCode(model, char2id, startSentence, limit=1000, temperature=1.):
    # model е инстанция на обучен LSTMLanguageModelPack обект
    # char2id е речник за символите, връщащ съответните индекси
    # startSentence е началния низ стартиращ със символа за начало '{'
    # limit е горна граница за дължината на поемата
    # temperature е температурата за промяна на разпределението за следващ символ
    
    result = startSentence[1:]
    
    id2char = dict(enumerate(char2id))

    #Правим функция, която да предсказва всяка следваща буква
    #по подобие на фиг. 1, следвайки фиг. 2 от заданието
    def predict(model, source, h=None):
        
        X = model.preparePaddedBatch(source)
        E = model.embed(X)
        source_lengths = [len(s) for s in source]

        if h != None:
            outputPacked, h = model.lstm(torch.nn.utils.rnn.pack_padded_sequence(E, source_lengths, enforce_sorted = False), h)
        else:
            outputPacked, h = model.lstm(torch.nn.utils.rnn.pack_padded_sequence(E, source_lengths, enforce_sorted = False))
        output,_ = torch.nn.utils.rnn.pad_packed_sequence(outputPacked)

        Z = model.projection(model.dropout(output.flatten(0, 1)))
        length = len(source) - 1
        p = torch.nn.functional.softmax(torch.div(Z, temperature), dim = 1).data
        p, topChar = p.topk(32)
        topChar = topChar.numpy().squeeze()
        p = p[length].numpy().squeeze()
        if type(topChar[length]) is np.ndarray:
            t = np.random.choice(topChar[length], p = p / np.sum(p))
        else:
            t = np.random.choice(topChar, p = p / np.sum(p))
        return id2char[t], h 

    #Проверяваме дали е въведена начална дума
    #Ако е въведена - добавяме отстояние след нея
    #Иначе генерираме случайна главна буква, с която да започнем
    if(len(startSentence) == 1):
        chars = list(char2id.keys())
        letters = chars[ord('a'):ord('z')] # ord() cast char to int
        startSentence += np.random.choice(letters)
    else:
        startSentence += " "

    initWordSize = len(result)
    
    python_function = [x for x in result] # текущото състояние на функцията
    output, h = predict(model, python_function)
    python_function.append(output)
    model.eval()
    
    size = initWordSize
    while not output == '}' and size <= limit :
        output, h = predict(model, python_function[size], h)
        python_function.append(output)
        size = size + 1

    result = ""
    for ch in python_function:
        result += ch

    #### Край на Вашия код
    #############################################################################

    return result

#############################################################################
### Търсене и извличане на информация. Приложение на дълбоко машинно обучение
### Стоян Михов
### Зимен семестър 2020/2021
#############################################################################
###
### Домашно задание 3
###
#############################################################################


#################################################################
####  LSTM с пакетиране на партида
#################################################################


class LSTMLanguageModelPack(torch.nn.Module):
    def preparePaddedBatch(self, source):
        device = next(self.parameters()).device
        m = max(len(s) for s in source)
        sents = [[self.word2ind.get(w, self.unkTokenIdx)
                  for w in s] for s in source]
        sents_padded = [s+(m-len(s))*[self.padTokenIdx] for s in sents]
        return torch.t(torch.tensor(sents_padded, dtype=torch.long, device=device))

    def save(self, fileName):
        torch.save(self.state_dict(), fileName)

    def load(self, fileName):
        self.load_state_dict(torch.load(fileName))

    def __init__(self, embed_size, hidden_size, word2ind, unkToken, padToken, endToken, lstm_layers, dropout):
        super(LSTMLanguageModelPack, self).__init__()
        #############################################################################
        ###  Тук следва да се имплементира инициализацията на обекта
        ###  За целта може да копирате съответния метод от програмата за упр. 13
        ###  като направите добавки за повече слоеве на РНН и dropout
        #############################################################################
        #### Начало на Вашия код.

        self.word2ind = word2ind
        self.unkTokenIdx = word2ind[unkToken]
        self.padTokenIdx = word2ind[padToken]
        self.endTokenIdx = word2ind[endToken]
        self.hidden_size = hidden_size
        self.lstm = torch.nn.LSTM(
            embed_size, hidden_size, lstm_layers, dropout=dropout)
        self.embed = torch.nn.Embedding(len(word2ind), embed_size)
        self.projection = torch.nn.Linear(hidden_size, len(word2ind))
        self.dropout = torch.nn.Dropout(dropout)

        #### Край на Вашия код
        #############################################################################

    def forward(self, source):
        #############################################################################
        ###  Тук следва да се имплементира forward метода на обекта
        ###  За целта може да копирате съответния метод от програмата за упр. 13
        ###  като направите добавка за dropout
        #############################################################################
        #### Начало на Вашия код.

        X = self.preparePaddedBatch(source)
        E = self.embed(X[:-1])
        source_lengths = [len(s)-1 for s in source]
        outputPacked, _ = self.lstm(torch.nn.utils.rnn.pack_padded_sequence(
            E, source_lengths, enforce_sorted=False))
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(outputPacked)

        Z = self.projection(self.dropout(output.flatten(0, 1)))
        Y_bar = X[1:].flatten(0, 1)
        Y_bar[Y_bar == self.endTokenIdx] = self.padTokenIdx
        H = torch.nn.functional.cross_entropy(
            Z, Y_bar, ignore_index=self.padTokenIdx)
        return H

        #### Край на Вашия код
        #############################################################################


startChar = 'ш'
endChar = 'щ'
unkChar = 'ь'
padChar = 'ъ'

# prepare
testCorpus, trainCorpus, char2id = utils.prepareData(
    corpusFileName, startChar, endChar, unkChar, padChar)
pickle.dump(testCorpus, open(testDataFileName, 'wb'))
pickle.dump(trainCorpus, open(trainDataFileName, 'wb'))
pickle.dump(char2id, open(char2idFileName, 'wb'))
print('Data prepared.')


# train
testCorpus = pickle.load(open(testDataFileName, 'rb'))
trainCorpus = pickle.load(open(trainDataFileName, 'rb'))
char2id = pickle.load(open(char2idFileName, 'rb'))

lm = model.LSTMLanguageModelPack(char_emb_size, hid_size, char2id, unkChar,
                                    padChar, endChar, lstm_layers=lstm_layers, dropout=dropout).to(device)
if len(sys.argv) > 2:
    lm.load(sys.argv[2])

optimizer = torch.optim.Adam(lm.parameters(), lr=learning_rate)
train.trainModel(trainCorpus, lm, optimizer, epochs, batchSize)
lm.save(modelFileName)
print('Model perplexity: ', train.perplexity(lm, testCorpus, batchSize))

# perplexity
testCorpus = pickle.load(open(testDataFileName, 'rb'))
char2id = pickle.load(open(char2idFileName, 'rb'))
lm = model.LSTMLanguageModelPack(char_emb_size, hid_size, char2id,
                                    unkChar, padChar, endChar, lstm_layers=lstm_layers, dropout=dropout)
lm.load(modelFileName)
print('Model perplexity: ', train.perplexity(lm, testCorpus, batchSize))


# generate
if len(sys.argv) > 2:
    seed = sys.argv[2]
else:
    seed = startChar

assert seed[0] == startChar

if len(sys.argv) > 3:
    temperature = float(sys.argv[3])
else:
    temperature = defaultTemperature

char2id = pickle.load(open(char2idFileName, 'rb'))
lm = model.LSTMLanguageModelPack(char_emb_size, hid_size, char2id,
                                    unkChar, padChar, endChar, lstm_layers=lstm_layers, dropout=dropout)
lm.load(modelFileName)
print(generator.generateCode(lm, char2id, seed, temperature=temperature))


def trainModel(trainCorpus, lm, optimizer, epochs, batchSize):
    idx = np.arange(len(trainCorpus), dtype='int32')
    lm.train()
    for epoch in range(epochs):
        np.random.shuffle(idx)
        for b in range(0, len(idx), batchSize):
            batch = [trainCorpus[i] for i in idx[b:min(b+batchSize, len(idx))]]
            H = lm(batch)
            optimizer.zero_grad()
            H.backward()
            optimizer.step()
            print("Epoch:", epoch, '/', epochs, ", Batch:", b //
                  batchSize, '/', len(idx) // batchSize, ", loss: ", H.item())


def perplexity(lm, testCorpus, batchSize):
    lm.eval()
    H = 0.
    c = 0
    for b in range(0, len(testCorpus), batchSize):
        batch = testCorpus[b:min(b+batchSize, len(testCorpus))]
        l = sum(len(s)-1 for s in batch)
        c += l
        with torch.no_grad():
            H += l * lm(batch)
    return math.exp(H/c)

#############################################################################
### Търсене и извличане на информация. Приложение на дълбоко машинно обучение
### Стоян Михов
### Зимен семестър 2020/2021
#############################################################################
###
### Упражнение 13
###
#############################################################################
