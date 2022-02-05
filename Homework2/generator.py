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


def generateCode(model, char2id, startSentence, limit=1000, temperature=1.):
    # model е инстанция на обучен LSTMLanguageModelPack обект
    # char2id е речник за символите, връщащ съответните индекси
    # startSentence е началния низ стартиращ със символа за начало '{'
    # limit е горна граница за дължината на поемата
    # temperature е температурата за промяна на разпределението за следващ символ

    result = startSentence[1:]

    #############################################################################
    ### Тук следва да се имплементира генерацията на текста
    #############################################################################
    ### Начало на Вашия код.

    # Функция, която се опитва да предскаже всяка следваща буква
    id2char = dict(enumerate(char2id))
    if len(startSentence) < 2:
        startSentence += "def " # kinda hardcoded a word but it doesn't           
        result += "def "        # produce anything meaningful withou this in the beggining 

    def predict(model, source, h=None):

        X = model.preparePaddedBatch(source)
        E = model.embed(X)
        source_lengths = [len(s) for s in source]

        if h != None:
            output_packed, h = model.lstm(torch.nn.utils.rnn.pack_padded_sequence(
                E, source_lengths, enforce_sorted=False), h)
        else:
            output_packed, h = model.lstm(torch.nn.utils.rnn.pack_padded_sequence(
                E, source_lengths, enforce_sorted=False))
            
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output_packed)

        Z = model.projection(model.dropout(output.flatten(0, 1)))
        length = len(source) - 1
        p = torch.nn.functional.softmax(torch.div(Z, temperature), dim=1).data
        p, topChar = p.topk(32)
        topChar = topChar.numpy().squeeze()
        p = p[length].numpy().squeeze()
        
        if type(topChar[length]) is np.ndarray:
            t = np.random.choice(topChar[length], p=p / np.sum(p))
        else:
            t = np.random.choice(topChar, p=p / np.sum(p))
        return id2char[t], h

    # Проверяваме дали е въведена начална дума
    # Ако е въведена - добавяме space
    # Иначе започваме с "def"
    if(len(startSentence) == 1):
        chars = list(char2id.keys())
        alphabet = chars[1:126] 
        startSentence += np.random.choice(alphabet)
    else:
        startSentence += " "

    initWordSize = len(result)

    gibberish_function = [x for x in result]  # текущото състояние
    output, h = predict(model, gibberish_function)
    gibberish_function.append(output)
    model.eval()

    size = initWordSize
    while not output == 'щ' and size <= limit:
        output, h = predict(model, gibberish_function[size], h)
        gibberish_function.append(output)
        size = size + 1

    result = ""
    for ch in gibberish_function:
        result += ch

    #### Край на Вашия код
    #############################################################################

    return result
