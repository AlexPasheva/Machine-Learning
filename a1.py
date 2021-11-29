#############################################################################
### Търсене и извличане на информация. Приложение на дълбоко машинно обучение
### Стоян Михов
### Зимен семестър 2021/2022
#############################################################################

### Домашно задание 1
###
### Преди да се стартира програмата е необходимо да се активира съответното обкръжение с командата:
### conda activate tii
###
### Ако все още нямате създадено обкръжение прочетете файла README.txt за инструкции


########################################
import numpy as np
import random

alphabet = ['а', 'б', 'в', 'г', 'д', 'е', 'ж', 'з', 'и', 'й', 'к', 'л', 'м', 'н', 'о', 'п', 'р', 'с', 'т', 'у', 'ф', 'х', 'ц', 'ч', 'ш', 'щ', 'ъ', 'ь', 'ю', 'я']

def extractDictionary(corpus):
    dictionary = set()
    for doc in corpus:
        for w in doc:
            if w not in dictionary: dictionary.add(w)
    return dictionary


def levenshteinDistanceMatrix(s1, s2):
    cols = len(s1)
    rows = len(s2)
    d = np.zeros((cols + 1, rows + 1), dtype=int)
    for i in range(cols + 1):
        d[i, 0] = i
    for j in range(rows + 1):
        d[0, j] = j
    for i in range(cols):
        for j in range(rows):
            if s1[i] == s2[j]:
                cost = 0
            else:
                cost = 1
            d[i+1, j+1] = min(d[i, j+1] + 1,                             # insert
                              d[i+1, j] + 1,                             # delete
                              d[i, j] + cost)                            # substitute
            if i > 0 and j > 0 and s1[i] == s2[j-1] and s1[i-1] == s2[j]:
                d[i+1, j+1] = min(d[i+1, j+1], d[i-1, j-1] + cost)       # transpose
    return d
    
def editDistance(s1, s2):
    #### функцията намира разстоянието на Левенщайн-Дамерау между два низа
    #### вход: низовете s1 и s2
    #### изход: минималният брой на елементарните операции ( вмъкване, изтриване, субституция и транспоциция на символи) необходими, за да се получи от единия низ другия

    #############################################################################
    #### Начало на Вашия код. На мястото на pass се очакват 10-25 реда
    
    resultMatrix = levenshteinDistanceMatrix(s1, s2)
    i,j=resultMatrix.shape
    
    return resultMatrix[i-1][j-1]

    #### Край на Вашия код
    #############################################################################
 
def editOperations(s1, s2):
    # функцията намира елементарни редакции, неободими за получаването на един низ от друг
    # вход: низовете s1 и s2
    # изход: списък с елементарните редакции ( идентитет, вмъкване, изтриване, субституция и транспоzиция на символи) необходими, за да се получи втория низ от първия

    # Например: editOperations('котка', 'октава') би следвало да връща списъка:
    ####    [('ко', 'ок'), ('т','т'), ('', 'а'), ('к', 'в'), ('а','а')]
    # |ко   |т |   |к  |а |
    # |ок   |т |а  |в  |а |
    # |Trans|Id|Ins|Sub|Id|
    ####
    # Можете да преизползвате и модифицирате кода на функцията editDistance
    #############################################################################
    # Начало на Вашия код.

    distMatrix = levenshteinDistanceMatrix(s1, s2)
    i, j = distMatrix.shape
    i -= 1
    j -= 1

    ops = list()

    while i != 0 and j != 0:
        if i > 1 and j > 1 and s1[i-1] == s2[j-2] and s1[i-2] == s2[j-1]:
            if distMatrix[i-2, j-2] < distMatrix[i, j]:
                # transpose
                ops.insert(
                    0, (s1[i - 2] + s1[i - 1], s1[i - 1] + s1[i - 2]))
                i -= 2
                j -= 2
                continue

        index = np.argmin(
            [distMatrix[i-1, j-1], distMatrix[i, j-1], distMatrix[i-1, j]])

        if index == 0:
            if distMatrix[i, j] > distMatrix[i-1, j-1]:
                ops.insert(0, (s1[i - 1], s2[j - 1]))     # replace
            else:
                ops.insert(0, (s1[i - 1], s2[j - 1]))     # substitute
            i -= 1
            j -= 1

        elif index == 1:
            ops.insert(0, ("", s2[j - 1]))                # insert
            j -= 1
        elif index == 2:
            ops.insert(0, (s1[i - 1], ""))                # delete
            i -= 1
    return ops

    # Край на Вашия код
    #############################################################################


def flatten(listOfLists):
    out = []
    for sublist in listOfLists:
        out.extend(sublist)
    return out
    

def computeOperationProbs(corrected_corpus, uncorrected_corpus, smoothing = 0.2):
    #### Функцията computeOperationProbs изчислява теглата на дадени елементарни операции (редакции)
    #### Теглото зависи от конкретните символи. Използвайки корпусите, извлечете статистика. Използвайте принципа за максимално правдоподобие. Използвайте изглаждане. 
    #### Вход: Корпус без грешки, Корпус с грешки, параметър за изглаждане. С цел простота може да се счете, че j-тата дума в i-тото изречение на корпуса с грешки е на разстояние не повече от 2 (по Левенщайн-Дамерау) от  j-тата дума в i-тото изречение на корпуса без грешки.
    #### Следва да се използват функциите generateCandidates, editOperations, 
    #### Помислете как ще изберете кандидат за поправка измежду всички възможни.
    #### Важно! При изтриване и вмъкване се предполага, че празния низ е представен с ''
    #### Изход: Речник, който по зададена наредена двойка от низове връща теглото за операцията.
    
    #### Първоначално ще трябва да преброите и запишете в речника operations броя на редакциите от всеки вид нужни, за да се поправи корпуса с грешки. След това изчислете съответните вероятности.
    
    operations = {} # Брой срещания за всяка елементарна операция + изглаждане
    operationsProb = {} # Емпирична вероятност за всяка елементарна операция
    for c in alphabet:
        operations[(c,'')] = smoothing                   # deletions
        operations[('',c)] = smoothing                   # insertions
        for s in alphabet:
            operations[(c,s)] = smoothing                # substitution and identity
            if c == s:    
                continue
            operations[(c+s,s+c)] = smoothing            # transposition

    #############################################################################
    #### Начало на Вашия код.
    
    corrected_corpus_words = flatten(corrected_corpus)
    uncorrected_corpus_words = flatten(uncorrected_corpus)
          
    for index in range(len(uncorrected_corpus_words)):
        result = editOperations(uncorrected_corpus_words[index-1], corrected_corpus_words[index-1])
        for op in result:
            if op in operations:
                operations[op] += 1
    
    keyList = list(operations.keys())
    valList = list(operations.values())
    Op = len(operations)
    for index in range(Op):
        op = keyList[index]
        operationsProb[op] = valList[index] / sum(valList)

    #### Край на Вашия код.
    #############################################################################
    return operationsProb

def operationWeight(a,b,operationProbs):
    #### Функцията operationWeight връща теглото на дадена елементарна операция
    #### Вход: Двата низа a,b, определящи операцията.
    ####       Речник с вероятностите на елементарните операции.
    #### Важно! При изтриване и вмъкване се предполага, че празния низ е представен с ''
    #### изход: Теглото за операцията
    
    if (a,b) in operationProbs.keys():
        return -np.log(operationProbs[(a,b)])
    else:
        print("Wrong parameters ({},{}) of operationWeight call encountered!".format(a,b))


def editWeight(s1, s2, operationProbs):
    #### функцията editWeight намира теглото между два низа
    #### За намиране на елеметарните тегла следва да се извиква функцията operationWeight
    #### вход: низовете s1 и s2 и речник с вероятностите на елементарните операции.
    #### изход: минималното тегло за подравняване, за да се получи втория низ от първия низ
    
    #############################################################################
    #### Начало на Вашия код. На мястото на pass се очакват 10-25 реда

    cols  = len(s1) + 1
    rows = len(s2) + 1
    d = np.zeros((rows , cols))

    for i in range(1,cols):
        d[0][i] = d[0][i - 1] + operationWeight('', s1[i-1], operationProbs)

    for i in range(1,rows):
        d[i][0] = d[i - 1][0] + operationWeight(s2[i - 1], '', operationProbs)
    
    for i in range(1, rows):
        for j in range(1, cols):
            d[i][j] = min( d[i-1][j-1] + operationWeight(s2[i-1], s1[j-1] , operationProbs),
                            d[i-1][j] + operationWeight(s2[i - 1], '', operationProbs),
                            d[i][j-1] + operationWeight('', s1[j - 1], operationProbs)
            )

            if i > 1 and j > 1 and s1[j - 2] == s2[i - 1] and s1[j - 1] == s2[i - 2]:
                swap_op = d[i - 2][j - 2] + operationWeight(s1[j-2:j], s2[i-2:i], operationProbs)
                d[i][j] = min(d[i][j], swap_op)
    
    return d[rows - 1][cols - 1]

    #### Край на Вашия код. 
    #############################################################################

def generateEdits(q):
    ### помощната функция, generateEdits по зададена заявка генерира всички възможни редакции на разстояние едно от тази заявка.
    ### Вход: заявка като низ q
    ### Изход: Списък от низове на разстояние 1 по Левенщайн-Дамерау от заявката
    ###
    ### В тази функция вероятно ще трябва да използвате азбука, която е дефинирана с alphabet
    ###
    #############################################################################
    #### Начало на Вашия код. На мястото на pass се очакват 10-15 реда

    splits     = [(q[:i], q[i:])             for i in range(len(q) + 1)]
    deletes    = [L + R[1:]                  for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:]    for L, R in splits if len(R)>1]
    replaces   = [L + c + R[1:]              for L, R in splits if R for c in alphabet if c != R[0]]
    inserts    = [L + c + R                  for L, R in splits for c in alphabet]
    return set(deletes + transposes + replaces + inserts)

    #### Край на Вашия код
    #############################################################################

def generateCandidates(query, dictionary, operationProbs):
    ### Започва от заявката query и генерира всички низове НА РАЗСТОЯНИЕ <= 2, за да се получат кандидатите за корекция. Връщат се единствено кандидати, които са в речника dictionary.
        
    ### Вход:
    ###     Входен низ query
    ###     Речник с допустими (правилни) думи: dictionary
    ###     речник с вероятностите на елементарните операции.

    ### Изход:
    ###     Списък от двойки (candidate, candidate_edit_log_probability), където candidate е низ на кандидат, а candidate_edit_log_probability е логаритъм от вероятността за редакция -- минус теглото.
    
    #############################################################################
    #### Начало на Вашия код. На мястото на pass се очакват 10-15 реда
    
    words = set (e2 for e1 in generateEdits(query) for e2 in generateEdits(e1))
    
    resultWords = []
    for w in words:
        if w in dictionary:
            resultWords.append((w, editWeight(query, w, operationProbs)))
            
    return resultWords 

    #### Край на Вашия код
    #############################################################################

def correctSpelling(r, dictionary, operationProbs):
    ### Функцията поправя корпус съдържащ евентуално сгрешени думи
    ### Генераторът на кандидати връща и вероятността за редактиране.
    ###
    ### Вход:
    ###    заявка: r - корпус от думи
    ###    речник с правилни думи: dictionary,
    ###    речник с вероятностите на елементарните операции: operationProbs
    ### 
    ### Изход: поправен корпус

    #############################################################################
    #### Начало на Вашия код. На мястото на pass се очакват 5-15 реда

    correctedQuery = []

    for sentence in r:
        for i, w in enumerate(sentence):
            if str(w).isalpha():
                candidates = dict(generateCandidates(str(w), dictionary, operationProbs))
                correctedWord = min(candidates, key=candidates.get)
                sentence[i] = correctedWord
        correctedQuery.append(sentence)
            
    return correctedQuery
    #### Край на Вашия код
    #############################################################################
    