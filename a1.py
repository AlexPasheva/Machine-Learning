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
import nltk
#nltk.download('punkt')
#nltk.download('stopwords')
#nltk.download('wordnet')

corpus_typos = open('corpus_typos.txt').read()
corpus_original = open('corpus_original.txt').read()

alphabet = ['а', 'б', 'в', 'г', 'д', 'е', 'ж', 'з', 'и', 'й', 'к', 'л', 'м', 'н', 'о', 'п', 'р', 'с', 'т', 'у', 'ф', 'х', 'ц', 'ч', 'ш', 'щ', 'ъ', 'ь', 'ю', 'я']

from nltk.tokenize import sent_tokenize
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk import WordNetLemmatizer
def clean_text(text):
    """
    This function takes as input a text on which several 
    NLTK algorithms will be applied in order to preprocess it
    """
    tokens = word_tokenize(text)
    # Remove the punctuations
    tokens = [word for word in tokens if word.isalpha()]
    # Lower the tokens
    tokens = [word.lower() for word in tokens]
    # Remove stopword
    tokens = [word for word in tokens if not word in stopwords.words("english")]
    # Lemmatize
    lemma = WordNetLemmatizer()
    tokens = [lemma.lemmatize(word, pos = "v") for word in tokens]
    tokens = [lemma.lemmatize(word, pos = "n") for word in tokens]
    return tokens

# neshto si se obyrkal
def extractDictionary(corpus):
    dictionary = set()
    for doc in corpus:
        for w in doc:
            if w not in dictionary: dictionary.add(w)
    return dictionary


def levenshteinDistanceMatrix(s1, s2):
    n1 = len(s1)
    n2 = len(s2)
    d = np.zeros((n1 + 1, n2 + 1), dtype=int)
    for i in range(n1 + 1):
        d[i, 0] = i
    for j in range(n2 + 1):
        d[0, j] = j
    for i in range(n1):
        for j in range(n2):
            if s1[i] == s2[j]:
                cost = 0
            else:
                cost = 1
            d[i+1, j+1] = min(d[i, j+1] + 1, # insert
                              d[i+1, j] + 1, # delete
                              d[i, j] + cost) # replace
            if i > 0 and j > 0 and s1[i] == s2[j-1] and s1[i-1] == s2[j]:
                d[i+1, j+1] = min(d[i+1, j+1], d[i-1, j-1] + cost) # transpose
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
    #### функцията намира елементарни редакции, неободими за получаването на един низ от друг
    #### вход: низовете s1 и s2
    #### изход: списък с елементарните редакции ( идентитет, вмъкване, изтриване, субституция и транспоzиция на символи) необходими, за да се получи втория низ от първия
    
    #### Например: editOperations('котка', 'октава') би следвало да връща списъка:
    ####    [('ко', 'ок'), ('т','т'), ('', 'а'), ('к', 'в'), ('а','а')]
    ####        |ко   |т |   |к  |а |
    ####        |ок   |т |а  |в  |а |
    ####        |Trans|Id|Ins|Sub|Id|
    ####
    #### Можете да преизползвате и модифицирате кода на функцията editDistance
    #############################################################################
    #### Начало на Вашия код.
    
    distMatrix = levenshteinDistanceMatrix(s1, s2)
    i, j = distMatrix.shape
    i -= 1
    j -= 1
  
    ops = list()
    
    while i != 0 and j != 0:
        if i > 1 and j > 1 and s1[i-1] == s2[j-2] and s1[i-2] == s2[j-1]:
            if distMatrix[i-2, j-2] < distMatrix[i, j]:
                ops.insert(0, (s2[i-1] + s2[i-2], s1[i - 1] + s1[i-2])) # transpose
                i -= 2
                j -= 2
                continue
        
        index = np.argmin(
            [distMatrix[i-1, j-1], distMatrix[i, j-1], distMatrix[i-1, j]])
        
        if index == 0:
            if distMatrix[i, j] > distMatrix[i-1, j-1]:
                ops.insert(0, (s2[i - 1], s1[j - 1])) # replace
            else:
                ops.insert(0, (s1[i-1], s2[j - 1])) # identity
            i -= 1
            j -= 1
            
        elif index == 1:
            ops.insert(0, ("", s2[j - 1])) # insert
            j -= 1
        elif index == 2:
            ops.insert(0, (s1[i - 1], "")) # delete
            i -= 1
    return ops

    #### Край на Вашия код
    #############################################################################

def computeOperationProbs(corrected_corpus, uncorrected_corpus, smoothing = 0.2):
    #### Функцията computeOperationProbs изчислява теглата на дадени елементарни операции (редакции)
    #### Теглото зависи от конкретните символи. Използвайки корпусите, извлечете статистика. Използвайте принципа за максимално правдоподобие. Използвайте изглаждане. 
    #### Вход: Корпус без грешки, Корпус с грешки, параметър за изглаждане. С цел простота може да се счете, че j-тата дума в i-тото изречение на корпуса с грешки е на разстояние не повече от 2 (по Левенщайн-Дамерау) от  j-тата дума в i-тото изречение на корпуса без грешки.
    #### Следва да се използват функциите generateCandidates, editOperations, 
    #### Помислете как ще изберете кандидат за поправка измежду всички възможни.
    #### Важно! При изтриване и вмъкване се предполага, че празния низ е представен с ''
    #### Изход: Речник, който по зададена наредена двойка от низове връща теглото за операцията.
    
    #### Първоначално ще трябва да преброите и запишете в речника operations броя на редакциите от всеки вид нужни, за да се поправи корпуса с грешки. След това изчислете съответните вероятности.
    
    #probability of (op) = (0.2 + number of op in corpus) / (0.2 * len(operations) + sum(op in operations in corpus))
    
    operations = {} # Брой срещания за всяка елементарна операция + изглаждане
    operationsProb = {} # Емпирична вероятност за всяка елементарна операция
    for c in alphabet:
        operations[(c,'')] = smoothing    # deletions
        operations[('',c)] = smoothing    # insertions
        for s in alphabet:
            operations[(c,s)] = smoothing    # substitution and identity
            if c == s:    
                continue
            operations[(c+s,s+c)] = smoothing    # transposition

    
    #############################################################################
    #### Начало на Вашия код.
    
    print(len(uncorrected_corpus))
    print(len(corrected_corpus))
          
    for index in range(13530):
        print(uncorrected_corpus[index-1], corrected_corpus[index-1])
        result = editOperations(uncorrected_corpus[index-1], corrected_corpus[index-1])
        #print(uncorrected_corpus[index-1])
        for op in result:
            if op in operations:
                operations[op] += 1
        
    #print(operations)       

    

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

    pass

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

    # All edits that are one edit away from `word`.

    splits     = [(q[:i], q[i:])    for i in range(len(q) + 1)]
    deletes    = [L + R[1:]               for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in alphabet]
    inserts    = [L + c + R               for L, R in splits for c in alphabet]
    return set(deletes + transposes + replaces + inserts)

    #### Край на Вашия код
    #############################################################################

def generateCandidates(query,dictionary,operationProbs):
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
    #operationProbs = computeOperationProbs(corrected_corpus, uncorrected_corpus) - np.log(a,b,operationProbs)
    #print(words)
    resultWords = []
    for w in words:
        if w in dictionary:
            resultWords.append(w)
            
    return resultWords
    
if __name__== "__main__" :
    #clean_orig = clean_text(corpus_original)
    #clean_typos = clean_text(corpus_typos)
    #print(editDistance("иван", "иванн"))
    #print(editOperations("мария", "амрия"))
    #print(editOperations("иван", "иванн"))
    #print(editOperations("обича", "убичъ"))
    #print(editOperations("мн", "много"))
    #print(generateCandidates("дам", clean_text(corpus), {}))
    #computeOperationProbs(clean_text(corpus_original), clean_text(corpus_typos))
    #print(clean_orig[13530] + " " + clean_typos[13530])
    #print(clean_orig[13531] + " " + clean_typos[13531])
    #print(clean_orig[13532] + " " + clean_typos[13532])
    #print(clean_orig[13533] + " " + clean_typos[13533])
    #print(clean_orig[13534] + " " + clean_typos[13534])
    #print(clean_orig[13535] + " " + clean_typos[13535])
    print(editOperations("печици", "печиъи"))
    print(editOperations("им", "ми"))   
    print(editOperations("викат", "викат"))
    print(editOperations("печатница", "печатница"))
    print(editOperations("димитър", "димитър"))
    print(editOperations("благоев", "благоев"))
    
    
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

    pass

    #### Край на Вашия код
    #############################################################################
