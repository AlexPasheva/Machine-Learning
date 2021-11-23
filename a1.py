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

def distanceHelper(str1, str2, m, n):
  
    # If first string is empty, the only option is to
    # insert all characters of second string into first
    if m == 0:
        return n
  
    # If second string is empty, the only option is to
    # remove all characters of first string
    if n == 0:
        return m
  
    # If last characters of two strings are same, nothing
    # much to do. Ignore last characters and get count for
    # remaining strings.
    if str1[m-1] == str2[n-1]:
        return distanceHelper(str1, str2, m-1, n-1)
  
    # If last characters are not same, consider all three
    # operations on last character of first string, recursively
    # compute minimum cost for all three operations and take
    # minimum of three values.
    return 1 + min(distanceHelper(str1, str2, m, n-1),    # Insert
                   distanceHelper(str1, str2, m-1, n),    # Remove
                   distanceHelper(str1, str2, m-1, n-1)    # Replace
                   )
    
def editDistance(s1, s2):
    #### функцията намира разстоянието на Левенщайн-Дамерау между два низа
    #### вход: низовете s1 и s2
    #### изход: минималният брой на елементарните операции ( вмъкване, изтриване, субституция и транспоциция на символи) необходими, за да се получи от единия низ другия

    #############################################################################
    #### Начало на Вашия код. На мястото на pass се очакват 10-25 реда

    print(distanceHelper(s1, s2, len(s1), len(s2)))

    #### Край на Вашия код
    #############################################################################
    


def editOperationHelper(s1, s2, m, n, dictionary):
    m = len(s1)
    n = len(s2)
    #dictionary(zip('', s2))
    # If first string is empty, the only option is to
    # insert all characters of second string into first
    if m == 0:
        return {'': s2}
  
    # If second string is empty, the only option is to
    # remove all characters of first string
    if n == 0:
        return {s1: ''}
  
    # If last characters of two strings are same, nothing
    # much to do. Ignore last characters and get count for
    # remaining strings.
    # adsds adprd
    # {a: a}
    # {ad: ad} 
    if s1[m-1] == s2[n-1]:
        return editDistance(s1, s2, m-1, n-1, dictionary.indexOf())
  
    # If last characters are not same, consider all three
    # operations on last character of first string, recursively
    # compute minimum cost for all three operations and take
    # minimum of three values.
    return 1 + min(editDistance(s1, s2, m, n-1),    # Insert
                   editDistance(s1, s2, m-1, n),    # Remove
                   editDistance(s1, s2, m-1, n-1)    # Replace
                   )

 
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
    print(editOperationHelper(s1, s2, len(s1), len(s2), ""))
    #### Край на Вашия код
    #############################################################################
    
if __name__== "__main__" :
    editOperations("", "дъмбълдо")

def computeOperationProbs(corrected_corpus,uncorrected_corpus,smoothing = 0.2):
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
        operations[(c,'')] = smoothing    # deletions
        operations[('',c)] = smoothing    # insertions
        for s in alphabet:
            operations[(c,s)] = smoothing    # substitution and identity
            if c == s:    
                continue
            operations[(c+s,s+c)] = smoothing    # transposition

    #############################################################################
    #### Начало на Вашия код.

    pass

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

    pass

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

    pass

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
