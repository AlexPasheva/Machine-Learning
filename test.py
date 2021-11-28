#############################################################################
### Търсене и извличане на информация. Приложение на дълбоко машинно обучение
### Стоян Михов
### Зимен семестър 2021/2022
#############################################################################
### За да работи програмата трябва корпусите corpus_original.txt, corpus_typos.txt и corpus_correct.txt да се намират в директорията, в която е програмата.

import a1 as a1
import nltk
nltk.download('punkt')
from nltk.corpus import PlaintextCorpusReader
import pickle

#############################################################################
#### Начало на тестовете
#### ВНИМАНИЕ! Тези тестове са повърхностни и тяхното успешно преминаване е само предпоставка за приемането, но не означава задължително, че програмата Ви ще бъде приета. За приемане на заданието Вашата програма ще бъде подложена на по-задълбочена серия тестове.
#############################################################################

L1 = ['заявката','заявката','заявката','заявката','заявката']       #'заявката', ne me kefqt
L2 = ['заявката','заявьата','завякатва','вя','язвката']             #'заявкатаа',
C = [0,1,2,7,3] #, 1
O = [[('з', 'з'),  ('а', 'а'),  ('я', 'я'),  ('в', 'в'),  ('к', 'к'),  ('а', 'а'),  ('т', 'т'),  ('а', 'а')], [('з', 'з'),  ('а', 'а'),  ('я', 'я'),  ('в', 'в'),  ('к', 'ь'),  ('а', 'а'),  ('т', 'т'),  ('а', 'а')], [('з', 'з'),  ('а', 'а'),  ('яв', 'вя'),  ('к', 'к'),  ('а', 'а'),  ('т', 'т'),  ('', 'в'),  ('а', 'а')]] #[('з', 'з'),  ('а', 'а'),  ('я', 'я'),  ('в', 'в'),  ('к', 'к'),  ('а', 'а'),  ('т', 'т'),  ('', 'а'),  ('а', 'а')]
D = [22.75, 34.06, 35.93, 62.03, 43.71] # 32.02, kopie ot vtori arg 32.06

#### Тест на editDistance
for s1,s2,d in zip(L1,L2,C):
    assert a1.editDistance(s1,s2) == d, "Разстоянието между '{}' и '{}' следва да е '{}'".format(s1,s2,d)
print("Функцията editDistance премина теста.")

##### Тест на editOperations
for s1,s2,o in zip(L1,L2,O):
    assert a1.editOperations(s1,s2) == o, "Операциите редактиращи '{}' до '{}' следва да са '{}'".format(s1,s2,o)
print("Функцията editOperations премина теста.")

print('Прочитане на корпуса от текстове...')
corpus_root = '.'
original = PlaintextCorpusReader(corpus_root, 'corpus_original.txt')
fullSentCorpusOriginal = [[w.lower() for w in sent] for sent in original.sents()]
typos = PlaintextCorpusReader(corpus_root, 'corpus_typos.txt')
fullSentCorpusTypos = [[w.lower() for w in sent] for sent in typos.sents()]
print('Готово.')


#### Тест на computeOperationProbs

print('Пресмятане вероятностите на елементарните операции...')
operationProbs = a1.computeOperationProbs(fullSentCorpusOriginal,fullSentCorpusTypos)
print('Готово.')
ps = [operationProbs[k] for k in operationProbs.keys()]

assert max(ps) < 0.2, "Не би следвало  да има елементарна операция с толкова голяма вероятност."
assert min(ps) > 0, "Използвайте изглаждане."
id_prob = 0
for k in operationProbs.keys():
    if k[0]==k[1]:
        id_prob += operationProbs[k]
assert id_prob > 0.95, "Би следвало операцията идентитет да има най-голяма вероятност."
print("Функцията computeOperationProbs премина теста.")

print("Запис на вероятностите във файл probabilities.pkl ...")
opfile = open("probabilities.pkl", "wb")
pickle.dump(operationProbs, opfile)
opfile.close()
print('Готово.')


#### Тест на editWeight
for s1,s2,d in zip(L1,L2,D):
    assert abs(a1.editWeight(s1,s2,operationProbs) - d) <  1, "Теглото между '{}' и '{}' следва да е приблизително '{}'".format(s1,s2,d)
print("Функцията editWeight премина теста.")

#### Тест на generate_edits
assert len(set(a1.generateEdits("тест"))) == 269, "Броят на елементарните редакции \"тест\"  следва да е 269"
print("Функцията generateEdits премина теста.")

dictionary = a1.extractDictionary(fullSentCorpusOriginal)
#### Тест на generate_candidates
a1.generateCandidates("такяива", dictionary, operationProbs)
assert len(set(a1.generateCandidates("такяива",dictionary,operationProbs))) == 4, "Броят на генерираните кандидати следва да е 4"
print("Функцията generateCandidates премина теста.")

#### Тест на correct_spelling
print(a1.correctSpelling(fullSentCorpusTypos[3668:3669], dictionary, operationProbs))
corr = a1.correctSpelling(fullSentCorpusTypos[3668:3669],dictionary,operationProbs)
assert ' '.join(corr[0]) == 'третата група ( нареченската ) бе ударила на камък : поради курортния характер на селото пръчовете от наречен били още миналата година премахнати , защото замърсявали околната среда със силната си миризма и създавали у чужденците впечатление за първобитност .', "Коригираната заявка следва да е 'третата група ( нареченската ) бе ударила на камък : поради курортния характер на селото пръчовете от наречен били още миналата година премахнати , защото замърсявали околната среда със силната си миризма и създавали у чужденците впечатление за първобитност .'."
print("Функцията correctSpelling премина теста.")

correct = PlaintextCorpusReader(corpus_root, 'corpus_correct.txt')
fullSentCorpusCorrect = [[w.lower() for w in sent] for sent in correct.sents()]
corpus_corrected = a1.correctSpelling(fullSentCorpusCorrect,dictionary,operationProbs)
corpus_corrected = '\n'.join(' '.join(s) for s in corpus_corrected)
with open('corrected.txt', 'w') as f:
    f.write(corpus_corrected)