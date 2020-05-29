import pickle
import operator
import pandas as pd
import numpy as np
from pathlib import Path
from sense2vec import Sense2Vec
from collections import Counter
from sklearn.preprocessing import Normalizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

folder = input('Insert embedding folder : ')
name = input('Insert domain : ')
top_n = int(input('Insert N (top) : '))
most_similar_n = int(input('Insert n (most similar) : '))
selection_mode = 'SVM' if int(input('SVM? ')) == 1 else 'Similarity' 
if selection_mode == 'SVM':
    mode = 'Concatenated Word Vector' if int(input('Concatenate? ')) == 1 else 'Word Vector' 

pos_aspect = [line.rstrip() for line in open('labelled data/{}/full-aspect-{}'.format(name, name), 'r')]
pos_opinion = [line.rstrip() for line in open('labelled data/{}/full-opinion-{}'.format(name, name), 'r')]
neg_example = [line.rstrip() for line in open('labelled data/{}/full-neg-{}'.format(name, name), 'r')]
text_file = open('{}/cleaned-{}.s2v'.format(folder, name), 'r')
corpus = text_file.read()
text_file.close()

word_counter = Counter(corpus.split())
most_common = word_counter.most_common()
i = count = 0
most_common_list = []
while count < top_n:
    word = most_common[i][0]
    if word in pos_aspect or word in pos_opinion:
        most_common_list.append(word)
        count+=1
    i+=1
print(most_common_list)
print(len(most_common_list))

s2v = Sense2Vec().from_disk(folder)

if selection_mode == 'SVM':
    model_file = Path('{}/model.pkl'.format(folder))
    if model_file.is_file():
        most_similar = []
        for word in most_common_list:
            most_similar.extend(s2v.most_similar(word, n=most_similar_n))
        most_similar_words = [word[0] for word in most_similar]
        most_similar_words.extend(most_common_list)
        most_similar_vectors = [s2v[word] for word in most_similar_words]
        labels = []
        for word in most_similar_words:
            label = 1 if word in pos_aspect else 0 if word in neg_example else -1
            labels.append(label)
        test_df = pd.DataFrame({'Word': most_similar_words, 'Vector': most_similar_vectors, 'Label': labels})
        print(test_df)
        print(len(test_df))

        test_df['Vector'] = list(Normalizer().fit_transform(np.stack(test_df['Vector'], axis=0)))
        y_test = test_df['Label']
        with open('{}/model.pkl'.format(folder), 'rb') as file:
            svm = pickle.load(file)
        y_predict = svm.predict(list(test_df['Vector']))
        test_df['Predict'] = y_predict

        print(confusion_matrix(y_test, y_predict))
        print('SVC Accuracy:', accuracy_score(y_test, y_predict))
        print(classification_report(y_test, y_predict, digits=5)+'\n')

        test_length = top_n * most_similar_n
        domain = 'Restaurant' if name == 'resto' else 'Digital Camera' if name == 'camera' else 'Handphone'
        # with open('DSL/Top-N + Top-N Most Similar/Top-{}x{}/SVM/{}/{}/aspect-{}.pkl'.format(top_n, most_similar_n, mode, domain, name), 'wb') as aspect_file:
        with open('DSL/Top-N + Top-N Most Similar/Top-{}x{}/{}/aspect-{}.pkl'.format(top_n, most_similar_n, domain, name), 'wb') as aspect_file:    
            aspect_list = list(test_df['Word'].loc[(test_df['Label'] == 1) & (test_df['Predict'] == 1)])
            aspect_list = [word.strip().split('|')[0].replace('_', ' ') for word in aspect_list]
            print('Aspect with duplicates :', len(aspect_list))
            aspect_list = list(set(aspect_list))
            print('Aspect without duplicates :', len(aspect_list))
            pickle.dump(aspect_list, aspect_file)
            print()

        with open('DSL/Top-N + Top-N Most Similar/Top-{}x{}/{}/neg-{}.pkl'.format(top_n, most_similar_n, domain, name), 'wb') as neg_file:
            neg_list = list(test_df['Word'].loc[(test_df['Label'] == 0) & (test_df['Predict'] == 0)])
            neg_list = [word.strip().split('|')[0].replace('_', ' ') for word in neg_list]
            print('Other with duplicates :', len(neg_list))
            neg_list = list(set(neg_list))
            print('Other without duplicates :', len(neg_list))
            pickle.dump(neg_list, neg_file)
            print()

        with open('DSL/Top-N + Top-N Most Similar/Top-{}x{}/{}/opinion-{}.pkl'.format(top_n, most_similar_n, domain, name), 'wb') as opinion_file:
            opinion_list = list(test_df['Word'].loc[(test_df['Label'] == -1) & (test_df['Predict'] == -1)])
            opinion_list = [word.strip().split('|')[0].replace('_', ' ') for word in opinion_list]
            print('Opinion with duplicates :', len(opinion_list))
            opinion_list = list(set(opinion_list))
            print('Opinion without duplicates :', len(opinion_list))
            pickle.dump(opinion_list, opinion_file)
            print()
    else:
        print("Model hasn't been trained yet. Please train it using classify_with_top_n.py first")

else:
    aspects, opinions, negs, y_predict, y_test = [], [], [], [], []
    for word in most_common_list:
        most_similar = s2v.most_similar(word, n=most_similar_n)
        
        for x in most_similar:
            real_label = 'aspect' if x[0] in pos_aspect else 'opinion' if x[0] in pos_opinion else 'neg'
            similarities = {'aspect': s2v.similarity(pos_aspect, x[0]), 'opinion': s2v.similarity(pos_opinion, x[0]), 'neg': s2v.similarity(neg_example, x[0])}
            predicted = max(similarities.items(), key=operator.itemgetter(1))[0]
            
            y_test.append(real_label)
            y_predict.append(predicted)
            if predicted == 'aspect' and real_label == 'aspect':
                aspects.append(x[0] + ',' + word + ',' + str(x[1]))
            elif predicted == 'opinion' and real_label == 'opinion':
                opinions.append(x[0] + ',' + word + ',' + str(x[1]))
            elif predicted == 'neg' and real_label == 'neg':
                negs.append(x[0] + ',' + word + ',' + str(x[1]))

    print(confusion_matrix(y_test, y_predict))
    print('Similarity Accuracy:', accuracy_score(y_test, y_predict))
    print(classification_report(y_test, y_predict, digits=5)+'\n')
    # test_df['Predict'] = y_predict
    # print(test_df[['Word','Label','Predict']])    
    test_length = top_n * most_similar_n
    domain = 'Restaurant' if name == 'resto' else 'Digital Camera' if name == 'camera' else 'Handphone'
    with open('DSL/Top-N + Top-N Most Similar/Top-{}/Similarity/{}/aspect-{}.pkl'.format(test_length, domain, name), 'wb') as file:
        aspect_list = []
        for word in most_common_list:
            if word in pos_aspect:
                aspect_list.append(word)
        for aspect in aspects:
            aspect_list.append(aspect)
        aspect_list = [word.strip().split('|')[0].replace('_', ' ') for word in aspect_list]
        aspect_list = list(set(aspect_list))
        print('Aspect count :', len(aspect_list))
        pickle.dump(aspect_list, file)

    with open('DSL/Top-N + Top-N Most Similar/Top-{}/Similarity/{}/opinion-{}.pkl'.format(test_length, domain, name), 'wb') as file:
        opinion_list = []
        for word in most_common_list:
            if word in pos_opinion:
                opinion_list.append(word)
        for opinion in opinions:
            opinion_list.append(opinion)
        opinion_list = [word.strip().split('|')[0].replace('_', ' ') for word in opinion_list]
        opinion_list = list(set(opinion_list))
        print('Opinion count :', len(opinion_list))
        pickle.dump(opinion_list, file)

    with open('DSL/Top-N + Top-N Most Similar/Top-{}/Similarity/{}/neg-{}.pkl'.format(test_length, domain, name), 'wb') as file:
        neg_list = []
        for word in most_common_list:
            if word in neg_example:
                neg_list.append(word)
        for neg in negs:
            neg_list.append(neg)
        neg_list = [word.strip().split('|')[0].replace('_', ' ') for word in neg_list]
        neg_list = list(set(neg_list))
        print('Other count :', len(neg_list))
        pickle.dump(neg_list, file)