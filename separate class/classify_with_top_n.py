import pickle
import pprint
import operator
import pandas as pd
import numpy as np
from sense2vec import Sense2Vec
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.svm import SVC, LinearSVC
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold, cross_validate
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import PCA
from itertools import chain
from pathlib import Path
from collections import Counter

pd.set_option('display.max_rows', None)

def predict(word):
    similarities = {1: s2v.similarity(pos_train_aspect, word), -1: s2v.similarity(pos_train_opinion, word), 0: s2v.similarity(neg_train_example, word)}
    return max(similarities.items(), key=operator.itemgetter(1))[0]

def get_nearest(s2v, word, aspect, opinion, neg):
    n_word = 50
    nearest_aspect, nearest_opinion, nearest_negative = None, None, None
    filled = False
    while not filled:
        most_similar = s2v.most_similar(word, n=n_word)
        # print()
        # print(most_similar)
        if n_word > 50:
            most_similar = most_similar[n_word-50:]
        for w in most_similar:
            if nearest_aspect is None and w[0] in aspect:
                nearest_aspect = w[0]
            elif nearest_opinion is None and w[0] in opinion:
                nearest_opinion = w[0]
            elif nearest_negative is None and w[0] in neg:
                nearest_negative = w[0]
            if nearest_aspect is not None and nearest_opinion is not None and nearest_negative is not None:
                filled = True
                break
        n_word+=50
    # print(word, nearest_aspect, nearest_opinion, nearest_negative)
    return s2v[nearest_aspect], s2v[nearest_opinion], s2v[nearest_negative]

def concatenate(vector, s2v, word, aspect, opinion, neg):
    # print(word)
    nearest_aspect_vec, nearest_opinion_vec, nearest_negative_vec = get_nearest(s2v, word, aspect, opinion, neg)
    # print(vector[0:10])
    vector = Normalizer().fit_transform([vector])
    # print(vector[0][0:10])
    nearest_aspect_vec = Normalizer().fit_transform([nearest_aspect_vec])
    nearest_opinion_vec = Normalizer().fit_transform([nearest_opinion_vec])
    nearest_negative_vec = Normalizer().fit_transform([nearest_negative_vec])
    concatenated_vector = np.concatenate((vector[0], nearest_aspect_vec[0], nearest_opinion_vec[0], nearest_negative_vec[0]))
    # print(len(concatenated_vector))
    return concatenated_vector

def average(vector, s2v, word, aspect, opinion, neg):
    nearest_aspect_vec, nearest_opinion_vec, nearest_negative_vec = get_nearest(s2v, word, aspect, opinion, neg)
    # print(vector[0:10])
    # vector = Normalizer().fit_transform([vector])
    # # print(vector[0][0:10])
    # nearest_aspect_vec = Normalizer().fit_transform([nearest_aspect_vec])
    # nearest_opinion_vec = Normalizer().fit_transform([nearest_opinion_vec])
    # nearest_negative_vec = Normalizer().fit_transform([nearest_negative_vec])
    # concatenated_vector = np.concatenate((vector[0], nearest_aspect_vec[0], nearest_opinion_vec[0], nearest_negative_vec[0]))
    averaged_vector = np.mean([nearest_aspect_vec, nearest_opinion_vec, nearest_negative_vec], axis=0)
    # print(len(averaged_vector))
    concatenated_vector = np.concatenate((vector, averaged_vector))
    return concatenated_vector

name = input('Please input domain name (camera/hp/resto) : ')
folder = input('Please input embedding folder name :')
selection_mode = 'SVM' if int(input('SVM? ')) == 1 else 'Similarity' 
if selection_mode == 'SVM':
    mode = 'Concatenated Word Vector' if int(input('Concatenate? ')) == 1 else 'Word Vector'
n = int(input('Insert N (max 1000): '))
text_file = open('{}/cleaned-{}.s2v'.format(folder, name), 'r')
corpus = text_file.read()
text_file.close()

word_counter = Counter(corpus.split())
most_common = word_counter.most_common(n)
most_common = [pair[0] for pair in most_common]

pos_train_aspect = [line.rstrip() for line in open('labelled data/{}/train-aspect-{}'.format(name, name), 'r')]
pos_train_opinion = [line.rstrip() for line in open('labelled data/{}/train-opinion-{}'.format(name, name), 'r')]
neg_train_example = [line.rstrip() for line in open('labelled data/{}/train-neg-{}'.format(name, name), 'r')]
pos_aspect = [line.rstrip() for line in open('labelled data/{}/aspect-{}'.format(name, name), 'r')]
pos_aspect = [word for word in pos_aspect if word in most_common]
pos_opinion = [line.rstrip() for line in open('labelled data/{}/opinion-{}'.format(name, name), 'r')]
pos_opinion = [word for word in pos_opinion if word in most_common]
neg_example = [line.rstrip() for line in open('labelled data/{}/neg-{}'.format(name, name), 'r')]
neg_example = [word for word in neg_example if word in most_common]

s2v = Sense2Vec().from_disk(folder)
pos_aspect_vectors = [s2v[word] for word in pos_aspect]
pos_train_aspect_vectors = [s2v[word] for word in pos_train_aspect]
# debug = [word for word in pos_aspect if s2v[word] is None]
# print(debug)
aspect_df = pd.DataFrame({'Word': pos_aspect, 'Vector': pos_aspect_vectors, 'Label': 1})
train_aspect_df = pd.DataFrame({'Word': pos_train_aspect, 'Vector': pos_train_aspect_vectors, 'Label': 1})
neg_vectors = [s2v[word] for word in neg_example]
train_neg_vectors = [s2v[word] for word in neg_train_example]
neg_df = pd.DataFrame({'Word': neg_example, 'Vector': neg_vectors, 'Label': 0})
train_neg_df = pd.DataFrame({'Word': neg_train_example, 'Vector': train_neg_vectors, 'Label': 0})
pos_opinion_vectors = [s2v[word] for word in pos_opinion]
pos_train_opinion_vectors = [s2v[word] for word in pos_train_opinion]
opinion_df = pd.DataFrame({'Word': pos_opinion, 'Vector': pos_opinion_vectors, 'Label': -1})
train_opinion_df = pd.DataFrame({'Word': pos_train_opinion, 'Vector': pos_train_opinion_vectors, 'Label': -1})
train_df = pd.concat([train_aspect_df, train_neg_df, train_opinion_df], ignore_index=True, sort=False)
test_df = pd.concat([aspect_df, neg_df, opinion_df], ignore_index=True, sort=False)
test_length = len(test_df)

if selection_mode == 'SVM':
    if mode == 'Word Vector':
        train_df['Vector'] = list(Normalizer().fit_transform(np.stack(train_df['Vector'], axis=0)))
        test_df['Vector'] = list(Normalizer().fit_transform(np.stack(test_df['Vector'], axis=0)))

        x_train = list(train_df['Vector'])
        x_test = list(test_df['Vector'])
    else:
        x_train_file = Path('pickled_data/{}_x_train_{}.pkl'.format(name, test_length))
        x_test_file = Path('pickled_data/{}_x_test_{}.pkl'.format(name, test_length))
        if x_train_file.is_file() and x_test_file.is_file():
            with open('pickled_data/{}_x_train_{}.pkl'.format(name, test_length), 'rb') as loaded_x_train:
                x_train = pickle.load(loaded_x_train)
            
            with open('pickled_data/{}_x_test_{}.pkl'.format(name, test_length), 'rb') as loaded_x_test:
                x_test = pickle.load(loaded_x_test)
        else:
            aspect = list(train_df.loc[train_df['Label'] == 1, 'Word'])
            opinion = list(train_df.loc[train_df['Label'] == -1, 'Word'])
            neg = list(train_df.loc[train_df['Label'] == 0, 'Word'])

            x_train = list(train_df.apply(lambda x: concatenate(x['Vector'], s2v, x['Word'], aspect, opinion, neg), axis=1))
            print('finished concatenating train similarities')

            with open('pickled_data/{}_x_train_{}.pkl'.format(name, test_length), 'wb') as x_train_file:
                pickle.dump(x_train, x_train_file)

            x_test = list(test_df.apply(lambda x: concatenate(x['Vector'], s2v, x['Word'], aspect, opinion, neg), axis=1))
            print('finished concatenating test similarities')

            with open('pickled_data/{}_x_test_{}.pkl'.format(name, test_length), 'wb') as x_test_file:
                pickle.dump(x_test, x_test_file)

    y_train = train_df['Label']
    y_test = test_df['Label']

    weights = compute_class_weight('balanced', np.unique(y_train), y_train)
    weights = dict(zip(np.unique(y_train), weights))

    # param_grid = {'C': [0.1, 1, 10, 100], 'kernel': ['linear', 'rbf']}
    model = SVC(class_weight=weights, random_state=0)
    # grid = GridSearchCV(model, param_grid, cv=5, verbose=3)
    # grid.fit(x_train, y_train)
    # print(grid.best_params_)
    model.fit(x_train, y_train)
    with open('{}/model.pkl'.format(folder), 'wb') as file:
        # pickle.dump(grid, file)
        pickle.dump(model, file)
    # y_predict = grid.predict(x_test)
    y_predict = model.predict(x_test)
    print(confusion_matrix(y_test, y_predict))
    print('SVC Accuracy:', accuracy_score(y_test, y_predict))
    print(classification_report(y_test, y_predict, digits=5)+'\n')
    test_df['Predict'] = y_predict
    # print(test_df[['Word','Label','Predict']])

    domain = 'Restaurant' if name == 'resto' else 'Digital Camera' if name == 'camera' else 'Handphone'
    with open('DSL/Top-N/Top-{}/SVM/{}/{}/aspect-{}.pkl'.format(test_length, mode, domain, name), 'wb') as aspect_file:
        aspect_list = list(test_df['Word'].loc[(test_df['Label'] == 1) & (test_df['Predict'] == 1)])
        aspect_list = [word.strip().split('|')[0].replace('_', ' ') for word in aspect_list]
        print('Aspect with duplicates :', len(aspect_list))
        aspect_list = list(set(aspect_list))
        print('Aspect without duplicates :', len(aspect_list))
        pickle.dump(aspect_list, aspect_file)
        print()

    with open('DSL/Top-N/Top-{}/SVM/{}/{}/neg-{}.pkl'.format(test_length, mode, domain, name), 'wb') as neg_file:
        neg_list = list(test_df['Word'].loc[(test_df['Label'] == 0) & (test_df['Predict'] == 0)])
        neg_list = [word.strip().split('|')[0].replace('_', ' ') for word in neg_list]
        print('Other with duplicates :', len(neg_list))
        neg_list = list(set(neg_list))
        print('Other without duplicates :', len(neg_list))
        pickle.dump(neg_list, neg_file)
        print()

    with open('DSL/Top-N/Top-{}/SVM/{}/{}/opinion-{}.pkl'.format(test_length, mode, domain, name), 'wb') as opinion_file:
        opinion_list = list(test_df['Word'].loc[(test_df['Label'] == -1) & (test_df['Predict'] == -1)])
        opinion_list = [word.strip().split('|')[0].replace('_', ' ') for word in opinion_list]
        print('Opinion with duplicates :', len(opinion_list))
        opinion_list = list(set(opinion_list))
        print('Opinion without duplicates :', len(opinion_list))
        pickle.dump(opinion_list, opinion_file)
        print()
else:
    y_test = test_df['Label']
    y_predict = test_df['Word'].apply(lambda x : predict(x))
    print(confusion_matrix(y_test, y_predict))
    print('Similarity Accuracy:', accuracy_score(y_test, y_predict))
    print(classification_report(y_test, y_predict, digits=5)+'\n')
    test_df['Predict'] = y_predict
    # print(test_df[['Word','Label','Predict']])

    domain = 'Restaurant' if name == 'resto' else 'Digital Camera' if name == 'camera' else 'Handphone'
    with open('DSL/Top-N/Top-{}/Similarity/{}/aspect-{}.pkl'.format(test_length, domain, name), 'wb') as aspect_file:
        aspect_list = list(test_df['Word'].loc[(test_df['Label'] == 1) & (test_df['Predict'] == 1)])
        aspect_list = [word.strip().split('|')[0].replace('_', ' ') for word in aspect_list]
        print('Aspect with duplicates :', len(aspect_list))
        aspect_list = list(set(aspect_list))
        print('Aspect without duplicates :', len(aspect_list))
        pickle.dump(aspect_list, aspect_file)
        print()

    with open('DSL/Top-N/Top-{}/Similarity/{}/neg-{}.pkl'.format(test_length, domain, name), 'wb') as neg_file:
        neg_list = list(test_df['Word'].loc[(test_df['Label'] == 0) & (test_df['Predict'] == 0)])
        neg_list = [word.strip().split('|')[0].replace('_', ' ') for word in neg_list]
        print('Other with duplicates :', len(neg_list))
        neg_list = list(set(neg_list))
        print('Other without duplicates :', len(neg_list))
        pickle.dump(neg_list, neg_file)
        print()

    with open('DSL/Top-N/Top-{}/Similarity/{}/opinion-{}.pkl'.format(test_length, domain, name), 'wb') as opinion_file:
        opinion_list = list(test_df['Word'].loc[(test_df['Label'] == -1) & (test_df['Predict'] == -1)])
        opinion_list = [word.strip().split('|')[0].replace('_', ' ') for word in opinion_list]
        print('Opinion with duplicates :', len(opinion_list))
        opinion_list = list(set(opinion_list))
        print('Opinion without duplicates :', len(opinion_list))
        pickle.dump(opinion_list, opinion_file)
        print()