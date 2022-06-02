import pandas as pd
import ast
from collections import Counter
import scipy
import csv

import eli5
import time
import string
import sklearn_crfsuite
from sklearn_crfsuite import metrics
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer
from sklearn.model_selection import train_test_split

from models.result_logs import ResultLogs
from models.tagging_systems import TaggingSystem


pd.options.mode.chained_assignment = None  # default='warn'


class CRFModel():

    def __init__(self, data_file_path, data_file_name, tagging_system, pos_tagging_used):
        self.data_file_path = data_file_path
        self.data_file_name = data_file_name
        self.tagging_system = tagging_system
        self.pos_tagging_used = pos_tagging_used
        self.cv_iterations = 10

        if tagging_system == TaggingSystem.IOB1:
            self.labels = ['I', 'B']
        elif tagging_system == TaggingSystem.IOB2:
            self.labels = ['I', 'B']
        elif tagging_system == TaggingSystem.BIOES:
            self.labels = ['I', 'B', 'E', 'S']
        else:
            self.labels = None

        if "extensive_preprocessing" in data_file_name:
            self.extensive_preprocessing = True
        else:
            self.extensive_preprocessing = False

        self.logs = []
        self.data = None
        self.model = None
        self.train = None
        self.test = None

        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.y_pred = None


    def read_data(self):
        """
        Reads the data from the csv file
        """

        self.data = pd.read_csv(self.data_file_path)  # .head(10)
        self.data[self.tagging_system.value] = self.data[self.tagging_system.value].apply(
            lambda x: ast.literal_eval(x))  # convert IOB column to list

        self.data['POS_tagging'] = self.data['POS_tagging'].apply(lambda x: ast.literal_eval(x))

        # concat aspect tagging with POS tagging in case POS tagging is used
        self.data['tags_concat'] = self.data.apply(
            lambda row: self.tags_concat(row[self.tagging_system.value], row['POS_tagging']), axis=1)


    def prepare_data(self):
        """
        Creates X_train, y_train, X_test and y_test sets
        """

        # If pos_tagging is True, then POS tag feature is also included, thus we use the
        # column with name 'tags_concat', that contains both the IOB tagging and the pos tagging
        if self.pos_tagging_used:
            self.train = self.train['tags_concat'].tolist()
        else:
            self.train = self.train[self.tagging_system.value].tolist()

        self.X_train = [self.sent2features(s) for s in self.train]
        self.y_train = [self.sent2labels(s) for s in self.train]


        if self.pos_tagging_used:
            self.test = self.test['tags_concat'].tolist()
        else:
            self.test = self.test[self.tagging_system.value].tolist()

        self.X_test = [self.sent2features(s) for s in self.test]
        self.y_test = [self.sent2labels(s) for s in self.test]


    def train_test_split_data(self):
        self.train, self.test = train_test_split(self.data, test_size=0.3, random_state=8, shuffle=True)


    def tags_concat(self, iob, pos):
        tags = []
        for el1, el2 in zip(iob, pos):
            tags.append((str(el1[0]), el1[1], el2[1]))
        return tags


    def sent2features(self, sent):
        return [self.word2features(sent, i) for i in range(len(sent))]


    def sent2labels(self, sent):
        if self.pos_tagging_used:
            return [label for token, label, postag in sent]
        else:
            return [label for token, label in sent]


    def sent2tokens(self, sent):
        return [token for token, label, postag in sent]


    def word2features(self, sent, i):
        """
        Creates the features for the CRF model
        """
        #print(sent)
        word = sent[i][0]

        features = {
            'bias': 1.0,
            'word.lower()': word.lower(),
            'word.isupper()': word.isupper(),
            'word.isdigit()': word.isdigit(),
            'word.istitle()': word.istitle(),
            'word[-3:]': word[-3:],
            'word[-2:]': word[-2:],
        }

        if self.pos_tagging_used:
            postag = sent[i][2]  # the POS tag
            features.update({
                'postag': postag
            })

        if self.extensive_preprocessing:
            features.update({
                'word.ispunctuation': (word in string.punctuation)
            })


        # For the next word
        if i > 0:
            word1 = sent[i - 1][0]

            features.update({
                '-1:word.lower()': word1.lower(),
                '-1:word.isupper()': word1.isupper(),
                '-1word.isdigit()': word1.isdigit(),
                '-1:word.istitle()': word1.istitle()
            })

            if self.pos_tagging_used:
                postag1 = sent[i - 1][2]
                features.update({
                    '-1:postag': postag1
                })

            if self.extensive_preprocessing:
                features.update({
                    '-1:word.ispunctuation': (word1 in string.punctuation)
                })

        else:
            features['BOS'] = True


        # For the previous word
        if i < len(sent) - 1:
            word1 = sent[i + 1][0]

            features.update({
                '+1:word.lower()': word1.lower(),
                '+1:word.isupper()': word1.isupper(),
                '+1:word.isdigit()': word1.isdigit(),
                '+1:word.istitle()': word1.istitle()
            })

            if self.pos_tagging_used:
                postag1 = sent[i + 1][2]
                features.update({
                    '+1:postag': postag1
                })

            if self.extensive_preprocessing:
                features.update({
                    '+1:word.ispunctuation': (word1 in string.punctuation)
                })

        else:
            features['EOS'] = True

        return features


    def fit(self):
        self.model.fit(self.X_train, self.y_train)


    def predict(self):
        self.y_pred = self.model.predict(self.X_test)
        f1 = metrics.flat_f1_score(self.y_test, self.y_pred,
                                   average='weighted', labels=self.labels)

        #print("\n\nF1-score: ", f1)
        self.logs.append("\n\nF1-score: " + str(f1))


    def print_classification_report(self):
        # group B and I results
        sorted_labels = sorted(
            self.labels,
            key=lambda name: (name[1:], name[0])
        )

        classification_report = (metrics.flat_classification_report(
            self.y_test, self.y_pred, labels=sorted_labels, digits=4
        ))

        #print(classification_report)
        self.logs.append('\n\n-------------------- Classification Report -------------------')
        self.logs.append('==============================================================')
        self.logs.append(str(classification_report))


    def print_eli5(self):
        expl = eli5.explain_weights(self.model, top=5, targets=self.labels.append('O'))
        # shw = eli5.show_weights(self.model, top=8, targets=self.labels.append('O'))
        # print(eli5.format_as_text(expl))

        self.logs.append('\n\n------------------------ ELI5 Weights ------------------------')
        self.logs.append('==============================================================')
        self.logs.append(eli5.format_as_text(expl))


    def create_eli5_html_visualization(self):
        html_obj = eli5.show_weights(self.model, top=20)

        html_file_name = '../../results/CRF/parts_1_2_3/ELI5/eli5_weights_' + self.data_file_name + '_' \
                         + self.tagging_system.value + '_' + str(self.pos_tagging_used) + '.html'

        # Write html object to a file (adjust file path; Windows path is used here)
        with open(html_file_name, 'wb') as f:
            f.write(html_obj.data.encode("UTF-8"))


    def print_transitions(self):
        self.logs.append('\n\n------------------------ Transitions -------------------------')
        self.logs.append('==============================================================')

        self.logs.append("\nTop likely transitions:")
        trans_features = Counter(self.model.transition_features_).most_common(20)
        for (label_from, label_to), weight in trans_features:
            self.logs.append("%-6s -> %-7s %0.6f" % (label_from, label_to, weight))

        self.logs.append("\nTop unlikely transitions:")
        trans_features = Counter(self.model.transition_features_).most_common()[-20:]
        for (label_from, label_to), weight in trans_features:
            self.logs.append("%-6s -> %-7s %0.6f" % (label_from, label_to, weight))


    def print_state_features(self):
        self.logs.append('\n\n----------------------- State Features -----------------------')
        self.logs.append('==============================================================')

        self.logs.append("\nTop positive:")
        state_features = Counter(self.model.state_features_).most_common(30)
        for (attr, label), weight in state_features:
            self.logs.append("%0.6f %-8s %s" % (weight, label, attr))

        self.logs.append("\nTop negative:")
        state_features = Counter(self.model.state_features_).most_common()[-30:]
        for (attr, label), weight in state_features:
            self.logs.append("%0.6f %-8s %s" % (weight, label, attr))


    def run_crf_grid_search(self):

        self.model = sklearn_crfsuite.CRF(
            max_iterations=100,
            all_possible_transitions=True
        )

        params_space = [
            {'algorithm': ['lbfgs'], 'c1': scipy.stats.expon(scale=0.5), 'c2': scipy.stats.expon(scale=0.05),
             'linesearch': ['MoreThuente', 'Backtracking', 'StrongBacktracking']},
            {'algorithm': ['pa'], 'pa_type': [0, 1, 2]},
            {'algorithm': ['l2sgd'], 'c2': scipy.stats.expon(scale=0.05)},
            {'algorithm': ['ap', 'arow']}
        ]

        # use the same metric for evaluation
        f1_scorer = make_scorer(metrics.flat_f1_score,
                                average='weighted', labels=self.labels)

        # grid search
        rs = RandomizedSearchCV(self.model, params_space,
                                cv=self.cv_iterations,
                                verbose=1,
                                n_jobs=-1,
                                n_iter=50,
                                scoring=f1_scorer)

        # fit the model
        rs.fit(self.X_train, self.y_train)

        # access the classifier with the best parameters
        self.model = rs.best_estimator_

        # get the results
        self.logs.append('\n\n-------------------- Grid Search Results ---------------------')
        self.logs.append('==============================================================')
        self.logs.append('\nBest parameters: ' + str(rs.best_params_))
        self.logs.append('Best CV score: ' + str(rs.best_score_))
        self.logs.append('Model size: {:0.2f}M'.format(rs.best_estimator_.size_ / 1000000))



    def train_model(self):
        start_time = time.time()

        self.read_data()
        self.logs.append('\n==============================================================')
        self.logs.append('==============================================================')
        self.logs.append('\n---------------------------- Info ----------------------------')
        self.logs.append('==============================================================')
        self.logs.append('\nFile: ' + self.data_file_path)
        self.logs.append('Total dataframe rows: ' + str(self.data.shape[0]))
        self.logs.append('Tagging system: ' + self.tagging_system.value)
        self.logs.append('POS tagging applied: ' + str(self.pos_tagging_used))
        self.logs.append(str(self.cv_iterations) +'-fold Cross Validation ')


        self.train_test_split_data()
        self.prepare_data()
        self.run_crf_grid_search()
        self.predict()
        self.print_classification_report()
        self.print_eli5()
        self.print_transitions()
        self.print_state_features()
        # self.create_eli5_html_visualization()


        elapsed_time = time.time() - start_time
        self.logs.append('\n------------------------')
        self.logs.append('Elapsed time: ' + time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
        self.logs.append('------------------------')


    def train_custom_model(self):
        """
        Runs a custom experiment in which the testing data derive from Part 1 only!!!
        """
        start_time = time.time()

        self.read_data()
        self.logs.append('\n==============================================================')
        self.logs.append('==============================================================')
        self.logs.append('\n---------------------------- Info ----------------------------')
        self.logs.append('==============================================================')
        self.logs.append('\nFile: ' + self.data_file_path)
        self.logs.append('Total dataframe rows: ' + str(self.data.shape[0]))
        self.logs.append('Tagging system: ' + self.tagging_system.value)
        self.logs.append('POS tagging applied: ' + str(self.pos_tagging_used))
        self.logs.append(str(self.cv_iterations) +'-fold Cross Validation ')


        # CUSTOM train test split data
        # random state is a seed value
        # the first 2400 rows derive from Part 1
        self.test = self.data.head(2400).sample(frac=0.4, random_state=200)
        self.train = self.data.drop(self.test.index)
        print(len( self.test))
        print(len(self.train))


        self.prepare_data()
        self.run_crf_grid_search()
        self.predict()
        self.print_classification_report()
        self.print_eli5()
        self.print_transitions()
        self.print_state_features()
        self.create_eli5_html_visualization()


        elapsed_time = time.time() - start_time
        self.logs.append('\n------------------------')
        self.logs.append('Elapsed time: ' + time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
        self.logs.append('------------------------')



