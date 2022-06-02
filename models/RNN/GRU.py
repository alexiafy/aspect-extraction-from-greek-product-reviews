import warnings
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.layers import Dropout, Bidirectional, GRU
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn_crfsuite import metrics
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import ast
import os

# from crf_layer import CRF
from models.RNN.crf_layer.crf_layer import CRF
from models.tagging_schemes import TaggingSystem
from embeddings.word2vec_embeddings import *


warnings.filterwarnings("ignore")


class GRUClassifier:

    def __init__(self, model_parameters, FILENAME, RESULTS_FOLDER_PATH, tagging_system):
        self.FILENAME = FILENAME
        self.RESULTS_FOLDER_PATH = RESULTS_FOLDER_PATH

        self.EMBEDDING_SIZE = model_parameters['EMBEDDING_SIZE']
        self.use_bidirectional = model_parameters['Bidirectional']
        self.use_crf_layer = model_parameters['crf_layer']

        try:
            self.n_units = model_parameters['n_units']
        except:
            self.n_units = 128

        try:
            self.batch_size = model_parameters['batch_size']
        except:
            self.batch_size = 32

        try:
            self.dropout_rate = model_parameters['dropout_rate']
        except:
            self.dropout_rate = 0.4

        try:
            self.epochs = model_parameters['epochs']
        except:
            self.dropout_rate = 40


        self.VOCABULARY_SIZE = 0
        self.MAX_SEQ_LENGTH = 0
        self.n_tags = 0
        self.embedding_weights = None

        self.tagging_system = tagging_system

        self.model = None
        self.history = None
        self.word_tokenizer = None

        self.data = None
        self.X_train = None
        self.Y_train = None
        self.X_test = None
        self.Y_test = None
        self.Y_pred = None
        self.X_validation = None
        self.Y_validation = None

        self.logs = []

        exp_params = ['_Bi' if self.use_bidirectional else '',
                      '_GRU']

        if self.tagging_system == TaggingSystem.IOB1:
            tag_scheme_str = 'IOB1'
        elif self.tagging_system == TaggingSystem.IOB2:
            tag_scheme_str = 'IOB2'
        elif self.tagging_system == TaggingSystem.BIOES:
            tag_scheme_str = 'BIOES'
        elif self.tagging_system == TaggingSystem.IOB1SentimentC3:
            tag_scheme_str = 'IOB1SentC3'
        elif self.tagging_system == TaggingSystem.IOB1SentimentC5:
            tag_scheme_str = 'IOB1SentC5'
        elif self.tagging_system == TaggingSystem.IOB2SentimentC3:
            tag_scheme_str = 'IOB2SentC3'
        elif self.tagging_system == TaggingSystem.IOB2SentimentC5:
            tag_scheme_str = 'IOB2SentC5'


        self.experiment_descr = self.FILENAME + '_' + tag_scheme_str + \
            ''.join(exp_params) + '_' + \
            str(self.batch_size) + 'batch_' + \
            str(self.n_units) + 'units_' + \
            str(self.dropout_rate) + 'dropout'


    def load_data(self, FOLDERNAME):
        self.data = pd.read_csv(FOLDERNAME + self.FILENAME + ".csv")  # .head(10)
        self.data[self.tagging_system.value] = self.data[self.tagging_system.value].apply(
            lambda x: ast.literal_eval(x))  # convert IOB column to list

        self.logs.append('\n==============================================================')
        self.logs.append('==============================================================')
        self.logs.append('\n---------------------------- Info ----------------------------')
        self.logs.append('==============================================================')
        self.logs.append('\nFile: ' + FOLDERNAME + self.FILENAME + ".csv")
        self.logs.append('Total dataframe rows: ' + str(self.data.shape[0]))
        self.logs.append('Tagging system: ' + self.tagging_system.value)


    def prepare_data(self):

        # Prepare Data
        # --------------------------------------------
        sentences = [sent for sent in self.data[self.tagging_system.value]]
        words = [word_tag_pair[0] for sent in self.data[self.tagging_system.value] for word_tag_pair in sent]
        self.tags = set([word_tag_pair[1] for sent in sentences for word_tag_pair in sent])  # get the unique tags

        n_unique_words = len(set([word.lower() for word in words]))
        n_words = len(words)
        self.n_tags = len(self.tags)

        print("Total number of tagged sentences: {}".format(len(sentences)))
        print("Total number of words: {}".format(n_words))
        print("Total number of unique words: {}".format(n_unique_words))
        print("Total number of tags: {}".format(self.n_tags))

        self.logs.append("Total number of tagged sentences: {}".format(len(sentences)))
        self.logs.append("Total number of words: {}".format(n_words))
        self.logs.append("Total number of unique words: {}".format(n_unique_words))
        self.logs.append("Total number of tags: {}".format(self.n_tags))
        # --------------------------------------------



        # Plot Histogram
        # --------------------------------------------
        # plt.hist([len(s) for s in sentences], bins=50)
        # plt.show()
        # --------------------------------------------



        # Create dictionaries
        # --------------------------------------------
        word2idx = {w: i for i, w in enumerate(words)}
        tag2idx = {t: i for i, t in enumerate(self.tags)}
        self.idx2tag = {i: w for w, i in tag2idx.items()}
        # --------------------------------------------



        # Divide data into words (X) and tags (Y)
        # --------------------------------------------
        X = [[item[0] for item in sent] for sent in sentences]
        Y = [[item[1] for item in sent] for sent in sentences]

        print('\nSample X: ', X[0], '\n')
        print('Sample Y: ', Y[0], '\n')
        # --------------------------------------------



        # Train test split
        # --------------------------------------------
        # split data into training and testing sets
        TEST_SIZE = 0.20
        X_train, X_test, Y_train, Y_test = train_test_split(np.array(X), np.array(Y), test_size=TEST_SIZE,
                                                            random_state=8)

        # split training data into training and validation sets
        VALID_SIZE = 0.15
        X_train, X_validation, Y_train, Y_validation = train_test_split(np.array(X_train), np.array(Y_train),
                                                                        test_size=VALID_SIZE, random_state=8)
        # print number of samples in each set
        self.logs.append("\n" * 2 + "** Shape **" + "\n" + "-" * 100 + "\n")
        self.logs.append("TRAINING DATA")
        self.logs.append('Shape of input sequences: {}'.format(X_train.shape))
        self.logs.append('Shape of output sequences: {}'.format(Y_train.shape))
        self.logs.append("-" * 50)
        self.logs.append("VALIDATION DATA")
        self.logs.append('Shape of input sequences: {}'.format(X_validation.shape))
        self.logs.append('Shape of output sequences: {}'.format(Y_validation.shape))
        self.logs.append("-" * 50)
        self.logs.append("TESTING DATA")
        self.logs.append('Shape of input sequences: {}'.format(X_test.shape))
        self.logs.append('Shape of output sequences: {}'.format(Y_test.shape))
        # --------------------------------------------



        # Vectorize X and Y
        # --------------------------------------------
        # Tokenizer() from Keras library to encode text sequence to integer sequence

        # num_of_words = round(n_words / 50)
        # word_tokenizer = Tokenizer(num_words=num_of_words, oov_token="<UKN>")  # instantiate tokenizer

        # encode X
        self.word_tokenizer = Tokenizer(oov_token="<UKN>")  # instantiate tokenizer
        self.word_tokenizer.fit_on_texts(X_train)  # fit tokenizer on train data

        X_train_encoded = self.word_tokenizer.texts_to_sequences(
            X_train)  # use the tokenizer to encode train data into sequences
        X_test_encoded = self.word_tokenizer.texts_to_sequences(X_test)
        X_validation_encoded = self.word_tokenizer.texts_to_sequences(X_validation)


        # count number of <UKN> in test and validation set
        X_test_unk_counter = len([1 for sentence in X_test_encoded for word_idx in sentence if word_idx == 1])
        X_test_total_words_counter = len([1 for sentence in X_test_encoded for word_idx in sentence])

        X_validation_unk_counter = len(
            [1 for sentence in X_validation_encoded for word_idx in sentence if word_idx == 1])
        X_validation_total_words_counter = len([1 for sentence in X_validation_encoded for word_idx in sentence])

        # print(word_tokenizer.word_index)
        self.logs.append(
            '\nPercentage of <UKN> words in test set {}'.format(X_test_unk_counter / X_test_total_words_counter * 100))
        self.logs.append('Percentage of <UKN> words in validation set {}'.format(
            X_validation_unk_counter / X_validation_total_words_counter * 100))


        # encode Y
        Y_train_encoded = [[tag2idx[tag] for tag in instance] for instance in
                           Y_train]  # use this instead of Tokenizer so that tags start from 0 instead of 1
        Y_test_encoded = [[tag2idx[tag] for tag in instance] for instance in Y_test]
        Y_validation_encoded = [[tag2idx[tag] for tag in instance] for instance in Y_validation]

        # tag_tokenizer = Tokenizer()
        # tag_tokenizer.fit_on_texts(Y)
        # Y_encoded = tag_tokenizer.texts_to_sequences(Y)

        # look at first encoded data point
        print("\n" * 2, "** Raw data point **", "\n", "-" * 100, "\n")
        print('X: ', X_train_encoded[0], '\n')
        print('Y: ', Y_train_encoded[0], '\n')
        print()
        print("\n" * 2, "** Encoded data point **", "\n", "-" * 100, "\n")
        print('X: ', X_train_encoded[0], '\n')
        print('Y: ', Y_train_encoded[0], '\n')

        print('X: {}'.format(X_train_encoded[0]))

        # make sure that each sequence of input and output is same length
        different_length = [1 if len(input) != len(output) else 0 for input, output in
                            zip(X_train_encoded, Y_train_encoded)]
        print("{} sentences have disparate input-output lengths".format(sum(different_length)))
        # --------------------------------------------



        # Pad sequences
        # --------------------------------------------
        # Pad each sequence to MAX_SEQ_LENGTH using KERAS pad_sequences() function.
        # Sentences longer than MAX_SEQ_LENGTH are truncated.
        # Sentences shorter than MAX_SEQ_LENGTH are padded with zeroes.

        # Truncation and padding can either be 'pre' or 'post'.
        # For padding we are using 'post' padding type, that is, add zeroes on the right side.
        # For truncation, we are using 'post', that is, truncate a sentence from right side.

        self.MAX_SEQ_LENGTH = 70  # sequences greater than MAX_SEQ_LENGTH in length will be truncated

        X_train_padded = pad_sequences(X_train_encoded, maxlen=self.MAX_SEQ_LENGTH, padding="post", truncating="post")
        X_test_padded = pad_sequences(X_test_encoded, maxlen=self.MAX_SEQ_LENGTH, padding="post", truncating="post")
        X_validation_padded = pad_sequences(X_validation_encoded, maxlen=self.MAX_SEQ_LENGTH, padding="post",
                                            truncating="post")

        Y_train_padded = pad_sequences(Y_train_encoded, maxlen=self.MAX_SEQ_LENGTH, padding="post", truncating="post",
                                       value=tag2idx["O"])
        Y_test_padded = pad_sequences(Y_test_encoded, maxlen=self.MAX_SEQ_LENGTH, padding="post", truncating="post",
                                      value=tag2idx["O"])
        Y_validation_padded = pad_sequences(Y_validation_encoded, maxlen=self.MAX_SEQ_LENGTH, padding="post",
                                            truncating="post",
                                            value=tag2idx["O"])

        # print the first sequence
        print("\n" * 2, "** Padded data point **", "\n", "-" * 100, "\n")
        print(X_train_padded[0], "\n")
        print(Y_train_padded[0])
        # --------------------------------------------



        # One hot encoding
        # --------------------------------------------
        # use Keras to_categorical function to one-hot encode Y
        self.Y_train = to_categorical(Y_train_padded)
        self.Y_test = to_categorical(Y_test_padded)
        self.Y_validation = to_categorical(Y_validation_padded)

        # define X data
        self.X_train = X_train_padded
        self.X_test = X_test_padded
        self.X_validation = X_validation_padded

        # print number of samples in each set
        self.logs.append("\n" * 2 + "** Final Shape **" + "\n" + "-" * 100 + "\n")
        self.logs.append("TRAINING DATA")
        self.logs.append('Shape of input sequences: {}'.format(X_train.shape))
        self.logs.append('Shape of output sequences: {}'.format(Y_train.shape))
        self.logs.append("-" * 50)
        self.logs.append("VALIDATION DATA")
        self.logs.append('Shape of input sequences: {}'.format(X_validation.shape))
        self.logs.append('Shape of output sequences: {}'.format(Y_validation.shape))
        self.logs.append("-" * 50)
        self.logs.append("TESTING DATA")
        self.logs.append('Shape of input sequences: {}'.format(X_test.shape))
        self.logs.append('Shape of output sequences: {}'.format(Y_test.shape))
        # --------------------------------------------


    def create_embeddings(self):
        # Embeddings
        # --------------------------------------------
        # EMBEDDING_SIZE = 100
        self.VOCABULARY_SIZE = len(self.word_tokenizer.word_index) + 1  # +1 for the padding word
        self.embedding_weights = load_word2vec_pretrained(self.VOCABULARY_SIZE, self.EMBEDDING_SIZE,
                                                          self.word_tokenizer)

        # check embedding dimension
        self.logs.append("Embeddings shape: {}".format(self.embedding_weights.shape))

        # print word2vec examples
        # print(word2vec.most_similar(positive=["βασιλιάς", "γυναίκα"], negative=["αντρας"]))
        # print(word2vec.most_similar("άρωμα"))
        # print(word2vec.most_similar("ενυδατωση"))
        # print(word2vec.most_similar("καφεσ"))
        # --------------------------------------------


    def define_model(self):

        lr = 0.0001

        # create architecture
        inputs = Input(shape=(self.MAX_SEQ_LENGTH,))

        # vocabulary size — number of unique words in data
        # length of vector with which each word is represented
        emb = Embedding(input_dim=self.VOCABULARY_SIZE,
                                 output_dim=self.EMBEDDING_SIZE,
                                 # length of input sequence
                                 input_length=self.MAX_SEQ_LENGTH,
                                 # word embedding matrix
                                 weights=[self.embedding_weights],
                                 # True — update embeddings_weight matrix
                                 trainable=True
                                 )(inputs)

        #output = Dropout(self.dropout_rate)(inputs)

        # True — return whole sequence; False — return single output of the end of the sequence
        if self.use_bidirectional:
            output = Bidirectional(GRU(self.n_units, return_sequences=True, recurrent_dropout=self.dropout_rate))(emb)
            output = Bidirectional(GRU(self.n_units, return_sequences=True, recurrent_dropout=self.dropout_rate))(output)
        else:
            self.model.add(GRU(self.n_units, return_sequences=True, recurrent_dropout=self.dropout_rate))
            self.model.add(GRU(self.n_units, return_sequences=True, recurrent_dropout=self.dropout_rate))



        #self.model.add(Dropout(self.dropout_rate))
        #self.model.add(Dense(self.n_units))

        opt = Adam(learning_rate=lr)  # optimizers: Adam, RMSprop, sgd, adamax

        if self.use_crf_layer:

            crf = CRF(self.n_tags, learn_mode='marginal', unroll=True)
            x = crf(output)

            self.model = Model(inputs=inputs, outputs=x)
            self.model.compile(optimizer=opt, loss=crf.loss_function, metrics=['acc'])

        else:
            self.model.add(TimeDistributed(Dense(self.n_tags, activation="softmax")))
            self.model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['acc']) # loss functions: categorical_crossentropy, mean_squared_error

        self.model.summary()

        self.logs.append('\nVocabulary size: {}'.format(self.VOCABULARY_SIZE))
        self.logs.append('Embedding size: {}'.format(self.EMBEDDING_SIZE))
        self.logs.append('Batch size: {}'.format(self.batch_size))
        self.logs.append('Units size: {}'.format(self.n_units))
        self.logs.append('Dropout: {}'.format(self.dropout_rate))
        self.logs.append('Optimizer: {}'.format(opt))
        self.logs.append('Learning rate: {}'.format(lr))
        self.logs.append('\n')

        stringlist = []
        self.model.summary(print_fn=lambda x: stringlist.append(x))
        short_model_summary = "\n".join(stringlist)
        self.logs.append(short_model_summary)

        return self.model


    def fit_model(self):
        # checkpoint
        filepath = self.RESULTS_FOLDER_PATH + 'weights/' + self.experiment_descr + '_weights.best.hdf5' #'../../results/RNN/weights/' + self.experiment_tag + '_weights.best.hdf5'
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_freq="epoch",
                                     save_best_only=True, mode='max')
        es_callback = EarlyStopping(monitor='val_loss', patience=5, verbose=1)

        '''
        if self.use_crf_layer:
            checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_freq="epoch",
                                         save_best_only=True, mode='max')
        else:
            checkpoint = ModelCheckpoint(filepath, monitor='val_viterbi_acc', verbose=1, save_freq="epoch", save_best_only=True, mode='max')

        # use early stopping
        if self.use_crf_layer:
            es_callback = EarlyStopping(monitor='val_crf_loss', patience=3, verbose=1)
        else:
            es_callback = EarlyStopping(monitor='val_loss', patience=3, verbose=1)
        '''

        callbacks_list = [es_callback, checkpoint]
        self.history = self.model.fit(self.X_train, self.Y_train, batch_size=self.batch_size, epochs=self.epochs,
                                      validation_data=(self.X_validation, self.Y_validation), callbacks=callbacks_list, shuffle=True)

        self.logs.append('\nEarly stopping in epoch {} '.format(es_callback.stopped_epoch))


    def plot_acc_loss(self):

        os.makedirs(self.RESULTS_FOLDER_PATH + 'plots/', exist_ok=True)

        #if self.use_crf_layer:
        #    acc = self.history.history['viterbi_accuracy']
        #    val_acc = self.history.history['val_viterbi_accuracy']
        #else:
        #    acc = self.history.history['acc']
        #    val_acc = self.history.history['val_acc']

        acc = self.history.history['acc']
        val_acc = self.history.history['val_acc']
        epoch_values = list(range(0, len(acc), 5))  # [0,1,2,3]

        # Plot accuracy
        f, ax = plt.subplots(1)
        plt.plot(acc, color="royalblue", linewidth=1)  # summarize history for accuracy
        plt.plot(val_acc, color='#f55142', linewidth=1)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.tick_params(left=False)
        ax.set_xticks(epoch_values)
        ax.set_xlim(xmin=0)
        # ax.set_ylim(ymin=0)

        plt.title('Accuracy Plot')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')

        legend = plt.legend(['Train', 'Validation'], loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
        legend.get_frame().set_edgecolor("gainsboro")

        ax.grid(axis='y', color="#E8E5E5")
        plt.tight_layout()

        plt.savefig(self.RESULTS_FOLDER_PATH + 'plots/' + self.experiment_descr + '_accuracy.png')
        plt.show()


        # Plot loss
        f, ax = plt.subplots(1)
        plt.plot(self.history.history['loss'], color="royalblue", linewidth=1)  # summarize history for loss
        plt.plot(self.history.history['val_loss'], color='#f55142', linewidth=1)

        epoch_values = list(range(0, len(self.history.history['val_loss']), 5))  # [0,1,2,3]

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.tick_params(left=False)
        ax.set_xticks(epoch_values)
        ax.set_xlim(xmin=0)
        # ax.set_ylim(ymin=0)

        plt.title('Loss Plot')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')

        legend = plt.legend(['Train', 'Validation'], loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
        legend.get_frame().set_edgecolor("gainsboro")

        ax.grid(axis='y', color="#E8E5E5")
        plt.tight_layout()

        plt.savefig(self.RESULTS_FOLDER_PATH + 'plots/' + self.experiment_descr + '_loss.png')
        plt.show()


    def evaluate(self):

        # Print accuracy and loss
        # --------------------------------------------
        self.logs.append('\n')
        # loss, accuracy = self.model.evaluate(self.X_test, self.Y_test, verbose=1)
        # self.logs.append("Loss: {0}, \nAccuracy: {1}".format(loss, accuracy))
        # --------------------------------------------

        # Evaluation
        # --------------------------------------------
        self.logs.append('\n')
        y_pred = self.model.predict(self.X_test, verbose=1)

        y_pred = np.argmax(y_pred, axis=-1)
        y_test_true = np.argmax(self.Y_test, axis=-1)

        # Convert the index to tag
        y_pred = [[self.idx2tag[i] for i in row] for row in y_pred]
        y_test_true = [[self.idx2tag[i] for i in row] for row in y_test_true]

        self.tags.remove('O')  # remove the 'O' tag in order to properly evaluate the model
        sorted_labels = sorted(
            self.tags,
            key=lambda name: (name[1:], name[0])
        )

        report = metrics.flat_classification_report(y_pred=y_pred, y_true=y_test_true, labels=sorted_labels, digits=4)
        self.logs.append(report)
        print(report)
        # --------------------------------------------


    def print_random_prediction(self):

        # At every execution model picks some random test sample from test set.
        i = np.random.randint(0, self.X_test.shape[0])  # choose a random number between 0 and len(X_te)b
        p = self.model.predict(np.array([self.X_test[i]]))
        p = np.argmax(p, axis=-1)
        true = np.argmax(self.Y_test[i], -1)

        tokenizer_dict = self.word_tokenizer.word_index

        self.logs.append("Sample number {} of {} (Test Set)".format(i, self.X_test.shape[0]))
        # Visualization
        self.logs.append("{:15}||{:5}||{}".format("Word", "True", "Pred"))
        self.logs.append(30 * "=")
        for w, t, pred in zip(self.X_test[i], true, p[0]):
            if w != 0:
                word = list(tokenizer_dict.keys())[list(tokenizer_dict.values()).index(w)]
                self.logs.append("{:15}: {:5} {}".format(word, self.idx2tag[t], self.idx2tag[pred]))
