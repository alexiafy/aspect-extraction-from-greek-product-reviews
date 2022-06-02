import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional, Input
from tensorflow.keras.models import Model
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from sklearn_crfsuite import metrics
from tensorflow.keras.preprocessing.text import Tokenizer
from gensim.models import KeyedVectors
import langid
from nltk.corpus import words
import nltk
import re


nltk.download('words')


plt.style.use("ggplot")



FOLDERNAME = 'aspect_extraction_datasets'
FILENAME = 'aspect_extraction_data_part_1_2'





def pred2label(pred):
    out = []
    for pred_i in pred:
        out_i = []
        for p in pred_i:
            p_i = np.argmax(p)
            out_i.append(idx2tag[p_i].replace("PADword", "O"))
        out.append(out_i)

    return out


def test2label(pred):
    out = []
    for pred_i in pred:
        out_i = []
        for p in pred_i:
            out_i.append(idx2tag[p].replace("PADword", "O"))
        out.append(out_i)
    return out


def print_word2vec_pretrained_stats():
    # Load Data
    # --------------------------------------------
    FOLDERNAME = "../../../datasets/aspect_extraction_datasets/parts_1_2/"
    FILENAME = 'aspect_extraction_data_parts_1_2'
    data = pd.read_csv(FOLDERNAME + FILENAME + ".csv")  # .head(10)
    data['IOB2_tagging'] = data['IOB2_tagging'].apply(lambda x: ast.literal_eval(x))  # convert IOB column to list
    # --------------------------------------------

    # Prepara Data
    # --------------------------------------------
    sentences = [s for s in data['IOB2_tagging']]
    our_words = [element[0] for sent in data['IOB2_tagging'] for element in sent]
    tags = set([element[1] for sent in sentences for element in sent])  # get the unique tags

    n_words = len(our_words)
    n_tags = len(tags)

    # print(words)
    # print(tags)
    print('Number of words: ', n_words)
    print('Number of unique words', len(set(our_words)))
    print('Number of tags: ', n_tags)
    # --------------------------------------------


    # Create dictionaries
    # --------------------------------------------
    tag2idx = {t: i for i, t in enumerate(tags)}
    # --------------------------------------------

    # Tokenize  !!! SOS todo only on train data!!!
    # --------------------------------------------
    X = [[el[0] for el in sent] for sent in sentences]
    y = [[el[1] for el in sent] for sent in sentences]

    # Encode X
    word_tokenizer = Tokenizer(num_words=25005, oov_token="<UKN>")  # instantiate tokeniser
    word_tokenizer.fit_on_texts(X)  # fit tokeniser on data

    # use the tokenizer to encode input sequence
    X_encoded = word_tokenizer.texts_to_sequences(X)
    # print(X_encoded)

    # --------------------------------------------

    # Pad sequences
    # --------------------------------------------
    max_len = 80
    X_padded = pad_sequences(sequences=X_encoded, maxlen=max_len, padding="post", value=n_words - 1)
    # --------------------------------------------

    # Embeddings
    # --------------------------------------------
    # word2vec
    word2vec_path = 'pretrained_embeddings/word2vec_pretrained/model.bin'  # From http://vectors.nlpl.eu/repository/
    word2vec = KeyedVectors.load_word2vec_format(word2vec_path, binary=True)  # load word2vec

    # assign word vectors from word2vec model

    EMBEDDING_SIZE = 100
    VOCABULARY_SIZE = len(word_tokenizer.word_index) + 1

    embedding_weights = np.zeros((VOCABULARY_SIZE, EMBEDDING_SIZE))  # create an empty embedding matix
    word2id = word_tokenizer.word_index  # create a word to index dictionary mapping

    print("len(word2id) ", len(word2id))

    embedding_found_counter = 0
    english_emb_found_counter = 0
    fail_counter = 0
    english_emb_fail_counter = 0
    greek_emb_fail_counter = 0
    # print(word2id.items())
    index = 0

    english_words_counter = 0
    greek_words_counter = 0
    not_greek_not_english_counter = 0
    not_greek_not_english_fail_counter = 0
    greek_check = re.compile(r'[α-ωΑ-Ω]')

    # copy vectors from word2vec model to the words present in corpus
    for word, index in word2id.items():
        index += 1
        # print("[", index, "/", len(word2id), "]")

        if word in words.words():
            english_words_counter += 1
        else:
            if greek_check.match(word):
                greek_words_counter += 1
            else:
                not_greek_not_english_counter += 1

        try:
            embedding_weights[index, :] = word2vec[word]
            embedding_found_counter += 1

            if word in words.words():
                english_emb_found_counter += 1

        except KeyError:
            fail_counter += 1
            if word in words.words():
                # print(word)
                english_emb_fail_counter += 1
            else:
                if greek_check.match(word):
                    print(word)
                    greek_emb_fail_counter += 1
                else:

                    not_greek_not_english_fail_counter += 1
            # print(word)
            pass

    print('Total number of tokenizer words: ', len(word2id))
    print('Number of words for which embedding exists: ', embedding_found_counter)
    print('Number of english words for which embedding exists: ', english_emb_found_counter)
    print('Number of words for which embedding does not exist: ', fail_counter)
    print('Number of english words for which embedding does not exist: ', english_emb_fail_counter)

    print('\nPercentage of successful embedding found: ', embedding_found_counter / len(word2id) * 100)
    print('Percentage of english successful embedding found: ',
          english_emb_found_counter / embedding_found_counter * 100)
    print('Percentage of english successful embedding found  2: ',
          english_emb_found_counter / (english_emb_found_counter + english_emb_fail_counter) * 100)
    print('Percentage of fail on embeddings: ', fail_counter / len(word2id) * 100)
    print('Percentage of english word fail on embeddings: ', english_emb_fail_counter / len(word2id) * 100)
    print('Percentage of english word fail on embeddings dsds: ', english_emb_fail_counter / fail_counter * 100)

    print("\n Total number of english words: ", english_words_counter)
    print("Total number of greek words: ", greek_words_counter)
    print("English word fail percentage: ", english_emb_fail_counter / english_words_counter * 100)
    print("Greek word fail percentage: ", greek_emb_fail_counter / greek_words_counter * 100)
    print("greek_emb_fail_counter ", greek_emb_fail_counter)
    print("not_greek_not_english_counter ", not_greek_not_english_counter)
    print("not_greek_not_english_fail_counter ", not_greek_not_english_fail_counter)


if __name__ == '__main__':
    print_word2vec_pretrained_stats()