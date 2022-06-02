import numpy as np
from gensim.models import KeyedVectors


def load_word2vec_pretrained(VOCABULARY_SIZE, EMBEDDING_SIZE, word_tokenizer):
    '''
    From http://vectors.nlpl.eu/repository/

    :return:
    '''
    # word2vec
    word2vec_path = 'embeddings/pretrained_embeddings/word2vec_pretrained/model.bin'  # From http://vectors.nlpl.eu/repository/
    word2vec = KeyedVectors.load_word2vec_format(word2vec_path, binary=True)  # load word2vec

    # assign word vectors from word2vec model
    embedding_weights = np.zeros((VOCABULARY_SIZE, EMBEDDING_SIZE))  # create an empty embedding matrix
    word2id = word_tokenizer.word_index  # create a word to index dictionary mapping

    # copy vectors from word2vec model to the words present in corpus
    for word, index in word2id.items():

        try:
            embedding_weights[index, :] = word2vec[word]
            # print(word)
        except KeyError:
            pass

    return embedding_weights


if __name__ == '__main__':
    print("main")
