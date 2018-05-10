import logging
import numpy as np
import nltk
import torch
from collections import Counter

FORMAT = '%(asctime)-15s %(message)s'
logging.basicConfig(level=logging.DEBUG, format=FORMAT)

min_word_occurrence = 10
window_size = 4
batch_size = 32
dim = 300
x_max = 100
alpha = 0.75
n_epoches = 10
lr = 0.05
save_every = 5
filepath = "text8"

# parse text
class WordsIndexer():
    # input: list text, int min_occurrence
    def __init__(self, text, min_word_occurrence, window_size, oov="oov"):
        self.min_word_occurrence = min_word_occurrence
        self.oov = oov
        self.index_to_word = [oov]
        self.word_to_index = {oov: 0}
        self.num_words = 1
        self.text = self.fit_transform(text)

    # get index of one word
    # input: string word
    # output: int index
    def get_or_set_index(self, word):
        try:
            return self.word_to_index[word]
        except:
            self.index_to_word.append(word)
            self.word_to_index[word] = self.num_words
            self.num_words += 1
            return self.num_words - 1

    # replace the words whose occurrences are lower than min_word_occurrence with oov
    # transform text to indexes
    def fit_transform(self, text):
        n_occurrences = Counter()
        self.oov_index = 0
        for word in text:
            n_occurrences[word] += 1

        filtered_text = [self.get_or_set_index(word) if n_occurrences[word] > self.min_word_occurrence
                else self.oov_index for word in text]
        return filtered_text

# capsule glove data and parameter
class GloveDataset():
    def __init__(self, text):
        self.indexer = WordsIndexer(text=text, min_word_occurrence=min_word_occurrence)

        self.l_vecs = torch.load("l_vecs")
        self.r_vecs = torch.load("r_vecs")
        self.l_biases = torch.load("l_biases")
        self.r_biases = torch.load("r_biases")

def readfile(filepath):
    with open(filepath) as f:
        text = f.read().lower()
    return nltk.word_tokenize(text)

# compute two vectors' similarity
# input: vector w1, vector w2
# output: double similarity
def similarity(w1, w2):
    return np.linalg.norm(w1 - w2) # return euclidean distance of two vectors

# find top size most similar words
# input: string word, dict w_to_i , int size
# output: array of string
def find_top_similary(word, w_to_i, size):
    vec = w[w_to_i[word]]
    sorted_words = sorted(words, key=lambda x: similarity(w[w_to_i[x]],vec))
    return sorted_words[:size]


if __name__ == "__main__":
    text = readfile(filepath)
    data = GloveDataset(text)
    words = data.indexer.word_to_index
    np.save("words",np.array(words))
    w_to_i = data.indexer.index_to_word
    w = data.l_vecs.data + data.r_vecs.data
    s = input()
    while s != "0":
        ws = find_top_similary(s, w_to_i, 10)
        for word in ws:
            print(word, similarity(w[w_to_i[word]], w[w_to_i[s]]))
        s = input()
