import torch
import numpy as np
from collections import Counter
from tqdm import tqdm
import time
import math
import logging
import nltk

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
        self.window_size = window_size
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

    # get co_occurrence
    # input: string text
    # output: array left, array right, array n_occurrence
    def get_co_occurrence(self):
        co_occurrence = Counter()
        length = len(self.text)
        for l in range(length):
            if text[l] != self.oov_index:
                for r in range(1, self.window_size+1):
                    if l+r < length and text[l+r] != self.oov_index:
                        co_occurrence[(self.text[l], self.text[l+r])] += 1 / r

        return zip(*[(left, right, value) for (left, right), value in co_occurrence.items()])

# capsule glove data and parameter
class GloveDataset():
    def __init__(self, text, dim, x_max, alpha, device):
        self.indexer = WordsIndexer(text=text, min_word_occurrence=min_word_occurrence,
                                     window_size=window_size)
        self.x_max = x_max
        self.alpha = alpha

        self.l_vecs = torch.rand(self.indexer.num_words, dim, requires_grad=True, device=device)
        self.r_vecs = torch.rand(self.indexer.num_words, dim, requires_grad=True, device=device)
        self.l_biases = torch.rand(self.indexer.num_words, requires_grad=True, device=device)
        self.r_biases = torch.rand(self.indexer.num_words, requires_grad=True, device=device)
        self.all_params = (self.l_vecs, self.r_vecs, self.l_biases, self.r_biases)

        self.l_indexes, self.r_indexes, self.co_occurrences = self.indexer.get_co_occurrence()

        self.ngrams_size = len(self.co_occurrences)
        self.l_indexes = torch.tensor(self.l_indexes, device=device)
        self.r_indexes = torch.tensor(self.r_indexes, device=device)
        self.weights = torch.tensor([self.wf(co_occurrence) for co_occurrence in self.co_occurrences],
                                    device=device)

        self.log_co_occurrences = torch.tensor([np.log(co_occurrence) for co_occurrence in self.co_occurrences],
                                            device=device)

    # compute weight
    def wf(self, x):
        if x < self.x_max:
            return (x / self.x_max) ** self.alpha
        return 1

    # generate batches
    def gen_batches(self, batch_size):
        indices = torch.randperm(self.ngrams_size)
        for idx in range(0, self.ngrams_size - batch_size + 1, batch_size):
            sample = indices[idx: idx + batch_size]
            yield self.l_vecs[self.l_indexes[sample]], self.r_vecs[self.r_indexes[sample]],\
                  self.l_biases[self.l_indexes[sample]], self.r_biases[self.r_indexes[sample]],\
                  self.weights[sample], self.log_co_occurrences[sample]

    # save parameters as file
    def save(self):
        torch.save(self.l_vecs, "l_vecs")
        torch.save(self.r_vecs, "r_vecs")
        torch.save(self.l_biases, "l_biases")
        torch.save(self.r_biases, "r_biases")

# transform time number to string
# input: int s
# output: string m
def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return "%dm %ds" % (m, s)

# transform percent to string
# input: time start, float percent
# output: string r
def timeSince(start, percent):
    now = time.time()
    s = now - start
    r = s / (percent)
    return "%s (- %s)" % (asMinutes(s), asMinutes(r - s))

def readfile(filepath):
    with open(filepath) as f:
        text = f.read().lower()
    return nltk.word_tokenize(text)

def get_loss(l_vecs, r_vecs, l_biases, r_biases, weights, log_co_occurrences):
    sim = (l_vecs * r_vecs).sum(1).view(-1)
    x = (sim + l_biases + r_biases - log_co_occurrences) ** 2
    loss = torch.mul(x, weights)
    return loss.mean()

# train model
def train(data, n_epoches, batch_size, lr):
    optimizer = torch.optim.SGD(data.all_params, lr=lr)
    optimizer.zero_grad()
    num_batches = int(data.ngrams_size / batch_size)
    start = time.time()
    for epoch in tqdm(range(1,n_epoches+1)):
        logging.info("start epoch %i", epoch)
        aver_loss = 0
        for batch in tqdm(data.gen_batches(batch_size), total=num_batches, mininterval=1):
            optimizer.zero_grad()
            loss = get_loss(*batch)
            loss.backward()
            aver_loss += loss.item() / num_batches
            optimizer.step()
        logging.info("%s (%d %d%%) %.4f" % (timeSince(start, epoch / n_epoches),
                                         epoch, epoch / n_epoches * 100, aver_loss))
        if epoch % save_every == 0:
            data.save()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("fetching data...")
    text = readfile(filepath)
    logging.info("building dataset...")
    data = GloveDataset(text, dim, x_max, alpha, device)

    logging.info("# words: %s", data.indexer.num_words)
    logging.info("# ngrams: %s", data.ngrams_size)
    logging.info("start training...")
    train(data, n_epoches, batch_size, lr)