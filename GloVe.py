import torch
import torch.optim as optim
import nltk
import numpy as np
import math
import time

window_size = 3
dim = 2
batch_size = 20
filepath = "short_story.txt"
alpha = 0.75
x_max = 2
iteraters = 20
lr = 0.001
print_every = 1

# input: file path
# output: string array of corpus
def readfile(filepath):
    with open(filepath) as f:
        corpus = f.read().lower()
    return nltk.word_tokenize(corpus) # word_tokenize: split the string reasonably automatically

# input: index array of corpus, size of window, number of words
# output: matrix X
def get_X(index_corpus, window_size, num_words):
    X = np.zeros((num_words, num_words))
    length = len(index_corpus)
    for i in range(length):
        for j in range(1, window_size+1):
            l = index_corpus[i]
            if i-j >= 0:
                X[l][index_corpus[i-j]] += 1/j
            if i+j < length:
                X[l][index_corpus[i+j]] += 1/j
    return X

# input: x
# output: 1 if x >= x_max, (x / x_max)**alpha if x < x_max
def wf(x):
    if x >= x_max:
        return 1
    return (x / x_max) ** alpha

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

# # input: array of all samples
# # output: array of batch_size randomly selected samples
# def get_batch(samples):
#     return np.random.choice(samples, batch_size)


if __name__ == "__main__":

    # read file
    corpus = readfile(filepath)

    # delete the same words
    words = np.unique(corpus)

    # get w_to_i
    num_words = len(words)
    w_to_i = {words[i]: i for i in range(num_words)}

    # get X
    index_corpus = [w_to_i[word] for word in corpus]
    X = get_X(index_corpus, window_size, num_words)

    # get all samples
    samples = np.transpose(np.nonzero(X)) # np.nonzero returns indexes of nonzero elements
    num_samples = len(samples)
    # initialize parameters
    w1 = torch.rand(num_words, dim, requires_grad=True)
    w2 = torch.rand(num_words, dim, requires_grad=True)
    b1 = torch.rand(num_words, requires_grad=True)
    b2 = torch.rand(num_words, requires_grad=True)

    # initialize optimizer
    optimizer = optim.Adam([w1, w2, b1, b2], lr=lr)

    avg_loss = 0
    num_batches = int(num_samples / batch_size)
    start = time.time()
    for it in range(1, iteraters+1):
        for batch in range(num_batches):
            optimizer.zero_grad()
            batch_samples_indexes = np.random.choice(np.arange(num_samples), batch_size, replace=False)
            loss = 0
            for id in batch_samples_indexes:
                l = samples[id][0]
                r = samples[id][1]
                loss += torch.mul((torch.dot(w1[l], w2[r])
                        + b1[l] + b2[r] - np.log(X[l][r])) ** 2, wf(X[l][r]))
            avg_loss += loss.item() / num_batches
            loss.backward()
            optimizer.step()

        if it % print_every == 0:
            print("%s (%d %d%%) %.4f" % (timeSince(start, it / iteraters),
                                         it, it / iteraters *100, avg_loss))
            avg_loss = 0

    torch.save(w1+w2,"w")
    np.save("words", words)
