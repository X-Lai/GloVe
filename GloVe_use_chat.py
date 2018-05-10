import torch
import torch.optim as optim
import nltk
import numpy as np
import math
import time

use_text8 = False
window_size = 3
dim = 200
batch_size = 16
filepath = "chat.txt"
alpha = 0.75
x_max = 1000
iteraters = 10
lr = 0.001
print_every = 1

# read file line by line
# input: file path
# output: string array of corpus line
def readfile(filepath):
    lines = []
    with open(filepath) as f:
        line = f.readline()
        while line:
            lines.append(line.split())
            line = f.readline()
    return lines

# get X from lines
# input: array of corpus, size of window, number of words
# output: matrix X
def get_X(lines, window_size, num_words):
    X = [{} for i in range(num_words)]
    for corpus in lines:
        length = len(corpus)
        for i in range(length):
            for j in range(1, window_size+1):
                l = w_to_i[corpus[i]]
                if i-j >= 0:
                    r = w_to_i[corpus[i-j]]
                    if r in X[l].keys():
                        X[l][r] += 1/j
                    else:
                        X[l][r] = 1/j
                if i+j < length:
                    r = w_to_i[corpus[i+j]]
                    if r in X[l].keys():
                        X[l][r] += 1/j
                    else:
                        X[l][r] = 1/j
    return X

# flatten lines
# input: lines
# output: words
def flatten(a):
    for each in a:
        if not isinstance(each, list):
            yield each
        else:
            yield from flatten(each)

# input: matrix X
# output: nparray of all samples
def get_samples(X):
    samples = []
    for w1 in range(len(X)):
        for w2 in X[w1].keys():
            samples.append((w1, w2))
    return samples

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
    # setup cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # read file
    lines = readfile(filepath)

    # delete the same words
    words = np.unique(list(flatten(lines)))

    # get w_to_i
    num_words = len(words)
    w_to_i = {word: i for (i,word) in enumerate(words)}

    # get X
    X = get_X(lines, window_size, num_words)

    # get all samples
    # samples = np.transpose(np.nonzero(X)) # np.nonzero returns indexes of nonzero elements
    samples = get_samples(X)
    num_samples = len(samples)

    print(num_samples)

    # initialize parameters
    w1 = torch.rand(num_words, dim, requires_grad=True, device=device)
    w2 = torch.rand(num_words, dim, requires_grad=True, device=device)
    b1 = torch.rand(num_words, requires_grad=True, device=device)
    b2 = torch.rand(num_words, requires_grad=True, device=device)

    # initialize optimizer
    optimizer = optim.Adam([w1, w2, b1, b2], lr=lr)

    avg_loss = 0
    num_batches = int(num_samples / batch_size)
    start = time.time()
    print("training begins...")

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
            print("%s (%d %d%%) %.4f" % (timeSince(start, it / iteraters),
                                         it, it / iteraters *100, avg_loss))

        if it % print_every == 0:
            print("%s (%d %d%%) %.4f" % (timeSince(start, it / iteraters),
                                         it, it / iteraters *100, avg_loss))
            avg_loss = 0

    torch.save(w1,"w1")
    torch.save(w2,"w2")
    torch.save(b1,"b1")
    torch.save(b2,"b2")
    np.save("words", words)
