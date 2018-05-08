import numpy as np
import torch

words = np.load("words.npy")
w = torch.load("w").detach().numpy()

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
    w_to_i = {words[i]:i for i in range(len(words))}
    s = input()
    while s != "0":
        ws = find_top_similary(s, w_to_i, 10)
        for word in ws:
            print(word, similarity(w[w_to_i[word]], w[w_to_i[s]]))
        s = input()