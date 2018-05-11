import numpy as np
import torch

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
    words = np.load("words.npy")
    w_to_i = {word: i for (i,word) in enumerate(words)}

    l_vecs = torch.load("l_vecs", map_location="cpu")
    r_vecs = torch.load("r_vecs", map_location="cpu")
    l_biases = torch.load("l_biases", map_location="cpu")
    r_biases = torch.load("r_biases", map_location="cpu")

    w = l_vecs.data + r_vecs.data
    s = input("please input:")
    while s != "0":
        ws = find_top_similary(s, w_to_i, 10)
        for word in ws:
            print(word, similarity(w[w_to_i[word]], w[w_to_i[s]]))
        s = input("please input:")
