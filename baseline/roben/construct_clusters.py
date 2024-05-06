import numpy as np
from tqdm import tqdm
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import os
import json
from collections import defaultdict
import string
import argparse

import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx

from edit_dist_utils import get_all_edit_dist_one, get_sorted_word
from preprocess_vocab import preprocess_neighbors, preprocess_neighbors_intprm
from utils_roben import pkl_load, pkl_save


RELATIVE_GLOVE_PATH = 'glove/glove.6B.50d.txt'
RELATIVE_COCA_PATH = 'COCA/coca-1grams.json'

#Read in frequencies from COCA
def read_coca_freq(coca_path, sort = True):
    with open(coca_path, 'r', encoding="ISO-8859-1") as f:
        coca_freq_dict = json.load(f)
    frequencies = [(elem.split('_')[0], int(coca_freq_dict[elem])) for elem in coca_freq_dict]
    if sort:
        frequencies.sort(key = lambda x: x[1], reverse = True)
    frequencies = process_duplicates(frequencies)
    return frequencies

def process_duplicates(frequencies):
    frequency_dict = {}
    duplicates = set()
    for elem, freq in frequencies:
        if elem in frequency_dict:
            duplicates.add(elem)
            frequency_dict[elem] += int(freq)
        else:
            frequency_dict[elem] = int(freq)
    frequencies = [(elem, frequency_dict[elem]) for elem in frequency_dict]
    frequencies.sort(key = lambda x: x[1], reverse = True)
    return frequencies

def get_glove_vocab(glove_path, num_lines = 400000):
    print("Reading GloVe vectors from {}...".format(glove_path))
    embedding_map = {}
    glove_vocab = set()
    with open(glove_path, encoding = 'utf-8', mode = "r") as f:
        for i, line in tqdm(enumerate(f), total=num_lines):
            toks = line.strip().split(' ')
            word = toks[0]
            glove_vocab.add(word)
    return glove_vocab


class Clusterer():
    def __init__(self, vocab_size = 100000, perturb_type = 'ed1'):
        #max_num_ret should be greater than number of initial vertices in the graph
        print("Getting word frequencies...")
        self.num_verts = vocab_size
        self.frequencies = read_coca_freq(os.path.join(args.data_dir, RELATIVE_COCA_PATH))
        self.word2freq = self._get_word2freq(self.frequencies)

        #verify frequencies are sorted
        assert self._is_sorted([float(f[1]) for f in self.frequencies], reverse = True)
        self.vertices = []
        self.edges = []

        #constraining vocab to be in glove
        glove_vocab = get_glove_vocab(os.path.join(args.data_dir, RELATIVE_GLOVE_PATH))
        self.frequencies = [elem for elem in self.frequencies if elem[0] in glove_vocab]

        self.word2cluster = {}
        self.cluster2representative = {}
        self.clusters = {}
        self.perturb_type = perturb_type

    def _is_sorted(self, lst, reverse = False):
        start, end, incr = 0, len(lst), 1
        if reverse:
            start, end, incr = len(lst) - 1, -1, -1
        prev_elem = float('-inf')
        for i in range(start, end, incr):
            elem = lst[i]
            if elem < prev_elem:
                return False
            prev_elem = elem
        return True

    def _get_word2freq(self, frequencies):
        word2freq = defaultdict(lambda: 0)
        for word, freq in frequencies:
            word2freq[word] += int(freq)
        return word2freq

    def construct_graph(self, perturb_type = 'ed1'):
        """
        Form a graph with nodes given by vocabulary in the constructor
        and edges between words that share a perturbation (using perturb_type)
        """

        self.perturb_type = perturb_type
        #Constrain vertex set. 
        self.vertices = list([self.frequencies[i][0] for i in range(self.num_verts)])
        self.word2freq = dict([(vtx, self.word2freq[vtx]) for vtx in self.vertices])


        if perturb_type == 'ed1':
            typo2words, neighbor_map, _ = preprocess_neighbors(self.vertices)
        elif perturb_type == 'intprm':
            typo2words, neighbor_map, _ = preprocess_neighbors_intprm(self.vertices)
        else:
            raise ValueError("Unsupported perturbation type")
        self.typo2words = typo2words

        print("Computing edges...")
        self.edges = self._filter_edges(neighbor_map)
        print("Generating edge matrix...")
        self.edge_mat = self._edges_to_matrix(self.vertices, self.edges)
        print("Finished constructing the graph")

    def _filter_edges(self, neighbor_map):
        """
        neighbor_map 
        """
        possible_edges = set()
        rejected_edges = set()
        similarities = []
        for vtx in neighbor_map:
            for vtx2 in neighbor_map[vtx]:
                if vtx == vtx2:
                    continue
                vtx_pair = [vtx, vtx2]
                vtx_pair.sort() #Graph is undirected, assume attack surface is symmetric
                vtx_pair = tuple(vtx_pair)
                if vtx_pair in possible_edges or vtx_pair in rejected_edges:
                    continue
                else:
                    possible_edges.add(vtx_pair)
        return possible_edges

    def _edges_to_matrix(self, vertices, edges):
        #exclusive_edges = [edge for edge in edges if edge[0] != edge[1]]
        #print("Num vertices: {}, edges: {}".format(vertices, exclusive_edges))
        edge_mat = np.zeros(shape = (len(vertices), len(vertices)), dtype = bool)
        vert2idx = dict([(vertices[i], i) for i in range(len(vertices))])
        #for edge in tqdm(edges):
        for edge in edges:
            vert1, vert2 = edge
            if vert1 == vert2:
                continue
            id1, id2 = vert2idx[vert1], vert2idx[vert2]
            edge_mat[id1][id2] = 1
            edge_mat[id2][id1] = 1
        return edge_mat


    def construct_clusters(self):
        if self.edge_mat is None:
            raise ValueError("Graph must already be computed...")
        self.cluster2elements = defaultdict(list)
        graph = csr_matrix(self.edge_mat)
        n_components, labels = connected_components(csgraph=graph, directed=False, return_labels=True)
        self.num_clusters = n_components
        for i in range(labels.shape[0]):
            label = labels[i]
            if label not in self.cluster2representative:
                representative = self.vertices[i] #Assumes vertices in sorted order of freq
                #print("Setting {} to representative".format(self.vertices[i]))
                self.cluster2representative[label] = representative
                self.clusters[label] = []
            #print("Adding {} to {}'s cluster".format(self.vertices[i], self.cluster2representative[label]))
            self.word2cluster[self.vertices[i]] = label
            self.clusters[label].append(self.vertices[i])
        self.typo2cluster = self._get_typo2cluster()

    def _get_typo2cluster(self):
        typo2cluster = {}
        typo2word = self._get_typo2word(self.vertices, self.word2freq)
        for typo in tqdm(typo2word):
            typo2cluster[typo] = self.word2cluster[typo2word[typo]]
        return typo2cluster

    def _get_typo2word(self, words, word2freq):
        typo2word = {}
        print("Getting typo2word")
        for typo in tqdm(self.typo2words):
            possible_words = self.typo2words[typo]
            typo_word_freq_list = [(word, word2freq[word]) for word in possible_words]
            typo_word_freq_list.sort(key = lambda x: x[1], reverse = True)
            most_frequent_word = typo_word_freq_list[0][0]
            typo2word[typo] = most_frequent_word

        #Word always recovers to it's own cluster
        for word in words:
            typo2word[word] = word

        return typo2word 

def save_clusterer(vocab_size = 100000, perturb_type = 'ed1', save_dir = 'clusterers',
                    check_perturb_size = False):

    filename = 'vocab{}_{}.pkl'.format(vocab_size, perturb_type)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, filename)
    print("Will save at: {}".format(save_path))

    #Initializing clusterer
    clusterer = Clusterer(vocab_size = vocab_size)
    #Initializing the graph
    clusterer.construct_graph(perturb_type = perturb_type)
    #Creating clusters.
    clusterer.construct_clusters()

    #Option to analyze number of perturbations, etc. 
    #if check_perturb_size:
    #    get_vocab_statistics(clusterer.vertices)
    #    return
    save_dict = {'cluster': clusterer.clusters,
                    'word2cluster': clusterer.word2cluster,
                    'cluster2representative': clusterer.cluster2representative,
                    'word2freq': clusterer.word2freq,
                    'typo2cluster': clusterer.typo2cluster}

    print("Saving everything at: ", save_path)
    pkl_save(save_dict, save_path)
    print("Number of clusters: {}, vocab size: {}".format(len(clusterer.clusters), vocab_size))

def get_vocab_statistics(vertices):
    #vertices correspond to words in the vocabulary.
    #prints stats on the number of perturbations each word has. 
    num_perturbations = []
    print("Total number of vertices: ", len(vertices))
    for vtx in tqdm(vertices):
        num_perturbations.append(len(get_all_edit_dist_one(vtx)))
    num_perturbations = np.array(num_perturbations)
    print("Mean: {} Min: {} Max: {}".format(num_perturbations.mean(), num_perturbations.min(), num_perturbations.max()))


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--data_dir", default='data', type=str, required=False,
                      help="The input data dir.")
  parser.add_argument("--output_dir", default='clusterers', type=str, required=False,
                      help="The output dir for the clusterer.")
  parser.add_argument('--vocab_size', type = int, default = 100000,
                        help = 'Size of the vocabulary used to make the clusters.')
  parser.add_argument('--perturb_type', choices=['ed1', 'intprm'], type = str, 
                        help = 'type of perturbation used to define clusters')
  return parser.parse_args()

if __name__ == '__main__':
    print("Starting the run...")
    args = parse_args()
    save_clusterer(vocab_size = args.vocab_size, perturb_type = args.perturb_type, save_dir=args.output_dir)



