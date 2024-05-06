import numpy as np
from tqdm import tqdm

from collections import defaultdict
from itertools import combinations
import pickle
import argparse
from edit_dist_utils import get_all_edit_dist_one, ed1_neighbors_mat
from datetime import datetime
import random
import os

def update_mu(C, f):
    weighted_cluster_assignments = C * f.reshape(-1, 1)
    cluster_weights = weighted_cluster_assignments.sum(axis = 0)
    zero_weights = np.where(cluster_weights == 0)[0]
    cluster_weights[zero_weights] = 1
    updated_mu = weighted_cluster_assignments / cluster_weights.reshape(1,-1) #Divide component-wise
    #Now, we want K x N, since we wnat to get mu_i by calling mu[i] 
    return updated_mu.T

def compute_centroid_distances(X, mu):
    #Returns an array of shape n_vocab_words x n_clusters. centroid_distances[i][j] is distance of word i to centroid j.
    n_vocab_words, vec_size = X.shape
    n_clusters, mu_vec_size = mu.shape
    assert mu_vec_size == vec_size
    centroid_distances = np.zeros(shape = (n_vocab_words, n_clusters))
    for i in range(n_clusters):
        centroid = mu[i]
        distances = np.linalg.norm(X - centroid, axis = 1)
        assert len(distances.shape) == 1
        assert distances.shape[0] == n_vocab_words
        centroid_distances[:, i] = distances
    return centroid_distances


def get_typo2cluster(words, word2freq, word2cluster):
    typo2cluster = {}
    typo2word = get_typo2word(words, word2freq)
    for typo in tqdm(typo2word):
        typo2cluster[typo] = word2cluster[typo2word[typo]]
    return typo2cluster

def get_typo2word(words, word2freq):
    typo2words = defaultdict(list)
    for word in words:
        word = word.lower()
        typos = get_all_edit_dist_one(word)
        for typo in typos:
            typo2words[typo].append((word, word2freq[word]))
    typo2word = {}
    for typo in typo2words:
        #Hard coding that vocab words always have to map to themselves.
        if typo in words:
            typo2word[typo] = typo
            continue
        typo2words[typo].sort(key = lambda x: x[1], reverse = True)
        most_frequent_word = typo2words[typo][0][0]
        typo2word[typo] = most_frequent_word
    return typo2word 

def compute_W(words, word2idx, word2freq):
    #TODO, want some way to verify clusterer is only ED2 clusters.
    typo2word = get_typo2word(words, word2freq)
    word2recoverwords = defaultdict(set)
    for word in words:
        word = word.lower()
        typos = get_all_edit_dist_one(word)
        for typo in typos:
            recover_word = typo2word[typo]
            word2recoverwords[word].add(recover_word)

    n_words = len(words)
    W = np.zeros(shape = (n_words, n_words))
    for word in word2recoverwords:
        word_idx = word2idx[word]
        for recovered_word in word2recoverwords[word]:
            recovered_idx = word2idx[recovered_word]
            W[word_idx][recovered_idx] = 1
    return W 

def apply_objective(X, mu, f, C, W, gamma = 0.5):
    n_vocab_words, n_clusters = C.shape
    centroid_distances = compute_centroid_distances(X, mu)
    A = f.T @ (C * centroid_distances).sum(axis = 1)

    num_words, num_words2 = W.shape
    assert num_words == num_words2
    assert num_words == n_vocab_words
    word_num_clusters = np.zeros(num_words)
    for i in range(num_words):
        dom_indices = W[i]
        relevant_clusters = C * dom_indices.reshape(-1, 1)
        num_relevant_clusters = relevant_clusters.max(axis = 0).sum()
        word_num_clusters[i] = num_relevant_clusters
    B = f.T @ word_num_clusters
    return gamma * A + (1 - gamma) * B, word_num_clusters

def clusters_from_verts(C_verts, C_fixed):
    n_verts, n_clusters = C_verts.shape
    n_words, n_fixed_clusters = C_fixed.shape
    assert n_fixed_clusters == n_clusters
    C = np.zeros(shape = C_fixed.shape)
    cluster_assignments = C_verts.argmax(axis = 1)
    for vert in range(n_verts):
        C[C_fixed[:, vert], cluster_assignments[vert]] = 1
    return C

def get_optimal_merge_efficient(X, mu, f, C, W, previous_objective, gamma = 0.5, combs = None):
    n_vocab_words, n_clusters = C.shape
    nonzero_clusters = np.where(np.count_nonzero(C, axis = 0) != 0)[0]
    cluster_freqs = np.array([np.dot(f, C[:, cluster_id]) for cluster_id in range(n_clusters)])

    current_best_clusters = C.copy()
    current_best_change = 0

    found_good_merge = False
    current_centroid_distances = compute_centroid_distances(X, mu)
    cluster_dominates = [(W * C[:, cluster_id]).max(axis = 1) for cluster_id in range(n_clusters)]
    combination_tuple = None
    
    for combination in combinations(nonzero_clusters, 2):

        cluster1, cluster2 = combination
        if combs is not None and (cluster1, cluster2) not in combs:
            continue

        cluster1_elems = C[:, cluster1]
        cluster2_elems = C[:, cluster2]

        combined_cluster_elems = np.logical_or(cluster1_elems, cluster2_elems)

        cluster1_centroid = mu[cluster1]
        cluster2_centroid = mu[cluster2]

        cluster1_freq = cluster_freqs[cluster1]
        cluster2_freq = cluster_freqs[cluster2]

        new_mu = (cluster1_centroid * cluster1_freq + cluster2_centroid * cluster2_freq) / (cluster1_freq + cluster2_freq)
      
        new_centroid_dist = compute_centroid_distances(X, new_mu.reshape(1, -1))
        
        new_weighted_centroid_dist = np.dot(f, new_centroid_dist[:, 0] * combined_cluster_elems)
        old_weighted_c1_dist = np.dot(f, current_centroid_distances[:, cluster1] * cluster1_elems)
        old_weighted_c2_dist = np.dot(f, current_centroid_distances[:, cluster2] * cluster2_elems)


        A_change = new_weighted_centroid_dist - old_weighted_c1_dist - old_weighted_c2_dist
        assert A_change >= 0

        dom_by_both = np.logical_and(cluster_dominates[cluster1], cluster_dominates[cluster2])
        B_change = -np.dot(f, dom_by_both)
        assert B_change <= 0
        # Gain back ambiguity from things that were previously dominated by original clusters
        objective_change = gamma * A_change + (1 - gamma) * B_change

        if objective_change < current_best_change:
            current_best_change = objective_change
            og_C = C.copy()
            combination_tuple = (cluster1, cluster2, cluster1)
            C[:,cluster1] = combined_cluster_elems
            C[:,cluster2] = np.zeros(n_vocab_words)
            current_best_clusters = C.copy()
            C = og_C
            found_good_merge = True

    return current_best_clusters, previous_objective + current_best_change, found_good_merge, combination_tuple


def get_allowable_combinations(edge_mat):
    allowable_combinations = set()
    num_vertices, num_vertices2 = edge_mat.shape
    assert num_vertices == num_vertices2
    for vtx in range(num_vertices):
        neighbors = np.where(edge_mat[vtx] != 0)[0]
        for neighbor in neighbors:
            if neighbor != vtx:
                allowable_combinations.add((vtx, neighbor))
    return allowable_combinations

def update_allowable_combinations(combination_tuple, prev_allowable_combinations):
    allowable_combinations = set()
    c1, c2, comb = combination_tuple
    assert comb == c1 or comb == c2
    for combination in prev_allowable_combinations:
        cluster1, cluster2 = combination
        if cluster1 in combination_tuple:
            if cluster2 in combination_tuple:
                #Two have been combined, don't need to read them
                continue
            else:
                allowable_combinations.add((comb, cluster2))
        elif cluster2 in combination_tuple:
            #Implies cluster one is not...
            allowable_combinations.add((cluster1, comb))
        else:
            allowable_combinations.add((cluster1, cluster2))
    return allowable_combinations


def merge_then_ilp(words, word2freq, gamma = 0.5, edge_mat = None, word2idx = None):
    if word2idx is None:
        word2idx = {word: i for i, word in enumerate(words)}
    n_words = len(words)
    f = np.zeros(n_words)
    for word in words:
        f[word2idx[word]] = word2freq[word]
    f = f / f.sum()

    X = np.identity(n_words)
    C = np.identity(n_words)
    W = compute_W(words, word2idx, word2freq)

    current_num_clusters = len(words)
    found_good_merge = True
    mu = update_mu(C, f)

    best_objective, word_num_clusters = apply_objective(X, mu, f, C, W, gamma = gamma)
    allowable_combinations = get_allowable_combinations(edge_mat)
    #Update while combining clusters still lowers the objective...
    while found_good_merge:
        mu = update_mu(C, f)
        og_C = C.copy()
        C, current_min_objective, found_good_merge, combination_tuple = get_optimal_merge_efficient(X, mu, f, C, W, best_objective, gamma = gamma, combs = allowable_combinations)
        if found_good_merge:
            allowable_combinations = update_allowable_combinations(combination_tuple, allowable_combinations)
            current_num_clusters -= 1
        best_objective = current_min_objective
    return C, best_objective, word2idx

def process_ilp_output(new_cluster_assignment, word2idx, word2freq):

    new_clusters = defaultdict(set)
    word2newcluster = {}
    cluster2newrepresentative = {}
    idx2word = dict([(word2idx[word], word) for word in word2idx])
    nonzero_clusters = np.where(new_cluster_assignment.max(axis = 0) != 0)[0]
    relevant_cluster_assignments = new_cluster_assignment[:, nonzero_clusters]
    new_cluster_assignments = relevant_cluster_assignments.argmax(axis = 1)
    assert len(new_cluster_assignments.shape) == 1
    for i in range(new_cluster_assignments.shape[0]):
        word = idx2word[i]
        cluster = new_cluster_assignments[i]
        new_clusters[cluster].add(word)
        word2newcluster[word] = cluster
    for cluster in new_clusters:
        word_freq_pairs = [(word, word2freq[word]) for word in new_clusters[cluster]]
        word_freq_pairs.sort(key = lambda x: x[1], reverse = True)
        cluster2newrepresentative[cluster] = word_freq_pairs[0][0] #Take the most frequent element
    return new_clusters, word2newcluster, cluster2newrepresentative


def new_cluster_assignments(clusterer_path, gamma = 0.3, 
                            toy = False, save = True, job_num = 0, total_jobs = 1):
    print("Loading clusterer dict")
    if toy:
        clusterer_dict = {'cluster': {0: ['stop', 'step'], 1: ['plain', 'pin', 'pun'], 2: ['ham']},
                        'word2cluster': {'stop': 0, 'step': 0, 'plain': 1, 'pin': 1, 'pun': 1, 'ham': 2},
                        'word2freq': {'stop': 100, 'step': 50, 'plain': 75, 'pin': 15, 'pun': 10, 'ham': 5},
                        'cluster2representative': {0: 'stop', 1: 'plain', 2: 'ham'}}
    else:
        with open(clusterer_path, 'rb') as f:
            clusterer_dict = pickle.load(f)
    word2cluster = clusterer_dict['word2cluster']
    clusters = clusterer_dict['cluster']
    word2freq = clusterer_dict['word2freq']
    cluster2representative = clusterer_dict['cluster2representative']
    words = list(word2cluster)

    cluster_id_iter = clusters
    if total_jobs > 1:
        num_cluster_elems = [(cluster_id, len(clusters[cluster_id])) for cluster_id in clusters]
        num_cluster_elems.sort(key = lambda x: x[1], reverse = True)
        sorted_cluster_ids = np.array([cluster_id for (cluster_id, n_elems) in num_cluster_elems])
        if total_jobs == 2:
            if job_num == 0:
                job_cluster_ids = sorted_cluster_ids[:2]
            elif job_num == 1:
                job_cluster_ids = sorted_cluster_ids[2:]
            else:
                raise ValueError("Invalid job id for total jobs 2")
        else:
            job_cluster_ids = sorted_cluster_ids[job_num::total_jobs]
        cluster_id_iter = list(job_cluster_ids)
        words = []
        for cluster_id in cluster_id_iter:
            words.extend(clusters[cluster_id])

    split_clusters = {}
    word2split_cluster = {}
    split_cluster2representative = {}
    num_clusters_added = 0

    #Will use different word2freqs for each cluster to speed up computation
    cluster_word2freqs = defaultdict(dict)
    
    for word in words:
        cluster_id = word2cluster[word]
        cluster_word2freqs[cluster_id][word] = word2freq[word]

    print("Starting the preprocessing")
    for cluster_id in tqdm(cluster_id_iter, desc = 'Ed2 Clusters'):
        cluster_words = clusters[cluster_id]
        cluster_word2freq = cluster_word2freqs[cluster_id]
        print("Starting preprocessing for cluster {}, which contains {} words: ".format(cluster2representative[cluster_id], len(cluster_words)))
        start = datetime.now()
        edge_mat = ed1_neighbors_mat(cluster_words)

        new_cluster_assignment, loss, word2idx = merge_then_ilp(cluster_words, cluster_word2freq, gamma = gamma, edge_mat = edge_mat)
        print("Fishished preprocessing for cluster {}. Total time: {}: ".format(cluster2representative[cluster_id], str(datetime.now() - start)))

        new_clusters, word2newcluster, newcluster2representative = process_ilp_output(new_cluster_assignment, word2idx, cluster_word2freq)
        print("New clusters: ", new_clusters)
        for new_local_cluster_id in new_clusters:
            new_global_cluster_id = new_local_cluster_id + num_clusters_added
            split_clusters[new_global_cluster_id] = new_clusters[new_local_cluster_id]
            split_cluster2representative[new_global_cluster_id] = newcluster2representative[new_local_cluster_id]
        for word in word2newcluster:
            word2split_cluster[word] = word2newcluster[word] + num_clusters_added
        num_clusters_added += len(new_clusters)

    print("Getting typo2cluster")
    typo2cluster = get_typo2cluster(words, word2freq, word2split_cluster)

    save_dict = {'cluster': split_clusters, 'word2cluster': word2split_cluster, 
                    'cluster2representative': split_cluster2representative, 
                    'word2freq': word2freq, 'typo2cluster': typo2cluster}
   # print("About to save: ", save_dict)


    print("Num clusters: ", len(split_clusters))
    print("Num words with clusters: ", len(word2split_cluster))
    print("Num cluster representatives: ", len(split_cluster2representative))
    if total_jobs == 1:
        split_clusterer_path = '{}_gamma{}pkl'.format(clusterer_path.strip('.pkl'), gamma)
        #if save and not toy:
        if save:
            print("Saving clusters for gamma = {} at {}".format(gamma, split_clusterer_path))
            with open(split_clusterer_path, 'wb') as f:
                pickle.dump(save_dict, f)
            print("Saved!")
    else:
        split_clusterer_path_dir = '{}_gamma{}'.format(clusterer_path.strip('.pkl'), gamma)
        #if save and not toy:
        if save:
            if not os.path.isdir(split_clusterer_path_dir):
                os.mkdir(split_clusterer_path_dir)
            print("Saving clusters for gamma = {} at {}".format(gamma, split_clusterer_path_dir))
            split_clusterer_path = os.path.join(split_clusterer_path_dir, 'job{}outof{}'.format(job_num, total_jobs))
            with open(split_clusterer_path, 'wb') as f:
                pickle.dump(save_dict, f)
            print("Saved!")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gamma', type = float, required = True,
                        help = 'How to weight the different objective functions')
    parser.add_argument('--clusterer_path', type = str, required = True,
                        help = 'Connected component clusterer that is to be split')
    parser.add_argument('--no_save', action = 'store_true',
                        help = 'Whether or not to avoid saving...')
    parser.add_argument('--toy', action = 'store_true', help = 'Use toy clusters for testing')
    parser.add_argument('--job_id', default = 0, type = int, help = 'Job number for parallelization')
    parser.add_argument('--num_jobs', default = 1, type = int, help ='Total number of parallelization jobs')

    args = parser.parse_args()
    return args



if __name__ == '__main__':
    #toy()
    args = parse_args()
    new_cluster_assignments(args.clusterer_path, gamma = args.gamma, toy = args.toy, 
                                save = not args.no_save, job_num = args.job_id, total_jobs = args.num_jobs)

