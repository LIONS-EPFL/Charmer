from functools import reduce
import itertools
import json
import numpy as np
import os
import pickle
import random
import torch
from tqdm import tqdm

from scRNN.corrector import ScRNNChecker
from utils_roben import OOV_CLUSTER, OOV_TOKEN
from utils_glue import InputExample
from edit_dist_utils import get_sorted_word

class Recoverer(object):
    """Clean up a possibly typo-ed string."""
    def __init__(self, cache_dir):
        self.cache_dir = cache_dir
        self.cache = {}
        self.name = None  # Subclasses should set this

    def _cache_path(self):
        return os.path.join(self.cache_dir, 'recoveryCache-{}.json'.format(self.name))

    def load_cache(self):
        path = self._cache_path()
        if os.path.exists(path):
            with open(self._cache_path()) as f:
                self.cache = json.load(f)
            print('Recoverer: loaded {} values from cache.'.format(len(self.cache)))
        else:
            print('Recoverer: no cache at {}.'.format(path))

    def save_cache(self, save = False):
        if save:
            cache_path = self._cache_path()
            print('Recoverer: saving {} cached values to {} .'.format(len(self.cache), cache_path))
            with open(cache_path, 'w') as f:
                json.dump(self.cache, f)


    def recover(self, text):
        """Recover |text| to a new string.
        
        Used at test time to preprocess possibly typo-ed input. 
        """
        if text in self.cache:
            return self.cache[text]
        recovered = self._recover(text)
        self.cache[text] = recovered
        return recovered

    def _recover(self, text):
        """Actually do the recovery for self.recover()."""
        raise NotImplementedError

    def get_possible_recoveries(self, text, attack_surface, max_num, analyze_res_attacks = False, ret_ball_stats = False):
        """For a clean string, return list of possible recovered strings, or None if too many.
        
        Used at certification time to exactly compute robust accuracy.

        Returns tuple (list_of_possibilities, num_possibilities)
        where list_of_possibilities is None if num_possibilities > max_num.
        """
        pass

    def recover_example(self, example):
        """Recover an InputExample |example| to a new InputExample.
        
        Used at test time to preprocess possibly typo-ed input. 
        """
        tokens = example.text_a.split()
        a_len = len(tokens)
        if example.text_b:
            tokens.extend(example.text_b.split())
        recovered_tokens = self.recover(' '.join(tokens)).split()
        a_new = ' '.join(recovered_tokens[:a_len])
        if example.text_b:
            b_new = ' '.join(recovered_tokens[a_len:])
        else:
            b_new = None
        return InputExample(example.guid, a_new, b_new, example.label)

    def get_possible_examples(self, example, attack_surface, max_num, analyze_res_attacks = False):
        """For a clean InputExample, return list of InputExample's you could recover to.
        
        Used at certification time to exactly compute robust accuracy.
        """
        tokens = example.text_a.split()
        a_len = len(tokens)
        if example.text_b:
            tokens.extend(example.text_b.split())
        possibilities, num_poss, perturb_counts = self.get_possible_recoveries(' '.join(tokens), attack_surface, max_num,
                                    analyze_res_attacks = analyze_res_attacks)
        if perturb_counts is not None:
            assert len(perturb_counts) == len(possibilities)
        if not possibilities:
            return (None, num_poss)
        out = []
        example_num = 0
        for i in range(len(possibilities)):
            poss = possibilities[i]
            poss_tokens = poss.split()
            a = ' '.join(poss_tokens[:a_len])
            if example.text_b:
                b = ' '.join(poss_tokens[a_len:])
            else:
                b = None
            if not analyze_res_attacks:
                poss_guid = '{}-{}'.format(example.guid, example_num)
            else:
                poss_guid = '{}-{}-{}'.format(example.guid, example_num, perturb_counts[i])
            out.append(InputExample('{}-{}'.format(poss_guid, example_num), a, b, example.label))
            example_num += 1
        return (out, len(out))


class IdentityRecoverer(Recoverer):
    def __init__(self, cache_dir):
        super(IdentityRecoverer, self).__init__(cache_dir)
        self.name = 'IdentityRecoverer'

    def recover(self, text):
        """Override self.recover() rather than self._recover() to avoid cache."""
        return text

class ClusterRecoverer(Recoverer):
    def __init__(self, cache_dir, clustering):
        super(ClusterRecoverer, self).__init__(cache_dir)
        self.clustering = clustering
        self.passthrough = False

    def get_possible_recoveries(self, text, attack_surface, max_num, analyze_res_attacks = False, ret_ball_stats = False):
        tokens = text.split()
        possibilities = []
        perturb_counts = []
        standard_clustering = np.array([self.clustering.map_token(token) for token in tokens])
        for token in tokens:
            cur_perturb = attack_surface.get_perturbations(token)
            perturb_counts.append(len(cur_perturb))
            poss_clusters = set()
            for pert in cur_perturb:
                clust_id = self.clustering.map_token(pert)
                poss_clusters.add(clust_id)
            possibilities.append(sorted(poss_clusters, key=str))  # sort for deterministic order
        if ret_ball_stats:
            return [len(pos_clusters) for pos_clusters in possibilities], perturb_counts
        num_pos = reduce(lambda x, y: x * y, [len(x) for x in possibilities])
        if num_pos > max_num:
            return (None, num_pos, None)
        poss_recoveries = []
        perturb_counts = None
        if analyze_res_attacks:
            perturb_counts = []
        num_zero = 0
        for clust_seq in itertools.product(*possibilities):
            if analyze_res_attacks:
                #print("Stand: ", standard_clustering)
                #print("Seq: ", clust_seq)
                #print("Lengths: {}, {}".format(len(standard_clustering), len(clust_seq)))
                #print("Types: {}, {}".format(type(np.array(clust_seq)[0]), type(standard_clustering[0])))
                #print("Comparison: ", np.array(clust_seq) != standard_clustering)
                #print("Inv comparison: ", np.array(clust_seq) == standard_clustering)
                num_different = (np.array(clust_seq) != standard_clustering).sum()
                if num_different == 0:
                    num_zero += 1
                #print(num_different)
                perturb_counts.append(num_different)
            poss_recoveries.append(self._recover_from_clusters(clust_seq))
        assert num_zero == 1 or not analyze_res_attacks
        return (poss_recoveries, len(poss_recoveries), perturb_counts)

    def _recover(self, text):
        tokens = text.split()
        clust_ids = [self.clustering.map_token(w, passthrough = self.passthrough) for w in tokens]
        return self._recover_from_clusters(clust_ids)

    def _recover_from_clusters(self, clust_ids):
        raise NotImplementedError


class ClusterRepRecoverer(ClusterRecoverer):
    def _recover_from_clusters(self, clust_ids):
        tokens = []
        for c in clust_ids:
            if c == OOV_CLUSTER:
                tokens.append('[MASK]')
            else:
                tokens.append(self.clustering.get_rep(c))
        return ' '.join(tokens)

class ClusterIntprmRecoverer(ClusterRepRecoverer):
    def get_possible_recoveries(self, text, attack_surface, max_num, analyze_res_attacks = False, ret_ball_stats = False):
        if analyze_res_attacks:
            raise NotImplementedError
        tokens = text.split()
        clusters = []
        for token in tokens:
            token_key = get_sorted_word(token) #Adversary can't modify sorted word, since attack is internal perturbations
            clust_id = self.clustering.map_token(token_key)
            clusters.append(clust_id)
        recovery = self._recover_from_clusters(clusters)
        return ([recovery], 1, None) #One possibility, and no perturb_counts since we don't analyze resticted attack yet.

    def _recover(self, text):
        tokens = text.split()
        clust_ids = [self.clustering.map_token(get_sorted_word(w), passthrough = False) for w in tokens]
        return self._recover_from_clusters(clust_ids)

class ScRNNRecoverer(Recoverer):
    def __init__(self, cache_dir, tc_dir, task_name):
        super(ScRNNRecoverer, self).__init__(cache_dir)
        self.checker = ScRNNChecker(tc_dir, unk_output=True, task_name=task_name)

    def _recover(self, text):
        return self.checker.correct_string(text)



RECOVERERS = {
        'identity': IdentityRecoverer,
        'clust-rep': ClusterRepRecoverer,
        'clust-intprm': ClusterIntprmRecoverer,
        'scrnn': ScRNNRecoverer,
}
