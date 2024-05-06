import pickle

#OOV_CLUSTER = '<OOV_CLUSTER>'
OOV_CLUSTER = -1 #Chnaged for 
OOV_TOKEN = '<UNK>'


class ModelRunner(object):
    """Object that can run a model on a given dataset."""
    def __init__(self, recoverer, output_mode, label_list, output_dir, device):
        self.recoverer = recoverer
        self.output_mode = output_mode
        self.label_list = label_list
        self.output_dir = output_dir
        self.device = device

    def train(self, train_data, args):
        """Given already-recovered data, train the model."""
        raise NotImplementedError

    def query(self, examples, batch_size, do_evaluate=True, return_logits=False, 
              do_recover=True, use_tqdm=True):
        """Run the recoverer on raw data and query the model on examples."""
        raise NotImplementedError

class Clustering(object):
    """Object representing an assignment of words to clusters.
    
    Provides some utilities for dealing with words, typos, and clusters.
    """
    def __init__(self, clusterer_dict, max_num_possibilities=None, passthrough=False):
        self.cluster2elements = clusterer_dict['cluster']
        self.word2cluster = clusterer_dict['word2cluster']
        self.cluster2representative = clusterer_dict['cluster2representative']
        self.word2freq = clusterer_dict['word2freq']
        self.typo2cluster = clusterer_dict['typo2cluster']
        if max_num_possibilities:
            self.cluster2elements = self.filter_possibilities(max_num_possibilities)

    def filter_possibilities(self, max_num_possibilities):
        filtered_cluster2elements = {}
        for cluster in self.cluster2elements:
            elements = self.cluster2elements[cluster]
            frequency_list = [(elem, self.word2freq[elem]) for elem in elements]
            frequency_list.sort(key = lambda x: x[1], reverse = True)
            filtered_elements = [pair[0] for pair in frequency_list[:max_num_possibilities]]
            filtered_cluster2elements[cluster] = filtered_elements
        return filtered_cluster2elements

    @classmethod
    def from_pickle(cls, path, **kwargs):
        with open(path, 'rb') as f:
            clusterer_dict = pickle.load(f)
        return cls(clusterer_dict, **kwargs)

    def get_words(self, cluster):
        if cluster == OOV_CLUSTER:
            return [OOV_TOKEN]
        return self.cluster2elements[cluster]

    def in_vocab(self, word):
        return word in self.word2cluster

    def get_cluster(self, word):
        """Get cluster of a word, or OOV_CLUSTER if out of vocabulary."""
        word = word.lower()
        if word in self.word2cluster:
            return self.word2cluster[word]
        return OOV_CLUSTER

    def get_rep(self, cluster):
        """Get representative for a cluster."""
        if cluster == OOV_CLUSTER:
            return OOV_TOKEN
        return self.cluster2representative[cluster]

    def get_freq(self, word):
        return self.word2freq[word]

    def map_token(self, token, remap_vocab=True, passthrough = False):
        """Map a token (possibly a typo) to a cluster.
        
        Args:
            token: a token, possibly a typo
            remap_vocab: if False, always map vocab words to themselves,
                because perturbing vocab words has been disallowed.
            passthrough: Allow OOV to go to downstream model...
        """
        token = token.lower()
        if token in self.word2cluster and not remap_vocab:
            return self.get_cluster(token)
        if token in self.typo2cluster:
            return self.typo2cluster[token]
        if passthrough:
            return token
        return OOV_CLUSTER


def pkl_save(obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)

def pkl_load(filename):
    with open(filename, 'rb') as f:
        obj = pickle.load(f)
        return obj

