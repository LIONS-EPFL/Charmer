import argparse
import sys
import itertools
from collections import defaultdict
from tqdm import tqdm

from utils_roben import pkl_save, pkl_load
from edit_dist_utils import get_all_edit_dist_one, get_all_internal_permutations, get_sorted_word

TOY_VOCAB = ['cat', 'bat', 'car', 'bar', 'airplane!!!']
GLOVE_PATH = 'data/glove/glove.6B.50d.txt'

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--vocab-type', choices=['lm', 'glove'], default = 'glove',
                        help = 'Where to get the vocab from')
  parser.add_argument('--save-root', type = str, default = '',
                        help = 'Name used to format vocab preprocessing output')
  parser.add_argument('--filetype', type = int, default = 1111,
                        help = 'insert, delete, substitute, swap')
  parser.add_argument('--perturb_type', default = 'ed1', type = str, help = 'Type of perturbation to make dict for.')
  return parser.parse_args()


def load_glove_vocab(glove_path, num_lines = 400000):
    print("Reading GloVe vectors from {}...".format(glove_path))
    vocab = []
    with open(glove_path) as f:
        for i, line in tqdm(enumerate(f), total=num_lines):
            toks = line.strip().split(' ')
            word = toks[0]
            vocab.append(word)
    return vocab

def vocab_from_lm(lm):
    print("Possible vocab size: ", len(lm.word_to_idx))
    vocab = list(lm.word_to_idx)
    vocab = [word for word in  vocab if word.isalpha() and word == word.lower()]
    print("Vocab size after flitering: ", len(vocab))
    return vocab

def preprocess_neighbors_intprm(vocab):
    neighbor_trans_map = None
    sorted2word = defaultdict(set)
    vocab = [word.lower() for word in vocab]
    print("Grouping by sorted word")
    for word in tqdm(vocab):
        sorted_word = get_sorted_word(word)
        sorted2word[sorted_word].add(word)

    neighbor_trans_map = None
    print("Constructing edges...")
    neighbor_map = defaultdict(set)
    for sorted_word in tqdm(sorted2word):
        permutations = itertools.permutations(sorted2word[sorted_word], r = 2)
        for src, dest in permutations:
            neighbor_map[src].add(dest)
        #Allow self-edges
        for src in sorted2word[sorted_word]:
            neighbor_map[src].add(src)
    return sorted2word, neighbor_map, neighbor_trans_map



def preprocess_neighbors(vocab, filetype = 1111, sub_restrict = None):
    #For efficiency, assume edit distance 1 is symmetric. Not true for certain filetypes, so perturbations act accordingly...
    typo2vocab = defaultdict(set)
    print("Making typo dict...")
    for word in tqdm(vocab):
        perturbations = get_all_edit_dist_one(word, filetype = filetype, sub_restrict = sub_restrict)

        for typo in perturbations:
            typo2vocab[typo].add(word)

    print("Constructing edges...")
    neighbor_map = defaultdict(set)
    neighbor_trans_map = defaultdict(set)
    for typo in tqdm(typo2vocab):
        permutations = itertools.permutations(typo2vocab[typo], r = 2)
        for src, dest in permutations:
            neighbor_map[src].add(dest)
            neighbor_trans_map[(src, dest)].add(typo)
        #Allow self-edges
        for src in typo2vocab[typo]:
            neighbor_map[src].add(src)
            neighbor_trans_map[(src, src)].add(src)
    return typo2vocab, neighbor_map, neighbor_trans_map



def preprocess_vocab(args):
    if args.vocab_type == 'glove':
        vocab = load_glove_vocab(GLOVE_PATH)
    elif args.vocab_type == 'lm':
        query_handler = load_language_model()
        vocab = vocab_from_lm(query_handler)
    else:
        raise ValueError("Invalid vocab type of {}".format(args.vocab_type))
    print("Vocab sample: ", vocab[300:500])
    sub_dict = None
    if args.modify_end:
        print("Modifying the end...")
    if args.perturb_type == 'ed1':
        typo2vocab, ed2_neighbors, neighbor_trans_map = preprocess_neighbors(vocab, filetype = args.filetype, 
                                                                        sub_restrict = sub_dict)
    elif args.perturb_type == 'intprm':
        typo2vocab, ed2_neighbors, neighbor_trans_map = preprocess_neighbors_intprm(vocab)
    else:
        raise NotImplementedError
    pkl_save(ed2_neighbors, 'ed2_neighbors{}pt{}.pkl'.format(args.save_root, args.perturb_type))
    print("Saved ed2")

def get_neighbors(args):
    neighbor_path = 'ed2_neighbors{}pt{}.pkl'.format(args.save_root, args.perturb_type)
    neighbor_dict = pkl_load(neighbor_path)
    while True:
        print('broad' in neighbor_dict['bold'])
        inpt = input("Enter a word: ")
        inpt = inpt.lower()
        if inpt not in neighbor_dict:
            print("Word not preprocessed...")
        else:
            print("Neighbors for {}:".format(inpt))
            print(neighbor_dict[inpt])

if __name__ == '__main__':
    args = parse_args()
    get_neighbors(args)
