import torch
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
from nltk import word_tokenize
import json
import os
from tqdm import tqdm
from random import sample

from edit_dist_utils import get_all_edit_dist_one, sample_random_internal_permutations
from utils_glue import InputExample, PROCESSORS, OUTPUT_MODES

class AttackSurface(object):
    def get_perturbations(self, word):
        raise NotImplementedError


class ED1AttackSurface(AttackSurface):
    def get_perturbations(self, word):
        return get_all_edit_dist_one(word)


class Attacker():
    #Going to use attack to cache mapping from clean examples to attack, then run normally in their script. 
    def __init__(self, attack_name, task, model_runner, save_dir, args, max_num_words = None, perturbations_per_word = 4):
        ATTACK2CLASS = {'DeleteSubFirstAttack': DeleteSubFirstAttack, 'DeleteAttack': DeleteAttack, 
                        'RandomPerturbationAttack': RandomPerturbationAttack, 'LongDeleteShortAll': LongDeleteShortAll}
        if attack_name not in ATTACK2CLASS:
            raise ValueError("Invalid attack name: {}".format(attack_name))
        attack = ATTACK2CLASS[attack_name]()
        if attack_name == 'RandomPerturbationAttack':
            #TODO, should do something better in constructor, but this should work
            attack.perturbations_per_word = perturbations_per_word
            if args.attack_type.lower() == 'intprm':
                attack.attack_type = 'intprm'
        elif args.attack_type.lower() == 'intprm':
            raise NotImplementedError

        self.label_map = {label : i for i, label in enumerate(model_runner.label_list)}
        self.task = task
        self.model_runner = model_runner
        self.attack = attack
        self.args = args
        self.max_num_words = max_num_words
        self.save_dir = args.attack_save_dir
        self.total_count = 0
        self.attacked_count = 0

    def attack_dataset(self, dataset): #, force_new = False):
        adv_dataset = []
        for example in tqdm(dataset):
            num_a_words = len(example.text_a.split())

            perturbed_example = self._attack_example(example, max_num_words = self.max_num_words, verbose = False)
            adv_dataset.append(perturbed_example)
            if self.total_count % 100 == 0:
                print("Performance so far: successfully attacked {}/{} total = {}".format(self.attacked_count, self.total_count, self.attacked_count / self.total_count))

        return adv_dataset

    def _example_to_words(self, example):
        exists_b = example.text_b is not None
        split_a = example.text_a.split()
        words = split_a.copy()
        if exists_b:
            split_b = example.text_b.split()
            words.extend(split_b)
        return words, len(split_a)

    def _attack_example(self, clean_example, max_num_words = None, max_attack_attempts = 1, verbose = True):
        self.total_count += 1
        label = clean_example.label
        exists_b = clean_example.text_b is not None
        words, num_in_a = self._example_to_words(clean_example)
        if max_num_words is None or max_num_words > len(words):
            max_num_words = len(words)
        perturb_word_idxs = np.random.choice(len(words), size = max_num_words, replace = False)
        to_be_attacked = [words]
        for perturbed_word_idx in perturb_word_idxs:
            perturbed_examples = []
            for words in to_be_attacked:
                word_to_perturb = words[perturbed_word_idx]
                word_perturbations = self.attack.get_perturbations(word_to_perturb)
                for prtbd_word in word_perturbations:
                    og_copy = words.copy()
                    og_copy[perturbed_word_idx] = prtbd_word
                    new_guid = '{}-{}'.format(clean_example.guid, len(perturbed_examples))
                    if not exists_b:
                        perturbed_examples.append(InputExample(new_guid, ' '.join(og_copy), label = label))
                    else:
                        perturbed_examples.append(InputExample(new_guid, ' '.join(og_copy[:num_in_a]), label = label, text_b = ' '.join(og_copy[num_in_a:])))
            #Labels should all be the same, sanity check

            preds = self.model_runner.query(
                perturbed_examples, self.args.eval_batch_size, do_evaluate=False, 
                return_logits=True, use_tqdm=False)
            worst_performing_indices, found_incorrect_pred = self._process_preds(preds, self.label_map[label])
            if found_incorrect_pred:
                assert len(worst_performing_indices) == 1
                worst_performing_idx = worst_performing_indices[0]
                self.attacked_count += 1
                if verbose:
                    print('')
                    og_example_str = 'Premise: {}\nHypothesis: {}'.format(clean_example.text_a, '' if not exists_b else clean_example.text_b)
                    print(og_example_str)
                    attacked_str = 'Premise: {}\nHypothesis: {}'.format(perturbed_examples[worst_performing_idx].text_a, '' if not exists_b else perturbed_examples[worst_performing_idx].text_b)
                    print(attacked_str)
                    print("Original label: {}".format(clean_example.label))
                    print("Attacked prediction: {}".format(self.model_runner.label_list[np.argmax(preds, axis = 1)[worst_performing_idx]]))

                return perturbed_examples[worst_performing_idx]
            else:
                to_be_attacked = []
                for idx in worst_performing_indices: 
                    new_words, _ = self._example_to_words(perturbed_examples[idx])
                    to_be_attacked.append(new_words)
        #Didn't find a successful attack, but still going to do worst case thing...
        if verbose:
            print('')
            print("Could not attack the following: ")
            og_example_str = 'Premise: {}\nHypothesis: {}'.format(clean_example.text_a, '' if not exists_b else clean_example.text_b)
            print(og_example_str)
        return perturbed_examples[worst_performing_indices[0]]

    def _process_preds(self, preds, label):
        #Should return a list of predictions, and whether or not a label is found...
        raise NotImplementedError

class BeamSearchAttacker(Attacker):
    def __init__(self, attack_name, task, model_runner, save_dir, args, max_num_words = None):
        super(BeamSearchAttacker, self).__init__(attack_name, task, model_runner, save_dir, args, max_num_words = max_num_words)
        self.beam_width = args.beam_width

    def _process_preds(self, preds, label):
        argmax_preds = np.argmax(preds, axis = 1)
        if not (argmax_preds == label).all():
            incorrect_idx = np.where(argmax_preds != label)[0][0]
            return [incorrect_idx], True
        if preds.shape[0] <= self.beam_width:
            return list(range(preds.shape[0])), False
        worst_performing_indices = np.argpartition(preds[:, label], self.beam_width)[:self.beam_width]
        return list(worst_performing_indices), False

class GreedyAttacker(Attacker):
    def _process_preds(self, preds, label):
        #Assumes if a pred changes the prediction, it's the only thing returned. Otherwise, returns list...
        argmax_preds = np.argmax(preds, axis = 1)
        if not (argmax_preds == label).all():
            incorrect_idx = np.where(argmax_preds != label)[0][0]
            return [incorrect_idx], True
        worst_performing_idx = np.argmin(preds[:,label])
        return [worst_performing_idx], False

class Attack():
    def get_perturbations(self, word):
        raise NotImplementedError()

    def name(self):
        raise NotImplementedError()


class LongDeleteShortAll(Attack):
    def __init__(self, perturbations_per_word = 4, max_insert_len = 4):
        self.cache = {}
        self.perturbations_per_word = perturbations_per_word
        self.max_insert_len = max_insert_len

    def get_perturbations(self, word):
        if word in self.cache:
            return self.cache[word]
        if len(word) > self.max_insert_len:
            perturbations = get_all_edit_dist_one(word, filetype = 100) #Just deletions
        else:
            perturbations = get_all_edit_dist_one(word)
            if len(perturbations) > self.perturbations_per_word:
                perturbations = set(sample(perturbations, self.perturbations_per_word))
        self.cache[word] = perturbations
        return perturbations

    def name(self):
        return 'LongDeleteShortAll'

class RandomPerturbationAttack(Attack):
    def __init__(self, perturbations_per_word = 5, attack_type = 'ed1'):
        self.cache = {}
        self.perturbations_per_word = perturbations_per_word
        self.attack_type = attack_type

    def get_perturbations(self, word):
        if word in self.cache:
            return self.cache[word]
        if self.attack_type == 'ed1':
            perturbations = get_all_edit_dist_one(word)
            if len(perturbations) > self.perturbations_per_word:
                pertubations = set(sample(perturbations, self.perturbations_per_word))
        elif self.attack_type == 'intprm':
            perturbations = sample_random_internal_permutations(word, n_perts = self.perturbations_per_word)
        else:
            raise NotImplementedError("Attack type: {} not implemented yet".format(self.attack_type))
        self.cache[word] = perturbations
        return perturbations

    def name(self):
        return 'RandomPerturbationAttack'


class DeleteSubFirstAttack(Attack):
    def __init__(self):
        self.cache = {}

    def get_perturbations(self, word):
        if len(word) < 3: #Min case where a substitution is possible
            return set([word])
        if word in self.cache:
            return self.cache[word]
        deletions = get_all_edit_dist_one(word, filetype = 100) #Just deletions
        substution_heads = get_all_edit_dist_one(word[:3], filetype = 10) #Just substitutions in pos 2 (can sub middle of first three letters)
        second_char_substitutions = set([head + word[3:] for head in substution_heads])
        perturbations = deletions.union(second_char_substitutions)
        self.cache[word] = perturbations
        return perturbations

    def name(self):
        return 'DeleteSubFirstAttack'

class DeleteAttack(Attack):
    def __init__(self):
        self.cache = {}

    def get_perturbations(self, word):
        if len(word) < 3:
            return set([word])
        if word in self.cache:
            return self.cache[word]
        deletions = get_all_edit_dist_one(word, filetype = 100)
        self.cache[word] = deletions
        return deletions

    def name(self):
        return 'DeleteAttack'


