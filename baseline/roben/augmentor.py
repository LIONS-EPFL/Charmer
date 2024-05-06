from random import sample

from utils_glue import InputExample
from attacks import ED1AttackSurface

class Augmentor():
    def __init__(self, attack_surface = None):
        if attack_surface is None:
            attack_surface = ED1AttackSurface()
        self.attack_surface = attack_surface

    def augment_dataset(self, dataset):
        augmented_examples = []
        for example in dataset:
            augmented = self._augment_example(example)
            augmented_examples.extend(augmented)
        return augmented_examples

    def _augment_example(self, example):
        #Should return a list, to allow for multiple 
        raise NotImplementedError

class IdentityAugmentor(Augmentor):
    def _augment_example(self, example):
        return [example]

class HalfAugmentor(Augmentor):
    #New training dataset is double the size, with half normal and half randomly augmented...
    def _augment_example(self, example):
        tokens = example.text_a.split()
        a_len = len(tokens)
        if example.text_b:
            tokens.extend(example.text_b.split())
        augmented_version = []
        for token in tokens:
            possible_perturbations = self.attack_surface.get_perturbations(token)
            augmented_version.append(sample(possible_perturbations, 1)[0])
        augmented_a = augmented_version[:a_len]
        a_aug = ' '.join(augmented_a)
        b_aug = None
        if example.text_b:
            augmented_b = augmented_version[a_len:]
            b_aug = ' '.join(augmented_b)
        augmented_example = InputExample('{}-AUG'.format(example.guid), a_aug, b_aug, example.label)
        return [example, augmented_example]

class KAugmentor(Augmentor):
    #TODO, should allow changing of k outside...
    def _augment_example(self, example, k = 4):
        tokens = example.text_a.split()
        a_len = len(tokens)
        if example.text_b:
            tokens.extend(example.text_b.split())
        augmented_examples = []
        for i in range(k):
            augmented_version = []
            for token in tokens:
                possible_perturbations = self.attack_surface.get_perturbations(token)
                augmented_version.append(sample(possible_perturbations, 1)[0])
            augmented_a = augmented_version[:a_len]
            a_aug = ' '.join(augmented_a)
            b_aug = None
            if example.text_b:
                augmented_b = augmented_version[a_len:]
                b_aug = ' '.join(augmented_b)
            augmented_example = InputExample('{}-AUG{}'.format(example.guid, i), a_aug, b_aug, example.label)
            augmented_examples.append(augmented_example)
        return [example, *augmented_examples]



AUGMENTORS = {'identity': IdentityAugmentor, 'half-aug': HalfAugmentor, 'k-aug': KAugmentor}

