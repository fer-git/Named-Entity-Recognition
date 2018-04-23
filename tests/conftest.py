# -*- coding: utf-8 -*-
import pytest

from ner.dataset.sentencepruner import Sentence
from ner.dataset.vocabbuilder import VocabBuilder


@pytest.fixture(scope='session')
def train_path():
    return './data/raw/conll_2012_train.txt'


@pytest.fixture(scope='session')
def dev_path():
    return './data/raw/conll_2012_dev.txt'


@pytest.fixture(scope='session')
def pretrained_folder():
    return './data/glove.6B'


class TestSentence(object):
    pass


@pytest.fixture(scope='session')
def sample_ontonotes():
    sentence_1 = TestSentence()
    sentence_1.words = ['This', 'is', 'a', 'sentence']
    sentence_1.pos_tags = ['DT', 'VBZ', 'DT', 'NN']
    sentence_1.named_entities = ['O', 'O', 'O', 'O']

    sentence_2 = TestSentence()
    sentence_2.words = ['Singapore', 'is', 'a', 'great', 'place', '.']
    sentence_2.pos_tags = ['NNP', 'VBZ', 'DT', 'JJ', 'NN', '.']
    sentence_2.named_entities = ['B-GPE', 'O', 'O', 'O', 'O', 'O']

    sentence_3 = TestSentence()
    sentence_3.words = ['This', 'will', 'cost', 'you', '1000', 'dollars']
    sentence_3.pos_tags = ['DT', 'MD', 'VB', 'PRP', 'CD', 'NNS']
    sentence_3.named_entities = ['O', 'O', 'O', 'O', 'B-CARDINAL', 'O']
    return [sentence_1, sentence_2, sentence_3]


@pytest.fixture(scope='session')
def sample_pruned():
    sentence_1 = Sentence(words=['this', 'is', 'a', 'sentence'],
                          chars=[['t', 'h', 'i', 's'], ['i', 's'], ['a'],
                                 ['s', 'e', 'n', 't', 'e', 'n', 'c', 'e']],
                          pos_tags=['DT', 'VBZ', 'DT', 'NN'],
                          named_entities=['O', 'O', 'O', 'O'])
    sentence_2 = Sentence(words=['singapore', 'is', 'a', 'great', 'place', '.'],
                          chars=[['s', 'i', 'n', 'g', 'a', 'p', 'o', 'r', 'e'],
                                 ['i', 's'], ['a'], ['g', 'r', 'e', 'a', 't'],
                                 ['p', 'l', 'a', 'c', 'e'], ['.']],
                          pos_tags=['NNP', 'VBZ', 'DT', 'JJ', 'NN', '.'],
                          named_entities=['B-GPE', 'O', 'O', 'O', 'O', 'O'])
    sentence_3 = Sentence(words=['this', 'will', 'cost', 'you', '<num>',
                                 'dollars'],
                          chars=[['t', 'h', 'i', 's'], ['w', 'i', 'l', 'l'],
                                 ['c', 'o', 's', 't'], ['y', 'o', 'u'],
                                 ['1', '0', '0', '0'], ['d', 'o', 'l', 'l', 'a',
                                                        'r', 's']],
                          pos_tags=['DT', 'MD', 'VB', 'PRP', 'CD', 'NNS'],
                          named_entities=['O', 'O', 'O', 'O', 'B-CARDINAL',
                                          'O'])
    return [sentence_1, sentence_2, sentence_3]


@pytest.fixture(scope='session')
def sample_vocab(sample_pruned):
    vocab = VocabBuilder()
    vocab.fit(sample_pruned)
    return vocab


@pytest.fixture(scope='session')
def sample_idxed_sentences(sample_pruned, sample_vocab):
    sample_vocab.fit(sample_pruned)
    sample_idxed = sample_vocab.transform(sample_pruned)
    return sample_idxed
