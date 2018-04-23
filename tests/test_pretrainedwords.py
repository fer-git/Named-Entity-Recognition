# -*- coding: utf-8 -*-
from ner.dataset.pretrainedwords import PretrainedWords


def test_sample_vocab_has_same_words(sample_vocab, pretrained_folder):
    word_vocab = sample_vocab.word_vocab.token2idx
    words = word_vocab.keys()
    pretrained = PretrainedWords(pretrained_folder, 50, word_vocab)
    assert all([word in pretrained.words for word in words])


def test_pruned_embedding_has_correct_dimension(sample_vocab,
                                                pretrained_folder):
    word_vocab = sample_vocab.word_vocab.token2idx
    num_words = len(word_vocab)
    pretrained = PretrainedWords(pretrained_folder, 50, word_vocab)
    assert pretrained.embedding_matrix.shape == (num_words, 50)


def test_pad_index_in_correct_position(sample_vocab, pretrained_folder):
    word_vocab = sample_vocab.word_vocab.token2idx
    pad_token = sample_vocab.word_vocab.pad_token
    pad_idx = word_vocab[pad_token]
    pretrained = PretrainedWords(pretrained_folder, 50, word_vocab)
    assert pretrained.words[pad_idx] == pad_token

