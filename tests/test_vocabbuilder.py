# -*- coding: utf-8 -*-
from ner.dataset.vocabbuilder import VocabBuilder, IndexedSentence


def test_vocab_builder_return_indexed_sentence(sample_pruned):
    vocab = VocabBuilder()
    vocab.fit(sample_pruned)
    idxed_sentences = vocab.transform(sample_pruned)
    assert isinstance(idxed_sentences, list)
    for idxed_sentence in idxed_sentences:
        assert isinstance(idxed_sentence, IndexedSentence)


def test_vocab_builder_return_all_numbers(sample_pruned):
    vocab = VocabBuilder()
    vocab.fit(sample_pruned)
    word_vocab = vocab.word_vocab.token2idx
    char_vocab = vocab.char_vocab.token2idx
    pos_tag_vocab = vocab.pos_tag_vocab.token2idx
    ne_vocab = vocab.ne_vocab.token2idx
    vocabs = [word_vocab, char_vocab, pos_tag_vocab, ne_vocab]
    for vocab in vocabs:
        for value in vocab.values():
            assert isinstance(value, int)


def test_word_char_vocabs_contain_unk(sample_pruned):
    vocab = VocabBuilder()
    vocab.fit(sample_pruned)
    word_vocab = vocab.word_vocab.token2idx
    word_unk_token = vocab.word_vocab.unk_token
    char_vocab = vocab.char_vocab.token2idx
    char_unk_token = vocab.char_vocab.unk_token
    assert word_unk_token in word_vocab
    assert char_unk_token in char_vocab


def test_all_vocabs_contain_pad(sample_pruned):
    vocab = VocabBuilder()
    vocab.fit(sample_pruned)
    word_vocab = vocab.word_vocab.token2idx
    char_vocab = vocab.char_vocab.token2idx
    pos_tag_vocab = vocab.pos_tag_vocab.token2idx
    ne_vocab = vocab.ne_vocab.token2idx
    word_pad_token = vocab.word_vocab.pad_token
    char_pad_token = vocab.char_vocab.pad_token
    pos_tag_token = vocab.pos_tag_vocab.pad_token
    ne_pad_token = vocab.ne_vocab.pad_token
    vocabs = [word_vocab, char_vocab, pos_tag_vocab, ne_vocab]
    pad_tokens = [word_pad_token, char_pad_token, pos_tag_token,
                  ne_pad_token]
    for pad_token, vocab in zip(pad_tokens, vocabs):
        assert pad_token in vocab


def test_all_words_exist_in_word_vocab(sample_pruned):
    vocab = VocabBuilder()
    vocab.fit(sample_pruned)
    word_vocab = vocab.word_vocab.token2idx
    for sample in sample_pruned:
        for word in sample.words:
            assert word in word_vocab


def test_all_chars_exist_in_char_vocab(sample_pruned):
    vocab = VocabBuilder()
    vocab.fit(sample_pruned)
    char_vocab = vocab.char_vocab.token2idx
    for sample in sample_pruned:
        for char in sample.chars:
            assert all([c in char_vocab for c in char])
