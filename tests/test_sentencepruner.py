# -*- coding: utf-8 -*-
from ner.dataset.sentencepruner import SentencePruner, Sentence


def test_pruner_return_list_of_sentences(sample_ontonotes):
    pruned_samples = SentencePruner().transform(sample_ontonotes)
    assert isinstance(pruned_samples, list)
    assert isinstance(pruned_samples[0], Sentence)


def test_pruner_lowercase_words(sample_ontonotes):
    pruned_samples = SentencePruner().transform(sample_ontonotes)
    combined_words = ''.join(pruned_samples[0].words)
    assert combined_words.islower()


def test_pruner_characters_all_singles(sample_ontonotes):
    pruned_samples = SentencePruner().transform(sample_ontonotes)
    chars = pruned_samples[0].chars
    for char in chars:
        assert all([len(c) == 1 for c in char])


def test_pruner_pos_tags_ne_same_length_as_words(sample_ontonotes):
    pruned_samples = SentencePruner().transform(sample_ontonotes)
    for pruned_sample in pruned_samples:
        words = pruned_sample.words
        pos_tags = pruned_sample.pos_tags
        named_entities = pruned_sample.named_entities
        assert len(words) == len(pos_tags)
        assert len(words) == len(named_entities)



