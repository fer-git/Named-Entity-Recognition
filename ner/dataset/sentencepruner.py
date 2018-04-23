# -*- coding: utf-8 -*-
from typing import List, NamedTuple


class Sentence(NamedTuple):
    words: List[str]
    chars: List[List[str]]
    pos_tags: List[str]
    named_entities: List[str]

    @property
    def num_words(self):
        return len(self.words)

    @property
    def num_chars(self):
        return len(self.chars)


class SentencePruner(object):
    """
    Preprocess each token of words, pos tags, and named_entities
    Generate characters feature from words
    """
    def __init__(self, preprocess_words=None, preprocess_chars=None,
                 preprocess_pos_tags=None, preprocess_named_entities=None):
        """
        Initialize sentence pruner object
        Parameters
        ----------
        preprocess_words: Function to be applied on each word
        preprocess_chars: Function to be applied on each char
        preprocess_pos_tags: Function to be applied on each pos tag
        preprocess_named_entities: Function to be applied on each named entity
        """
        if preprocess_words:
            self.preprocess_words = preprocess_words
        else:
            self.preprocess_words = self._prep_words

        if preprocess_chars:
            self.preprocess_chars = preprocess_chars
        else:
            self.preprocess_chars = self._prep_chars

        if preprocess_pos_tags:
            self.preprocess_pos_tags = preprocess_pos_tags
        else:
            self.preprocess_pos_tags = self._prep_pos_tags

        if preprocess_named_entities:
            self.preprocess_named_entities = preprocess_named_entities
        else:
            self.preprocess_named_entities = self._prep_named_entities

    def transform(self, sentences) -> List[Sentence]:
        """
        Preprocess each word, pos tag, and named entity from each sentence
        Parameters
        ----------
        sentences: List of sentence object that has attributes words, pos_tags,
        and named_entities

        Returns
        -------
        List of sentence object that has attributes words, chars, pos_tags,
        and named_entities

        """
        results = []
        for sentence in sentences:
            clean_words = self.preprocess_words(sentence.words)
            chars = [[char for char in words] for words in sentence.words]
            clean_chars = self.preprocess_chars(chars)
            clean_pos_tags = self.preprocess_pos_tags(sentence.pos_tags)
            clean_named_entities = self.preprocess_named_entities(
                sentence.named_entities)
            results.append(Sentence(clean_words, clean_chars, clean_pos_tags,
                                    clean_named_entities))
        return results

    @staticmethod
    def _prep_words(words: List[str]) -> List[str]:
        """
        Default word processor

        Lowercase and replace digit string with <num>
        Parameters
        ----------
        words: List of words from a sentence

        Returns
        -------
        List of preprocessed words
        """
        results = []
        for word in words:
            if word.isdigit():
                results.append('<num>')
            else:
                results.append(word.lower())
        return results

    @staticmethod
    def _prep_chars(chars: List[List[str]]) -> List[List[str]]:
        """
        Default char processor

        Do nothing on each char
        Parameters
        ----------
        chars: List of list of character from each word of each sentence

        Returns
        -------
        List of list of preprocessed chars

        """
        return chars

    @staticmethod
    def _prep_pos_tags(pos_tags: List[str]) -> List[str]:
        """
        Default pos tags processor

        Do nothing on each pos tag
        Parameters
        ----------
        pos_tags: List of pos tags from each sentence

        Returns
        -------
        List of preprocessed pos tags

        """
        return pos_tags

    @staticmethod
    def _prep_named_entities(named_entities: List[str]) -> List[str]:
        """
        Default pos tags processor

        Do nothing on each named entity
        Parameters
        ----------
        named_entities: List of named entities from each sentence

        Returns
        -------
        List of preprocessed named entities
        """
        return named_entities
