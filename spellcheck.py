__author__ = 'pkillewald'

import re
import collections


class SpellChecker:
    NWORDS = []

    alphabet = 'abcdefghijklmnopqrstuvwxyz'

    def __init__(self):
        with open('data/big.txt') as f:
            self.NWORDS = self._train(self._words(f.read()))

    @staticmethod
    def _words(text):
        return re.findall('[a-z]+', text.lower())

    @staticmethod
    def _train(features):
        model = collections.defaultdict(lambda: 1)
        for f in features:
            model[f] += 1
        return model

    def _edits1(self, word):
        s = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        deletes = [a + b[1:] for a, b in s if b]
        transposes = [a + b[1] + b[0] + b[2:] for a, b in s if len(b) > 1]
        replaces = [a + c + b[1:] for a, b in s for c in self.alphabet if b]
        inserts = [a + c + b for a, b in s for c in self.alphabet]
        return set(deletes + transposes + replaces + inserts)

    def _known_edits2(self, word):
        return set(e2 for e1 in self._edits1(word) for e2 in self._edits1(e1) if e2 in self.NWORDS)

    def _known(self, words):
        return set(w for w in words if w in self.NWORDS)

    def correct(self, word):
        return max(self._known([word])
                   or self._known(self._edits1(word))
                   or self._known_edits2(word)
                   or [word], key=self.NWORDS.get)
