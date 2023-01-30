#!/usr/bin/env python
__author__ = "Ricardo Jung"
__email__ = "s2458588@stud.uni-frankfurt.de"

# __copyright__ = ""
# __credits__ = [""]
# __license__ = ""
# __version__ = ""
# __maintainer__ = ""
# __status__ = ""

import numpy as np


class SequenceTokenizer:
    """Generate a series of possible segmentations from a given vocabulary (dict: {string:relative freqency}) for a
    given target token."""

    # segment_stack = []

    def __init__(self, vocab: dict, target: str):
        self.vocab = vocab
        self.target = target
        self.length = len(target)
        self.subvocab = [i for i in self.vocab if i in self.target]
        self.segmentations = []
        self.segmenter(self.target, stop=self.length)
        self.maxed = None
        self.maximize_segments()

    def fill_stack(self, s):
        self.segmentations.append(s)

    def segmenter(self, token, stop: int, start=0, segments=None, stack=None):
        if stack is None:  # avoiding mutables in default arguments
            stack = []
        if segments is None:
            segments = []
        if start == stop:
            stack.append(segments)
            self.segmentations.append(segments)
            # self.fill_stack(segments)

        else:
            new_morpheme = [i for i in self.subvocab if token.startswith(i) and len(i) > 1]
            for m in new_morpheme:
                start += len(m)
                segments.append(m)
                rest = token[len(m):]
                if len(rest) == 1:
                    start = stop
                    segments.append(rest)
                    self.segmentations.append(segments)
                    segments = segments[:-2]
                    start -= (len(m) + 1)
                else:
                    self.segmenter(token=rest, stop=stop, start=start, segments=segments.copy(), stack=stack.copy())
                    segments = segments[:-1]
                    start -= len(m)

    def maximize_segments(self):
        ws = dict()
        len_tk = self.length
        for s in self.segmentations:
            coverage = [len(i) / len_tk for i in s]  # how much % of the word is covered by this morpheme
            morpheme_length = [len(i) for i in s]  # how long is each morpheme
            rel_freq = [self.vocab[i] for i in s]  # relative frequency of the morpheme in the vocab
            n_o_segs = [len(s) for i in s]

            lex_bias = sum([(np.tanh(mlen) / nsegs) for mlen, nsegs, freq in zip(morpheme_length, n_o_segs, rel_freq)])

            ws[lex_bias] = s
        self.maxed = ws[max(ws.keys())]



