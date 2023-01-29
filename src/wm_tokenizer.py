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
    segment_stack = []

    def __init__(self, vocab: dict, target: str):
        self.vocab = vocab
        self.target = target
        self.subvocab = [i for i in self.vocab if i in self.target]
        self.run = self.segmenter(self.target, stop=len(target))
        self.maxed = self.maximize_segments(vocab=self.vocab, segmentations=self.segment_stack)

    def segmenter(self, token, segments=None, start=0, stop=int):

        if segments is None:
            segments = []
        if start == stop:
            self.segment_stack.append(segments)

            return segments
        else:
            new_morpheme = [i for i in self.subvocab if token.startswith(i) and len(i) > 1]
            for m in new_morpheme:
                rest = token[len(m):]
                start += len(m)
                segments.append(m)
                self.segmenter(token=rest, segments=segments.copy(), start=start, stop=stop)
                segments = segments[:-1]
                start -= len(m)

    def maximize_segments(self, vocab, segmentations):
        ws = dict()
        len_tk = len("".join(segmentations[0]))
        for s in segmentations:
            coverage = [len(i) / len_tk for i in s]  # how much % of the word is covered by this morpheme
            morpheme_length = [len(i) for i in s]  # how long is each morpheme
            rel_freq = [vocab[i] for i in s]  # relative frequency of the morpheme in the vocab
            n_o_segs = [len(s) for i in s]

            lex_bias = sum([(np.tanh(mlen) / nsegs) for mlen, nsegs, freq in zip(morpheme_length, n_o_segs, rel_freq)])

            ws[lex_bias] = s
        return ws[max(ws.keys())]
