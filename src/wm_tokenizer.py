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


# from HanTa import HanoverTagger as ht
# from transformers import BertTokenizer
# from tokenizers import pre_tokenizers

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


class WordmapTokenizer:
    def __init__(self, bert_pretokenizer, bert_tokenizer, hantatagger, vocab):
        self.bert_pretokenizer = bert_pretokenizer
        self.bert_tokenizer = bert_tokenizer
        self.hantatagger = hantatagger
        self.vocab = vocab
        self.bert_tokenizer.add_vocab(list(self.vocab.keys()))

    # from transformers import BertTokenizer
    # tk = BertTokenizer.from_pretrained("bert-base-german-cased")
    def wordmap2tokenizer(self, data: str, pos_tag: str, vocab, pt=None, tk=None, tg=None):
        """Takes a string, a STTS Pos-tag and a vocabulary: dict/list. Wordmap will only tokenize tokens with the
        given pos-tag. The rest will be done by BERT tokenizers.\n Requires:\n tk = BertTokenizer.from_pretrained(
        "bert-base-german-cased")\n tg = ht.HanoverTagger('morphmodel_ger.pgz')\n
        pt = pre_tokenizers.BertPreTokenizer()"""

        encoding = []
        sent = [s[0] for s in pt.pre_tokenize_str(data)]
        for tkn in sent:
            if tg.analyze(tkn)[1].startswith(pos_tag):
                wmt = SequenceTokenizer(vocab=vocab, target=tkn)
                tokenizer_format = [wmt.maxed[0]] + ["##" + i for i in wmt.maxed[1:]]
                tk.add_tokens(wmt.maxed[0])
                tk.add_tokens(tokenizer_format)
                encoding.extend(tokenizer_format)
            else:
                encoding.extend(tk.tokenize(tkn))
        return tk.encode_plus(encoding, return_token_type_ids=True, return_attention_mask=True,
                              padding=True)


if __name__ == '__main__':
    WordmapTokenizer()

    import sys

    input_file = sys.argv[1]
    output_file = sys.argv[2]
