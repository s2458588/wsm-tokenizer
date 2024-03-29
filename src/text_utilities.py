#!/usr/bin/env python
__author__ = "Ricardo Jung"
__email__ = "s2458588@stud.uni-frankfurt.de"

# __copyright__ = ""
# __credits__ = [""]
# __license__ = ""
# __version__ = ""
# __maintainer__ = ""
# __status__ = ""

import regex as re
import glob
import numpy as np
import collections as cl
from scipy import stats


class VerbDict:
    """Create a dictionary with format {token:relative frequency} from given crawled files in file paths.\n
    "../new_tokenizer/fun_vocab_raw.txt"
    "../new_tokenizer/lex_vocab_raw.txt"
    """

    def __init__(self, path_to_functional_morphemes="../new_tokenizer/fun_vocab_raw.txt", path_to_lexemic_morphemes="../new_tokenizer/lex_vocab_raw.txt"):
        self.path_to_functional_morphemes = path_to_functional_morphemes
        self.path_to_lexemic_morphemes = path_to_lexemic_morphemes
        self.lm_raw = None
        self.fm_raw = None
        self.lm_abs = None
        self.fm_abs = None
        self.lm_rel = None
        self.fm_rel = None
        self.lmfm = None
        self.vocab = None
        self.load_files()
        self.generate_hashmaps()

    def load_files(self):
        with open(self.path_to_lexemic_morphemes, encoding="utf8", mode="r") as lv:
            lm_raw = lv.read().split("\n")
            lv.close()
        self.lm_raw = lm_raw

        with open(self.path_to_functional_morphemes, encoding="utf8", mode="r") as fv:
            fm_raw = fv.read().split("\n")
            fv.close()
        self.fm_raw = fm_raw

    def generate_hashmaps(self):
        """Generates dictionaries for (1) functional morphemes (2) lexemic morphemes (3) both combined."""
        # cleaning and preselection of subvocabulary

        fm_clean = [i for i in self.fm_raw if i != ""]
        lm_clean = [i for i in self.lm_raw if
                    (len(i) > 1) and (i not in fm_clean)]  # select only morphemes longer than 1 character
        lm_ncount = cl.Counter(lm_clean).most_common()
        n_o_lm = len(lm_clean)
        lm_rel_pre = {k: (v / n_o_lm) + 1 if ((len(k) > 2) and (k not in fm_clean)) else 0 for k, v in
                      lm_ncount}  # len > 2 could be mean len of set(fm) -> int(2.47)
        # functional morphemes: remove any str with len above 2.5 std (normalized)

        fm_uni = list(set(fm_clean))
        fm_outliers = [fm_uni[i] for i in np.where(np.abs(stats.zscore([len(i) for i in fm_uni])) > 2.5)[0]]
        fm_uni_no_outliers = set(fm_uni) - set(fm_outliers)

        fm_ncount = cl.Counter(fm_clean).most_common()
        n_o_fm = len(fm_clean)
        fm_rel = {k: v / n_o_fm if (len(k) > 1) and (k in list(fm_uni_no_outliers)) else 0 for k, v in fm_ncount}

        # lexical morphemes: use everything below median + median absolute deviation
        lm_uni = list(set(lm_rel_pre))
        lm_len = [len(i) for i in lm_uni]
        lm_med = np.median(lm_len)
        lm_mad = stats.median_abs_deviation(lm_len)
        lm_outliers = set([lm_uni[i] for i in np.where(lm_med + lm_mad < lm_len)[0]])  # where(array > condition)
        lm_uni_no_outliers = set(lm_uni) - set(lm_outliers)

        lm_rel = {k: (v / n_o_lm) if (len(k) > 2) and (k not in fm_rel) and (k in list(lm_uni_no_outliers)) else 0 for
                  k, v in lm_ncount}
        self.vocab = {**lm_rel, **fm_rel}


def corpus_metrics(tokenset):
    all_chars = ""
    for sylnumber in tokenset:
        for v in tokenset[sylnumber]:
            all_chars += v

    n_maps = cl.Counter(list(all_chars)).most_common()
    char_array = np.array(n_maps)

    n_count = [i[1] for i in n_maps]
    total_count = sum(n_count)
    rel_freqs = [i / total_count for i in n_count]

    # ARRAY WITH CHARS, COUNTS and REL. FREQ
    full_array = np.column_stack((char_array, rel_freqs))
    return {j: (int(k), float(l)) for (j, k, l) in full_array}


class PosCorpus:
    """Corpus object generated from files in a folder"""

    def __init__(self, path):
        self.path = path
        self.types = from_path(path)
        self.counted_corpus = count_syllables(self.types)  # corpus sorted tokens into approx. syllable counts
        self.metrics = corpus_metrics(self.counted_corpus)  # contains character metrics; count, relative frequency


def from_path(path: str) -> set:
    """Returns a set of vocabulary from txt files in a directory (use wd)"""
    files = glob.glob(f'{path}/*.txt')
    words = []
    for file in files:
        with open(file, 'r', encoding='utf8') as f:
            words += f.read().split('\n')
    try:
        words.remove('')
    except ValueError:
        pass
    return set(words)


def count_syllables(text, pattern='[aeuioäöüAEIUOÄÖÜ][aeuioäöüAEIUOÄÖÜ]?'):
    """Counts chained vowels (max 2) from iterable word list and sorts them into a dict {vowelcount:{set of tokens}}"""

    if isinstance(text, set):
        count_dict = dict()
        for v in text:
            vowels = re.findall(pattern=pattern, string=v)
            if len(vowels) in count_dict:
                count_dict[len(vowels)].add(v)
            else:
                count_dict.update({len(vowels): {v}})
        try:
            del count_dict[0]
        except KeyError:
            pass
        return count_dict
    else:
        try:
            return len(re.findall(pattern=pattern, string=text))
        except TypeError:
            print(Exception)


def decode(target: str, m: str):
    """Returns a subword from a target string and a map. Yet to implement maps with 1 on both ends and returning the
    remaining string """
    if m.startswith("1"):
        return target[:m.count("1")] + "##", target[-m.count("0"):]
    elif m.endswith("1"):
        return target[-m.count("0"):], "##" + target[-m.count("1"):]


def match_ends(s1: str, s2: str):
    """Compares characters of two strings at index and returns a case."""
    cases = {
        "first": s1[0] == s2[0],
        "last": s1[-1] == s2[-1],
        "any": (s1[0] == s2[0] or s1[-1] == s2[-1])
    }
    return cases


def wordmap(longer, shorter, start=0):
    """Compares every character for a pair of strings. Takes start index as optional argument. Returns wordmap"""
    return [int(c1 == c2) for c1, c2 in zip(list(longer)[start::], list(shorter))]



def files_from_path(path: str, full_path=True) -> list:
    """Lists file in a directory"""
    if full_path:
        files = glob.glob(f'{path}/*.txt')
        return files
    else:
        files = glob.glob(f'{path}/*.txt')
        return [file[len(path)+1:] for file in files]
