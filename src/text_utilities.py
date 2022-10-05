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


def from_path(path: str) -> set:
    """Returns a set of vocabulary from txt files in a directory (use wd)"""
    files = glob.glob(f'{path}/*.txt')
    words = []
    for f in files:
        with open(f, 'r', encoding='utf8') as f:
            words += f.read().split('\n')
    try:
        words.remove('')
    except ValueError:
        pass
    return set(words)


def count_syllables(text: set, pattern='[aeuioäöüAEIUOÄÖÜ][aeuioäöüAEIUOÄÖÜ]?') -> dict:
    """Counts chained vowels (max 2) from iterable word list and sorts them into a dict {vowelcount:{set of tokens}}"""
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


def map_subword(target: str, map: str) -> str:
    """Returns a subword from a target string and a map. Yet to implement maps with 1 on both ends."""
    if map.startswith("1"):
        return target[:map.count("1")] + "##"
    elif map.endswith("1"):
        return "##" + target[-map.count("1"):]


print(map_subword("verstehen", "000000011"))
