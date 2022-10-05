#!/usr/bin/env python
__author__ = "Ricardo Jung"
__email__ = "s2458588@stud.uni-frankfurt.de"


# __copyright__ = ""
# __credits__ = [""]
# __license__ = ""
# __version__ = ""
# __maintainer__ = ""
# __status__ = ""


def word_mapping(target: str, token_dict: dict, syllables: int) -> list:
    """Compares target to tokens in dictionary {int:[t1, t2, t3]}, returns binary string of matching characters."""
    maps = []
    for v in token_dict[syllables]:
        pair = (v, target)
        smaller = min(pair, key=len)
        longer = max(pair, key=len)
        diff = len(longer) - len(smaller)
        m = []
        step = 0
        while step <= diff:
            if diff == 0:
                for i in range(len(smaller)):
                    m.append(1 if pair[0][i] == pair[1][i + step] else 0)
            else:
                for i in range(len(smaller)):
                    m.append(1 if smaller[i] == longer[i + step] else 0)
            if m[0] == 1 or m[-1] == 1:
                maps.append(m)
            step += 1
            m = []
    return maps

