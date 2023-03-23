from typing import List
import torch
from collections import Counter


def _flatten(t: List[List]) -> List:
    """flatten list of lists"""
    return [item for sublist in t for item in sublist]


def _get_tagname_to_id(target: List[List[str]]):
    """
    Assign id to each tag
    """
    tag_set = set()
    for tags_i in target:
        if isinstance(tags_i, list):
            tag_set.update(tags_i)
        elif isinstance(tags_i, str):
            tag_set.add(tags_i)
    tagname_to_tagid = {tag: i for i, tag in enumerate(list(sorted(tag_set)))}

    return tagname_to_tagid


def _get_target_weights(targets: List[List[str]], tagname_to_tagid: Dict[str, int]):
    flat_targets = _flatten(targets)
    counts = dict(Counter(flat_targets))
    # print(counts)
    n_entries = len(targets)
    proportions = torch.zeros(len(tagname_to_tagid), dtype=float)
    for tagname, tagid in tagname_to_tagid.items():
        proportions[tagid] = n_entries / (2 * counts[tagname])
    return proportions
