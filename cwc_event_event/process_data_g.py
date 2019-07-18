from pathlib import Path
import pickle
import sys
import argparse
from flexnlp import Document
from collections import defaultdict, Counter, OrderedDict
from itertools import combinations
from typing import Iterator, List, Mapping, Union, Optional, Set
import logging as log
import abc
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import random
random_seed = 7
sys.path.append(str(Path(__file__).parent.absolute()))
from ldcred import REDDoc, REDRelation, REDEntity
from ldctcr import NewDoc, NewRelation, NewEntity
from ldctbd import TBDDoc, TBDRelation, TBDEntity
from featureFuncs import create_pos_dict, token_idx

import os
#from nltk.parse import CoreNLPParser
#parser = CoreNLPParser(url='http://localhost:9000')


log.basicConfig(level=log.INFO)

@dataclass
class FlatRelation:
    doc_id: str
    id: str
    rev: bool
    left: REDEntity
    right: REDEntity
    doc: REDDoc
    report_dominance: bool
    rel_type: Optional[str] = None

def print_annotation_stats(data_dir: Path):
    """
    Prints stats such as relation types per each dataset split (train/test/dev)
    Args:
        data_dir:

    Returns:
    """

    for split in ('dev', 'test', 'train'):
        l1_types = defaultdict(int)
        l2_types = defaultdict(int)

        split_dir = data_dir / split
        for file in split_dir.glob('*/*.pkl'):
            with file.open('rb') as fh:
                doc: REDDoc = pickle.load(fh)
                for rel_id, rel in doc.relations.items():
                    l1_types[rel.type] += 1
                    for v in [_ for n, _ in rel.properties if n == 'type']:
                        l2_types[rel.type + '::' + v] += 1
        print(f"#{split} L1 ")
        for n, v in sorted(l1_types.items(), reverse=True, key=lambda x: x[0]):
            print(f'{n}\t{v}')

        print(f"##{split} L2 ")
        for n, v in sorted(l2_types.items(), reverse=True, key=lambda x: x[0]):
            print(f'{n}\t{v}')


def print_flat_relation_stats(data_dir: Path):
    """
    Same as `print_annotation_stats` but flattens the event-event relation annotations
    Args:
        data_dir:

    Returns:

    """
    for split in ('dev', 'test', 'train'):
        flat_rels = read_relations(data_dir, split)
        stats = Counter(r.rel_type for r in flat_rels)
        print(f"#{split} Flattened")
        for n, v in sorted(stats.items(), reverse=True, key=lambda x: x[0]):
            print(f'{n}\t{v}')


def parse_sents(raw_doc, pos_dict):
    # Inputs: 1. raw text of a document
    #         2. pos_dictionary with text span 
    # Output: 1. a dictionary with key = sentence span; value = Syntactic Tree
    #         2. a dictionary with key = sentence span; value = list of word span
    # Note: this is applicable ot TCR dataset so far

    sent_str = 0
    sent_end = 0
    
    syn_trees = {}
    sent2words = {}
    sent_spans = []
    for i in range(len(raw_doc)):
        if raw_doc[i] == '\n':
            sent_end = i
            sent_spans.append((sent_str, sent_end))
            sent_str = sent_end + 1
    
    for ss in sent_spans:
        sent = [(k, v[0]) for k,v in pos_dict.items() if ((int(k.split(':')[0][1:]) >= ss[0]) and 
                                                     (int(k.split(':')[1][:-1]) < ss[1]))]

        if len(sent) > 0:
            orig = [list(x) for x in zip(*sent)]
            
            tree = list(parser.parse(orig[1]))

            j = 0
            k = 0
            leaves = tree[0].leaves()
            temp = []
            chars = 0
            while j < len(leaves):

                # if both parsed results are the same, simply append
                if leaves[j] == orig[1][k] or orig[1][k] in ["\"", "(", ")", "[", "]", "{", "}"]:
                    temp.append(orig[0][k])
                    j += 1
                    k += 1
                # otherwise, find text spans based on the parsed in tree                
                else:
                    chars += len(leaves[j])
                    # copy original text span k times, will only lookup the first one later
                    temp.append(orig[0][k])
                    j += 1
                    if chars == len(orig[1][k]):
                        k += 1
                        chars = 0

            assert len(leaves) == len(temp)

            sent2words[ss] = temp
            syn_trees[ss] = tree[0]

    #if int(temp[-1].split(':')[1][:-1]) < len(raw_doc) - 20:
    #    print(leaves)
    return syn_trees, sent2words


def token_location(sent2words, left, right):
    # Inputs: 1. sentence to words map
    #         2. left event span
    #         3. right event span
    # Outputs: 1. left event sentence key + token idx
    #          2. right event sentence key + token idx

    left_str = "[%s:%s)" % (left[0], left[1] + 1)
    right_str = "[%s:%s)" % (right[0], right[1] + 1)

    #print(left_str)
    #print(right_str)

    for k in sent2words.keys():
        if left[0] >= k[0] and left[1] < k[1]:
            left_idx = (k, -1)
            try:
                left_idx = (k, sent2words[k].index(left_str))
            except:
                for j in range(len(sent2words[k])):
                    if int(sent2words[k][j].split(':')[1][:-1]) > left[0]:
                        left_idx = (k, j)
                        print('left')
                        print(sent2words[k][j])
                        break

        if right[0] >= k[0] and right[1] < k[1]:
            right_idx = (k, -1)
            try:
                right_idx = (k, sent2words[k].index(right_str))
            except:
                for j in range(len(sent2words[k])):
                    if int(sent2words[k][j].split(':')[1][:-1]) > right[0]:
                        right_idx = (k, j)
                        print('right')
                        print(k)
                        print(sent2words[k])
                        print(sent2words[k][j])
                        break
                
    assert left_idx, right_idx

    return left_idx, right_idx

def eventDominates(syn_tree, left, right):
    # assuming left is reporting event
    # determine if left dominates right event
    left_tree_loc = syn_tree.leaf_treeposition(left)
    right_tree_loc = syn_tree.leaf_treeposition(right)

    dominant_path = left_tree_loc[:-2]
    
    # two parents up is the VP node
    # return if left and right share the same VP node
    
    return dominant_path == right_tree_loc[:len(dominant_path)]

def read_relations(data_dir: Path, 
                   split: Optional[str] = None,
                   data_type: str = "red",
                   exclude_types: Optional[Set[str]] = None,
                   include_types: Optional[Set[str]] = None,
                   other_label: str='OTHER',
                   neg_rate: float=0,
                   neg_label: str='NONE', 
                   eval_list: list=[],
                   pred_window: int=200,
                   shuffle_all: bool=False,
                   use_grammar: bool=False,
                   joint: bool=False,
                   backward_sample: bool=False) -> Iterator[FlatRelation]:
    """
    reads flattened relations
    Args:
        split: optional split name filter. if not given, split filter will not be applied
        exclude_types: relation types o exclude (not useful when
        include_types: relation types to include
        other_label: if other_label=None, then the filtered types are skipped, if other_label is given
        neg_label: label for the negative samples, default=None
        neg_rate: ratio of negative samples to positives,
                0 means no negatives, 1 means negatives make up 50% of the total examples.
                (Note: this rate is an approximation within documents, the rate is upper bounded)
        data_dir: directory where pickles are stored

    Returns: an Iterator[FlatRelation]

    """
    
    if data_type == "red":
        doc_type = REDDoc
        ent_type = REDEntity
        relation_type = REDRelation
        label_map = red_label_map
    elif data_type == "new":
        doc_type = NewDoc
        ent_type = NewEntity
        relation_type = NewRelation
        label_map = new_label_map

        # load all causal pairs here                                                                               
        if joint:
            clink_file = open(str(data_dir) + "/allClinks.txt", 'r')
            clinks = {}
            for line in clink_file:
                clink = line.strip().split('\t')
                doc = clink[0]
                event_pair = (clink[1], clink[2])
                label = clink[3]
                if doc in clinks.keys():
                    clinks[doc][event_pair] = label
                else:
                    clinks[doc] = {event_pair:label}
    elif data_type in ['matres', 'tbd']:
        doc_type = TBDDoc
        ent_type = TBDEntity
        relation_type = TBDRelation
        if data_type =='tbd':
            label_map = tbd_label_map
        else:
            label_map = matres_label_map
    
    if include_types and exclude_types:
        log.warning("Either include_types or exclude_types filter is recommended; but both given")
        if include_types & exclude_types:
            raise Exception(f"Ambiguous filters: {include_types} {exclude_types}")
    split_dir = data_dir / split if split else data_dir
    print("%s processing %s %s" % ("="*10, split, "="*10))

    doc_dict = {}
    all_samples = []

    pos_counter = 0
    neg_counter = 0
    causal_counter = 0

    for file in split_dir.glob('**/*.pkl'):
        with file.open('rb') as fh:
            doc = pickle.load(fh)
        
        all_events = [k for k,v in doc.entities.items() if v.type in ['EVENT', 'TIMEX3']]
    
        all_timex = [v for _,v in doc.entities.items() if v.type in ['TIMEX3']]
        pos_dict = create_pos_dict(doc.nlp_ann.pos_tags())

        '''
        all_timex = sorted(all_timex, key = lambda x: x.span[0])
        to_replace = OrderedDict()
        swap = 1
        for tv in all_timex:
            for k,v in pos_dict.items():
                lidx = int(k.split(':')[0][1:])
                ridx = int(k.split(':')[1][:-1])
                
                if lidx >= tv.span[0] and ridx <= tv.span[1]:
                    to_replace[k] = swap
                    swap = 0
                if ridx > tv.span[1]:
                    swap = 1
                    break
            
        new_pos = []
        tc = 0
        for k,v in pos_dict.items():
            if k in to_replace.keys():
                # swap with timex item
                if to_replace[k] == 1:
                    tt = all_timex[tc]
                    key = "[%s:%s)" % (tt.span[0], tt.span[1])
                    value = (tt.text, 'TIMEX')
                    
                    new_pos.append((key, value))
                    tc += 1
                else:
                    continue
            else:
                new_pos.append((k,v))
        pos_dict = OrderedDict(new_pos)
        '''
        doc.pos_dict = pos_dict
        
        # store document only once                                                                                         
        doc_dict[doc.id] = doc
        
        if use_grammar:
            syn_trees, sent2words = parse_sents(doc.raw_text, pos_dict)

        pos_count = 0
        neg_count = 0
        pos_relations = set()   # remember to exclude these from randomly generated negatives
#        print(pos_dict)
#        print(sent2words)
        for rel_id, rel in doc.relations.items():
            rel_type = rel.type
            other_type = ((include_types and rel_type not in include_types)
                          or (exclude_types and rel_type in exclude_types))

            if other_type:
                rel_type = other_label

            elif rel.type == 'ALINK' or rel.type == 'TLINK':
                assert 'type' in rel.properties and len(rel.properties['type']) == 1
                    
                rel_type = rel.properties['type'][0]
                rel_type = label_map[rel_type]
        
            # find the events associated with this relation
            events = [(n, v)
                      for n, vs in rel.properties.items()
                      for v in vs
                      if isinstance(v, ent_type)]

            if len(events) < 2:
                log.warning(f'SKIP: Doc: {doc.id} Relation: {rel_id} of type {rel_type} has'
                            f' {len(events)} events, but 2 needed. \n\n{events}')
                continue
            
            report_dominance = False
            first_event_name, first_event = events[0]
            # all the second relations should have same name
            assert len(set(x for x, y in events[1:])) == 1
            for second_event_name, second_event in events[1:]:

                all_keys, lidx_start, lidx_end, ridx_start, ridx_end = token_idx(first_event.span, second_event.span, pos_dict)
                first_mid = (lidx_start + lidx_end) / 2.0
                second_mid = (ridx_start + ridx_end) / 2.0
                tok_dist = np.abs(first_mid - second_mid)
        
                # only consider the case: first_event is reporting and second event not timex
                if use_grammar and first_event.type == "REPORTING" and second_event.type not in ['DATE','TIME', 'DURATION']:

                    left_loc, right_loc = token_location(sent2words, first_event.span, second_event.span)
                
                    # use parse_tree to extract event dominance 
                    # both events have to be in the same sentecen
                    if left_loc[0] == right_loc[0] and left_loc[1] > 0 and right_loc[1] > 0: 
                        #print(first_event.type, second_event.type)
                        #print(left_loc, right_loc)

                        report_dominance = eventDominates(syn_trees[left_loc[0]], left_loc[1], right_loc[1])
                        #if report_dominance:
                        #    print("Dominant Case Found!")
                        #    print(left_loc, right_loc)
                        #    print([pos_dict[x] for x in all_keys[lidx_start:ridx_end+1]])
                        #    kill
                # later we want to see if we want to remove out of window in prediction too
                if tok_dist > pred_window:
                    label = neg_label
                    #label = rel_type
                else:
                    label = rel_type

                if rel_type is not None:
                    #rev_ind = False
                    
                    if first_mid > second_mid:
                        #assert data_type == "red"
                        left = second_event
                        right = first_event
                        rev_ind = True
                    # weird case where both events are the same text, but can be timex and event
                    elif first_mid == second_mid:
                        print(label)
                        print(first_event)
                        print(second_event)
                        print('*'*50)
                        
                    else:
                        left = first_event
                        right = second_event
                        rev_ind = False
                    
                    # later we want to see if we want to remove out of window in prediction too                                                            
                    if tok_dist > pred_window:
                        label = neg_label
                        label = rel_type                                                                                                                  
                    else:
                        label = rel_type
                        pos_count += 1
                    
                    if backward_sample:
                        all_samples.append(("P%s" % pos_counter, rev_ind, right, left, doc.id, label + "_rev"))
                        pos_relations.add((right.id, left.id))
                    else:
                        all_samples.append(("P%s" % pos_counter, rev_ind, left, right, doc.id, report_dominance, label))
                        pos_relations.add((left.id, right.id))

                    if joint and data_type == "new" and left.id[0] == 'e' and right.id[0] == 'e':
                        # collect causal pairs                                                                     
                        left_id = left.id.split('i')[0] + left.id.split('i')[1]
                        right_id = right.id.split('i')[0] + right.id.split('i')[1]
                        # use the positive sample index for causal pairs
                        # this is useful for global inference; pair lookup for causal constraints
                        if (left_id, right_id) in clinks[doc.id].keys():
                            all_samples.append(("C%s" % pos_counter, rev_ind, left, right, doc.id, report_dominance, clinks[doc.id][(left_id, right_id)]))
                            causal_counter += 1
                    pos_counter += 1
        # negative samples
        neg_sample_size = int(neg_rate * pos_count)

        if neg_sample_size > 0:
            all_neg = [(l_id, r_id) for l_id, r_id in combinations(all_events, 2)
                       if (l_id, r_id) not in pos_relations]# and (r_id, l_id) not in pos_relations]

            if split in eval_list:
                neg_sample_size = len(all_neg)

            random.Random(random_seed).shuffle(all_neg)
            for left_id, right_id in all_neg:#[:neg_sample_size]:
                # fitler out entities that are too long
                if int(doc.entities[left_id].span[1]) - int(doc.entities[left_id].span[0]) > 50:
                    continue
                if int(doc.entities[right_id].span[1]) - int(doc.entities[right_id].span[0]) > 50:
                    continue

                # filter out entities that are the same: could be both as time and event
                if (int(doc.entities[right_id].span[0]) == int(doc.entities[left_id].span[0])) and (int(doc.entities[right_id].span[1]) == int(doc.entities[left_id].span[1])):
                    continue
                
                # exclude rels that are more than 2*ngbrs token-dist away
                all_keys, lidx_start, lidx_end, ridx_start, ridx_end = token_idx(doc.entities[left_id].span, doc.entities[right_id].span, pos_dict)

                left_mid = (lidx_start + lidx_end) / 2.0
                right_mid = (ridx_start + ridx_end) / 2.0
                rev_ind = False
                
                ## be really careful this is only for one seq RNN model
                if left_mid > right_mid:
                    #assert data_type == "red"
                    left = doc.entities[right_id]
                    right = doc.entities[left_id]
                    rev_ind = True
                elif left_mid == right_mid:
                    continue
                else:
                    left = doc.entities[left_id]
                    right = doc.entities[right_id]
                    rev_ind = False
                
                if np.abs(left_mid - right_mid) > pred_window:
                    continue
                if neg_count >= neg_sample_size:
                    break
                else:
                    neg_count += 1
                    all_samples.append(("N%s"%neg_counter, rev_ind, left, right, doc.id, neg_label))
                    neg_counter += 1

    print("Total sample positive size is: %s" % pos_counter)
    print("Total sample negative size is: %s" % neg_counter)
    print("Total causal sample size is: %s" % causal_counter)
    #with open("%s/%s_docs.txt" % (str(data_dir), split), 'w') as file:
    #    for k in doc_dict.keys():
    #        file.write(k)
    #        file.write('\n')
    #file.close()

    if shuffle_all:
        random.Random(random_seed).shuffle(all_samples)
    for s in all_samples:
        yield FlatRelation(s[4], s[0], s[1], s[2], s[3], doc_dict[s[4]], s[5], s[6])

all_red_labels = [##'WHOLE/PART',
                  'TERMINATES',
                  'SIMULTANEOUS',
                  ##'SET/MEMBER',
                  ##'REPORTING',
                  'REINITIATES',
                  'OVERLAP/PRECONDITION',
                  'OVERLAP/CAUSES',
                  'OVERLAP',
                  'INITIATES',
                  ##'IDENTICAL',
                  'ENDS-ON',
                  'CONTINUES',
                  'CONTAINS-SUBEVENT',
                  'CONTAINS',
                  ##'BRIDGING',
                  'BEGINS-ON',
                  'BEFORE/PRECONDITION',
                  'BEFORE/CAUSES',
                  'BEFORE',
                  ##'APPOSITIVE',
                  'NONE']

new_label_map = OrderedDict([('BEFORE', 'BEFORE'), 
                             ('AFTER', 'AFTER'), 
                             ('SIMULTANEOUS', 'SIMULTANEOUS'),
                             ('VAGUE', 'VAGUE')
                             ])

causal_label_map = OrderedDict([('cause', 'causes'),
                                ('caused_by', 'caused_by')
                               ])

matres_label_map = OrderedDict([('VAGUE', 'VAGUE'),
                             ('BEFORE', 'BEFORE'),
                             ('AFTER', 'AFTER'),
                             ('SIMULTANEOUS', 'SIMULTANEOUS')
                         ])

tbd_label_map = OrderedDict([('VAGUE', 'VAGUE'),
                             ('BEFORE', 'BEFORE'),
                             ('AFTER', 'AFTER'),
                             ('SIMULTANEOUS', 'SIMULTANEOUS'),
                             ('INCLUDES', 'INCLUDES'),
                             ('IS_INCLUDED', 'IS_INCLUDED')
                       ])

red_label_map = OrderedDict([('TERMINATES', 'TERMINATES'),
             ('SIMULTANEOUS', 'SIMULTANEOUS'),
             ('REINITIATES',  'REINITIATES'),
             ('OVERLAP/PRECONDITION', 'OVERLAP'),
             ('OVERLAP/CAUSES', 'OVERLAP'),
             ('OVERLAP', 'OVERLAP'),
             ('INITIATES', 'INITIATES'),
             ('ENDS-ON', 'ENDS-ON'),
             ('CONTINUES', 'CONTINUES'),
             ('CONTAINS-SUBEVENT', 'CONTAINS'),
             ('CONTAINS', 'CONTAINS'),
             ('BEGINS-ON', 'BEGINS-ON'),
             ('BEFORE/PRECONDITION', 'BEFORE'),
             ('BEFORE/CAUSES', 'BEFORE'),
             ('BEFORE', 'BEFORE'),
             ('NONE', 'NONE')])

rev_map = OrderedDict([('BEFORE_rev', 'BEFORE'),
           ('BEGINS-ON_rev', 'BEGINS-ON'),
           ('CONTAINS_rev', 'CONTAINS'),
           ('CONTINUES_rev', 'CONTINUES'),
           ('ENDS-ON_rev', 'ENDS-ON'),
           ('INITIATES_rev', 'INITIATES'),
           ('OVERLAP_rev', 'OVERLAP'),
           ('REINITIATES_rev', 'REINITIATES'),
           ('SIMULTANEOUS_rev', 'SIMULTANEOUS'),
           ('TERMINATES_rev', 'TERMINATES'),
           ('NONE', 'NONE')])

class REDEveEveRelModel(abc.ABC):

    def __init__(self, labels: List[str] = all_red_labels):
        self._id_to_label, self._label_to_id = None, None
        self.labels = labels

    @property
    def name(self):
        return type(self).__name__

    @property
    def labels(self):
        return self._id_to_label

    @labels.setter
    def labels(self, labels: List[str]):
        self._id_to_label = labels
        self._label_to_id = {label: i for i, label in enumerate(labels)}

    @abc.abstractmethod
    def predict(self, left: REDEntity, right: REDEntity) -> str:
        raise NotImplementedError()

    @abc.abstractmethod
    def train_epoch(self, train_data: Iterator[FlatRelation]) -> float:
        raise NotImplementedError()

    def __call__(self, left: REDEntity, right: REDEntity) -> str:
        return self.predict(left, right)


@dataclass()
class MajorityBaselineModel(REDEveEveRelModel):
    """
    A simple baseline model which assigns majority class label as the label for every example
    """

    # manually learned by inspecting the result of `print_flat_relation_stats()` on training split
    majority_label_str: str = 'IDENTICAL'

    def train_epoch(self, train_data: Iterator[FlatRelation]) -> float:
        label_freq = Counter(rel.rel_type for rel in train_data)
        self.majority_label_str, _ = sorted(label_freq.items(), key=lambda x: x[1])[-1]

        self._id_to_label = sorted(label_freq.keys())
        self._label_to_id = {label: i for i, label in enumerate(self._id_to_label)}
        return -1.0

    def predict(self, left: REDEntity, right: REDEntity) -> str:
        return self.majority_label_str


@dataclass()
class RandomBaselineModel(REDEveEveRelModel):
    """
    A simple baseline model which assigns a random class label to each event-event pair
    """
    label_probs: Optional[List[float]] = None
    weighted: bool = False

    @property
    def name(self):
        return type(self).__name__ + (':Weighted' if self.weighted else ':Uniform')

    def train_epoch(self, train_data: Iterator[FlatRelation]) -> float:
        label_freq = Counter(rel.rel_type for rel in train_data)
        self.labels = sorted(label_freq.keys())
        if self.weighted:
            total = sum(label_freq.values())
            self.label_probs = [label_freq[l] / total for l in self.labels]
        return -1.0

    def predict(self, left: REDEntity, right: REDEntity) -> str:
        return np.random.choice(self.labels, p=self.label_probs)


class ClassificationReport:

    def __init__(self, name, true_labels: List[Union[int, str]],
                 pred_labels: List[Union[int, str]]):

        assert len(true_labels) == len(pred_labels)
        self.num_tests = len([x for x in true_labels if x != 'NONE'])
        self.total_truths = Counter(true_labels)
        self.total_predictions = Counter(pred_labels)
        self.name = name
        self.labels = sorted(set(true_labels) | set(pred_labels))
        self.confusion_mat = self.confusion_matrix(true_labels, pred_labels)
        self.accuracy = sum(y == y_ for y, y_ in zip(true_labels, pred_labels)) / len(true_labels)
        self.trim_label_width = 15

    @staticmethod
    def confusion_matrix(true_labels: List[str], predicted_labels: List[str]) \
            -> Mapping[str, Mapping[str, int]]:
        mat = defaultdict(lambda: defaultdict(int))
        for truth, prediction in zip(true_labels, predicted_labels):
            mat[truth][prediction] += 1
        return mat

    def __repr__(self):
        res = f'Name: {self.name}\t Created: {datetime.now().isoformat()}\t'
        res += f'Total Labels: {len(self.labels)} \t Total Tests: {self.num_tests}\n'
        display_labels = [label[:self.trim_label_width] for label in self.labels]
        label_widths = [len(l) + 1 for l in display_labels]
        max_label_width = max(label_widths)
        header = [l.ljust(w) for w, l in zip(label_widths, display_labels)]
        header.insert(0, ''.ljust(max_label_width))
        res += ''.join(header) + '\n'
        for true_label, true_disp_label in zip(self.labels, display_labels):
            predictions = self.confusion_mat[true_label]
            row = [true_disp_label.ljust(max_label_width)]
            for pred_label, width in zip(self.labels, label_widths):
                row.append(str(predictions[pred_label]).ljust(width))
            res += ''.join(row) + '\n'
        res += '\n'

        def safe_division(numr, denr, on_err=0.0):
            return on_err if denr == 0.0 else numr / denr

        def num_to_str(num):
            return '0' if num == 0 else str(num) if type(num) is int else f'{num:.4f}'

        n_correct = 0
        n_true = 0
        n_pred = 0

        all_scores = []
        header = ['Total  ', 'Predictions', 'Correct', 'Precision', 'Recall  ', 'F1-Measure']
        res += ''.ljust(max_label_width + 2) + '  '.join(header) + '\n'
        head_width = [len(h) for h in header]
        
        for label, width, display_label in zip(self.labels, label_widths, display_labels):
            if label not in ['NONE']:#, 'VAGUE']:
                total_count = self.total_truths.get(label, 0)
                pred_count = self.total_predictions.get(label, 0)
        
                n_true += total_count
                n_pred += pred_count

                correct_count = self.confusion_mat[label][label]
                n_correct += correct_count

                precision = safe_division(correct_count, pred_count)
                recall = safe_division(correct_count, total_count)
                f1_score = safe_division(2 * precision * recall, precision + recall)
                all_scores.append((precision, recall, f1_score))

                row = [total_count, pred_count, correct_count, precision, recall, f1_score]
                row = [num_to_str(cell).ljust(w) for cell, w in zip(row, head_width)]
                row.insert(0, display_label.rjust(max_label_width))
                res += '  '.join(row) + '\n'

        # weighing by the truth label's frequency
        label_weights = [safe_division(self.total_truths.get(label, 0), self.num_tests)
                         for label in self.labels if label not in ['NONE', 'VAGUE']]
        weighted_scores = [(w * p, w * r, w * f) for w, (p, r, f) in zip(label_weights, all_scores)]
        assert len(label_weights) == len(weighted_scores)

        res += '\n'
        res += '  '.join(['Weighted Avg'.rjust(max_label_width),
                          ''.ljust(head_width[0]),
                          ''.ljust(head_width[1]),
                          ''.ljust(head_width[2]),
                          num_to_str(sum(p for p, _, _ in weighted_scores)).ljust(head_width[3]),
                          num_to_str(sum(r for _, r, _ in weighted_scores)).ljust(head_width[4]),
                          num_to_str(sum(f for _, _, f in weighted_scores)).ljust(head_width[5])])

        print(n_correct, n_pred, n_true)
        precision = safe_division(n_correct, n_pred)
        recall = safe_division(n_correct, n_true)
        f1_score = safe_division(2 * precision * recall, precision + recall)

        res += f'\n Total Examples: {self.num_tests}'
        res += f'\n Overall Precision: {num_to_str(precision)}'
        res += f'\n Overall Recall: {num_to_str(recall)}'
        res += f'\n Overall F1: {num_to_str(f1_score)} '
        return res


@dataclass
class REDEvaluator:
    model: REDEveEveRelModel

    def evaluate(self, test_data: Iterator[FlatRelation]):
        pairs = [(ex.rel_type, self.model(ex.left, ex.right)) for ex in test_data]
        true_labels, pred_labels = zip(*pairs)
        return ClassificationReport(self.model.name, true_labels, pred_labels)


def main(data_dir, tempo_filter=False, skip_other=False, neg_rate=0):
    # Task 1: print stats of annotations
    # print_annotation_stats(_data_dir)
    # Task 2: print stats of expanded relations
    # print_flat_relation_stats(_data_dir)

    # we are focussed on this type only
    opt_args = {}
    if tempo_filter:
        opt_args['include_types'] = {'TLINK'}
    if skip_other:
        opt_args['other_label'] = None
    if neg_rate > 0:
        opt_args['neg_rate'] = neg_rate
    log.info(f"Reading datasets to memory from {data_dir}")
    
    # buffering data in memory --> it could cause OOM
    train_data = list(read_relations(data_dir, 'train', **opt_args))
    test_data = list(read_relations(data_dir, 'test', **opt_args))
    models = [MajorityBaselineModel(),
              RandomBaselineModel(weighted=False),
              RandomBaselineModel(weighted=True)]
    for model in models:
        print(f"\n======={model.name}=====\n")
        model.train_epoch(train_data)
        evaluator = REDEvaluator(model)
        report = evaluator.evaluate(test_data)
        print(report)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('data_dir', type=Path,
                   help='Path to directory having RED data. This should be output of '
                        '"ldcred.py flexnlp"')
    p.add_argument('--tempo-filter', action='store_true',
                   help='Include Causal and Temporal relations only. By default all relations are'
                        ' included. When --filter is specified, non temporal and causal relations '
                        'will be labelled as OTHER')
    p.add_argument('--skip-other', action='store_true',
                   help='when --tempo-filter is applied, the excluded types are marked as OTHER.'
                        'By enabling --skip-other, OTHER types are skipped.')
    p.add_argument('-nr', '--neg-rate', type=float, default=0.,
                   help='Negative sample rate.')

    args = p.parse_args()
    main(**vars(args))
