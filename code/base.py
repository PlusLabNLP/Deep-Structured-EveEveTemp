from pathlib import Path
import pickle
import sys
import argparse
from collections import defaultdict, Counter, OrderedDict
from itertools import combinations
from typing import Iterator, List, Mapping, Union, Optional, Set
import logging as log
import abc
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import random
sys.path.append(str(Path(__file__).parent.absolute()))

new_label_map = OrderedDict([('BEFORE', 'BEFORE'), 
                             ('AFTER', 'AFTER'), 
                             ('SIMULTANEOUS', 'SIMULTANEOUS'),
                             ('VAGUE', 'VAGUE')
                             ])

causal_label_map = OrderedDict([('causes', 'causes'),
                                ('caused_by', 'caused_by')
                               ])

rev_causal_map = OrderedDict([('causes', 'caused_by'),
                              ('caused_by', 'causes')
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

rev_map = OrderedDict([('VAGUE', 'VAGUE'),
                       ('BEFORE', 'AFTER'),
                       ('AFTER', 'BEFORE'),
                       ('SIMULTANEOUS', 'SIMULTANEOUS'),
                       ('INCLUDES', 'IS_INCLUDED'),
                       ('IS_INCLUDED', 'INCLUDES')
                       ])

class EveEveRelModel(abc.ABC):
    def __init__(self):
        pass
    
    @property
    def name(self):
        return type(self).__name__

class ClassificationReport:

    def __init__(self, name, true_labels: List[Union[int, str]],
                 pred_labels: List[Union[int, str]], exclude_vague=True):

        assert len(true_labels) == len(pred_labels)
        self.num_tests = len([x for x in true_labels if x != 'NONE'])
        self.total_truths = Counter(true_labels)
        self.total_predictions = Counter(pred_labels)
        self.name = name
        self.labels = sorted(set(true_labels) | set(pred_labels))
        self.exclude_labels = ['NONE','VAGUE'] if exclude_vague else ['NONE']
        self.confusion_mat = self.confusion_matrix(true_labels, pred_labels)
        self.accuracy = float(sum(y == y_ for y, y_ in zip(true_labels, pred_labels)))/float(len(true_labels))
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
            if label not in self.exclude_labels:
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
                         for label in self.labels if label not in self.exclude_labels]
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

        precision = safe_division(n_correct, n_pred)
        recall = safe_division(n_correct, n_true)
        f1_score = safe_division(2 * precision * recall, precision + recall)

        res += f'\n Total Examples: {self.num_tests}'
        res += f'\n Overall Precision: {num_to_str(precision)}'
        res += f'\n Overall Recall: {num_to_str(recall)}'
        res += f'\n Overall F1: {num_to_str(f1_score)} '
        return res

