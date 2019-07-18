#!/usr/bin/env python                                                                                           \
                                                                                                                 
# Author : RJ ; Created: Oct 26, 2018                                                                           \
                                                                                                                 
import os
from dataclasses import dataclass
from typing import (List, Iterator, Tuple, Union, Dict, Mapping, Callable, Set, Type, TextIO, Optional)
from collections import defaultdict as ddict
import logging as log
from lxml import etree
from pathlib import Path
import re
import json
import pickle
import random
from collections import OrderedDict, Counter
import numpy as np

def parse_ettt(fl, caevo_preds):
    # take a input .tml file and caevo outputs
    # output pairs composed by valid e and t in the .tml file
    # note this is only the first filter

    et_count = 0
    tt_count = 0
    ettt = {}
    with open(str(args.dir) + '/' + fl, encoding='utf-8') as f:
        
        xml_tree = etree.parse(f)
        events = xml_tree.xpath('.//MAKEINSTANCE')

        eids = {e.attrib['eiid']: e.attrib['eventID'] for e in events}
        
        timex = [t.attrib['tid'] for t in xml_tree.xpath('.//TIMEX3')]
        #print(eids)
        #print(timex)

        for (s, t), rel in caevo_preds[fl.replace(".tml", "")].items():

            if s[0] == 'e':
                s = s[0] + 'i' + s[1:]
            if t[0] == 'e':
                t = t[0] + 'i' + t[1:]

            if s in timex:
                if t in timex:
                    ettt[(s, t)] = rel
                    tt_count +=1
                elif t in eids:
                    ettt[(s, eids[t])] = rel
                    et_count += 1

            if s in eids:
                if t in timex:
                    ettt[(eids[s], t)] = rel
                    et_count +=1

    print("ET: %s, TT: %s" % (et_count, tt_count))
    return(ettt)

        
def validate(ettt, gold_file):
    # take an et_tt pair dictionary
    # output valid pairs as in the Gold TBDense file

    with open(gold_file, 'rb') as fh:
        gold = pickle.load(fh)


    gold_totl, caevo_totl, caevo_matched = 0, 0, 0

    gold_new = {}
    caevo_new = {}
    reversed_matched = 0
    for k, v in ettt.items():
        print(k)
        matched = 0
        caevo_new[k] = {}
        gold_new[k] = {}
        for kk, vv in ettt[k].items():
            if kk in gold[k]:
                matched += 1
                caevo_new[k][kk] = vv
                gold_new[k][kk] = gold[k][kk]
            elif (kk[1], kk[0]) in gold[k]:
                matched += 1
                key = (kk[1], kk[0])
                caevo_new[k][kk] = vv
                gold_new[k][key] = gold[k][key]
                reversed_matched += 1

        print("Gold: ", len(gold_new[k]))
        print("CAEVO: ", len(caevo_new[k]))
        print("Matched: ", matched)

        gold_totl +=  len(gold_new[k])
        caevo_totl += len(caevo_new[k])
        caevo_matched += matched
    print(gold_totl, caevo_totl, caevo_matched, reversed_matched)

    return caevo_new, gold_new
def main(args):

    src_docs = os.listdir((str(args.dir)))
    ettt = {}
    for doc in src_docs:
        print(str(doc).split('/')[-1])
        ettt[doc.replace('.tml', '')] = parse_ettt(doc, args.caevo_preds)
    ettt_new, gold_new = validate(ettt, args.gold_dir)
    out_path = str(args.out) + "/caevo_test_ettt.pkl"
    with open(out_path, 'wb') as fh:
        pickle.dump(ettt_new, fh)

    with open(args.gold_dir, 'wb') as fh:
        pickle.dump(gold_new, fh)

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('-dir', help='Path to RED directory (extracted .tgz)', type=Path)
    p.add_argument('-out', help='output File (if task=json) or Folder (if task=flexnlp)', type=Path)
    p.add_argument('-gold_dir', type=str)
    p.add_argument('-caevo_preds', type=dict)
    args = p.parse_args()

    args.gold_dir = '/nas/home/rujunhan/data/TBDense/caevo_test_ettt.pkl'
    args.dir = Path('/nas/home/rujunhan/CAEVO/test/')
    args.out = Path('/nas/home/rujunhan/CAEVO/')
    args.caevo_preds = np.load(str(args.out) + '/caevo.npy').item()
    #print(args)
    main(args)
