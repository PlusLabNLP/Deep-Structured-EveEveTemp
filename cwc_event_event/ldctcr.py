#!/usr/bin/env python                                                                                                   
# Author : RJ ; Created: Sep 21, 2018                                                                                  
import os
from dataclasses import dataclass
from typing import (List, Iterator, Tuple, Union, Dict, Mapping, Callable, Set, Type, TextIO, Optional)
from collections import defaultdict as ddict
import glob
import logging as log
from lxml import etree
from pathlib import Path
import re
import json
from flexnlp import Pipeline, Document
from flexnlp.integrations.flexnlp import PlainTextIngester
from flexnlp.integrations.spacy import SpacyAnnotator
from flexnlp.utils.misc_utils import WithId
import pickle
import random
from collections import deque
from collections import OrderedDict
log.basicConfig(level=log.DEBUG)


@dataclass
class NewEntity():
    id: str
    type: str
    text: str
    tense: str
    aspect: str
    polarity: str
    span: Tuple[int, int]

    @classmethod
    def new(self, cls, fts, counter):
        res = {}
        res['text'] = cls.text
        res['span'] = [counter+1, counter+len(res['text'])]
        
        if cls.tag == "EVENT":
            res['id'] = "ei%s" % cls.attrib['eid'][1:]
            res['type'] = cls.attrib['class']
            res['tense'] = fts[cls.attrib['eid']][0]
            res['aspect'] = fts[cls.attrib['eid']][2]
            res['polarity'] = fts[cls.attrib['eid']][1]
        elif cls.tag == "TIMEX3":
            res['id'] = cls.attrib['tid']
            res['type'] = cls.attrib['type']
            res['tense'] = None
            res['polarity'] = None
            res['aspect'] = None
        return NewEntity(**res)

@dataclass
class NewRelation():
    id: str
    type: str
    properties: Dict[str, List[Union[str, NewEntity]]]
    
    def new(cls, entities):

        res = {}
        res['id'] = cls.attrib['lid']
        # relation category
        res['type'] = cls.tag
        # relation type
        id_name = 'eventInstanceID' if 'eventInstanceID' in cls.attrib.keys() else 'timeID'
        res['properties'] = {}
        res['properties']['type'] = [cls.attrib['relType']]
        res['properties']['source'] = [entities[cls.attrib[id_name]]]
        try:
            res['properties']['target'] = [entities[cls.attrib['relatedToEventInstance']]]
        except:
            res['properties']['target'] = [entities[cls.attrib['relatedToTime']]]
        return NewRelation(**res)

@dataclass
class NewDoc:
    id: str
    raw_text: str = ""
    entities: Mapping[str, NewEntity] = None
    relations: Mapping[str, NewRelation] = None
    nlp_ann: Optional[Document] = None

    def parse_entities(self, entities, all_text, event_fts={}):

        res = []
        
        # this new dataset doesn't have span indicators, so we need to create it manually
        q, raw_text = self.build_queue(all_text)
        
        # case timex -- the first event is doc time -- excluded for now!!!
        if event_fts == {}:
            entities = entities[1:]
        for e in entities:
            m = q.popleft()
            while m[0] != e.text:
                m = q.popleft()
            entity = NewEntity.new(e, event_fts, m[1])
            # make sure span created is correct
            assert raw_text[entity.span[0] : entity.span[1] + 1] == e.text
            res.append(entity)
            
        return res, raw_text
    
    def build_queue(self, all_text):
        raw_text = ""
        counter = -1
        q = deque()
        for tx in all_text:
            q.append((tx, counter))
            counter += len(tx)
            raw_text += tx

        return q, raw_text

    def parse(self, id: str, text_path: Path, resolve_refs=True):

        print(id)
        with text_path.open(encoding='utf-8') as f:
            xml_tree = etree.parse(f)
            
            events = xml_tree.xpath('.//EVENT')
            timex = xml_tree.xpath('.//TIMEX3')
            
            # need to figure a way to add doc
            alinks = []

            event_fts = {e.attrib['eventID']: (e.attrib['tense'], e.attrib['polarity'], e.attrib['aspect']) for e in xml_tree.xpath('.//MAKEINSTANCE')}
            all_text = list(xml_tree.xpath('.//TEXT')[0].itertext())
            
            events, raw_text = self.parse_entities(events, all_text, event_fts)
            timexs, _ = self.parse_entities(timex, all_text)
            entities = events #+ timexs
            entities = OrderedDict([(e.id, e) for e in entities])
        
            relations = []
            pos_pairs = []

            total_count = 0
            missing_count = 0
            for el in xml_tree.xpath('.//TLINK'):
                id_name = 'eventInstanceID' if 'eventInstanceID' in el.attrib.keys() else 'timeID'
                
                if el.attrib[id_name] not in entities.keys():
                    total_count += 1
                    missing_count += 1
                    continue
                # if events are related to time
                # exclude t0 for now
                elif 'relatedToTime' in el.attrib.keys():
                    if el.attrib['relatedToTime'] == "t0" or el.attrib['relatedToTime'] not in entities.keys():
                        if el.attrib['relatedToTime'] != "t0":
                            total_count += 1
                            missing_count += 1
                        continue
                    else:
                        total_count += 1
                        relations.append(NewRelation.new(el, entities))
                        pos_pairs.append((el.attrib[id_name], el.attrib['relatedToTime']))
                else:
                    if el.attrib['relatedToEventInstance'] not in entities.keys():
                        total_count += 1
                        missing_count +=1
                        continue
                    else:
                        total_count += 1
                        relations.append(NewRelation.new(el, entities))
                        pos_pairs.append((el.attrib[id_name], el.attrib['relatedToEventInstance']))
            
            print("total positive samples: %s" % len(relations))

            # We need to construct vague pairs that are not more than one sentence distance away 
            # Also append Causal relations here.

            vague_count = 0
            ent_keys = list(entities.keys())

            for i in range(len(ent_keys) - 1):
                for j in range(i+1, len(ent_keys)):
                    lkey = ent_keys[i]
                    rkey = ent_keys[j]
                    if (entities[lkey].id, entities[rkey].id) in pos_pairs:
                        continue
                    # only need to compare starting time.
                    # ensure right event / time occur after left entity
                    if  entities[lkey].span[0] > entities[rkey].span[0]:
                        continue
                    # ensure events are within one sentence distance away
                    elif raw_text[entities[lkey].span[0] : entities[rkey].span[1]].count('\n') > 1:
                        continue
                    # also exclude both are time
                    elif entities[lkey].id[0] == 't' and entities[rkey].id[0] == 't':
                        continue
                    else:
                        res = {}
                        res['id'] = 'vid%s' % vague_count
                        vague_count += 1
                        # relation category                                                                                            
                        res['type'] = 'TLINK'
                        # relation type
                        res['properties'] = {}
                        res['properties']['type'] = ['VAGUE']
                        res['properties']['source'] = [entities[lkey]]
                        res['properties']['target'] = [entities[rkey]]
                        
                        relations.append(NewRelation(**res))

            print("Total vague samples are: %s" % vague_count)
            relations = {r.id: r for r in relations}
            
            return NewDoc(id, raw_text, entities, relations)


class PackageReader:

    def __init__(self, dir_path: str):
        self.root = os.path.abspath(dir_path)
        assert os.path.exists(self.root) and os.path.isdir(self.root)

        split_indx = 20
        tmpr_dir = f'{self.root}/TemporalPart/*'
        src_docs = glob.glob(tmpr_dir)
        
        assert len(src_docs) == 25
        
        random.seed(7)
        random.shuffle(src_docs)

        suffix = ".tml"
        src_to_id = {src:src.replace(suffix, "")[len(tmpr_dir)-1:] for src in src_docs}
 
        test_files = ['2010.01.13.google.china.exit',
                      '2010.01.13.mexico.human.traffic.drug',
                      '2010.01.08.facebook.bra.color',
                      '2010.01.12.haiti.earthquake',
                      '2010.01.12.turkey.israel']

        self.train_files = {k:v for k,v in list(src_to_id.items()) if v not in test_files}
        self.test_files = {k:v for k,v in list(src_to_id.items()) if v in test_files}

        assert len(self.train_files) + len(self.test_files) == len(src_to_id)

    def read_split(self, split_name) -> Iterator[NewDoc]:
        assert split_name in ('train', 'test')

        src_id_map = {'train': self.train_files, 'test': self.test_files}[split_name]
        
        for src_file, doc_id in src_id_map.items():
            doc = NewDoc(doc_id)
            yield doc.parse(doc_id, Path(src_file))


class FlexNLPAnnotator:

    def __init__(self):
        self.pipeline = (Pipeline.builder()
                         .add(PlainTextIngester())
                         .add(SpacyAnnotator.create_for_language(SpacyAnnotator.ENGLISH,
                                                                 use_regions=False,
                                                                 respect_existing_sentences=False))
                         .build())

    def __call__(self, doc:NewDoc):
        return self.pipeline.process(WithId(doc.id, doc.raw_text))


def flexnlp_annotate(reader: PackageReader, out_dir: Path,
                     splits: Iterator[str] = ('dev', 'test', 'train')):
    """                                                                                                                             
    Annotate docs using FlexNLP Pipeline                                                                                            
    Args:                                                                                                                           
        reader: reader to access                                                                                                    
        out_dir: directory to store files                                                                                           
        splits: data splits such as test, dev, train                                                                                
                                                                                                                                    
    Returns:                                                                                                                        
                                                                                                                                    
    """
    annotate = FlexNLPAnnotator()
    for split in splits:
        for doc in reader.read_split(split):
            doc.nlp_ann = annotate(doc)
            out_path = out_dir / split / f'{doc.id}.pkl'
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with out_path.open('wb') as fh:
                pickle.dump(doc, fh)

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('-task', choices={'json', 'flexnlp'}, type=str,
                   help='Tasks. "json" will simply exports documents to jsonline file;'
                        ' "flexnlp" will annotate docs and dump them as pickle files')
    p.add_argument('-dir', help='Path to RED directory (extracted .tgz)', type=Path)
    p.add_argument('-out', help='output File (if task=json) or Folder (if task=flexnlp)', type=Path)

    args = p.parse_args()
    
    args.task = 'flexnlp'
    args.dir = Path('/nas/home/rujunhan/data/TemporalCausalReasoning/')
    args.out = Path('/nas/home/rujunhan/tcr_output/')
    

    print(args)
    
    pr = PackageReader(args.dir)
    
    if args.task == 'json':
        assert not args.out.exists() or args.out.is_file()
        with args.out.open('w', encoding='utf-8', errors='replace') as out:
            export_json_lines(pr, out)
    elif args.task == 'flexnlp':
        assert not args.out.exists() or args.out.is_dir()
        flexnlp_annotate(pr, args.out, splits = ["train", "test"])
    
