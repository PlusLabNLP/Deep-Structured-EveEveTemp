#!/usr/bin/env python                                                                                                   
# Author : RJ ; Created: Oct 26, 2018                                                                                  
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
from collections import OrderedDict, Counter
log.basicConfig(level=log.DEBUG)


label_map = OrderedDict([('v','VAGUE'), 
                         ('b', 'BEFORE'), 
                         ('a', 'AFTER'), 
                         ('ii', 'IS_INCLUDED'), 
                         ('i', 'INCLUDES'), 
                         ('s', 'SIMULTANEOUS')])

@dataclass
class TESVEntity():
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
            res['id'] = fts[cls.attrib['eid']][2]
            res['type'] = cls.attrib['class']
            res['tense'] = fts[cls.attrib['eid']][0]
            res['aspect'] = None
            res['polarity'] = fts[cls.attrib['eid']][1]
        elif cls.tag == "TIMEX3":
            res['id'] = cls.attrib['tid']
            res['type'] = cls.attrib['type']
            res['aspect'] = None
            res['tense'] = None
            res['polarity'] = None
        return TESVEntity(**res)

@dataclass
class TESVRelation():
    id: str
    type: str
    properties: Dict[str, List[Union[str, TESVEntity]]]
    
    def new(pair, label, entities, id, id_counter):
        
        res = {}
        # relation type
        
        # TBDense data doesn't supply relation id
        res['properties'] = {}
        res['id'] = id + '_' + str(id_counter)
        res['type'] = 'TLINK'
        source, target = pair
        try:
            source = source[0] + source[1:] if source[0] == 'e' else source
        except:
            print(pair)
            print(label)
            print(source)
            kill

        target = target[0] + target[1:] if target[0] =='e' else target

        res['properties']['source'] = [entities[source]]
        res['properties']['target'] = [entities[target]]
        res['properties']['type'] = [label]
        return TESVRelation(**res)

@dataclass
class TESVDoc:
    id: str
    raw_text: str = ""
    entities: Mapping[str, TESVEntity] = None
    relations: Mapping[str, TESVRelation] = None
    nlp_ann: Optional[Document] = None

    def parse_entities(self, entities, all_text, event_fts={}):

        res = []
        
        # this new dataset doesn't have span indicators, so we need to create it manually
        q, raw_text = self.build_queue(all_text)
        
        for e in entities:
            if e.attrib['eid'] in event_fts:
                m = q.popleft()
                while m[0] != e.text:
                    m = q.popleft()
                
                entity = TESVEntity.new(e, event_fts, m[1])
                # make sure span created is correct
                assert raw_text[entity.span[0] : entity.span[1] + 1] == e.text
                res.append(entity)
            else:
                # exclude entities that are missing in the <MAKEINSTANCE>
                continue
            
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

    def parse(self, id: str, text_path: Path):
        
        print(text_path, id)

        with text_path.open(encoding='utf-8') as f:
            xml_tree = etree.parse(f)
            
            events = xml_tree.xpath('.//EVENT')
            #timex = [t for t in xml_tree.xpath('.//TIMEX3') if t.attrib['functionInDocument'] != 'CREATION_TIME']
            
            event_fts = {e.attrib['eventID']: (e.attrib['tense'], e.attrib['polarity'], e.attrib['eiid']) for e in xml_tree.xpath('.//MAKEINSTANCE')}
            all_text = list(xml_tree.xpath('.//TEXT')[0].itertext())

            events, raw_text = self.parse_entities(events, all_text, event_fts)
            #timexs, _ = self.parse_entities(timex, all_text)
            entities = events #+ timexs
            entities = OrderedDict([(e.id, e) for e in entities])

            relations = []
            pos_pairs = []

            total_count = 0
            missing_count = 0
            
            total = 0
            event = 0
            timex = 0
            missing = 0

            id_counter = 1
            for el in xml_tree.xpath('.//TLINK'):
                id_name = 'eventInstanceID' if 'eventInstanceID' in el.attrib.keys() else 'timeID'

                total += 1
                if el.attrib[id_name] not in entities.keys():
                    missing += 1
                    continue

                # only take events for now 
                elif 'relatedToTime' in el.attrib.keys():
                    timex += 1
                else:
                    if el.attrib['relatedToEventInstance'] not in entities.keys():
                        missing += 1
                        continue
                    else:
                        event += 1
                        pair = (el.attrib['eventInstanceID'], el.attrib['relatedToEventInstance'])
                        relations.append(TESVRelation.new(pair, 'NONE', entities, id, id_counter))
                        id_counter += 1
            
            print(total, event, timex, missing)
            relations = {r.id: r for r in relations}
            return TESVDoc(id, raw_text, entities, relations)


class PackageReader:

    def __init__(self, dir_path: str):
        self.root = os.path.abspath(dir_path)

        assert os.path.exists(self.root) and os.path.isdir(self.root)

        raw_dir = f'{self.root}/**/*'
        src_docs = glob.glob(raw_dir, recursive=True)

        print('Total TE3-Silver files: %s' % len(src_docs))

        suffix = ".tml"
        
        # we only need to doc name as unique key, and first 6 elements are directories
        src_to_id = {src:src.replace(suffix, "")[len(raw_dir)+14:] for src in src_docs[6:]}

        self.src_to_id = src_to_id

        assert len(src_to_id) == len(src_docs) - 6

    def read_split(self, split_name) -> Iterator[TESVDoc]:
        assert split_name in ('train', 'dev', 'test')
        
        for src_file, doc_id in self.src_to_id.items():
            doc = TESVDoc(doc_id)
            yield doc.parse(doc_id, Path(src_file))


class FlexNLPAnnotator:

    def __init__(self):
        self.pipeline = (Pipeline.builder()
                         .add(PlainTextIngester())
                         .add(SpacyAnnotator.create_for_language(SpacyAnnotator.ENGLISH,
                                                                 use_regions=False,
                                                                 respect_existing_sentences=False))
                         .build())

    def __call__(self, doc:TESVDoc):
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
    args.dir = Path('/nas/home/rujunhan/StructTempRel-EMNLP17/data/TempEval3/Training/TE3-Silver')
    args.out = Path('/nas/home/rujunhan/te3sv_output/')
    print(args)

    pr = PackageReader(args.dir)

    if args.task == 'json':
        assert not args.out.exists() or args.out.is_file()
        with args.out.open('w', encoding='utf-8', errors='replace') as out:
            export_json_lines(pr, out)
    elif args.task == 'flexnlp':
        assert not args.out.exists() or args.out.is_dir()
        flexnlp_annotate(pr, args.out, splits = ["train"])
    
