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
from collections import OrderedDict
log.basicConfig(level=log.DEBUG)


label_map = OrderedDict([('VAGUE','VAGUE'),
                         ('BEFORE', 'BEFORE'),
                         ('AFTER', 'AFTER'), 
                         ('EQUAL', 'SIMULTANEOUS'),
                         ])

@dataclass
class TBDEntity():
    id: str
    type: str
    text: str
    tense: str
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
            res['polarity'] = fts[cls.attrib['eid']][1]
        elif cls.tag == "TIMEX3":
            res['id'] = cls.attrib['tid']
            res['type'] = cls.attrib['type']
            res['tense'] = None
            res['polarity'] = None
        return TBDEntity(**res)

@dataclass
class TBDRelation():
    id: str
    type: str
    properties: Dict[str, List[Union[str, TBDEntity]]]
    
    def new(pair, pair_text, label, entities, id, id_counter):
        
        res = {}
        # relation type
        
        # TBDense data doesn't supply relation id
        res['properties'] = {}
        res['id'] = id + '_' + str(id_counter)
        res['type'] = 'TLINK'
        source, target = pair
        try:
            source = source[0] + 'i' + source[1:] if source[0] == 'e' else source
        except:
            print(pair)
            print(label)
            print(source)
            kill

        target = target[0] + 'i' + target[1:] if target[0] =='e' else target

        print(target, source)
        assert entities[source].text == pair_text[0]
        assert entities[target].text == pair_text[1]

        res['properties']['source'] = [entities[source]]
        res['properties']['target'] = [entities[target]]
        res['properties']['type'] = [label_map[label]]
        return TBDRelation(**res)

@dataclass
class TBDDoc:
    id: str
    raw_text: str = ""
    entities: Mapping[str, TBDEntity] = None
    relations: Mapping[str, TBDRelation] = None
    nlp_ann: Optional[Document] = None

    def parse_entities(self, entities, all_text, event_fts={}):

        res = []
        
        # this new dataset doesn't have span indicators, so we need to create it manually
        q, raw_text = self.build_queue(all_text)
        
        for e in entities:
            m = q.popleft()
            while m[0] != e.text:
                m = q.popleft()
            entity = TBDEntity.new(e, event_fts, m[1])
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

    def parse(self, id: str, text_path: Path, dense_pairs: {}):
        
        print(id)
        with text_path.open(encoding='utf-8') as f:
            xml_tree = etree.parse(f)
            
            events = xml_tree.xpath('.//EVENT')
            timex = [t for t in xml_tree.xpath('.//TIMEX3') if t.attrib['functionInDocument'] != 'CREATION_TIME']
            
            event_fts = {e.attrib['eventID']: (e.attrib['tense'], e.attrib['polarity'], e.attrib['eiid']) for e in xml_tree.xpath('.//MAKEINSTANCE')}
            all_text = list(xml_tree.xpath('.//TEXT')[0].itertext())

            events, raw_text = self.parse_entities(events, all_text, event_fts)
            #timexs, _ = self.parse_entities(timex, all_text)
            entities = events #+ timexs
            entities = OrderedDict([(e.id, e) for e in entities])
        
            #for k in entities.keys():
            #    print(k)

            #print(raw_text)
            #print(entities)

            relations = []
            pos_pairs = []

            total_count = 0
            missing_count = 0

            #print(len(dense_pairs[id]))
            #print(dense_pairs[id])
            
            num_doc_times = 0
            id_counter = 1
            for pair, (label, pair_text) in dense_pairs[id].items():
                # It seems t0 in the TBDense dataset denote doc_create_time
                if pair[0] == 't0' or pair[1] == 't0':
                    #print('Doc Time, skip')
                    #print(pair)
                    num_doc_times += 1
                else:
                    relations.append(TBDRelation.new(pair, pair_text, label, entities, id, id_counter))
                    id_counter += 1

            print(len(events))#, len(timex))
            print(num_doc_times)
            assert (len(relations) + num_doc_times) == len(dense_pairs[id]) 
            relations = {r.id: r for r in relations}
            
            return TBDDoc(id, raw_text, entities, relations)


class PackageReader:

    def __init__(self, dir_path: str, dense_pairs: {}):
        self.root = os.path.abspath(dir_path)

        assert os.path.exists(self.root) and os.path.isdir(self.root)

        raw_dir = f'{self.root}/timeml/*'
        src_docs = glob.glob(raw_dir)
        
        assert len(src_docs) == 183

        self.all_samples = dense_pairs

        suffix = ".tml"
        src_to_id = {src:src.replace(suffix, "")[len(raw_dir)-1:] for src in src_docs}
 
        dev_files = ["APW19980227.0487",
                     "CNN19980223.1130.0960", 
                     "NYT19980212.0019",
                     "PRI19980216.2000.0170", 
                     "ed980111.1130.0089"]

        test_files = ["APW19980227.0489",
                      "APW19980227.0494", 
                      "APW19980308.0201", 
                      "APW19980418.0210",
                      "CNN19980126.1600.1104", 
                      "CNN19980213.2130.0155",
                      "NYT19980402.0453", 
                      "PRI19980115.2000.0186",
                      "PRI19980306.2000.1675"]

        src_to_id = {k: v for k,v in src_to_id.items() if v in self.all_samples.keys()}

        #for x in sorted(list(src_to_id.values())):
        #    print(x)
            
        # MATRES is missing one docs from original TBDENSE
        # this docs contain very few event pairs (<10)
        assert len(src_to_id) == 35

        self.train_files = {k:v for k,v in list(src_to_id.items()) if v not in test_files}# + dev_files}
        #self.dev_files = {k:v for k,v in list(src_to_id.items()) if v in dev_files}
        self.test_files = {k:v for k,v in list(src_to_id.items()) if v in test_files}

        assert len(self.train_files)  + len(self.test_files) == len(src_to_id)


    def read_split(self, split_name) -> Iterator[TBDDoc]:
        assert split_name in ('train','test')

        src_id_map = {'train': self.train_files,  'test': self.test_files}[split_name]
        
        for src_file, doc_id in src_id_map.items():
            doc = TBDDoc(doc_id)
            yield doc.parse(doc_id, Path(src_file), self.all_samples)


class FlexNLPAnnotator:

    def __init__(self):
        self.pipeline = (Pipeline.builder()
                         .add(PlainTextIngester())
                         .add(SpacyAnnotator.create_for_language(SpacyAnnotator.ENGLISH,
                                                                 use_regions=False,
                                                                 respect_existing_sentences=False))
                         .build())

    def __call__(self, doc:TBDDoc):
        return self.pipeline.process(WithId(doc.id, doc.raw_text))


def flexnlp_annotate(reader: PackageReader, out_dir: Path,
                     splits: Iterator[str] = ('test', 'train')):
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


def tb_dense(dir_path):
    # tb_dense dataset only uses a subset of time bank dataset: 36 / 183                                
    # input: txt file specifying: doc, ent1, ent2, rel_type
    #output: dictionary - {doc: {(ent1, ent2):rel_type}}

    tbd = open(str(dir_path) + "/pairs.txt", 'r')
    count = 0
    tbd_samples = {}
    for line in tbd:
        line = line.strip().split('\t')
        line = [x for x in line if len(x) > 0]
        doc_id = line[0]
        ent1_id = line[1]
        ent2_id = line[2]
        label = line[3]

        if doc_id in tbd_samples.keys():
            tbd_samples[doc_id][(ent1_id, ent2_id)] = label
        else:
            tbd_samples[doc_id] = {(ent1_id, ent2_id): label}
        count += 1

    docs = list(tbd_samples.keys())
    print("Total %s TBD samples processed" % count)
    print("Total %s TBD docs processed" % len(docs))

    
    samples = open(str(dir_path) + "/matres.txt", 'r')
    count = 0
    all_samples = {}
    for line in samples:
        line = line.strip().split('\t')
        line = [x for x in line if len(x) > 0]
        doc_id = line[0]
        ent1 = line[1]
        ent2 = line[2]
        ent1_id = "e" + line[3]
        ent2_id = "e" + line[4]
        label = line[5]

        if doc_id in docs:
            if doc_id in all_samples.keys():
                all_samples[doc_id][(ent1_id, ent2_id)] = (label, (ent1, ent2))
            else:
                all_samples[doc_id] = {(ent1_id, ent2_id): (label, (ent1, ent2))}
            count += 1

    docs_m = list(all_samples.keys())
    for d in docs:
        if d not in docs_m:
            print("%s is missing" %d)
    
    print("Total %s MATRES samples included" % count)
    print("Total %s MATRES docs processed" % len(docs_m))
    
    return all_samples


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('-task', choices={'json', 'flexnlp'}, type=str,
                   help='Tasks. "json" will simply exports documents to jsonline file;'
                        ' "flexnlp" will annotate docs and dump them as pickle files')
    p.add_argument('-dir', help='Path to RED directory (extracted .tgz)', type=Path)
    p.add_argument('-out', help='output File (if task=json) or Folder (if task=flexnlp)', type=Path)
    p.add_argument('-dense_pairs', help='all samples in TBDense', type=dict)
    
    args = p.parse_args()
    
    args.task = 'flexnlp'
    args.dir = Path('/nas/home/rujunhan/data/TBDense/')
    args.out = Path('/nas/home/rujunhan/matres_output/')
    print(args)

    args.dense_pairs = tb_dense(args.dir)

    pr = PackageReader(args.dir, args.dense_pairs)
    
    if args.task == 'json':
        assert not args.out.exists() or args.out.is_file()
        with args.out.open('w', encoding='utf-8', errors='replace') as out:
            export_json_lines(pr, out)
    elif args.task == 'flexnlp':
        assert not args.out.exists() or args.out.is_dir()
        flexnlp_annotate(pr, args.out, splits = ["train",  "test"])
    
