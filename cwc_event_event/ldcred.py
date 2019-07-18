#!/usr/bin/env python
# Author : TG ; Created: July 30, 2018

import os
from dataclasses import dataclass
from typing import (List, Iterator, Tuple, Union, Dict, Mapping, Callable, Set, Type, TextIO,
                    Optional)
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

log.basicConfig(level=log.DEBUG)


def camel_to_snake(name: str) -> str:
    name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    name = re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()
    return name.replace('__', '_')


@dataclass
class REDAnnBase:
    """Base class of RED annotations """
    id: str
    type: str
    parents_type: str
    properties: Dict[str, List[str]]

    @classmethod
    def parse_xml(cls, node: etree.Element):
        res = {
            'id': node.xpath('id/text()')[0],
            'type': node.xpath('type/text()')[0],
            'parents_type': node.xpath('parentsType/text()')[0]
        }

        pairs = [(el.tag, el.text) for el in node.xpath('properties/*')]
        pairs = [(camel_to_snake(name), None if val == 'NONE' else val) for name, val in pairs]

        props = ddict(list)
        for name, val in pairs:
            props[name].append(val)
        res['properties'] = dict(props)
        return res


@dataclass
class REDEntity(REDAnnBase):
    text: str
    span: Tuple[int, int]

    @classmethod
    def parse_xml(cls, node: etree.Element):
        res = super(REDEntity, cls).parse_xml(node)
        span_strs = node.xpath('span/text()')
        res['span'] = tuple(map(int, span_strs[0].split(',')))[:2]
        return res

    @classmethod
    def new(cls, node: etree.Element, raw_text: str):
        res = cls.parse_xml(node)
        start, end = res['span']
        res['text'] = raw_text[start: end]
        return cls(**res)


@dataclass
class REDRelation(REDAnnBase):
    properties: Dict[str, List[Union[str, REDEntity]]]

    @classmethod
    def new(cls, node: etree.Element):
        return cls(**cls.parse_xml(node))


@dataclass
class REDDoc:
    id: str
    raw_text: str
    entities: Mapping[str, REDEntity]
    relations: Mapping[str, REDRelation]
    nlp_ann: Optional[Document] = None

    @staticmethod
    def parse(id: str, text_path: Path, ann_path: Path, resolve_refs=True):
        with open(text_path, encoding="utf-8") as fh:
            raw_text = fh.read()
        with ann_path.open() as f:
            xml_tree = etree.parse(f)
            entities = [REDEntity.new(el, raw_text) for el in xml_tree.xpath('.//entity')]
            relations = [REDRelation.new(el) for el in xml_tree.xpath('.//relation')]
        entities = {e.id: e for e in entities}
        relations = {r.id: r for r in relations}
        doc = REDDoc(id, raw_text, entities, relations)
        if resolve_refs:
            doc = doc.resolve_references()
        return doc

    def resolve_references(self):
        for rel in self.relations.values():
            for name, vals in rel.properties.items():
                rel.properties[name] = [self.entities[val] if type(val) is str and '@e@' in val
                                        else val for val in vals]
        return self

    def pprint(self, entity_filter: Callable[[REDEntity], bool] = None,
               relation_filter: Callable[[REDRelation], bool] = None):
        print('ID: ' + self.id)
        es = list(filter(entity_filter, self.entities.values())) if entity_filter \
            else self.entities.values()
        if es:
            print("Entities: ")
            for e in es:
                print(f"- {e}")

        rels = list(filter(relation_filter, self.relations.values())) if relation_filter \
            else self.relations.values()
        if rels:
            print("Relations: ")
            for rel in rels:
                print(f' - {rel.id}  {rel.type}  {rel.parents_type}')
                for name, val in rel.properties:
                    print(f'--\t\t {name} ::  {val}')


class FlexNLPAnnotator:

    def __init__(self):
        self.pipeline = (Pipeline.builder()
                         .add(PlainTextIngester())
                         .add(SpacyAnnotator.create_for_language(SpacyAnnotator.ENGLISH,
                                                                 use_regions=False,
                                                                 respect_existing_sentences=False))
                         .build())

    def __call__(self, doc):
        return self.pipeline.process(WithId(doc.id, doc.raw_text))


class PackageReader:

    def __init__(self, dir_path: str):
        self.root = os.path.abspath(dir_path)
        assert os.path.exists(self.root) and os.path.isdir(self.root)
        src_docs = glob.glob(f'{self.root}/data/source/*/*')
        assert src_docs
        suffix = '.RED-Relation.gold.completed.xml'

        src_to_ann = [(src, f'{src.replace("source", "annotation")}{suffix}') for src in src_docs]
        missing_docs = [t for t in src_to_ann if not os.path.exists(t[1])]
        if missing_docs:
            raise Exception(f"Missing source to annotation mapping for {missing_docs}")
        src_to_ann = dict(src_to_ann)

        with open(f'{self.root}/docs/splits.txt') as fh:
            lines = [line.strip() for line in fh]
        lines = [line for line in lines if line]

        dev_files = lines[1: lines.index('test set:')]  # first few lines are dev
        test_files = lines[lines.index('test set:') + 1:]  # last few lines are test
        dir_pref = f'{self.root}/data/source/'
        dev_files = set(dir_pref + f for f in dev_files)
        test_files = set(dir_pref + f for f in test_files)

        log.info(f"Found {len(src_docs)} total docs, splits: {len(test_files)} "
                 f"test docs and {len(dev_files)} dev docs")
        self.dev_files = {f: src_to_ann[f] for f in dev_files}
        self.test_files = {f: src_to_ann[f] for f in test_files}
        self.train_files = {f: src_to_ann[f] for f in src_to_ann.keys() - dev_files - test_files}
        assert len(self.dev_files) + len(self.test_files) + len(self.train_files) == len(src_to_ann)

    def read_split(self, split_name) -> Iterator[REDDoc]:
        assert split_name in ('train', 'dev', 'test')
        src_ann_map = {'train': self.train_files, 'dev': self.dev_files, 'test': self.test_files}[
            split_name]
        for src_file, ann_file in src_ann_map.items():
            doc_id = '/'.join(src_file.split('/')[-2:])
            yield REDDoc.parse(doc_id, Path(src_file), Path(ann_file))


@dataclass
class JsonSerializer:
    types: Set[Type]

    def __call__(self, obj):
        if type(obj) is list:
            return [self(x) for x in obj]
        elif type(obj) is dict:
            return {k: self(v) for k, v in obj.items()}
        elif type(obj) in self.types:
            return self(vars(obj))
        return obj


def export_json_lines(reader: PackageReader, out: TextIO,
                      splits: Iterator[str] = ('dev', 'test', 'train')):
    serializer = JsonSerializer({REDEntity, REDRelation, REDDoc})
    count = 0
    for split in splits:
        for doc in reader.read_split(split):
            doc = vars(doc)
            doc['split'] = split
            line = json.dumps(doc, ensure_ascii=False, default=serializer)
            out.write(line)
            out.write('\n')
            count += 1
    log.info(f"wrote {count} docs")


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
    p.add_argument('task', choices={'json', 'flexnlp'}, type=str,
                   help='Tasks. "json" will simply exports documents to jsonline file;'
                        ' "flexnlp" will annotate docs and dump them as pickle files')
    p.add_argument('dir', help='Path to RED directory (extracted .tgz)', type=Path)
    p.add_argument('out', help='output File (if task=json) or Folder (if task=flexnlp)', type=Path)

    args = p.parse_args()
    pr = PackageReader(args.dir)
    if args.task == 'json':
        assert not args.out.exists() or args.out.is_file()
        with args.out.open('w', encoding='utf-8', errors='replace') as out:
            export_json_lines(pr, out)
    elif args.task == 'flexnlp':
        assert not args.out.exists() or args.out.is_dir()
        flexnlp_annotate(pr, args.out)
