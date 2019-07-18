# 1. Load processed data
# 2. featurize
# 3. shuffle
# 4. save for model training

from pathlib import Path
import pickle
import argparse
from flexnlp import Document
from collections import defaultdict, Counter, OrderedDict
from itertools import combinations
from typing import Iterator, List, Mapping, Union, Optional, Set
import random
from baseline import FlatRelation, print_annotation_stats, print_flat_relation_stats, read_relations, all_red_labels, REDEveEveRelModel, ClassificationReport, matres_label_map, tbd_label_map, new_label_map, red_label_map, rev_map, causal_label_map
from featureFuncs import *
import multiprocessing as mp
from functools import partial
from sklearn.model_selection import KFold, ParameterGrid, train_test_split
import logging as log
from ldctcr import NewDoc, NewRelation, NewEntity
from ldctbd import TBDDoc, TBDRelation, TBDEntity
from ldcte3sv import TESVDoc, TESVRelation, TESVEntity
from pytorch_pretrained_bert.modeling import BertModel, BertConfig, PreTrainedBertModel
from pytorch_pretrained_bert.tokenization import BertTokenizer
from sklearn.utils import resample
import os
import torch
from torch.utils import data
import time

def create_features(ex, args):

    pos2idx = args.pos2idx
    pos_dict = ex.doc.pos_dict #create_pos_dict(ex.doc.nlp_ann.pos_tags())                                      
    all_keys, lidx_start, lidx_end, ridx_start, ridx_end = token_idx(ex.left.span, ex.right.span, pos_dict)

    # create an indicator of whether two events are outside of pred_window                                      
    left_mid = (lidx_start + lidx_end) / 2.0
    right_mid = (ridx_start + ridx_end) / 2.0
    pred_ind = False if np.abs(left_mid - right_mid) > args.pred_win else True

    # compute the exact number of left and right neighbors                                                     \
                                                                                                                    
    if lidx_start != lidx_end:
        lngbrs = args.ngbrs - math.floor((lidx_end - lidx_start) / 2)
        rngbrs = args.ngbrs - math.ceil((lidx_end - lidx_start) / 2)
    else:
        lngbrs = args.ngbrs
        rngbrs = args.ngbrs

    end_pad = rngbrs
    # left_ngbrs out of range, pad left                                                                        \
                                                                                                                    
    if lidx_start - lngbrs < 0:
        lsent_pos = [('<pad>', '<unk>') for x in range(lngbrs - lidx_start)] + [pos_dict[x] for x in all_keys[:lidx_end+1+rngbrs]]
    # right_nbrs out of range pad right                                                                        \
                                                                                                                    
    elif lidx_end + rngbrs > len(pos_dict) - 1:
        lsent_pos = [pos_dict[x] for x in all_keys[lidx_start - lngbrs:]] + [('<pad>', '<unk>') for x in range(rngbrs + lidx_end - (len(pos_dict) - 1))]
    # regular cases                                                                                            \
    else:                                                                                                      \
        lsent_pos = [pos_dict[x] for x in all_keys[lidx_start - lngbrs : lidx_end + 1 + rngbrs]]

    # adjust target token index in left sentence                                                               \
                                                                                                                    
    lidx_end_s = (lidx_end - lidx_start) + lngbrs
    lidx_start_s = lngbrs

    assert lidx_start_s >= 0
    assert lidx_end_s < args.ngbrs * 2

    # need to figure out exact number of left and right neighbors                                              \
                                                                                                                    
    if ridx_start != ridx_end:
        lngbrs = args.ngbrs - math.floor((ridx_end - ridx_start) / 2)
        rngbrs = args.ngbrs - math.ceil((ridx_end - ridx_start) / 2)
    else:
        lngbrs = args.ngbrs
        rngbrs = args.ngbrs

    end_pad += lngbrs

    end_pad -= (np.abs(lidx_end - ridx_start) - 1)

    # left_ngbrs out of range, pad left                                                                        \
                                                                                                                    
    if ridx_start - lngbrs < 0:
        rsent_pos = [('<pad>', '<unk>') for x in range(lngbrs - ridx_start)] + [pos_dict[x] for x in all_keys[:ridx_end+1+rngbrs]]
    # right_nbrs out of range pad right                                                                        \
    elif ridx_end + rngbrs > len(pos_dict) - 1:
        rsent_pos = [pos_dict[x] for x in all_keys[ridx_start - lngbrs:]] + [('<pad>', '<unk>') for x in range(\
rngbrs + ridx_end - (len(pos_dict) - 1))]
    # regular cases
    else:
        rsent_pos = [pos_dict[x] for x in all_keys[ridx_start - lngbrs : ridx_end + 1 + rngbrs]]

    assert len(lsent_pos) == 2 * args.ngbrs + 1
    assert len(rsent_pos) == 2 * args.ngbrs + 1

    # adjust target token index in right sentence                                                              \
    ridx_end_s = (ridx_end - ridx_start) + lngbrs
    ridx_start_s = lngbrs

    assert ridx_start_s >= 0
    assert ridx_end_s < args.ngbrs * 2

    if lidx_end_s + (ridx_start - lidx_end) < len(lsent_pos):
        lsent_pos = lsent_pos[:lidx_end_s + (ridx_start - lidx_end)]
        rsent_pos = rsent_pos[ridx_start_s:]
        end_pad = ['<pad>' for _ in range(end_pad)]

        ridx_start_s = lidx_end_s + (ridx_start - lidx_end)
        ridx_end_s = lidx_end_s + (ridx_end - lidx_end)
    else:
        end_pad = []
        ridx_start_s += len(lsent_pos)
        ridx_end_s += len(lsent_pos)

    # lookup token idx                                                                                         \
                                                                                                                    
    lsent = [x[0].lower() for x in lsent_pos]
    lsent = [args.w2i[t] if t in args.w2i.keys() else 1 for t in lsent]

    rsent = [x[0].lower() for x in rsent_pos]
    rsent = [args.w2i[t] if t in args.w2i.keys() else 1 for t in rsent]
        
    psent = [args.w2i[p] for p in end_pad]

    # lookup pos_tag idx                                                                                       \

    lpos = [pos2idx[p] if p in pos2idx.keys() else len(pos2idx) for _, p in lsent_pos]
    rpos = [pos2idx[p] if p in pos2idx.keys() else len(pos2idx) for _, p in rsent_pos]
    ppos = [len(pos2idx) for p in end_pad]
    
    assert len(lsent) == len(lpos)
    assert len(rsent) == len(rpos)

    sent = lsent + rsent + psent
    pos = lpos + rpos + ppos

    assert len(sent) == args.ngbrs * 4 + 2
    assert len(pos) == args.ngbrs * 4 + 2

    return (sent, pos, ex.rev, lidx_start_s, lidx_end_s, ridx_start_s, ridx_end_s, pred_ind)


def create_features_tbd(ex, args):
    ### create a new feature creation method for TCR dataset                                                  
    # compute token index                                                                                      \
                                                                                                                    
    pos2idx = args.pos2idx
    pos_dict = ex.doc.pos_dict

    all_keys, lidx_start, lidx_end, ridx_start, ridx_end = token_idx(ex.left.span, ex.right.span, pos_dict)

    # find the start and end point of a sentence                                                               \
                                                                                                                    
    left_seq = [pos_dict[x][0] for x in all_keys[:lidx_start]]
    right_seq = [pos_dict[x][0] for x in all_keys[ridx_end + 1:]]

    try:
        sent_start = max(loc for loc, val in enumerate(left_seq) if val == '.') + 1
    except:
        sent_start = 0

    # sent_end token will not be included                                                                      \
    
    try:
        sent_end = ridx_end + 1 + min(loc for loc, val in enumerate(right_seq) if val == '.')
    except:
        sent_end = len(pos_dict)

    assert sent_start < sent_end
    assert sent_start <= lidx_start
    assert ridx_end <= sent_end

    sent_key = all_keys[sent_start:sent_end]
    # construct sentence and pos sequence with index 

    sent = [args.w2i[t] if t in args.w2i.keys() else 1 for t in [pos_dict[x][0].lower() for x in sent_key]]
    pos = [pos2idx[k] if k in pos2idx.keys() else len(pos2idx) for k in [pos_dict[x][1] for x in sent_key]]

    pred_ind = True

    # calculate events' index in sentence                                                                      \         
    lidx_start_s = lidx_start - sent_start
    lidx_end_s = lidx_end - sent_start
    ridx_start_s = ridx_start - sent_start
    ridx_end_s = ridx_end - sent_start

    # create lexical features for the model                                                                    \
    new_fts = []
    if args.bert_fts:
        # 1. construct bert tokenized input                                                                              
        bert_sent = []
        for x in [pos_dict[x][0].lower() for x in sent_key]:
            try:
                bert_sent += args.tokenizer.convert_tokens_to_ids([x])
            except:
                bert_sent += args.tokenizer.convert_tokens_to_ids(['[UNK]'])
        bert_sent = torch.LongTensor(np.array([bert_sent]).transpose())
        # 2. computer bert embeddings                                                                                    
        bert_fts = bert_features(lidx_start_s, lidx_end_s, ridx_start_s, ridx_end_s, bert_sent, args.bert_model)
        # 3. map bert embeddings to lower dimension features space
        #    with learned mapping from unlabled data
        new_fts.extend(dim_reduction(bert_fts, args.mapping, args.n_fts))
        if args.ling_fts:
            new_fts.append(-distance_features(lidx_start, lidx_end, ridx_start, ridx_end))
            new_fts.extend(polarity_features(ex.left, ex.right))
            new_fts.extend(tense_features(ex.left, ex.right))
    else:
        new_fts.append(-distance_features(lidx_start, lidx_end, ridx_start, ridx_end))
        new_fts.extend(polarity_features(ex.left, ex.right))
        new_fts.extend(tense_features(ex.left, ex.right))

    return (sent, pos, new_fts, ex.rev, lidx_start_s, lidx_end_s, ridx_start_s, ridx_end_s, pred_ind)


def create_features_tcr(ex, args):
    ### create a new feature creation method for TCR dataset                                                    

    # compute token index                                                                                       
    pos2idx = args.pos2idx
    pos_dict = ex.doc.pos_dict                                                                                 \

    all_keys, lidx_start, lidx_end, ridx_start, ridx_end = token_idx(ex.left.span, ex.right.span, pos_dict)

    # find the start and end point of a sentence                                                                
    left_seq = [pos_dict[x][0] for x in all_keys[:lidx_start]]
    right_seq = [pos_dict[x][0] for x in all_keys[ridx_end + 1:]]

    try:
        sent_start = max(loc for loc, val in enumerate(left_seq) if val == '.') + 1
    except:
        sent_start = 0

    # sent_end token will not be included                                                                       
    try:
        sent_end = ridx_end + 1 + min(loc for loc, val in enumerate(right_seq) if val == '.')
    except:
        sent_end = len(pos_dict)

    assert sent_start < sent_end
    assert sent_start <= lidx_start
    assert ridx_end <= sent_end

    sent_key = all_keys[sent_start:sent_end]
    # construct sentence and pos sequence with index                                                            
    sent = [args.w2i[t] if t in args.w2i.keys() else 1 for t in [pos_dict[x][0].lower() for x in sent_key]]
    pos = [pos2idx[k] if k in pos2idx.keys() else len(pos2idx) for k in [pos_dict[x][1] for x in sent_key]]

    # prediction is always true for TCR dataset                                                                 
    pred_ind = True

    # calculate events' index in sentence                                                                                                              
    lidx_start_s = lidx_start - sent_start
    lidx_end_s = lidx_end - sent_start
    ridx_start_s = ridx_start - sent_start
    ridx_end_s = ridx_end - sent_start

    # create lexical features for the model                                                                                   
    new_fts = []
    if args.bert_fts:
        # 1. construct bert tokenized input
        bert_sent = []
        for x in [pos_dict[x][0].lower() for x in sent_key]:
            try:
                bert_sent += args.tokenizer.convert_tokens_to_ids([x])
            except:
                bert_sent += args.tokenizer.convert_tokens_to_ids(['[UNK]'])
        bert_sent = torch.LongTensor(np.array([bert_sent]).transpose())

        # 2. computer bert embeddings                                                                                        
        bert_fts = bert_features(lidx_start_s, lidx_end_s, ridx_start_s, ridx_end_s, bert_sent, args.bert_model)
        #print(bert_fts)                                                                                                      
        # 3. map bert embeddings to lower dimension features space                                                           
        #    with learned mapping from unlabled data                                                                         
        new_fts.extend(dim_reduction(bert_fts, args.mapping, args.n_fts))
    else:
        new_fts.append(-distance_features(lidx_start, lidx_end, ridx_start, ridx_end))
        new_fts.extend(polarity_features(ex.left, ex.right))
        new_fts.extend(tense_features(ex.left, ex.right))

    return (sent, pos, new_fts, ex.rev, lidx_start_s, lidx_end_s, ridx_start_s, ridx_end_s, pred_ind)


def compute_mapping(data, args):
    # compute mapping with unlabeled data using torch.svd()

    tokenizer = args.tokenizer
    bert_model = args.bert_model

    start_time = time.time()
    all_berts = []
    for ex in data:
        pos_dict = ex.doc.pos_dict                                                          

        all_keys, lidx_start, lidx_end, ridx_start, ridx_end = token_idx(ex.left.span, ex.right.span, pos_dict)

        # find the start and end point of a sentence                                                                  \
        left_seq = [pos_dict[x][0] for x in all_keys[:lidx_start]]
        right_seq = [pos_dict[x][0] for x in all_keys[ridx_end + 1:]]
        try:
            sent_start = max(loc for loc, val in enumerate(left_seq) if val == '.') + 1
        except:
            sent_start = 0

        # sent_end token will not be included                                                                         \
                                                                                                                       
        try:
            sent_end = ridx_end + 1 + min(loc for loc, val in enumerate(right_seq) if val == '.')
        except:
            sent_end = len(pos_dict)

        assert sent_start < sent_end
        assert sent_start <= lidx_start
        assert ridx_end <= sent_end

        sent_key = all_keys[sent_start:sent_end]

        # construct sentence with bert tokenizer index                                                                \
        # NOTE: this is a quick way to hack bert tokenization                                                          
        # need to figure out a way in data ingestion process                                                           
        sent = []
        for x in [pos_dict[x][0].lower() for x in sent_key]:
            try:
                sent += tokenizer.convert_tokens_to_ids([x])
            except:
                sent += tokenizer.convert_tokens_to_ids(['[UNK]'])

        sent = torch.LongTensor(np.array([sent]).transpose())

        # calculate events' index in sentence
        lidx_start_s = lidx_start - sent_start
        lidx_end_s = lidx_end - sent_start
        ridx_start_s = ridx_start - sent_start
        ridx_end_s = ridx_end - sent_start

        # create lexical features for the model                                                             
        all_berts.append(bert_features(lidx_start_s, lidx_end_s, ridx_start_s, ridx_end_s, sent, bert_model))

    end_time = time.time()

    print("%s seconds elapsed computing bert embeddings for %s samples" % (end_time - start_time, len(data)))
    assert len(all_berts) == len(data)
    a = torch.FloatTensor(all_berts)
    u,s,v = torch.svd(a)
    print(u.size(), s.size(), v.size())
    print(torch.dist(a, torch.mm(torch.mm(u, torch.diag(s)), v.t())))
    return v 


def dim_reduction(bert_fts, mapping, dim):
    
    fts = torch.mm(torch.FloatTensor(bert_fts).reshape(1, -1), mapping[:, :dim])
    #print(fts.size())
    return list(fts.data.numpy())

def data_split(train_docs, eval_docs, data):

    train_set = []
    eval_set = []

    for s in data:
        # dev-set doesn't require unlabled data
        if s[0] in eval_docs:
            # 0:doc_id, 1:ex.id, 2:(ex.left.id, ex.right.id), 3:label_id, 4:features  
            eval_set.append(s) 
        elif s[1][0] in ['L', 'C']:
            train_set.append(s) 

    return train_set, eval_set

def parallel(ex, args):

    if ex.rel_type == 'NONE':
        label_id = -1
    else:
        if args.data_type == "new" and ex.id[0] == 'C':
            label_id = args._label_to_id_c[ex.rel_type]
        else:
            label_id = args._label_to_id[ex.rel_type]

    # Features created for unlabeled data should be the same as labeled samples                     
    if args.data_type == "red":
        return ex.doc_id, ex.id, (ex.left.id, ex.right.id), label_id, create_features(ex, args)
    elif args.data_type == "new":
        return ex.doc_id, ex.id, (ex.left.id, ex.right.id), label_id, create_features_tcr(ex, args)
    # matres dataset is based on tbd data                                                           
    elif args.data_type in ["matres", "tbd"]:
        return ex.doc_id, ex.id, (ex.left.id, ex.right.id), label_id, create_features_tbd(ex, args)


def split_and_save(train_docs, dev_docs, data, data_u, seed, save_dir):
    # first split labeled into train and dev                                                                             
    # then append unlabed data into train                                                                                
    train_data, dev_data = data_split(train_docs, dev_docs, data)
    print(len(train_data), len(dev_data))

    # add unlabeled data only to train data                                                                              
    train_data = data_u + train_data
    # shuffle     
    #random.Random(seed).shuffle(train_data)

    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    with open(save_dir + '/train.pickle', 'wb') as handle:
        pickle.dump(train_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    handle.close()

    with open(save_dir + '/dev.pickle', 'wb') as handle:
        pickle.dump(dev_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    handle.close()

    return


def bootstrap_and_save(data, data_u, seed, save_dir):
    bs_samples = resample(data, replace=True, random_state=seed)
    keys = [x[1] for x in bs_samples]

    assert len(bs_samples) == len(data)

    # use not sampled data as dev data
    dev_data = [x for x in data if x[1] not in keys]
    print(seed, len(dev_data))

    # append unlabled data into labeled data and shuffle
    train_data = data_u + bs_samples
    random.Random(seed).shuffle(train_data)

    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    with open(save_dir + '/train.pickle', 'wb') as handle:
        pickle.dump(train_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    handle.close()

    with open(save_dir + '/dev.pickle', 'wb') as handle:
        pickle.dump(dev_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    handle.close()
    
    return


def reduce_vocab(data, save_dir, w2i, glove):
    # sent in data is index by original GloVe emb
    # 1. need to output a mappting from GloVe index to reduce index: glove2vocab
    # 2. a reduced emb saved in npy

    glove2vocab = {0:0, 1:1}
    count = 2
    emb = []
    i2w = {v:k for k,v in w2i.items()}

    for x in data:
        for t in x[4][0]:
            if t not in glove2vocab:
                glove2vocab[t] = count
                count += 1
                emb.append(glove[i2w[t]])
    
    emb = np.array(emb)
    print(emb.shape)
    assert emb.shape[1] == len(glove['the'])
    assert emb.shape[0] + 2 == len(glove2vocab)
    
    np.save(save_dir + '/emb_reduced.npy', emb)
    np.save(save_dir + '/glove2vocab.npy', glove2vocab)

    return


def compute_context_id(data):
    # compute context (1 - 2 sentences) _id map for cross validataion
    # context: doc_id + context_id
    
    context_id_map = {}
    count = 0
    for x in data:
        context = tuple(x[4][0])
        if context not in context_id_map:
            context_id_map[context] = 'Context_' + str(count)
            count += 1

    print(len(context_id_map))
    return context_id_map


def main(args):

    data_dir = args.data_dir
    opt_args = {}

    if args.tempo_filter:
        opt_args['include_types'] = args.include_types
    if args.skip_other:
        opt_args['other_label'] = None
    if args.nr > 0:
        opt_args['neg_rate'] = args.nr
        opt_args['eval_list'] = args.eval_list

    opt_args['data_type'] = args.data_type
    opt_args['pred_window'] = args.pred_win
    opt_args['shuffle_all'] = args.shuffle_all
    opt_args['backward_sample'] = args.backward_sample
    opt_args['joint'] = args.joint
    log.info(f"Reading datasets to memory from {data_dir}")

    # buffering data in memory --> it could cause OOM                                                                          
    train_data = list(read_relations(Path(data_dir), 'train', **opt_args))
        
    dev_data = []

    if args.data_type in ["red", "tbd", "matres"]:
        dev_data = list(read_relations(Path(data_dir), 'dev', **opt_args))

    # combine train and dev                                                                                                    
    # either CV or split by specific file name later                                                                           
    train_data += dev_data

    test_data = list(read_relations(Path(data_dir), 'test', **opt_args))

    train_data_u = []
    # load unlabeled data                                                                                                      
    if args.loss_u:
        opt_args['data_type'] = 'te3sv'
        train_data_u = list(read_relations(Path(args.data_dir_u), 'train', **opt_args))

    print("Total Unlabeled Data: %s" % (len(train_data_u)))

    if args.data_type == "red":
        label_map = red_label_map
    elif args.data_type == "matres":
        label_map = matres_label_map
    elif args.data_type == "tbd":
        label_map = tbd_label_map
    else:
        label_map = new_label_map

    all_labels = list(OrderedDict.fromkeys(label_map.values()))

    if args.joint:
        label_map_c = causal_label_map
        # in order to perserve order of unique keys                                                                   
        all_labels_c =  list(OrderedDict.fromkeys(label_map_c.values()))
        args._label_to_id_c = OrderedDict([(all_labels_c[l],l) for l in range(len(all_labels_c))])
        args._id_to_label_c = OrderedDict([(l,all_labels_c[l]) for l in range(len(all_labels_c))])
        print(args._label_to_id_c)

    args._label_to_id = OrderedDict([(all_labels[l],l) for l in range(len(all_labels))])
    args._id_to_label = OrderedDict([(l,all_labels[l]) for l in range(len(all_labels))])
    print(args._label_to_id)
    
    args.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    args.bert_model = BertModel.from_pretrained('bert-base-uncased')

    if args.bert_fts:
        #args.mapping = compute_mapping(train_data_u, args)
        #np.save(args.save_data_dir + '/bert_reduction_mapping.npy', args.mapping.data.numpy())
        
        # load pre-computed mapping
        args.mapping = torch.FloatTensor(np.load('/nas/home/rujunhan/tcr_output/bert_reduction_mapping.npy'))
        print(args.mapping.size())

    # featurize train + dev data
    with mp.Pool(processes=2) as pool:
        data = pool.map(partial(parallel, args = args), train_data)
        print(len(data))

    #context_map_id = compute_context_id(data)
    #np.save(args.save_data_dir + '/context_map_id.npy', context_map_id)
    #kill

    # featurize unlabeled data                                                                                        
    data_u = []
    if args.loss_u:
        with mp.Pool(processes=2) as pool:
            data_u = pool.map(partial(parallel, args = args), train_data_u)
        print(len(data_u))
    
    # doc splits
    if args.data_type in ['new']:
        train_docs, dev_docs = train_test_split(args.train_docs, test_size=0.2, random_state=args.seed)
        # Both RED and TBDense data have give splits on train/dev/test                                              
    else:
        train_docs = args.train_docs
        dev_docs = args.dev_docs

    if not os.path.isdir(args.save_data_dir):
        os.mkdir(args.save_data_dir)

    if 'all' in args.split:
        
        # featurize test data -- only once for 'all' since test data should always be fixed 
        with mp.Pool(processes=2) as pool:
            test_data = pool.map(partial(parallel, args = args), test_data)

        with open(args.save_data_dir + '/test.pickle', 'wb') as handle:
            pickle.dump(test_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.close()

        split_and_save(train_docs, dev_docs, data, data_u, args.seed, args.save_data_dir)

        # quick trick to reduce number of tokens in GloVe
        reduce_vocab(data + test_data, args.save_data_dir, args.w2i, args.glove)

    elif 'shuffle' in args.split:
        # shuffle by context_id for cv
        context_id_map = np.load(args.data_dir + 'all/context_map_id.npy').item()
        
        #random.Random(args.seed).shuffle(data) 

        kf = KFold(n_splits=args.cv_folds, random_state=args.seed, shuffle=True)
        all_context_ids = list(context_id_map.values())
        all_splits = [{'train':trn, 'dev':dev} for trn,dev in kf.split(all_context_ids)]

        assert len(all_splits) == args.cv_folds

        for k, x in enumerate(all_splits):

            save_data_dir = "%s/fold%s" % (args.save_data_dir, k)

            train_context = [all_context_ids[i] for i in x['train']]
            dev_context = [all_context_ids[i] for i in x['dev']]

            train_data = [x for x in data if context_id_map[tuple(x[4][0])] in train_context]
            dev_data = [x for x in data if context_id_map[tuple(x[4][0])] in dev_context]
            
            print(len(train_data), len(dev_data))
            assert len(train_data) + len(dev_data) == len(data)

            save_data_dir = "%s/fold%s" % (args.save_data_dir, k)
            
            if not os.path.isdir(save_data_dir):
                os.mkdir(save_data_dir)

            with open(save_data_dir + '/train.pickle', 'wb') as handle:
                pickle.dump(train_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.close()

            with open(save_data_dir + '/dev.pickle', 'wb') as handle:
                pickle.dump(dev_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.close()

    elif 'cv' in args.split:

        kf = KFold(n_splits=args.cv_folds, random_state=args.seed)
        
        all_docs = train_docs + dev_docs

        all_splits = [{'train':trn, 'eval':evl} for trn,evl in kf.split(all_docs)]
        assert len(all_splits) == args.cv_folds
        
        for k, x in enumerate(all_splits):

            save_data_dir = "%s/fold%s" % (args.save_data_dir, k)
            train_docs = [all_docs[i] for i in x['train']]
            dev_docs = [all_docs[i] for i in x['eval']]

            split_and_save(train_docs, dev_docs, data, data_u, args.seed, save_data_dir)

    elif 'bs' in args.split:
        
        if not os.path.isdir(args.save_data_dir):
            os.mkdir(args.save_data_dir)
        count = 0
        for seed in range(0, 400, 10):
            save_data_dir = "%s/bootstrap%s" % (args.save_data_dir, count)
            bootstrap_and_save(data, data_u, seed, save_data_dir)
            count += 1


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    # arguments for data processing                                                                                                   
    p.add_argument('-data_dir', type=str,
                   help='Path to directory having RED data. This should be output of '
                        '"ldcred.py flexnlp"')
    p.add_argument('--tempo_filter', action='store_true',
                   help='Include Causal and Temporal relations only. By default all relations are'
                        ' included. When --filter is specified, non temporal and causal relations '
                        'will be labelled as OTHER')
    p.add_argument('--skip_other', action='store_true',
                   help='when --tempo-filter is applied, the excluded types are marked as OTHER.'
                        'By enabling --skip-other, OTHER types are skipped.')
    p.add_argument('-nr', '--neg-rate', type=float, default=0.0,help='Negative sample rate.')
    p.add_argument('-include_types', type=set, default={'TLINK'})
    p.add_argument('-eval_list', type=list, default=[])
    p.add_argument('-shuffle_all', type=bool, default=False)
    p.add_argument('-backward_sample', type=bool, default=False)
    p.add_argument('-data_dir_u', type=str,
                   help='Path to directory of unlabeld data (TE3-Silver). This should be output of '
                        '"ldcred.py flexnlp"')
    p.add_argument('-pred_win', type=int, default=200)
    p.add_argument('-data_type', type=str, default="red")
    p.add_argument('-joint', type=bool, default=False)
    p.add_argument('-loss_u', type=str, default='')
    p.add_argument('-emb', type=int, default=300)
    p.add_argument('-split', type=str, default='all') # cv / all / all_bert_15fts/ cv_bert_15fts'/ 'cv_bert_30fts' / bs (bootstrap) / bs_bert
    p.add_argument('-cv_folds', type=int, default=5) # use together with split = cv
    p.add_argument('-seed', type=int, default=7)
    p.add_argument('-bert_fts', type=bool, default=False)
    p.add_argument('-ling_fts', type=bool, default=False)
    p.add_argument('-n_fts', type=int, default=15)
    args = p.parse_args()
    
    args.eval_list = []
    args.data_type = "matres"
    if args.data_type == "red":
        args.data_dir = "/nas/home/rujunhan/red_output/"
        args.train_docs = [x.strip() for x in open("%strain_docs.txt" % args.data_dir, 'r')]
        args.dev_docs = [x.strip() for x in open("%sdev_docs.txt" % args.data_dir, 'r')]
    elif args.data_type == "new":
        args.data_dir = "/nas/home/rujunhan/tcr_output/"
        args.train_docs = [x.strip() for x in open("%strain_docs.txt" % args.data_dir, 'r')]
    elif args.data_type == "matres":
        args.data_dir = "/nas/home/rujunhan/matres_output/"
        args.train_docs = [x.strip() for x in open("%strain_docs.txt" % args.data_dir, 'r')]
        args.dev_docs = [x.strip() for x in open("%sdev_docs.txt" % args.data_dir, 'r')]
    elif args.data_type == "tbd":
        args.data_dir = "/nas/home/rujunhan/tbd_output/"
        args.train_docs = [x.strip() for x in open("%strain_docs.txt" % args.data_dir, 'r')]
        args.dev_docs = [x.strip() for x in open("%sdev_docs.txt" % args.data_dir, 'r')]

    args.data_dir_u = "/nas/home/rujunhan/te3sv_output/"
    args.save_data_dir = args.data_dir + args.split

    args.nr = 0.0
    args.tempo_filter = True
    args.skip_other = True

    args.glove = read_glove("/nas/home/rujunhan/red_output/glove/glove.6B.%sd.txt" % args.emb)
    print(len(args.glove))
    vocab = np.array(['<pad>', '<unk>'] + list(args.glove.keys()))
    args.w2i = OrderedDict((vocab[i], i) for i in range(len(vocab)))

    tags = open("/nas/home/rujunhan/tcr_output/pos_tags.txt")
    pos2idx = {}
    idx = 0
    for tag in tags:
        tag = tag.strip()
        pos2idx[tag] = idx
        idx += 1
    args.pos2idx = pos2idx

    main(args)

