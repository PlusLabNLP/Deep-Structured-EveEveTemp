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
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn import Parameter
import math
import time
import copy
from torch.utils import data
from process_data_g import FlatRelation, print_annotation_stats, print_flat_relation_stats, read_relations, all_red_labels, REDEveEveRelModel, matres_label_map, tbd_label_map, new_label_map, red_label_map, causal_label_map
from baseline import  ClassificationReport, rev_map
from featureFuncs import *
from gurobi_inference import Gurobi_Inference
import multiprocessing as mp
from functools import partial
from sklearn.model_selection import KFold, ParameterGrid, train_test_split
from ldctcr import NewDoc, NewRelation, NewEntity
from ldctbd import TBDDoc, TBDRelation, TBDEntity
from temporal_evaluation import *
from nn_model import BiLSTM
 
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(123) # 123, 1, 10, 100, 200, 1000

class EventDataset(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, data_dir, data_split, glove2vocab, data_dir_rev=""):
        'Initialization'

        self.glove2vocab = glove2vocab
        with open(data_dir + data_split + '.pickle', 'rb') as handle:
            self.data = pickle.load(handle)
        handle.close()

        if data_dir_rev:
            with open(data_dir_rev + data_split + '.pickle', 'rb') as handle:
                data_rev = pickle.load(handle)
            handle.close()
            self.data += data_rev

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.data)

    def __getitem__(self, idx):
        'Generates one sample of data'

        sample = self.data[idx]
        doc_id = sample[0]
        sample_id = sample[1]
        pair = sample[2]
        label = torch.LongTensor([sample[3]])
        # use a reduced version of vocabulary and index                                                     
        sent = torch.LongTensor([self.glove2vocab[x] for x in sample[4][0]])
        pos = torch.LongTensor(sample[4][1])
        fts = torch.FloatTensor(sample[4][2])
        rev = sample[4][3]
        lidx_start_s = sample[4][4]
        lidx_end_s = sample[4][5]
        ridx_start_s = sample[4][6]
        ridx_end_s = sample[4][7]
        pred_ind = sample[4][8]

        return doc_id, sample_id, pair, label, sent, pos, fts, rev, lidx_start_s, lidx_end_s, ridx_start_s,\
 ridx_end_s, pred_ind


@dataclass()
class NNClassifier(REDEveEveRelModel):
    """                                                                                                                            
    A simple baseline model which assigns a random class label to each event-event pair                                            
    """
    label_probs: Optional[List[float]] = None
    _label_to_id_c = {}
    _id_to_label_c = {}

    def predict(self, model, data, args):
        model.eval()
        eval_batch = 1

        count = 1
        correct = 0
        
        labels, probs, probs_c, losses_t, losses_c = [], [], [], [], []
        labels_rev, probs_rev, labels_c = [], [], []
        eval_pairs = []
        # if flipped sample included, we need to determined which one to use based on max_prob              

        rev_count = 0
        rev_indx = []

        for doc_id, data_id, pairs, label, sents, poss, fts, rev, lidx_start, lidx_end, ridx_start, ridx_end, pred_ind in data:
            if rev:
                rev_count += 1

            label = label.reshape(eval_batch)
            sents = sents.reshape(-1, args.batch)
            poss = poss.reshape(-1, args.batch)

            if args.bert_fts:
                fts = fts.reshape(-1, args.n_fts)
            sents = (sents, poss, fts)

            data_id = data_id[0]
            rev = rev.tolist()[0]
            left = pairs[0][0]
            right = pairs[1][0]

            lidx_start = lidx_start.tolist()[0]
            lidx_end = lidx_end.tolist()[0]
            ridx_start = ridx_start.tolist()[0]
            ridx_end = ridx_end.tolist()[0]
            is_causal = (data_id[0] == 'C')

            _, prob = model(label, sents, lidx_start, lidx_end, ridx_start, ridx_end, pred_ind, flip=rev, causal=is_causal)
   
            # temporal case                                                                                 
            if not is_causal:
                eval_pairs.append(("%s_%s"%(doc_id[0], left), "%s_%s"%(doc_id[0], right)))
                if rev:
                    probs_rev.append(prob)
                    labels_rev.append(label)
                else:
                    probs.append(prob)
                    labels.append(label)
            # causal case                                                                                   
            else:
                predicted = (prob.data.max(1)[1]).long().view(-1)
                correct += (predicted == label.data).sum()
                probs_c.append(prob)
                labels_c.append(label)

            count += 1
            if count % 1000 == 0:
                print("finished evaluating %s samples" % count)
        
        # collect forward and backward
        all_probs = probs + probs_rev
        prob_table = torch.cat(all_probs).data.numpy()

        # initialize causal
        prob_table_c = np.zeros((0, 0))

        if len(probs_c) > 0:
            probs_c = torch.cat(probs_c, dim=0)
            prob_table_c = probs_c.data.numpy()
    
        # collect forward and backward labels
        all_labels = torch.cat(labels+labels_rev+labels_c, dim=0)
        true_labels = list(all_labels.data.numpy())

        # always put causal pairs after temporal pairs
        assert len(eval_pairs)  + len(labels_c) == len(true_labels)

        best_pred_idx, _, predictions, _ = self.global_prediction(eval_pairs, prob_table, prob_table_c, evaluate=True, true_labels = true_labels, flip=args.backward_sample)

        mask = torch.ByteTensor(best_pred_idx)

        assert mask.data.numpy().shape == prob_table.shape

        #predictions = torch.masked_select(probs, mask)
        loss = self.loss_func(best_pred_idx, all_labels.data.numpy()[:prob_table.shape[0]], torch.cat(all_probs,dim=0), args.margin)
        
        if args.backward_sample:
            # devide probs and preds by half and use
            # best_pred_idx to select best_prob
            assert len(labels)*2 == len(data)
            assert len(probs)*2 == len(data)

            # mask forward and backward
            mask_f = mask[:rev_count, :]
            mask_b = mask[rev_count:, :]

            # Prob still uses local model, but corresponds to global best
            max_probs_f = torch.masked_select(torch.cat(probs,dim=0), mask_f).reshape(-1, 1)
            max_probs_b = torch.masked_select(torch.cat(probs_rev,dim=0), mask_b).reshape(-1, 1)
            #print(max_probs_f)
            #print(max_probs_b)
            max_probs = torch.cat((max_probs_f, max_probs_b), 1)
            #print(max_probs.shape)
            
            # but label is given by gloabl -- different from pairwise model
            max_label_f = torch.LongTensor(predictions[:rev_count]).reshape(-1, 1)
            max_label_b = torch.LongTensor(predictions[rev_count:]).reshape(-1, 1)
            max_label = torch.cat((max_label_f, max_label_b), 1)

            # mask decides to take forward or backward                                                             
            mask = list(max_probs.max(1)[1].view(-1).numpy())
            
            
            def get_forward_label(idx_backward):
                label_backward = self._id_to_label[idx_backward]
                idx_forward = self._label_to_id[rev_map[label_backward]]                
                return idx_forward

            pred_labels = [max_label.data.numpy()[i, j] if j == 0 
                           else get_forward_label(max_label.data.numpy()[i, j])
                           for i,j in enumerate(mask)]

            # choose forward or backward labels                                                                    
            labels_f = torch.cat(labels, dim=0).reshape(-1, 1)
            labels_b = torch.cat(labels_rev, dim=0).reshape(-1, 1)
            max_label = torch.cat((labels_f, labels_b), 1)
            labels = [max_label.data.numpy()[i, j] if j == 0 
                      else get_forward_label(max_label.data.numpy()[i, j]) 
                      for i,j in enumerate(mask)] 

            assert len(pred_labels) == len(labels)

            return labels, pred_labels

        print("Evaluation loss: %.4f" % loss.data.numpy())
        #return predictions, loss
        return probs, predictions


    def _train(self, train_data, eval_data, emb, pos_emb, args, in_cv=False, test_data=None):
        
        #torch.manual_seed(args.seed)
        if args.model == 'rnn':                                                                                                         
            model = BiLSTM(emb, pos_emb, args)                                                                                          
        elif args.model == 'cnn':                                                                                                       
            model = CNN(emb, pos_emb, args) 

        if args.load_model == True:
            checkpoint = torch.load(args.ilp_dir + args.load_model_file)
            model.load_state_dict(checkpoint['state_dict'])
            best_eval_f1 = checkpoint['f1']
            print("Local best eval f1 is: %s" % best_eval_f1)

        best_eval_f1 = 0.0
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), 
                              lr = args.lr, momentum=args.momentum, weight_decay=args.decay)
        
        #optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
        best_model = copy.deepcopy(model)

        losses = [] 

        sents, poss, ftss, labels = [], [], [], []                                                                                                                                                 
        best_eval_loss = 100.0

        best_epoch = 1

        for epoch in range(args.epochs):
            print("Training Epoch #%s..." % (epoch + 1))
            model.train()
            train_pairs = []
            rev_indx = []
            correct = 0
            count = 1    
            loss_hist_t, loss_hist_c = [], []
            probs, probs_c, labels_c, all_labels = [], [], [], [] 
            start_time = time.time()
            for doc_id, data_id, pairs, labels, sents, poss, ftss, rev, lidx_start, lidx_end, ridx_start, ridx_end, _ in train_data:
                
                data_id = data_id[0]
                is_causal = (data_id[0] == 'C')

                if data_id[0] == 'U' and (args.skip_u or not args.loss_u):
                    continue

                sents = sents.reshape(-1, args.batch)
                poss = poss.reshape(-1, args.batch)

                if args.bert_fts:
                    ftss = ftss.reshape(-1, args.n_fts)

                sents = (sents, poss, ftss)

                data_id = data_id[0]
                rev = rev.tolist()[0]
                
                left = pairs[0][0]
                right = pairs[1][0]

                lidx_start = lidx_start.tolist()[0]
                lidx_end = lidx_end.tolist()[0]
                ridx_start = ridx_start.tolist()[0]
                ridx_end = ridx_end.tolist()[0]

                labels = labels.reshape(args.batch)
                model.zero_grad()       
                    
                out, prob = model(labels, sents, lidx_start, lidx_end, ridx_start, ridx_end, flip=rev, causal=is_causal)

                # temporal case                                                                                         
                if not is_causal:
                    train_pairs.append(("%s_%s"%(doc_id[0], left), "%s_%s"%(doc_id[0], right)))
                    probs.append(prob)
                    all_labels.append(labels)
                # causal case                                                                                        
                else:
                    predicted = (prob.data.max(1)[1]).long().view(-1)
                    correct += (predicted == labels.data).sum()
                    probs_c.append(prob)
                    labels_c.append(labels)

                count += 1 

            # perform global inference
            # true_labels = [x[0] for x in train_data]

            # we need probs as Variable to build computation graph
            probs = torch.cat(probs, dim=0)
            prob_table = probs.data.numpy()

            prob_table_c = np.zeros((0, 0))
            if len(probs_c) > 0:
                probs_c = torch.cat(probs_c, dim=0)
                prob_table_c = probs_c.data.numpy()

            all_labels = torch.cat(all_labels+labels_c, dim=0)

            print(prob_table.shape, prob_table_c.shape, len(train_pairs))

            # find max prediction based on global prediction 
            best_pred_idx, best_pred_idx_c = self.global_prediction(train_pairs, prob_table, prob_table_c, flip=args.backward_sample)
            Nt = prob_table.shape[0]
            
            # find true label prediction
            loss = self.loss_func(best_pred_idx, all_labels.data.numpy()[:Nt], probs, args.margin)
            if args.joint:
                loss_c = self.loss_func(best_pred_idx_c, all_labels.data.numpy()[Nt:], probs_c, args.margin)
                loss = loss + loss_c
            loss.backward()                                                                                                                                
            #torch.nn.utils.clip_grad_norm(model.parameters(), args.clipper)
            optimizer.step()                               
                          
            all_labels = []

            print("Train loss: %.4f" % loss.data.numpy())
            print("*"*50)
            
            ###### Evaluate at the end of each epoch ##### 
            if len(eval_data) > 0:

                if args.backward_sample:
                    eval_labels, eval_preds = self.predict(model, eval_data, args)
                else:
                    _, eval_preds = self.predict(model, eval_data, args)
                    eval_labels = [x[3].tolist()[0][0] for x in eval_data if x[1][0][0] != 'C']

                eval_f1 = self.weighted_f1(eval_preds, eval_labels)

                #ta_f1 = temporal_awareness(eval_data, [self._id_to_label[x] for x in eval_preds])
                
                if eval_f1 > best_eval_f1:
                    best_eval_f1 = eval_f1
                    best_model = copy.deepcopy(model)
                    best_epoch = epoch + 1

                print("Evaluation F1: %.4f" % eval_f1)
                print("*"*50)

        print("Final Evaluation F1: %.4f" % best_eval_f1)
        print("Final Evaluation Loss: %.4f" % best_eval_loss)
        print("*"*50)

        if len(eval_data) > 0 :
            self.model = best_model
        else:
            self.model = copy.deepcopy(model)
        
        if args.save_model:
            torch.save({'args': args,
                        'state_dict': best_model.state_dict(),
                        'f1': best_eval_f1,
                        'optimizer' : optimizer.state_dict(),
                        }, 
                       "%s/global_best_%s.pth.tar" % (args.ilp_dir, args.save_stamp))
        
        return best_eval_f1, best_epoch
    
    def loss_func(self, best_pred_idx, all_labels, probs, margin):
        
        # max global prediction scores
        mask = torch.ByteTensor(best_pred_idx)
        assert mask.size() == probs.size()
        max_scores = torch.masked_select(probs, mask)

        globalNlocal = (probs.data.max(1)[0].view(-1) != max_scores.data.view(-1)).numpy()

        # true label scores
        N = probs.size()[0]
        C = probs.size()[1]
        idx_mat = np.zeros((N, C), dtype=int)

        for n in range(N):
            idx_mat[n][all_labels[n]] = 1
        mask = torch.ByteTensor(idx_mat)
        assert mask.size() == probs.size()
        label_scores = torch.masked_select(probs, mask)
    
        '''
        for n in range(N):
            if globalNlocal[n]:
                if label_scores[n].data == max_scores[n].data:
                    print("Global Assignment = Gold Label")
                else:
                    print("Global - Gold label score:")
                    print(max_scores[n].data.numpy() - label_scores[n].data.numpy())
        '''

        ### Implement SSVM Loss here
        # distance measure
        #delta = Variable(torch.FloatTensor([0.0 if label_scores[n].data == max_scores[n].data else margin for n in range(N)]))
        delta = torch.FloatTensor([margin for n in range(N)])
        losses = []
        
        diff = delta + max_scores - label_scores
        # loss should be non-negative
        count = 0
        for n in range(N):
            if diff[n].data.numpy() <= 0.0:
                losses.append(Variable(torch.FloatTensor([0.0])))
            else:
                count += 1
                #print(diff[n], n)
                losses.append(diff[n].reshape(1,))
        #print(losses)
        #print(count)
        #kill
        losses = torch.cat(losses)
        #print(losses)
        return torch.mean(losses) 
    
    def global_prediction(self, pairs, prob_table, prob_table_c, evaluate=False, true_labels=[], flip=False):
        # input:                                                                                                                              
        # 1. pairs: doc_id + entity_id                                                                                                        
        # 2. prob_table: numpy matrix of local predictions (N * C)
        # 3. evaluate: True - print classification report
        # 4. true_label: if evaluate is true, need to true_label to evaluate model
        # output:                                                                                                                             
        # 1. if evaluate, print classification report and return best global assignment  
        # 2. else, class selection for each sample store in matrix form                   
                                                          
        N, C = prob_table.shape
        Nc, Cc = prob_table_c.shape
        
        global_model = Gurobi_Inference(pairs, prob_table, prob_table_c, self._label_to_id, self._label_to_id_c, flip)
        global_model.run()
        global_model.predict()

        best_pred_idx = np.zeros((N, C), dtype=int)
        best_pred_idx_c = np.zeros((Nc, Cc), dtype=int)

        # temporal
        for n in range(N):
            best_pred_idx[n, global_model.pred_labels[n]] = 1
        # causal
        for n in range(Nc):
            best_pred_idx_c[n, global_model.pred_labels_c[n]] = 1

        if evaluate:
            assert len(true_labels) == N + Nc
            global_model.evaluate(true_labels)
            return best_pred_idx, best_pred_idx_c, global_model.pred_labels, global_model.pred_labels_c
        else:
            return best_pred_idx, best_pred_idx_c

    def data_split(self, train_docs, eval_docs, data, nr, pairs):

        train_set, eval_set = [], []
        
        for s in data:
            if s[0] in eval_docs:
                eval_set.append((s[1], s[2], s[3]))
            elif s[1][0] in ['P', 'C']:
                train_set.append((s[1], s[2], s[3]))
            # for training set, only keep nr negative samples
            # high = total negative / total positive
            # low = 0
            elif nr > np.random.uniform(high=10):
                train_set.append((s[2], s[3]))
        
        train_pairs, eval_pairs = [], []

        for pr in pairs:
            if pr['pairs'][0].split('_')[0] in eval_docs:
                eval_pairs.append(pr)
            else:
                train_pairs.append(pr)

        assert len(train_pairs) == len(train_set)
        assert len(eval_pairs) == len(eval_set)

        return train_set, eval_set, train_pairs, eval_pairs

    def cross_validation(self, emb, pos_emb, args, pairs):
        
        param_perf = []
        for param in ParameterGrid(args.params):
            param_str = ""
            for k,v in param.items():
                exec("args.%s=%s" % (k, v))
                param_str += "%s=%s" % (k, v)
                param_str += " "

            print("Train parameters: %s" % param_str)
            print("*" * 50)

            for k,v in param.items():
                exec("args.%s=%s" % (k, v))
            
            all_splits = [x for x in range(args.n_splits)]
            
            # multi-process over only data splits due to source limitation
            with mp.Pool(processes=args.n_splits) as pool:
                res = pool.map(partial(self.parallel_cv, emb=emb, pos_emb=pos_emb, args=args), all_splits)
            print(res)
            f1s = list(zip(*res))[0]
            best_epoch = list(zip(*res))[1]
            param_perf.append((param, np.mean(f1s), np.mean(best_epoch)))
            
        params, f1, epoch = sorted(param_perf, key=lambda x: x[1], reverse=True)[0]
        print(sorted(param_perf, key=lambda x: x[1], reverse=True))
        print("Best Average F1: %s" % f1)
        print("Best Parameters Are: %s " % params)
        print("Best Epoch is: %s" % int(epoch))
        params['epochs'] = int(epoch)
        return params

    def parallel_cv(self, split, emb = np.array([]), pos_emb = [], args=None):
        params = {'batch_size': args.batch}

        if args.bert_fts:
            type_dir = "cv_bert_%sfts" % args.n_fts
        else:
            type_dir = "cv_shuffle" if args.cv_shuffle else 'cv'
        print(type_dir)

        backward_dir = ""
        if args.backward_sample:
            backward_dir = "%s/cv_backward/fold%s/" % (args.data_dir, split)
        print(backward_dir)

        train_data = EventDataset(args.data_dir + '%s/fold%s/' % (type_dir, split), "train", args.glove2vocab, backward_dir)
        train_generator = data.DataLoader(train_data, **params)

        dev_data = EventDataset(args.data_dir + '%s/fold%s/' % (type_dir, split), "dev", args.glove2vocab, backward_dir)
        dev_generator = data.DataLoader(dev_data, **params)


        return self._train(train_generator, dev_generator, emb, pos_emb, args, in_cv=True)
                  
    def train_epoch(self, train_data, dev_data, args, test_data = None):

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
            self._label_to_id_c = OrderedDict([(all_labels_c[l],l) for l in range(len(all_labels_c))])
            self._id_to_label_c = OrderedDict([(l,all_labels_c[l]) for l in range(len(all_labels_c))])
            print(self._label_to_id_c)
            print(self._id_to_label_c)

        self._label_to_id = OrderedDict([(all_labels[l],l) for l in range(len(all_labels))])
        self._id_to_label = OrderedDict([(l,all_labels[l]) for l in range(len(all_labels))])

        print(self._label_to_id)
        print(self._id_to_label)

        args.label_to_id = self._label_to_id
        
        emb = args.emb_array
        # fix random seed, the impact is minimal, but useful for perfect replication                        
        np.random.seed(args.seed)
        emb = np.vstack((np.random.uniform(0, 1, (2, emb.shape[1])), emb))

        assert emb.shape[0] == len(args.glove2vocab)

        pos_emb= np.zeros((len(args.pos2idx) + 1, len(args.pos2idx) + 1))
        for i in range(pos_emb.shape[0]):
            pos_emb[i, i] = 1.0
       
        if args.cv == True:
            best_params = self.cross_validation(emb, pos_emb, args, pairs)
            print(best_params)
            ### retrain on the best parameters
            print("To refit...")
            args.refit = True

            for k,v in best_params.items():
                exec("args.%s=%s" % (k, v))

        # refit on all training data
        if args.refit_all:
            train_docs = train_docs + dev_docs
            dev_docs = []
            print("Refit on all %s docs." % len(train_docs))

        #train_data, dev_data, train_pairs, dev_pairs = self.data_split(train_docs, dev_docs, data, args.nr, pairs)
        #args.save_model = True
        best_f1, _ = self._train(train_data, dev_data, emb, pos_emb, args)
        print("Final Dev F1: %.4f" % best_f1)
        return -1.0

    def weighted_f1(self, pred_labels, true_labels):
        def safe_division(numr, denr, on_err=0.0):
            return on_err if denr == 0.0 else numr / denr

        assert len(pred_labels) == len(true_labels)

        weighted_f1_scores = {}
        if 'NONE' in self._label_to_id.keys():
            num_tests = len([x for x in true_labels if x != self._label_to_id['NONE']])
        else:
            num_tests = len([x for x in true_labels])

        print("Total samples to eval: %s" % num_tests)
        total_true = Counter(true_labels)
        total_pred = Counter(pred_labels)
        labels = list(self._id_to_label.keys())
        
        n_correct = 0
        n_true = 0
        n_pred = 0

        exclude_labels = ['NONE', 'VAGUE'] if len(self._label_to_id) == 4 else ['NONE']
        for label in labels:
            if self._id_to_label[label] not in exclude_labels:
                
                true_count = total_true.get(label, 0)
                pred_count = total_pred.get(label, 0)
                n_true += true_count
                n_pred += pred_count

                correct_count = 0
                for l in range(len(pred_labels)):
                    if pred_labels[l] == true_labels[l] and pred_labels[l] == label:
                        correct_count += 1
                n_correct += correct_count

        print(n_correct, n_pred, n_true)
        precision = safe_division(n_correct, n_pred)                                                       
        recall = safe_division(n_correct, n_true)
        f1_score = safe_division(2 * precision * recall, precision + recall)
        return f1_score
        #return sum(list(weighted_f1_scores.values()))

@dataclass
class REDEvaluator:
    model: REDEveEveRelModel
    def evaluate(self, test_data, args):
        # load test data first since it needs to be executed twice in this function                    
                  
        # rearrange pairs order for global inference (All T + All C)                                                                      # if causal in empty, the order doesn't change         
        
        if args.backward_sample:
            true_labels, pred_labels = self.model.predict(self.model.model, test_data, args)
        else:
            _, pred_labels = self.model.predict(self.model.model, test_data, args)
            
            true_labels = [x[3].tolist()[0][0] for x in test_data if x[1][0][0] != 'C']
            print(len(true_labels), len(pred_labels))

        pred_labels = [self.model._id_to_label[x] for x in pred_labels]
        true_labels = [self.model._id_to_label[x] for x in true_labels]

        #print(pred_labels)
        #print(true_labels)
        assert len(pred_labels) == len(true_labels)

        if args.data_type in ['tbd']:
            test_data = [(x[0][0], x[2][0][0], x[2][1][0], true_labels[k])
                         for k, x in enumerate(test_data) if k < len(true_labels)]
            temporal_awareness(test_data, pred_labels, args.data_type, args.eval_with_timex)

        #print(mcnemar_test(pred_labels, true_labels))
        
        # pred_glob_labels
        ids = [x[1][0] for x in test_data if x[1][0][0] != 'C']

        #self.for_analysis(ids, true_labels, pred_labels, test_data, args.ilp_dir+'matres_global_all.tsv')
        
        return ClassificationReport(self.model.name, true_labels, pred_labels)


    #def mcnemar_test(pred_labels, true_labels):
    #    pred_and_true = np.sum([_ for k,v in enumerate(pred_labels) if v == true_labels[k]])
    #    pred_not_true = np.sume([])
    #    return pvalue


    def for_analysis(self, ids, golds, preds, test_data, outfile):
        with open(outfile, 'w') as file:
            file.write('\t'.join(['doc_id', 'pair_id', 'label', 'pred', 'left_text', 'right_text', 'context']))
            file.write('\n')
            i = 0
            i2w = np.load('i2w.npy').item()
            v2g = np.load('v2g.npy').item()
            for ex in test_data:
                #print(ex[0])                                                                                                 
                #print(ex[2])                                                                                                 
                left_s = ex[8].tolist()[0]
                left_e = ex[9].tolist()[0]
                right_s = ex[10].tolist()[0]
                right_e = ex[11].tolist()[0]
                #print(left_s, left_e, right_s, right_e)                                                                      
                #print(len(i2w))                                                                                             
                #print(len(v2g))                                                                                              
                context = [i2w[v2g[x]] for x in ex[4][0].tolist()]
                #print(context)                                                                                               
                left_text = context[left_s : left_e + 1][0]
                right_text  = context[right_s : right_e + 1][0]
                #print(left_text, right_text)                                                                                 
                #kill                                                                                                         
                context = ' '.join(context)
                file.write('\t'.join([ex[0][0],
                                      ids[i],
                                      golds[i],
                                      preds[i],
                                      left_text,
                                      right_text,
                                      context]))
                file.write('\n')
                i += 1
                print(i)
        file.close()
        return
    '''
    def for_analysis(self, ids, golds, preds, outfile):

        #i2w = {v:k for k,v in w2i.items()}

        with open(outfile, 'w') as file:
            file.write('\t'.join(['pair_id', 'label', 'pred']))
            file.write('\n')
            for i in range(len(preds)):
                file.write('\t'.join([ids[i], golds[i], preds[i]]))
                file.write('\n')
        file.close()
        return
    '''

def temporal_awareness(data, pred_labels, data_type, with_timex=False):
    
    gold_rels = {}
    
    for i, ex in enumerate(data):
        # Do not evaluate on VAGUE class for MATRES dataset
        if data_type == 'matres' and ex.rel_type == 'VAGUE':
            continue
        #if dev:
        if ex[0] in gold_rels:
            gold_rels[ex[0]].append((i, ex[1], ex[2], ex[3]))
        else:
            gold_rels[ex[0]] = [(i, ex[1], ex[2], ex[3])]

    idx2docs = {}
    for k, vs in gold_rels.items():
        for v in vs:
            idx2docs[v[0]] = (k, v[1], v[2], v[3])
    
    # for debug
    #for k,v in gold_rels.items():
    #    print(k)
    #     print(len(v))
    ### append ET and TT pairs
    
    if data_type == 'tbd' and with_timex:
        print("TBDense Gold")
        with open("/nas/home/rujunhan/data/TBDense/caevo_test_ettt.pkl", "rb") as fl:
            gold = pickle.load(fl)
            for k, v in gold.items():
                print(k, len(v))
                #gold_rels[k] = [(0, kk[0], kk[1], vv) for kk,vv in v.items()]
                gold_rels[k].extend([(0, kk[0], kk[1], vv) for kk,vv in v.items()])
                print(len(gold_rels[k]))


    pred_rels = {}
   
    for i, pl in enumerate(pred_labels):
        try:
            if idx2docs[i][0] in pred_rels:
                pred_rels[idx2docs[i][0]].append((idx2docs[i][1], idx2docs[i][2], pl))
            else:
                pred_rels[idx2docs[i][0]] = [(idx2docs[i][1], idx2docs[i][2], pl)]
        except:
            # VAGUE pairs in matres, excluded
            continue

    
    #for k,v in pred_rels.items():
    #    print(k, len(v))
    #    print(v)

    if data_type == 'tbd' and with_timex:
        ### append ET and TT pairs 
        print("CAEVO Predictions")
        with open("/nas/home/rujunhan/CAEVO/caevo_test_ettt.pkl", "rb") as fl:
            pred = pickle.load(fl)
            for k, v in pred.items():
                #pred_rels[k] = [(kk[0], kk[1], vv) for kk,vv in v.items()]
                pred_rels[k].extend([(kk[0], kk[1], vv) for kk,vv in v.items()])
    
    #for k,v in pred_rels.items():
    #    print(k, len(v))

    return evaluate_all(gold_rels, pred_rels)

def main(args):

    data_dir = args.data_dir
    opt_args = {}

    params = {'batch_size': args.batch,
              'shuffle': False}

    if args.bert_fts:
        type_dir = "all_bert_%sfts/" % args.n_fts
    else:
        type_dir = "all/"

    data_dir_back = ""
    if args.backward_sample:
        data_dir_back = args.data_dir + "all_backward/"

    train_data = EventDataset(args.data_dir + type_dir, "train", args.glove2vocab, data_dir_back)
    train_generator = data.DataLoader(train_data, **params)

    dev_data = EventDataset(args.data_dir + type_dir, "dev", args.glove2vocab, data_dir_back)
    dev_generator = data.DataLoader(dev_data, **params)
    
    test_data = EventDataset(args.data_dir + type_dir, "test", args.glove2vocab, data_dir_back)
    test_generator = data.DataLoader(test_data, **params)

    print(len(train_data), len(dev_data), len(test_data))
    
    models = [NNClassifier()]
    for model in models:
        print(f"\n======={model.name}=====")
        print(f"======={args.model}=====\n")

        if args.bootstrap:
            model.train_epoch(train_generator, dev_generator, args, test_data = test_generator)
            print("Finished Bootstrap Testing")
        else:
            model.train_epoch(train_generator, dev_generator, args)
            evaluator = REDEvaluator(model)
            print("Testing Data: %s" % args.data_type)
            #if args.data_type in ["red", "tbd"]:
            #    print(evaluator.evaluate(dev_data, args))
            print(evaluator.evaluate(test_generator, args))
    
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
    
    p.add_argument('-nr', '--neg-rate', type=float, default=0.,
                   help='Negative sample rate.')

    p.add_argument('-include_types', type=set, default={'TLINK'})
    p.add_argument('-eval_list', type=list, default=[])
    p.add_argument('-shuffle_all', type=bool, default=False)
    # select model
    p.add_argument('-model', type=str, default='rnn')

    # arguments for RNN model
    p.add_argument('-emb', type=int, default=300)
    p.add_argument('-hid', type=int, default=50)
    p.add_argument('-num_layers', type=int, default=1)
    p.add_argument('-batch', type=int, default=1)
    p.add_argument('-data_type', type=str, default="red")
    p.add_argument('-epochs', type=int, default=3)
    p.add_argument('--lr', type=float, default=0.01) # 0.01 for TBD, 0.1 for MATRES, 0.05 for TCR
    p.add_argument('--decay', type=float, default=0.4) # 0.9 for TBD, 0.4 for MATRES, 0.7 for TCR
    p.add_argument('--momentum', type=float, default=0.9) # 0.9 for TBD, 0.9 for MATRES, 0.7 for TCR
    p.add_argument('-num_classes', type=int, default=2) # get updated in main()
    p.add_argument('-dropout', type=float, default=0.4)
    p.add_argument('-ngbrs', type=int, default = 15)                                   
    p.add_argument('-pos2idx', type=dict, default = {})
    p.add_argument('-emb_array', type=np.array)
    p.add_argument('-glove2vocab', type=OrderedDict)
    p.add_argument('-cuda', type=bool, default=False)
    p.add_argument('-refit_all', type=bool, default=False)
    p.add_argument('-params', type=dict, default={})
    p.add_argument('-n_splits', type=int, default=3)
    p.add_argument('-pred_win', type=int, default=200)
    p.add_argument('-n_fts', type=int, default=15)
    p.add_argument('-unlabeled_weight', type=float, default=0.1)
    p.add_argument('-eval_with_timex', type=bool, default=True)
    # arguments for CNN model
    p.add_argument('-stride', type=int, default = 1)
    p.add_argument('-kernel', type=int, default = 5)
    p.add_argument('-train_docs', type=list, default=[])
    p.add_argument('-dev_docs', type=list, default=[])
    p.add_argument('-cv', type=bool, default=False)
    p.add_argument('-cv_shuffle', type=bool, default=False)
    p.add_argument('-attention', type=bool, default=False)
    p.add_argument('-backward_sample', type=bool, default=True) # TODO: check this
    p.add_argument('-save_model', type=bool, default=True)
    p.add_argument('--margin', type=float, default=0.3) # TODO: check this
    p.add_argument('-ilp_dir', type=str, default="../ILP/")
    p.add_argument('-load_model', type=bool, default=True)
    p.add_argument('-load_model_file', type=str, default= '0226_tbd_local_50_0.4.pth.tar')
                   #'matres_1124_hid30_dropout40.pth.tar')# 'tbd_1121.pth.tar')
    p.add_argument('-save_stamp', type=str, default='0302_tbd_global_best.pth.tar')
    p.add_argument('-joint', type=bool, default=False)
    p.add_argument('-num_causal', type=int, default=2)
    p.add_argument('-loss_u', type=str, default='')
    p.add_argument('-skip_u', type=bool, default=True)
    p.add_argument('-bert_fts', type=bool, default=False)
    # bootstrap options                                                                                     
    p.add_argument('-bootstrap', type=bool, default=False)
    p.add_argument('-bs_list', type=list, default=list(range(0, 5)))
    p.add_argument('-seed', type=int, default=9) # 9, 1, 10, 100, 200, 1000
    p.add_argument('-use_grammar', type=bool, default=False)
    args = p.parse_args()
    
    #args.eval_list = ['train', 'dev', 'test']
    
    args.eval_list = []
    #args.data_type = "tbd"
    if args.data_type == "red":
        #args.data_dir = "/nas/home/rujunhan/red_output/"
        args.data_dir = "../output_data/red_output/"
        args.train_docs = [x.strip() for x in open("%strain_docs.txt" % args.data_dir, 'r')]
        args.dev_docs = [x.strip() for x in open("%sdev_docs.txt" % args.data_dir, 'r')]
    elif args.data_type == "new":
        #args.data_dir = "/nas/home/rujunhan/tcr_output/"
        args.data_dir = "../output_data/tcr_output/"
        args.train_docs = [x.strip() for x in open("%strain_docs.txt" % args.data_dir, 'r')] 
    elif args.data_type == "matres":
        #args.data_dir = "/nas/home/rujunhan/matres_output/"
        args.data_dir = "../output_data/matres_output/"
        args.train_docs = [x.strip() for x in open("%strain_docs.txt" % args.data_dir, 'r')]
        args.dev_docs = [x.strip() for x in open("%sdev_docs.txt" % args.data_dir, 'r')]
    elif args.data_type == "tbd":
        #args.data_dir = "/nas/home/rujunhan/tbd_output/"
        args.data_dir = "../output_data/tbd_output/"
        args.train_docs = [x.strip() for x in open("%strain_docs.txt" % args.data_dir, 'r')]
        args.dev_docs = [x.strip() for x in open("%sdev_docs.txt" % args.data_dir, 'r')]

    # create pos_tag and vocabulary dictionaries
    # make sure raw data files are stored in the same directory as train/dev/test data

    tags = open("../output_data/tcr_output/pos_tags.txt")
    pos2idx = {}
    idx = 0
    for tag in tags:
        tag = tag.strip()
        pos2idx[tag] = idx
        idx += 1
    args.pos2idx = pos2idx
    args.cuda = False
    
    args.pred_win = args.ngbrs * 2
    args.emb_array = np.load(args.data_dir + 'all' + '/emb_reduced.npy')
    args.glove2vocab = np.load(args.data_dir + 'all' + '/glove2vocab.npy').item()

    print(args.emb_array.shape)
    print(len(args.glove2vocab))

    args.nr = 0.0
    args.tempo_filter = True
    args.skip_other = True
    
    if not args.cv:
        print("learning_rate: %s; momentum: %s; decay: %s; margin: %s" % (args.lr, args.momentum, args.decay, args.margin))

    #args.params = {'lr': [0.01],  'momentum':[0.9], 'decay':[0.1, 0.5, 0.9]}
    args.params = {'lr': [args.lr],  'momentum':[args.momentum], 'decay':[args.decay]}

    main(args)
