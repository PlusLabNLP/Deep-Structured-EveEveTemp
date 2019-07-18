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
from random import shuffle
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn import Parameter
from torch.utils import data
import math
import time
import copy
from baseline import FlatRelation, print_annotation_stats, print_flat_relation_stats, read_relations, all_red_labels, REDEveEveRelModel, ClassificationReport, matres_label_map, tbd_label_map, new_label_map, red_label_map, rev_map, causal_label_map
from featureFuncs import *
from nn_models_oneSeq_cv import BiLSTM
import multiprocessing as mp
from functools import partial
from sklearn.model_selection import KFold, ParameterGrid, train_test_split
from ldctcr import NewDoc, NewRelation, NewEntity
from ldctbd import TBDDoc, TBDRelation, TBDEntity
from ldcte3sv import TESVDoc, TESVRelation, TESVEntity
from global_inference import temporal_awareness
from temporal_evaluation import *
#from featurize_data import EventDataset

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(123)

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

        return doc_id, sample_id, pair, label, sent, pos, fts, rev, lidx_start_s, lidx_end_s, ridx_start_s, ridx_end_s, pred_ind
def _l2_normalize(d):

    d /= (torch.norm(d, dim=2, keepdim=True) + 1e-16)

    return d

@dataclass()
class NNClassifier(REDEveEveRelModel):
    """                                                                                                                            
    A simple baseline model which assigns a random class label to each event-event pair                                            
    """
    label_probs: Optional[List[float]] = None

    def predict(self, model, data, args):
        # is using flip option; make sure #backard == #forward
        # and their order shall be the same

        model.eval()
        criterion = nn.CrossEntropyLoss()                                                                                                         
        eval_batch = 1

        count = 1
        correct = 0
        labels, probs, losses_t, losses_c = [], [], [], []
        
        # if flipped sample included, we need to determined which one to use based on max_prob
        if args.backward_sample:
            labels_rev, probs_rev = [], []
        rev_count = 0
        for _, data_id, _, label, sents, poss, fts, rev, lidx_start, lidx_end, ridx_start, ridx_end, pred_ind in data:

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
            lidx_start = lidx_start.tolist()[0]
            lidx_end = lidx_end.tolist()[0]
            ridx_start = ridx_start.tolist()[0]
            ridx_end = ridx_end.tolist()[0]
            is_causal = (data_id[0] == 'C')
            
            out, prob = model(label, sents, lidx_start, lidx_end, ridx_start, ridx_end, pred_ind, flip=rev, causal=is_causal)
            out = out.view(eval_batch, -1)
                
            loss = criterion(out, label)
            
            # temporal case
            if not is_causal:
                losses_t.append(loss.data)
                if rev:
                    probs_rev.append(prob)
                    labels_rev.append(label.data)
                else:
                    probs.append(prob)
                    labels.append(label.data)
            # causal case
            else:
                predicted = (prob.data.max(1)[1]).long().view(-1)
                correct += (predicted == label.data).sum() 
                losses_c.append(loss.data)
                
            count += 1
            if count % 1000 == 0:
                print("finished evaluating %s samples" % count)
        
        print(rev_count)
        if args.backward_sample:
            assert len(labels)*2 == len(data)
            assert len(probs)*2 == len(data)

            max_probs_f = (torch.cat(probs,dim=0).data.max(1)[0]).reshape(-1, 1)
            max_probs_b = (torch.cat(probs_rev,dim=0).data.max(1)[0]).reshape(-1, 1)
            max_probs = torch.cat((max_probs_f, max_probs_b), 1)

            max_label_f = (torch.cat(probs,dim=0).data.max(1)[1]).reshape(-1, 1)
            max_label_b = (torch.cat(probs_rev,dim=0).data.max(1)[1]).reshape(-1, 1)
            max_label = torch.cat((max_label_f, max_label_b), 1)
            

            def get_forward_label(idx_backward):
                label_backward = self._id_to_label[idx_backward]
                idx_forward = self._label_to_id[rev_map[label_backward]]
                return idx_forward

            # mask decides to take forward or backward
            mask = list(max_probs.max(1)[1].view(-1).numpy())
            pred_labels = torch.LongTensor([max_label[i, j] if j == 0
                                            else get_forward_label(max_label.data.numpy()[i, j])
                                            for i,j in enumerate(mask)])
            
            #print(pred_labels)
            
            # choose forward or backward labels
            labels_f = torch.cat(labels,dim=0).reshape(-1, 1)
            labels_b = torch.cat(labels_rev,dim=0).reshape(-1, 1)
            max_label = torch.cat((labels_f, labels_b), 1)
            labels = torch.LongTensor([max_label[i, j] if j == 0
                                       else get_forward_label(max_label.data.numpy()[i, j]) 
                                       for i,j in enumerate(mask)])
                
            
            return pred_labels, np.mean(losses_t), labels

        else:
            assert len(labels) + len(losses_c) == len(data)
            assert len(probs) + len(losses_c) == len(data)
            
            probs = torch.cat(probs,dim=0)
            labels = torch.cat(labels,dim=0)
                
        print("Evaluation temporal loss: %.4f" % np.mean(losses_t))
        if args.joint:
            print("Evaluation causal loss: %.4f; accuracy: %.4f" % (np.mean(losses_c), float(correct) / float(len(losses_c))))
        return probs.data, np.mean(losses_t), labels


    def _train(self, train_data, eval_data, emb, pos_emb, args, in_cv=False, test_data=None):
        # since we are training with single sample in each iteration,
        # we don't need to distinguish causal and temporal labels

        if args.model == 'rnn':                                                                                                         
            model = BiLSTM(emb, pos_emb, args)                                                                                          
        elif args.model == 'cnn':                                                                                                       
            model = CNN(emb, pos_emb, args)    
                          
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

        criterion = nn.CrossEntropyLoss()                                                                                                                                                
        losses = [] 

        sents, poss, ftss, labels = [], [], [], []                                                                                                                                                 
        if args.load_model == True:
            checkpoint = torch.load(args.ilp_dir + args.load_model_file)
            model.load_state_dict(checkpoint['state_dict'])
            epoch = checkpoint['epoch']
            best_eval_f1 = checkpoint['f1']
            print("Local best eval f1 is: %s" % best_eval_f1)

        best_eval_f1 = 0.0 
        best_epoch = 0

        for epoch in range(args.epochs):
            print("Training Epoch #%s..." % (epoch + 1))
            model.train()
            correct = 0
            count = 1
            loss_hist_t, loss_hist_u, loss_hist_c = [], [], []

            start_time = time.time()
            for _, data_id, _, labels, sents, poss, ftss, rev, lidx_start, lidx_end, ridx_start, ridx_end, _ in train_data:
                
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
                lidx_start = lidx_start.tolist()[0]
                lidx_end = lidx_end.tolist()[0]
                ridx_start = ridx_start.tolist()[0]
                ridx_end = ridx_end.tolist()[0]
            
                labels = labels.reshape(args.batch)

                model.zero_grad()

                # unsupervised loss - vat
                if args.loss_u == 'vat':# and data_id[0] == 'U':
                    loss_u = self.vat_loss(model, labels, sents, lidx_start, lidx_end, ridx_start, ridx_end, causal=is_causal, 
                                           xi=args.xi, eps=args.eps)

                model.zero_grad()
                out, prob = model(labels, sents, lidx_start, lidx_end, ridx_start, ridx_end, flip=rev, causal=is_causal)

                # unsupervised loss - entropy minimization
                if args.loss_u == 'entropy':
                    loss_u = self.entropy_loss(prob)

                # add supervised loss                                                                                            
                if data_id[0] in ['L', 'C']:
                    out = out.view(labels.size()[0], -1)
                    loss_s = criterion(out, labels)
                    
                    if args.loss_u:
                        loss = loss_s + args.unlabeled_weight * loss_u
                    else:
                        loss = loss_s
                    
                else:
                    loss = args.unlabeled_weight * loss_u

                loss.backward()                                                                                
                #torch.nn.utils.clip_grad_norm(model.parameters(), args.clipper)                                                                                                       
                optimizer.step()                                                                                                                                   
                if data_id[0] != 'U':
                    predicted = (prob.data.max(1)[1]).long().view(-1)
                    
                if is_causal:
                    correct += (predicted == labels.data).sum()                                                                                                                                                  
                if args.batch == 1:             
                    if is_causal:
                        loss_hist_c.append(loss_s.data.numpy())
                    else:
                        if data_id[0] == 'L':
                            loss_hist_t.append(loss_s.data.numpy())
                        if args.loss_u:# and data_id[0] == 'U':
                            loss_hist_u.append(loss_u.data.numpy())
                else:                       
                    if data_id[0] == 'L':
                        loss_hist_t.extend(loss_s.data.numpy().tolist())                                                                
                    if args.loss_u:# and data_id[0] == 'U':
                        loss_hist_u.extend(loss_u.data.numpy().tolist())                                                                                  
                if count % 1000 == 0:                                                                                                       
                    print("trained %s samples" % count)
                    print("Temporal loss is %.4f" % np.mean(loss_hist_t))
                    print("Unsupervised loss is %.4f" % np.mean(loss_hist_u))
                    if args.joint:
                        print("Causal loss is %.4f; accuracy is %.4f" % (np.mean(loss_hist_c), float(correct) / float(len(loss_hist_c)))) 
                    print("%.4f seconds elapsed" % (time.time() - start_time))
                count += 1

            # Evaluate at the end of each epoch                                 
                                                                                 
            print("*"*50)
            print("%s samples to evaluate..." % len(eval_data))
            # select model only based on temporal F1 if refit on train / eval split
            # if doing final fitting based on train and epoch from CV, then only use all train data
            if len(eval_data) > 0:
                # 0:doc_id, 1:ex.id, 2:(ex.left.id, ex.right.id), 3:label_id, 4:features 

                eval_preds, eval_loss, eval_labels = self.predict(model, eval_data, args)
                if args.backward_sample:
                    pred_labels = eval_preds
                else:
                    pred_labels = eval_preds.max(1)[1].long().view(-1)
            
                assert eval_labels.size() == pred_labels.size() 
                eval_correct = (pred_labels == eval_labels).sum()
                eval_acc =  float(eval_correct) / float(len(eval_labels))

                pred_labels = list(pred_labels.numpy())
                eval_labels = list(eval_labels.numpy())

                if args.data_type == '':
                    pred_labels = [self._id_to_label[x] for x in pred_labels]
                    # only pass in doc_id, left_id, right_id, and rel_type
                    eval_f1 = temporal_awareness([(x[0][0], x[2][0][0], x[2][1][0], 
                                                   self._id_to_label[x[3].data.numpy()[0][0]]) 
                                                  for x in eval_data], pred_labels, args.data_type)
                else:
                    eval_f1 = self.weighted_f1(pred_labels, eval_labels)
                
                if eval_f1 >= best_eval_f1:
                    best_eval_f1 = eval_f1
                    # do not copy intermedite model in CV to save memory
                    if not in_cv:
                        self.model = copy.deepcopy(model)
                    best_epoch = epoch

                print("Evaluation loss: %.4f; Evaluation F1: %.4f" % (eval_loss, eval_f1))
                print("*"*50)

        print("Final Evaluation F1: %.4f" % best_eval_f1)
        print("*"*50)

        # for refit all
        if len(eval_data) == 0 or args.load_model == True:
            self.model = copy.deepcopy(model)
            best_epoch = epoch

        if args.save_model == True:
            torch.save({'epoch': epoch,
                        'args': args,
                        'state_dict': self.model.state_dict(),
                        'f1': best_eval_f1,
                        'optimizer' : optimizer.state_dict()
                    }, "%s%s.pth.tar" % (args.ilp_dir, args.save_stamp))
        

        # if bootstrap testing, we want this function to return f1 score on test set
        if args.bootstrap:
            test_preds, test_loss, test_labels = self.predict(model, test_data, args)
            pred_labels = test_preds.max(1)[1].long().view(-1)

            assert test_labels.size() == pred_labels.size()

            pred_labels = list(pred_labels.numpy())
            test_labels = list(test_labels.numpy())

            if args.data_type == '':
                pred_labels = [self._id_to_label[x] for x in pred_labels]
                # only pass in doc_id, left_id, right_id, and rel_type                                                          
                test_f1 = temporal_awareness([(x[0], x[2][0], x[2][1], self._id_to_label[x[3]])
                                              for x in test_data], pred_labels, args.data_type, dev=True)
            else:
                test_f1 = self.weighted_f1(pred_labels, test_labels)
        
            return test_f1

        return best_eval_f1, best_epoch

    def entropy_loss(self, prob):
        B = prob.size()[0]
        N = prob.size()[1]
        
        log_prob = torch.log(prob).reshape(B, N, -1)
        prob = prob.reshape(B, -1, N)

        # Need to modify this with batch computation
        return -prob.bmm(log_prob).reshape(1, -1)


    def vat_loss(self, model, label, sent, lidx_start, lidx_end, ridx_start, ridx_end, 
                 pred_inds=[], flip = False, causal =False, xi=1e-6, eps=2.5, num_iters=1):
        # x is sent input tokens
        # so need to lookup in emb table

        # calculate actual prediction
        with torch.no_grad():
            _, y_pred = model(label, sent, lidx_start, lidx_end, ridx_start, ridx_end, pred_inds, flip, causal)

        x = model.emb(sent[0])

        d = torch.Tensor(x.size()).normal_()

        for i in range(num_iters):
            # d has to be a unit vector, i.e. normalized
            d = xi *_l2_normalize(d)
            d = Variable(d, requires_grad=True)
            y_hat, _ = model(label, (x + d, sent[1], sent[2]), lidx_start, lidx_end, 
                             ridx_start, ridx_end, pred_inds, flip, causal, vat=True)
            
            logp_hat = F.log_softmax(y_hat, dim=1)
            delta_kl = F.kl_div(logp_hat, y_pred)
            delta_kl.backward()

        d = d.grad.data.clone()
        model.zero_grad()
        
        d = _l2_normalize(d)
        d = Variable(d)
        r_adv = eps *d
        # compute lds
        y_hat, _ = model(label, (x + r_adv.detach(), sent[1], sent[2]), lidx_start, lidx_end, 
                         ridx_start, ridx_end, pred_inds, flip, causal, vat=True)
        logp_hat = F.log_softmax(y_hat, dim=1)
        delta_kl = F.kl_div(logp_hat, y_pred)
        return delta_kl
        
    
    def data_split(self, train_docs, eval_docs, data, nr):

        train_set = []
        eval_set = []
        
        for s in data:
            if s[0] in eval_docs:
                # 0:doc_id, 1:ex.id, 2:(ex.left.id, ex.right.id), 3:label_id, 4:features
                eval_set.append((s[0], s[1], s[2], s[3], s[4]))
            elif s[1][0] in ['L', 'C']:
                train_set.append((s[1], s[3], s[4]))
            # for training set, only keep nr negative samples
            # high = total negative / total positive
            # low = 0
            elif nr > np.random.uniform(high=10):
                train_set.append((s[1], s[3], s[4]))
        
        return train_set, eval_set

    def cross_validation(self, emb, pos_emb, args):

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
        print("Best Epoch is: %s" % epoch)
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

        #return 0, 0
        return self._train(train_generator, dev_generator, emb, pos_emb, args, in_cv=True)
 
    def parallel_bootstrap(self, bs_n, emb = np.array([]), pos_emb = [], args=None, test_data=None):

        params = {'batch_size': args.batch,
                  'shuffle': False}

        if args.bert_fts:
            type_dir = "bs_bert_%sfts" % args.n_fts
        else:
            type_dir = "bs"
            type_dir = "cv"

        #train_data = EventDataset(args.data_dir + '%s/bootstrap%s/' % (type_dir, bs_n), "train", args.glove2vocab,)
        train_data = EventDataset(args.data_dir + '%s/fold%s/' % (type_dir, bs_n), "train", args.glove2vocab,)
        train_generator = data.DataLoader(train_data, **params)

        #dev_data = EventDataset(args.data_dir + '%s/bootstrap%s/' % (type_dir, bs_n), "dev", args.glove2vocab,)
        dev_data = EventDataset(args.data_dir + '%s/fold%s/' % (type_dir, bs_n), "dev", args.glove2vocab,)
        dev_generator = data.DataLoader(dev_data, **params)

        return self._train(train_generator, dev_generator, emb, pos_emb, args, in_cv=True, test_data=test_data)

    def bootstrap(self, emb, pos_emb, args, test_data):

        # use args.n_splits as number of parallel jobs to save arguments
        n_jobs = args.n_splits

        with mp.Pool(processes=n_jobs) as pool:
            res = pool.map(partial(self.parallel_bootstrap, emb=emb, pos_emb=pos_emb, args=args, test_data=test_data), args.bs_list)
        print(res)

        return

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
            print(self._label_to_id_c)

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

        # we may want a separate function for clarity
        if args.bootstrap:
            self.bootstrap(emb, pos_emb, args, test_data)
            return

        if args.cv == True:
            best_params = self.cross_validation(emb, pos_emb, args)

            ### retrain on the best parameters
            print("To refit...")
            args.refit = True

            for k,v in best_params.items():
                exec("args.%s=%s" % (k, v))
        
        if args.refit_all:
            train_docs = args.train_docs
            dev_docs = []

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

        # f1 score is used for tcr and matres and hence exclude vague
        
        exclude_labels = ['NONE', 'VAGUE'] if len(self._label_to_id) == 4 else ['NONE']

        for label in labels:
            if self._id_to_label[label] not in exclude_labels:
                
                true_count = total_true.get(label, 0)
                pred_count = total_pred.get(label, 0)
        
                n_true += true_count
                n_pred += pred_count
                
                correct_count = len([l for l in range(len(pred_labels)) 
                                     if pred_labels[l] == true_labels[l] and pred_labels[l] == label])
                #for l in range(len(pred_labels)):
                #    if pred_labels[l] == true_labels[l] and pred_labels[l] == label:
                #        correct_count += 1
                n_correct += correct_count
                #precision = safe_division(correct_count, pred_count)
                #recall = safe_division(correct_count, true_count)

        precision = safe_division(n_correct, n_pred)                                                               
        recall = safe_division(n_correct, n_true)
        f1_score = safe_division(2 * precision * recall, precision + recall)
                #weight = safe_division(true_count, num_tests)
                #weighted_f1_scores[self._id_to_label[label]] = f1_score * weight
        return(f1_score)
        
        #print(weighted_f1_scores)
        #return sum(list(weighted_f1_scores.values()))

@dataclass
class REDEvaluator:
    model: REDEveEveRelModel
    def evaluate(self, test_data, args):
        # load test data first since it needs to be executed twice in this function                                                
        '''
        to_ilp = {}
        pairs = [(ex.doc.id+"_"+ex.left.id, ex.doc.id+"_"+ex.right.id) for ex in test_data]
         to_ilp['pairs'] = pairs
        to_ilp['labels'] = true_labels
        '''
        print("start testing...")
        preds, loss, true_labels = self.model.predict(self.model.model, test_data, args)
        
        '''
        to_ilp['probs'] = preds.numpy()

        out_path = Path(args.ilp_dir+'ilp.pkl')
        with out_path.open('wb') as fh:
            pickle.dump(to_ilp, fh)
        '''
        if args.backward_sample:
            preds = preds.view(-1)
        else:
            preds = (preds.max(1)[1]).long().view(-1)
        pred_labels = [self.model._id_to_label[x] for x in preds.numpy().tolist()]
        true_labels = [self.model._id_to_label[x] for x in true_labels.tolist()]

        if args.data_type in ['tbd']:
            test_data = [(x[0][0], x[2][0][0], x[2][1][0], true_labels[k])
                         for k, x in enumerate(test_data) if k < len(true_labels)]
            temporal_awareness(test_data, pred_labels, args.data_type, args.eval_with_timex)

        ids = [x[1][0] for x in test_data if x[1][0][0] != 'C']
        #self.for_analysis(ids, true_labels, pred_labels, test_data, '%s%s_local_all.tsv' % (args.ilp_dir, args.data_type))
        print("Done writing...")
        return ClassificationReport(self.model.name, true_labels, pred_labels)

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

    #for debug
    '''
    for doc_id, sample_id, pair, label, sent, pos, fts, rev, lidx_start_s, lidx_end_s, ridx_start_s, ridx_end_s, pred_ind in train_generator:
        print(doc_id)
        print(sample_id)
        print(label)
        print(fts)
        kill
    '''
    models = [NNClassifier()]
    for model in models:
        print(f"\n======={model.name}=====")
        print(f"======={args.model}=====\n")
        
        # if boostrap testing, we only output a list of test f1 scores 
        if args.bootstrap:
            model.train_epoch(train_generator, dev_generator, args, test_data = test_generator)
            print("Finished Bootstrap Testing")
        else:
            model.train_epoch(train_generator, dev_generator, args)
            evaluator = REDEvaluator(model)
            print("test...")
            #if args.data_type in ['tbd', 'matres']:
            #    print(evaluator.evaluate(dev_generator, args))
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
    p.add_argument('-data_dir_u', type=str,
                   help='Path to directory of unlabeld data (TE3-Silver). This should be output of '
                        '"ldcred.py flexnlp"')

    # select model
    p.add_argument('-model', type=str, default='rnn')

    # arguments for RNN model
    p.add_argument('-emb', type=int, default=300)
    p.add_argument('--hid', type=int, default=50)
    p.add_argument('-num_layers', type=int, default=1)
    p.add_argument('-batch', type=int, default=1)
    p.add_argument('-data_type', type=str, default="red")
    p.add_argument('-epochs', type=int, default=0)
    p.add_argument('-seed', type=int, default=123)
    p.add_argument('-lr', type=float, default=0.0005)
    p.add_argument('-num_classes', type=int, default=2) # get updated in main()
    p.add_argument('--dropout', type=float, default=0.6)
    p.add_argument('-ngbrs', type=int, default = 20)                                   
    p.add_argument('-pos2idx', type=dict, default = {})
    p.add_argument('-emb_array', type=np.array)
    p.add_argument('-glove2vocab', type=OrderedDict)
    p.add_argument('-cuda', type=bool, default=False)
    p.add_argument('-refit_all', type=bool, default=False)
    p.add_argument('-params', type=dict, default={})
    p.add_argument('-n_splits', type=int, default=5)
    p.add_argument('-pred_win', type=int, default=200)
    p.add_argument('-n_fts', type=int, default=15)
    p.add_argument('-unlabeled_weight', type=float, default=0.1)
    p.add_argument('-eval_with_timex', type=bool, default=True)
    p.add_argument('-shuffle', type=bool, default=False)
    p.add_argument('-xi', type=float, default=1e-3)
    p.add_argument('-eps',type=float, default=2.5) #[2.5]
    # arguments for CNN model
    p.add_argument('-stride', type=int, default = 1)
    p.add_argument('-kernel', type=int, default = 5)
    p.add_argument('-train_docs', type=list, default=[])
    p.add_argument('-dev_docs', type=list, default=[])
    p.add_argument('-cv', type=bool, default=False)
    p.add_argument('-cv_shuffle', type=bool, default=False)
    p.add_argument('-attention', type=bool, default=False)
    p.add_argument('-backward_sample', type=bool, default=True)
    p.add_argument('-save_model', type=bool, default=False)
    p.add_argument('--save_stamp', type=str, default="0301_matres_local_100_0.2_no_flip")
    p.add_argument('-ilp_dir', type=str, default="/nas/home/rujunhan/ILP/")
    p.add_argument('-load_model', type=bool, default=True)
    p.add_argument('-load_model_file', type=str, default= '0226_tbd_local_50_0.4.pth.tar') #tbd_1121.pth.tar')
    p.add_argument('-joint', type=bool, default=False)
    p.add_argument('-num_causal', type=int, default=2)
    p.add_argument('-loss_u', type=str, default='')
    p.add_argument('-skip_u', type=bool, default=True)
    p.add_argument('-bert_fts', type=bool, default=False)
    # bootstrap options
    p.add_argument('-bootstrap', type=bool, default=False)
    p.add_argument('-bs_list', type=list, default=list(range(0, 5)))
    args = p.parse_args()
    
    #args.eval_list = ['train', 'dev', 'test']
    args.eval_list = []
    args.data_type = "tbd"
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
    # create pos_tag and vocabulary dictionaries
    # make sure raw data files are stored in the same directory as train/dev/test data
    
    tags = open("/nas/home/rujunhan/tcr_output/pos_tags.txt")
    pos2idx = {}
    idx = 0
    for tag in tags:
        tag = tag.strip()
        pos2idx[tag] = idx
        idx += 1
    args.pos2idx = pos2idx
    args.cuda = False
    
    args.pred_win = args.ngbrs * 2

    '''
    glove = read_glove("/nas/home/rujunhan/red_output/glove/glove.6B.%sd.txt" % args.emb)
    vocab = np.array(['<pad>', '<unk>'] + list(glove.keys()))
    w2i = OrderedDict((vocab[i], i) for i in range(len(vocab)))
    i2w = {v:k for k,v in w2i.items()}
    np.save('i2w.npy', i2w)
    '''
    args.emb_array = np.load(args.data_dir + 'all' + '/emb_reduced.npy')
    args.glove2vocab = np.load(args.data_dir + 'all' + '/glove2vocab.npy').item()

    '''
    v2g = {v:k for k,v in args.glove2vocab.items()}
    np.save('v2g.npy', v2g)
    '''
    print(args.emb_array.shape)
    print(len(args.glove2vocab))

    args.nr = 0.0
    args.tempo_filter = True
    args.skip_other = True
    
    #args.params = {'hid': [40], 'unlabeled_weight': [0.1], 'dropout': [0.1, 0.2]}#, 0.3, 0.4]}#, 'xi':[1e-5, 1e-4, 1e-2, 1e-1]}
    args.params = {'hid': [args.hid], 'unlabeled_weight': [0.1], 'dropout': [args.dropout]}
    #args.params = {'hid': [30], 'unlabeled_weight': [0.1], 'dropout': [0.6, 0.7, 0.8, 0.9]}
    print(args.hid, args.dropout)
    main(args)
