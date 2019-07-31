from gurobipy import *
from pathlib import Path
from collections import OrderedDict
from typing import Iterator, List, Mapping, Union, Optional, Set
import numpy as np
import pickle
from baseline import rev_map, rev_causal_map, ClassificationReport

class Gurobi_Inference():
    
    def __init__(self, pairs, probs, pairs_c, probs_c, label2idx, label2idx_c):
        '''
        pairs: list of str tuple ; (docid_eventid, docid_eventid)
        probs: a numpy matrix of local prediction scores; (#instance, #classes)
        probs_c: a numpy matrix of local prediction scores; (#causal instance, #causal class)
        '''
        self.model = Model("event_event_rel")
        
        # temporal
        self.pairs = pairs
        self.idx2pair = {n: self.pairs[n] for n in range(len(pairs))}
        self.pair2idx = {v:k for k,v in self.idx2pair.items()}
        self.probs = probs
        self.label2idx = label2idx
        self.idx2label = OrderedDict([(v,k) for k,v in label2idx.items()])
        self.rev_map = rev_map
        self.rev_causal_map = rev_causal_map
        self.N, self.P = probs.shape
        self.pred_labels = list(np.argmax(probs, axis=1)) # size: self.N
        
        # causal
        self.pairs_c = pairs_c
        self.idx2pair_c = {n: self.pairs_c[n] for n in range(len(pairs_c))}
        self.pairs2idx_c = {v:k for k,v in self.idx2pair.items()}
        self.probs_c = probs_c
        self.label2idx_c = label2idx_c
        self.idx2label_c = OrderedDict([(v,k) for k,v in label2idx_c.items()])
        self.Nc, self.Pc = probs_c.shape
        self.pred_labels_c = []
        if self.Nc > 0:
            self.pred_labels_c = list(np.argmax(probs_c, axis=1))

    def define_vars(self):
        var_table = []
        # temporal variables
        for n in range(self.N):
            sample = []
            for p in range(self.P):
                sample.append(self.model.addVar(vtype=GRB.BINARY, name="y_%s_%s"%(n,p)))
            var_table.append(sample)

        # causal variables
        for n in range(self.Nc):
            sample = []
            for p in range(self.Pc):
                sample.append(self.model.addVar(vtype=GRB.BINARY, name="yc_%s_%s"%(n,p)))
            var_table.append(sample)
        return var_table # [self.N * self.P] + [self.Nc * self.Pc]
        
    def objective(self, samples, p_table, p_table_c):
    
        obj = 0.0

        assert len(samples) == self.N + self.Nc
        assert len(samples[0]) == self.P
        
        # temporal
        for n in range(self.N):
            for p in range(self.P):
                obj += samples[n][p] * p_table[n][p]

        # causal
        for n in range(self.N, self.N + self.Nc):
            for p in range(self.Pc):
                obj += samples[n][p] * p_table_c[n - self.N][p]

        return obj
    
    def single_label(self, sample):
        return sum(sample) == 1

    def grammar_rules(self, sample, label):
        return sample[self.label2idx[label]] == 1

    def transitivity_list(self):
        transitivity_samples = []
        pair2idx = self.pair2idx
        for k, (e1, e2) in self.idx2pair.items():
            for (re1, re2), i in pair2idx.items():
                if e2 == re1 and (e1, re2) in pair2idx.keys():
                    transitivity_samples.append((pair2idx[(e1, e2)], pair2idx[(re1, re2)], pair2idx[(e1, re2)]))
        return transitivity_samples
    
    def transitivity_criteria(self, tab, triplet):
        r1, r2, r3 = triplet
        ld = self.label2idx
        if 'INCLUDES' in ld.keys():
            return [
                    tab[r1][ld['BEFORE']] + tab[r2][ld['BEFORE']] - tab[r3][ld['BEFORE']],
                    tab[r1][ld['AFTER']] + tab[r2][ld['AFTER']] - tab[r3][ld['AFTER']],
                    tab[r1][ld['SIMULTANEOUS']] + tab[r2][ld['SIMULTANEOUS']] - tab[r3][ld['SIMULTANEOUS']],
                    tab[r1][ld['INCLUDES']] + tab[r2][ld['INCLUDES']] - tab[r3][ld['INCLUDES']],
                    tab[r1][ld['IS_INCLUDED']] + tab[r2][ld['IS_INCLUDED']] - tab[r3][ld['IS_INCLUDED']],
                    #tab[r1][ld['VAGUE']] + tab[r2][ld['VAGUE']] - tab[r3][ld['VAGUE']],
                    tab[r1][ld['BEFORE']] + tab[r2][ld['VAGUE']] - tab[r3][ld['BEFORE']] - tab[r3][ld['VAGUE']] - tab[r3][ld['INCLUDES']] - tab[r3][ld['IS_INCLUDED']],
                    tab[r1][ld['BEFORE']] + tab[r2][ld['INCLUDES']] - tab[r3][ld['BEFORE']] - tab[r3][ld['VAGUE']] - tab[r3][ld['INCLUDES']],
                    tab[r1][ld['BEFORE']] + tab[r2][ld['IS_INCLUDED']] - tab[r3][ld['BEFORE']] - tab[r3][ld['VAGUE']] - tab[r3][ld['IS_INCLUDED']],
                    tab[r1][ld['AFTER']] + tab[r2][ld['VAGUE']] - tab[r3][ld['AFTER']] - tab[r3][ld['VAGUE']] - tab[r3][ld['INCLUDES']] - tab[r3][ld['IS_INCLUDED']],
                    tab[r1][ld['AFTER']] + tab[r2][ld['INCLUDES']] - tab[r3][ld['AFTER']] - tab[r3][ld['VAGUE']] - tab[r3][ld['INCLUDES']],
                    tab[r1][ld['AFTER']] + tab[r2][ld['IS_INCLUDED']] - tab[r3][ld['AFTER']] - tab[r3][ld['VAGUE']] - tab[r3][ld['IS_INCLUDED']],
                    tab[r1][ld['INCLUDES']] + tab[r2][ld['VAGUE']] - tab[r3][ld['INCLUDES']] - tab[r3][ld['AFTER']] - tab[r3][ld['VAGUE']] - tab[r3][ld['BEFORE']],
                    tab[r1][ld['INCLUDES']] + tab[r2][ld['BEFORE']] - tab[r3][ld['INCLUDES']] - tab[r3][ld['VAGUE']] - tab[r3][ld['BEFORE']],
                    tab[r1][ld['INCLUDES']] + tab[r2][ld['AFTER']] - tab[r3][ld['INCLUDES']] - tab[r3][ld['VAGUE']] - tab[r3][ld['AFTER']],
                    tab[r1][ld['IS_INCLUDED']] + tab[r2][ld['VAGUE']] - tab[r3][ld['IS_INCLUDED']] - tab[r3][ld['VAGUE']] - tab[r3][ld['BEFORE']] - tab[r3][ld['AFTER']],
                    tab[r1][ld['IS_INCLUDED']] + tab[r2][ld['BEFORE']] - tab[r3][ld['BEFORE']] - tab[r3][ld['VAGUE']] - tab[r3][ld['IS_INCLUDED']],
                    tab[r1][ld['IS_INCLUDED']] + tab[r2][ld['AFTER']] - tab[r3][ld['AFTER']] - tab[r3][ld['VAGUE']] - tab[r3][ld['IS_INCLUDED']],
                    tab[r1][ld['VAGUE']] + tab[r2][ld['BEFORE']] - tab[r3][ld['BEFORE']] - tab[r3][ld['VAGUE']] - tab[r3][ld['INCLUDES']] - tab[r3][ld['IS_INCLUDED']],
                    tab[r1][ld['VAGUE']] + tab[r2][ld['AFTER']] - tab[r3][ld['AFTER']] - tab[r3][ld['VAGUE']] - tab[r3][ld['INCLUDES']] - tab[r3][ld['IS_INCLUDED']],
                    tab[r1][ld['VAGUE']] + tab[r2][ld['INCLUDES']] - tab[r3][ld['INCLUDES']] - tab[r3][ld['VAGUE']] - tab[r3][ld['BEFORE']] - tab[r3][ld['AFTER']],
                    tab[r1][ld['VAGUE']] + tab[r2][ld['IS_INCLUDED']] - tab[r3][ld['IS_INCLUDED']] - tab[r3][ld['VAGUE']] - tab[r3][ld['BEFORE']] - tab[r3][ld['AFTER']],
                    tab[r1][ld['BEFORE']] + tab[r2][ld['SIMULTANEOUS']] - tab[r3][ld['BEFORE']],
                    tab[r1][ld['AFTER']] + tab[r2][ld['SIMULTANEOUS']] - tab[r3][ld['AFTER']],
                    tab[r1][ld['INCLUDES']] + tab[r2][ld['SIMULTANEOUS']] - tab[r3][ld['INCLUDES']],
                    tab[r1][ld['IS_INCLUDED']] + tab[r2][ld['SIMULTANEOUS']] - tab[r3][ld['IS_INCLUDED']],
                   ]
        else:
            return [
                    tab[r1][ld['BEFORE']] + tab[r2][ld['BEFORE']] - tab[r3][ld['BEFORE']],
                    tab[r1][ld['AFTER']] + tab[r2][ld['AFTER']] - tab[r3][ld['AFTER']],
                    tab[r1][ld['SIMULTANEOUS']] + tab[r2][ld['SIMULTANEOUS']] - tab[r3][ld['SIMULTANEOUS']],
                    tab[r1][ld['BEFORE']] + tab[r2][ld['VAGUE']] - tab[r3][ld['BEFORE']] - tab[r3][ld['VAGUE']],
                    tab[r1][ld['AFTER']] + tab[r2][ld['VAGUE']] - tab[r3][ld['AFTER']] - tab[r3][ld['VAGUE']],
                    tab[r1][ld['VAGUE']] + tab[r2][ld['BEFORE']] - tab[r3][ld['BEFORE']] - tab[r3][ld['VAGUE']],
                    tab[r1][ld['VAGUE']] + tab[r2][ld['AFTER']] - tab[r3][ld['AFTER']] - tab[r3][ld['VAGUE']],
                    tab[r1][ld['BEFORE']] + tab[r2][ld['SIMULTANEOUS']] - tab[r3][ld['BEFORE']],
                    tab[r1][ld['AFTER']] + tab[r2][ld['SIMULTANEOUS']] - tab[r3][ld['AFTER']],
                   ]

    def symmetry_constraints(self, samples, n, causal=False):
        # Note: Here is assert samples is made up by (normal instance, reverse
        # insatnce)
        constraints = []
        if causal:
            offset = int(len(self.pairs_c) / 2)
            for label, idx in self.label2idx_c.items():
                rev_idx = self.label2idx_c[self.rev_causal_map[label]]
                constraints.append(samples[n][idx] == samples[n+offset][rev_idx])
        else:
            offset = int(len(self.pairs) / 2)
            for label, idx in self.label2idx.items():
                rev_idx = self.label2idx[self.rev_map[label]]
                constraints.append(samples[n][idx] == samples[n+offset][rev_idx])
        return constraints
        
    def tense_relation(self, n):
        label = None
        if self.report_dominance:
            if (self.tenses[n][0] in ['PRESENT']) and (self.tenses[n][1] in ['PAST']):
                label = 'AFTER'
            elif (self.tenses[n][0] in ['PRESENT']) and (self.tenses[n][1] in ['PRESENT']) and (self.aspect[n][1] in ['PERFECTIVE']):
                label = 'AFTER'
            elif (self.tenses[n][0] in ['PRESENT']) and (self.tenses[n][1] in ['FUTURE']):
                label = 'BEFORE'
            elif (self.tenses[n][0] in ['PAST']) and (self.tenses[n][1] in ['PAST']) and (self.aspect[n][1] in ['PERFECTIVE']):
                label = 'AFTER'
        return label


    def causal_temporal(self, samples, ti, ci):
        # if a relation is classified as "causes"; its temporal relation should be before
        return samples[ci][self.label2idx_c['causes']] <= samples[ti][self.label2idx['BEFORE']]

    def define_constraints(self, var_table):
        # Constraint 1: single label assignment
        for n in range(self.N + self.Nc):
            self.model.addConstr(self.single_label(var_table[n]), "c1_%s" % n)
        
        # Constraint 2: transitivity
        trans_triples = self.transitivity_list()
        t = 0
        for triple in trans_triples:
            for ci in self.transitivity_criteria(var_table, triple):
                self.model.addConstr(ci <= 1, "c2_%s" % t)
                t += 1
        
        # Constraint 3: Symmetry
        offset = int(len(self.pairs) / 2)
        for n in range(offset):
            for si in self.symmetry_constraints(var_table, n):
                self.model.addConstr(si,  "c3_%s" % n)
        offset = int(len(self.pairs_c) / 2)
        for n in range(offset):
            for si in self.symmetry_constraints(var_table, n+self.N, causal=True):
                self.model.addConstr(si,  "c3_1_%s" % n)
        
        # Constraint 3: grammar rules
        #for n in range(self.N):
        #    label = self.tense_relation(n)
        #    if label:
        #        self.model.addConstr(self.grammar_rules(var_table[n], label), "c3_%s" % n)
        
        # Constraint 4: Temporal + Causal
        if self.Nc > 0:
            for ci in range(self.Nc):
                ti = self.pair2idx[self.pairs_c[ci]] # lookup the same temporal pair as causal pair
                self.model.addConstr(self.causal_temporal(var_table, ti, ci+self.N), "c4_%s" % ci)
        return 
    
    def run(self):
        try:
            # Define variables
            var_table = self.define_vars()

            # Set objective 
            self.model.setObjective(self.objective(var_table, self.probs, self.probs_c), GRB.MAXIMIZE)
            
            # Define constrains
            self.define_constraints(var_table)

            # run model
            self.model.setParam('OutputFlag', False)
            self.model.optimize()
            
        except GurobiError:
            print('Error reported')

    def predict(self):
        count = 0
        for v in self.model.getVars():
            # sample type (T or C)
            is_causal = (v.varName.split('_')[0] == 'yc')
            # sample idx
            s_idx = int(v.varName.split('_')[1])
            # sample class index
            c_idx = int(v.varName.split('_')[2])

            if v.x == 1.0:
                if is_causal:
                    if self.pred_labels_c[s_idx] != c_idx:
                        self.pred_labels_c[s_idx] = c_idx
                        count += 1
                else:
                    if self.pred_labels[s_idx] != c_idx:
                        self.pred_labels[s_idx] = c_idx
                        count += 1
        print('# of global correction: %s' % count)
        print('Objective Function Value:', self.model.objVal)
        return 
    
    def evaluate(self, true_labels, exclude_vague=True, backward=True):
        assert len(true_labels) == len(self.pred_labels) + len(self.pred_labels_c)
        assert backward == True
        if backward:
            labels_t =  [self.idx2label[x.item()] for x in true_labels[:int(self.N/2)]]
            pred_labels =  [self.idx2label[x] for x in self.pred_labels[:int(self.N/2)]]
        else:
            labels_t =  [self.idx2label[x.item()] for x in true_labels[:self.N]]
            pred_labels =  [self.idx2label[x] for x in self.pred_labels]
                                                                    
        print(ClassificationReport("Event-Event-Rel-Global", labels_t, pred_labels, exclude_vague))

        if self.Nc > 0:
            labels_c = [self.idx2label_c[x.item()] for x in true_labels[self.N:]]
            pred_labels_c = [self.idx2label_c[x] for x in self.pred_labels_c]

            assert len(labels_c) == len(pred_labels_c)
            if backward:
                labels_c = labels_c[:int(self.Nc/2)]
                pred_labels_c = pred_labels_c[:int(self.Nc/2)]
            correct = [x for x in range(len(labels_c)) if pred_labels_c[x] == labels_c[x]]
            print("Causal Accurracy is: %.4f" % (float(len(correct)) / float(len(labels_c))))
        return true_labels[:int(self.N/2)].cpu().tolist(), self.pred_labels[:int(self.N/2)]
