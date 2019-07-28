import numpy as np
from torch.utils.data import DataLoader
import torch
from torch.nn.utils.rnn import pad_sequence

def get_data_loader(dataset, batch_size, shuffle=False):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=_collate_fn, num_workers=0)

def _collate_fn(l):
    # pad data
    doc_ids = []
    sample_ids = []
    pairs = []
    labels = []
    sents = []
    seq_lens = []
    poss = []
    fts = []
    revs = []
    l_starts = []
    l_ends = []
    r_starts = []
    r_ends = []
    pred_inds = []
    for doc, sample_id, pair, label, sent, pos, ft, rev, l_s, l_e, r_s, r_e, pred_ind in l:
        doc_ids.append(doc)
        sample_ids.append(sample_id)
        pairs.append(pair)
        labels.append(label)
        sents.append(sent)
        seq_lens.append(sent.size(0))
        poss.append(pos)
        fts.append(ft)
        revs.append(rev)
        l_starts.append(l_s)
        l_ends.append(l_e)
        r_starts.append(r_s)
        r_ends.append(r_e)
        pred_inds.append(pred_ind)
    labels = torch.LongTensor(labels)
    sents = pad_sequence(sents, batch_first=True, padding_value=0) # padding sent emb idx is 0
    seq_lens = torch.LongTensor(seq_lens)
    poss = pad_sequence(poss, batch_first=True, padding_value=37) # padding pos emb idx is 37
    fts = torch.stack(fts, dim=0)
    l_starts = torch.LongTensor(l_starts)
    l_ends = torch.LongTensor(l_ends)
    r_starts = torch.LongTensor(r_starts)
    r_ends = torch.LongTensor(r_ends)
    '''
    seq_lens is a LongTensor recording the sentences' length, (batch_size)
    sample_ids is a list of string, start with 'U'/ 'L'/ 'C'
    pairs is a list of tuple
    labels is a LongTensor recording the ground truth, (batch_size)
    sents is a padded LongTensor, (batch_size, max_batchseq_lens)
    poss is a padded LongTensor, (batch_size, max_batchseq_lens)
    fts is a FlotTensor, (batch_size, feature_size=15)
    revs  is a list of boolean, which recording whether this instance is reversed label or not
    l_starts/l_ends/r_starts/r_ends is a LongTensor which record the index to select,
        (batch_size)
    '''
    return (seq_lens, sample_ids, pairs, labels, sents, poss, fts, revs, l_starts,
            l_ends, r_starts, r_ends, pred_inds)

