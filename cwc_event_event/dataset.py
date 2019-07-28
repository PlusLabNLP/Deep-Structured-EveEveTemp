import torch
from torch.utils import data
import pickle
import numpy as np

class EventDataset(data.Dataset):
    def __init__(self, data_dir, data_split, glove2vocab, data_dir_rev=""):
        'Initialization'
        self.glove2vocab = glove2vocab
        with open(data_dir + data_split + '.pickle', 'rb') as handle:
            self.data = pickle.load(handle)
       
        if data_dir_rev:
            with open(data_dir_rev + data_split + '.pickle', 'rb') as handle:
                data_rev = pickle.load(handle)
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
        label = sample[3]
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

    def merge_dataset(self, dataset):
        self.data += dataset.data
