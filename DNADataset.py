# -*- coding: utf-8 -*-
"""
train and test dataset
Fault-tolerant coding(FTC)
author Yu-Hang Yin
"""
import os
import numpy as np
from torch.utils.data import Dataset
import torch

def embed0(seq, mapper, word_dim):
    mat = np.asarray([mapper[element] if element in mapper else np.ones(word_dim) for element in seq])
    return mat

def embed(seq, mapper, word_dim):
    mat = np.asarray([mapper[seq[i:i+2]] if seq[i:i+2] in mapper else np.ones(word_dim) for i in range(len(seq)-1)])
    return mat


class DNADataset(Dataset):
    def __init__(self, dataset_path, dataset_type="train", transform=None):
        filename = dataset_type + ".data"
        dataset_load = os.path.join(dataset_path, filename)
        print("Dataset load from ", dataset_load)
        with open(dataset_load, 'r') as dna_file:
            data = dna_file.readlines()
        self.seqs = []
        self.labels = []
        for line in data:
            temp_seq = line.strip().split()
            self.seqs.append(temp_seq[1])
            self.labels.append(temp_seq[2])
        self.transform = transform

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, index):

        mapper1 = {'A': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   'C': [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   'G': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   'T': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]}

        mapper2 = {
            'AA': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            'CA': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            'GA': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            'TA': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],

            'AC': [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            'CC': [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            'GC': [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            'TC': [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],

            'AG': [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            'CG': [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            'GG': [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            'TG': [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],

            'AT': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            'CT': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            'GT': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            'TT': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
        }

        mapper3 = {
            'AA': [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            'AC': [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            'AG': [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            'AT': [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],

            'CA': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            'CC': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            'CG': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            'CT': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],

            'GA': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            'GC': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            'GG': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            'GT': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],

            'TA': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            'TC': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            'TG': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            'TT': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        }

        seq1 = embed0(self.seqs[index], mapper1, len(mapper1['A']))
        seq2 = embed(self.seqs[index], mapper2, len(mapper2['AA']))
        seq3 = embed(self.seqs[index], mapper3, len(mapper3['AA']))

        seqtemp = np.append(seq1, seq2, axis=0)
        seq = np.append(seqtemp, seq3, axis=0)
        seq = np.asarray(seq)
        label = np.asarray(int(self.labels[index]))
        sample = {'seq': seq, 'label': label}

        if self.transform:
            sample = self.transform(sample)
        return sample


class ToTensor(object):

    def __call__(self, sample):
        seq, label = sample['seq'], sample['label']
        # numpy seq: H * W * C
        # torch seq: C * H * W
        seq = seq.transpose((1, 0))
        return {'seq': torch.from_numpy(seq),
                'label': torch.from_numpy(label)}


class DNADataset_for_prediction(Dataset):
    def __init__(self, seq_file, transform=None):
        print("Dataset load from ", seq_file)
        with open(seq_file, 'r') as dna_file:
            data = dna_file.readlines()
        self.seqs = []
        for line in data:
            x_str = "".join(line)
            if x_str.strip() == "":
                continue
            if x_str[0] == '>':
                continue
            # self.seqs.append(list(x_str.strip().split()[0]))
            temp_seq = line.strip().split()
            self.seqs.append(temp_seq[0])
        self.transform = transform

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, index):
        mapper1 = {'A': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   'C': [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   'G': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   'T': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]}

        mapper2 = {
            'AA': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            'CA': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            'GA': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            'TA': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],

            'AC': [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            'CC': [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            'GC': [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            'TC': [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],

            'AG': [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            'CG': [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            'GG': [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            'TG': [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],

            'AT': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            'CT': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            'GT': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            'TT': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
        }

        mapper3 = {
            'AA': [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            'AC': [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            'AG': [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            'AT': [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],

            'CA': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            'CC': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            'CG': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            'CT': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],

            'GA': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            'GC': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            'GG': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            'GT': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],

            'TA': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            'TC': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            'TG': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            'TT': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        }

        seq1 = embed0(self.seqs[index], mapper1, len(mapper1['A']))
        seq2 = embed(self.seqs[index], mapper2, len(mapper2['AA']))
        seq3 = embed(self.seqs[index], mapper3, len(mapper3['AA']))

        seqtemp = np.append(seq1, seq2, axis=0)
        seq = np.append(seqtemp, seq3, axis=0)
        seq = np.asarray(seq)

        sample = {'seq': seq}

        if self.transform:
            sample = self.transform(sample)
        return sample


class ToTensor_for_prediction(object):

    def __call__(self, sample):
        seq = sample['seq']
        # numpy seq: H * W * C
        # torch seq: C * H * W
        seq = seq.transpose((1, 0))
        return {'seq': torch.from_numpy(seq)}