import random
import itertools

import numpy as np
from sklearn.preprocessing import LabelEncoder
import h5py
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


def get_disentanglement_dataset(
        dataset: str,
        train: bool,
        batch_size: int,
        **kwargs
    ):
    datasets = {"DSPRITES": DSprites, "MPI3D": MPI3D, "SHAPES3D": Shapes3D}
    dataset = datasets[dataset.upper()](train=train, **kwargs)
    if train:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    model_kwargs = {
        'n_concepts': dataset.n_concepts,
        'dim_y': dataset.n_classes_task,
        'dim_c': dataset.dim_c,
        'continuous_c': False,
        'continuous_y': False,
        'ch_in': dataset.n_channels,
        'image_size': 64,
        'imbalance_ratio': dataset.imbalance_ratio,
    }
    attr_groups = [[i] for i in range(dataset.n_concepts)]
    random.shuffle(attr_groups)
    return dataloader, model_kwargs, attr_groups


class DisentanglementDataset(Dataset):

    def __init__(
            self,
            filepath,
            task,
            concepts,
            nuisances_task,
            n_samples=None,
            train=True,
            seed=42,
            flatten=False,
            return_nuisances=False,
            binarize_concepts=True,
            **kwargs
        ):
        
        self.all_factors = list(self.factors_nvalues.keys())

        self.task = task
        self.concepts = concepts
        self.nuisances_task = nuisances_task
        self.n_samples = n_samples
        self.train = train
        self.seed = seed
        self.flatten = flatten
        self.return_nuisances = return_nuisances
        self.binarize_concepts = binarize_concepts

        if binarize_concepts:
            self.n_concepts = sum([self.factors_nvalues[k] for k in concepts])
            self.dim_c = 1
        else:
            self.n_concepts = len(concepts)
            self.dim_c = [self.factors_nvalues[k] for k in concepts]
        self.n_classes_task = self.factors_nvalues[task]
        self.nuisances_nontask = list(set(self.all_factors) - \
            set([task] + concepts + nuisances_task))
        self._read_dataset(filepath)
        self._filter_samples_nuisances()
        self._select_samples()
        self._find_imbalance()


    def _read_dataset(self, npz_path):
        dataset = np.load(npz_path, allow_pickle=True)
        self.images = dataset['imgs'][:,None]
        self.labels = dataset['latents_classes'][:,1:]


    def _filter_samples_nuisances(self):
        random.seed(self.seed)
        def split_into_groups(data, n):
            k, r = divmod(len(data), n)
            groups = [data[i * k + min(i, r):(i + 1) * k + min(i + 1, r)] for i in range(n)]
            return groups
        
        factors_task = self.concepts + self.nuisances_task
        sets = [range(self.factors_nvalues[k]) for k in factors_task]
        combinations = list(itertools.product(*sets))
        random.shuffle(combinations)
        groups = split_into_groups(combinations, self.factors_nvalues[self.task])
        groups_task = [[i] + list(group) for i in range(self.factors_nvalues[self.task])\
            for group in groups[i]]

        idxs = [self.all_factors.index(i) for i in [self.task]+factors_task]
        #matches = (self.labels[:,idxs][:, None] == np.array(groups_task)).all(axis=2)
        #keep = list(matches.any(axis=1))

        def find_row_indices(A, B):
            A_view = np.core.records.fromarrays(A.T)
            B_view = np.core.records.fromarrays(B.T)
            return np.nonzero(np.in1d(A_view, B_view))[0]
        keep = find_row_indices(self.labels[:,idxs], np.array(groups_task))

        self.images = self.images[keep]
        self.labels = self.labels[keep]
    

    def _select_samples(self):
        if self.n_samples is None or self.n_samples>self.images.shape[0]:
            self.n_samples = self.images.shape[0]

        idxs = random.sample(range(self.images.shape[0]), self.n_samples)
        train_samples = self.n_samples*75//100
        idxs = idxs[:train_samples] if self.train else idxs[train_samples:]

        self.images = self.images[idxs]
        self.labels = self.labels[idxs]


    def _find_imbalance(self):
        concepts = torch.stack([self._get_label(
            torch.tensor(label), self.concepts, one_hot=self.binarize_concepts
        ) for label in self.labels])
        self.imbalance_ratio = concepts.shape[0] / torch.sum(concepts, axis=0) - 1


    def __len__(self):
        return len(self.images)


    def __getitem__(self, idx):
        factors = torch.tensor(self.labels[idx])
        image = torch.tensor(self.images[idx], dtype=torch.float32) / 255.
        if self.flatten:
            image = image.flatten()
        task = factors[self.all_factors.index(self.task)]
        concepts = self._get_label(factors, self.concepts, one_hot=self.binarize_concepts)
        if self.return_nuisances:
            if len(self.nuisances_task)>0:
                nuisances_task = self._get_label(factors, self.nuisances_task, one_hot=False)
            else:
                nuisances_task = torch.tensor([])
            if len(self.nuisances_nontask)>0:
                nuisances_nontask = self._get_label(factors, self.nuisances_nontask, one_hot=False)
            else:
                nuisances_nontask = torch.tensor([])
            return image, task, concepts, nuisances_task, nuisances_nontask
        else:
            return image, task, concepts
    

    def _get_label(self, factors, set_labels, one_hot=True):
        if one_hot:
            return torch.cat([
                F.one_hot(
                    factors[self.all_factors.index(i)], 
                    self.factors_nvalues[i]
                ) for i in set_labels
            ]).float()
        else:
            return torch.stack([
                factors[self.all_factors.index(i)] for i in set_labels
            ]).float()
    


class DSprites(DisentanglementDataset):

    factors_nvalues = {
        'shape': 3, 
        'scale': 6, 
        'orientation': 40, 
        'posX': 32, 
        'posY': 32,
    }
    n_channels = 1

    def _read_dataset(self, npz_path):
        dataset = np.load(npz_path, allow_pickle=True)
        self.images = dataset['imgs'][:,None]
        self.labels = dataset['latents_classes'][:,1:]



class MPI3D(DisentanglementDataset):

    factors_nvalues = {
        'object_color': 6, 
        'object_shape': 6, 
        'object_size': 2, 
        'camera_height': 3, 
        'background_color': 3, 
        'horizontal_axis': 40, 
        'vertical_axis': 40
    }
    n_channels = 3

    def _read_dataset(self, npz_path):
        dataset = np.load(npz_path, allow_pickle=True)
        self.images = dataset['images']
        self.labels = dataset['labels']



class Shapes3D(DisentanglementDataset):

    factors_nvalues = {
        'floor_hue': 10, 
        'wall_hue': 10, 
        'object_hue': 10, 
        'scale': 8, 
        'shape': 4, 
        'orientation': 15
    }    
    n_channels = 3

    def _read_dataset(self, h5_path):
        dataset = h5py.File(h5_path, 'r')
        self.images = np.array(dataset['images']).transpose(0, 3, 1, 2)
        labels = np.array(dataset['labels'])
        encoded_labels = np.zeros_like(labels, dtype=int)
        for i in range(labels.shape[1]):
            le = LabelEncoder()
            encoded_labels[:, i] = le.fit_transform(labels[:, i])
        self.labels = encoded_labels