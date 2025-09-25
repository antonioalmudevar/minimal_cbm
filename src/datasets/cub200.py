from typing import Callable, Optional, Tuple, Any
import os
import random
import pickle

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import torchvision.transforms as transforms


USED_ATTRIBUTES = [\
    1, 4, 6, 7, 10, 14, 15, 20, 21, 23, 25, 29, 30, 35, 36, 38, 40, 44, 45, 50, \
    51, 53, 54, 56, 57, 59, 63, 64, 69, 70, 72, 75, 80, 84, 90, 91, 93, 99, \
    101, 106, 110, 111, 116, 117, 119, 125, 126, 131, 132, 134, 145, 149, \
    151, 152, 153, 157, 158, 163, 164, 168, 172, 178, 179, 181, 183, 187, 188, 193, 194, 196, 198, \
    202, 203, 208, 209, 211, 212, 213, 218, 220, 221, 225, 235, 236, 238, 239, 240, 242, 243, 244, 249, \
    253, 254, 259, 260, 262, 268, 274, 277, 283, 289, 292, 293, 294, 298, 299, \
    304, 305, 308, 309, 310, 311]


def get_cub200(
        train: bool,
        batch_size: int,
        img_size: int=299,
        resampling: bool=True,
        **kwargs
    ):
    if train:
        transform = transforms.Compose([
            transforms.ColorJitter(brightness=32/255, saturation=(0.5, 1.5)),
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [2, 2, 2])
        ])
    else:
        transform = transforms.Compose([
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [2, 2, 2])
        ])
    dataset = CUB200(train=train, transform=transform, **kwargs)
    if train and resampling:
        #sampler = BatchSampler(ImbalancedDatasetSampler(dataset), batch_size=batch_size, drop_last=True)
        #dataloader = DataLoader(dataset, batch_sampler=sampler)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    model_kwargs = {
        'n_concepts': dataset.n_concepts,
        'dim_y': 200,
        'dim_c': 1,
        'continuous_y': False,
        'continuous_c': False,
        'ch_in': 3,
        'image_size': img_size,
        'imbalance_ratio': find_class_imbalance(dataset.data, True),
        #'imbalance_ratio': None,
    }
    attr_groups = dataset.concepts_groups
    return dataloader, model_kwargs, attr_groups



class CUB200(Dataset):
    """
    Modification of https://github.com/yewsiang/ConceptBottleneck/blob/master/CUB/dataset.py
    to read first all the images in memory to fasten the training process.
    """

    def __init__(
            self, 
            train: bool,
            imgs_dir: str,
            attr_dir: str, 
            pkls_dir: str,
            modify_pkls: Optional[bool]=False,
            n_groups_concepts: int=27,
            seed: int=42,
            return_nuisances=False,
            uncertain_label: bool=False,
            intervention_stage: bool=False,
            transform: Optional[Callable]=None,
        ) -> None:

        self.train = train
        self.imgs_dir = imgs_dir
        self.attr_dir = attr_dir
        self.pkls_dir = pkls_dir
        self.n_groups_concepts = n_groups_concepts
        self.seed = seed
        self.return_nuisances = return_nuisances
        self.modify_pkls = modify_pkls
        self.uncertain_label = uncertain_label
        self.intervention_stage = intervention_stage
        self.transform = transform

        self._read_pkls()
        self._set_attr_groups()
        self._set_concepts_nuisances()


    def __len__(self) -> int:
        return len(self.data)


    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        image = self.data[index]['img']
        task = self.data[index]['class_label']
        image = image if self.transform is None else self.transform(image)
        attr_label = self.data[index]['attribute_certainty'] if self.uncertain_label \
            else self.data[index]['attribute_label']
        concepts = torch.tensor(attr_label)[self.concepts_idxs].float()
        if self.return_nuisances:
            nuisances_task = torch.tensor(attr_label)[self.nuisances_idxs].float()
            nuisances_nontask = torch.tensor([]).float()
            return image, task, concepts, nuisances_task, nuisances_nontask
        else:
            return image, task, concepts
    

    def _read_pkls(self) -> None:
        pkl_file = os.path.join(self.pkls_dir, 'train.pkl' if self.train else 'test.pkl')
        self.data = pickle.load(open(pkl_file, 'rb'))
        for i in range(len(self.data)):
            if self.data[i]['img_path'][:len(self.imgs_dir)] != self.imgs_dir:
                path_split = self.data[i]['img_path'].split('/')
                self.data[i]['img_path'] = os.path.join(
                    self.imgs_dir, path_split[-2], path_split[-1]
                )
            if 'img' in self.data[i]:
                self.modify_pkls = False
            else:
                self.data[i]['img'] = Image.open(self.data[i]['img_path']).convert('RGB')
        if self.modify_pkls:
            with open(pkl_file, "wb") as f:
                pickle.dump(self.data, f)
        if self.intervention_stage:
            self._add_visibility_info()

    
    def _set_attr_groups(self):
        attr_file = os.path.join(self.attr_dir, 'attributes.txt')
        with open(attr_file, "r") as file:
            attr_labels = [i.split(" ")[1] for i in file.read().split("\n")[:-1] \
                if int(i.split(" ")[0]) in USED_ATTRIBUTES]
        attr_groups = {}
        for i, attr in enumerate(attr_labels):
            key = attr.split("::")[0]
            if key in attr_groups:
                attr_groups[key].append(i)
            else:
                attr_groups[key] = [i]
        self.attr_groups = list(attr_groups.values())


    def _set_concepts_nuisances(self):
        random.seed(self.seed)
        groups_concepts = random.sample(range(len(self.attr_groups)), self.n_groups_concepts)
        groups_concepts.sort()
        self.concepts_idxs = [j  for i, group in enumerate(self.attr_groups) \
            for j in group if i in groups_concepts]
        self.nuisances_idxs = [j  for i, group in enumerate(self.attr_groups) \
            for j in group if i not in groups_concepts]
        self.concepts_groups = [group \
            for i, group in enumerate(self.attr_groups) if i in groups_concepts]
        self.n_concepts = len(self.concepts_idxs)
    

    def _add_visibility_info(self):
        for datapoint in self.data:
            for attr in USED_ATTRIBUTES:
                if datapoint['attribute_certainty'][USED_ATTRIBUTES.index(attr)]==1:
                    datapoint['attribute_label'][USED_ATTRIBUTES.index(attr)] = 0


def find_class_imbalance(data, multiple_attr=False, attr_idx=-1):
    imbalance_ratio = []
    n = len(data)
    n_attr = len(data[0]['attribute_label'])
    if attr_idx >= 0:
        n_attr = 1
    if multiple_attr:
        n_ones = [0] * n_attr
        total = [n] * n_attr
    else:
        n_ones = [0]
        total = [n * n_attr]
    for d in data:
        labels = d['attribute_label']
        if multiple_attr:
            for i in range(n_attr):
                n_ones[i] += labels[i]
        else:
            if attr_idx >= 0:
                n_ones[0] += labels[attr_idx]
            else:
                n_ones[0] += sum(labels)
    for j in range(len(n_ones)):
        imbalance_ratio.append(total[j]/n_ones[j] - 1)
    if not multiple_attr: #e.g. [9.0] --> [9.0] * 312
        imbalance_ratio *= n_attr
    return imbalance_ratio



class ImbalancedDatasetSampler(Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
    """

    def __init__(self, dataset, indices=None):
        # if indices is not provided,
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) \
            if indices is None else indices

        # if num_samples is not provided,
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices)

        # distribution of classes in the dataset
        label_to_count = {}
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1

        # weight for each sample
        weights = [1.0 / label_to_count[self._get_label(dataset, idx)]
                   for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, idx):  # Note: for single attribute dataset
        return dataset.data[idx]['attribute_label'][0]

    def __iter__(self):
        idx = (self.indices[i] for i in torch.multinomial(
            self.weights, self.num_samples, replacement=True))
        return idx

    def __len__(self):
        return self.num_samples