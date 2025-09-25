import os
import pickle
import random
from PIL import Image

import torch
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def get_celeba(
        train: bool,
        batch_size: int,
        crop_size: int=128,
        **kwargs
    ):
    if train:
        transform = transforms.Compose([
            transforms.Resize(218),
            transforms.CenterCrop(178),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(218),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])
    dataset = ConceptCelebA(
        split="train" if train else "test", transform=transform, **kwargs)
    if train:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    model_kwargs = {
        'n_concepts': dataset.n_concepts,
        'dim_y': 1 if len(dataset.tasks)==1 else 2**len(dataset.tasks),
        'dim_c': 1,
        'continuous_c': False,
        'continuous_y': False,
        'ch_in': 3,
        'image_size': crop_size,
        'imbalance_ratio': None,
    }
    attr_groups = dataset.concepts_groups
    return dataloader, model_kwargs, attr_groups


class ConceptCelebA(datasets.CelebA):

    def __init__(
            self, 
            root,
            split,
            task,
            n_concepts,
            seed=42,
            return_nuisances=False,
            transform=None,
            n_samples=None,
            imgs_pkl_dir=None,
            **kwargs
        ):
        self.root = root
        self.split = split
        self.tasks = task if isinstance(task, list) else [task]
        self.n_concepts = n_concepts
        self.seed = seed
        self.return_nuisances = return_nuisances
        self.transform = transform
        self.imgs_pkl_dir = imgs_pkl_dir

        split_map = {
            "train": 0,
            "valid": 1,
            "test": 2,
            "all": None,
        }
        split_ = split_map[datasets.utils.verify_str_arg(
            split.lower(), "split", ("train", "valid", "test", "all"))]
        splits = self._load_csv("list_eval_partition.txt")
        self.attr_all= self._load_csv("list_attr_celeba.txt", header=1)

        mask = slice(None) if split_ is None else (splits.data == split_).squeeze()
        if mask == slice(None):  # if split == "all"
            self.filename = splits.index
        else:
            self.filename = [splits.index[i] for i in torch.squeeze(torch.nonzero(mask))]

        self.attr = self.attr_all.data[mask]
        self.attr = torch.div(self.attr + 1, 2, rounding_mode="floor")
        self.attr_names = self.attr_all.header[:-1]

        self.n_samples = len(self.attr) if n_samples is None else \
            min(n_samples, len(self.attr))
        
        self._set_concepts_nuisances()
        if self.imgs_pkl_dir is not None:
            self._load_images()


    def __getitem__(self, index):
        image = self._load_image(index) \
            if self.imgs_pkl_dir is None else self.images[index]
        if self.transform is not None:
            image = self.transform(image)
        
        attributes = self.attr[index]
        tasks_bits = [attributes[self.attr_names.index(task)].float() for task in self.tasks]
        tasks = int("".join(str(int(b.item())) for b in tasks_bits), 2)
        concepts = self._get_label(attributes, self.concepts)
        if self.return_nuisances:
            nuisances_task = self._get_label(attributes, self.nuisances_task)
            nuisances_nontask = self._get_label(attributes, self.nuisances_nontask)
            return image, tasks, concepts, nuisances_task, nuisances_nontask
        else:
            return image, tasks, concepts
        

    def __len__(self) -> int:
        return self.n_samples
    

    def _get_label(self, attributes, set_labels):
        label = [attributes[self.attr_names.index(i)] for i in set_labels]
        return torch.tensor([]).float() if len(label)==0 else torch.stack(label).float()
    

    def _get_attributes_correlations(self):
        attr_centered = self.attr_all.data.float() - \
            self.attr_all.data.float().mean(dim=0, keepdim=True)
        n = self.attr_all.data.float().shape[0]
        cov = (attr_centered.T @ attr_centered) / (n - 1)
        std = self.attr_all.data.float().std(dim=0, unbiased=True).view(-1, 1)
        return cov / (std @ std.T)

    
    def _set_concepts_nuisances(self):
        concepts = list(set(self.attr_names) - set(self.tasks))
        concepts.sort()
        random.seed(self.seed)
        random.shuffle(concepts)
        self.concepts = concepts[:self.n_concepts]
        self.nuisances_task = list(
            set([attr for _, attr in enumerate(self.attr_names)]) -\
            set(self.tasks + self.concepts)
        )
        self.concepts_groups = [[self.attr_names.index(c)] for c in self.concepts]
        self.nuisances_nontask = []


    def _load_image(self, index):
        img_path = os.path.join(
            self.root, self.base_folder, "img_align_celeba", self.filename[index])
        with Image.open(img_path) as img:
            return img.copy()


    def _load_images(self):
        pkl_path = os.path.join(self.imgs_pkl_dir, "img_align_celeba_{}.pkl".format(self.split))
        if os.path.exists(pkl_path):
            with open(pkl_path, 'rb') as f:
                self.images = pickle.load(f)
        else:
            self.images = [self._load_image(i) for i in range(len(self))]
            with open(pkl_path, 'wb') as f:
                pickle.dump(self.images, f)