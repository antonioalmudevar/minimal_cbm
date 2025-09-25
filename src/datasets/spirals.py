from typing import Tuple, Any

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


def get_spirals(
        train: bool,
        batch_size: int,
        resampling: bool=True,
        **kwargs
    ):
    if train and resampling:
        dataset = Spirals(train=True, **kwargs)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    else:
        dataset = Spirals(train=False, **kwargs)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    model_kwargs = {
        'n_concepts': dataset.n_concepts,
        'dim_y': 1,
        'dim_c': dataset.dim_c,
        'continuous_c': False,
        'continuous_y': True,
        'ch_in': 2,
        'image_size': 1,
        'imbalance_ratio': None,
    }
    attr_groups = None
    return dataloader, model_kwargs, attr_groups


class Spirals(Dataset):

    def __init__(
            self, 
            train,
            binarize_concepts=False,
            class_ratios=[1.0, 0.4, 0.05, 0.05],
            n_points_major=400,
            n_points_minor=400,
            seed=42,
            **kwargs
        ) -> None:

        self.train = train
        self.binarize_concepts = binarize_concepts
        self.n_concepts = 4 if binarize_concepts else 1
        self.dim_c = 1 if binarize_concepts else 4

        np.random.seed(seed)
        X_imb_4_extreme, y_imb_4_extreme = make_imbalanced_spiral_4class(
            n_points_major, n_points_minor, class_ratios
        )
        X_train_ext, X_test_ext, y_train_ext, y_test_ext = train_test_split(
            X_imb_4_extreme, y_imb_4_extreme, test_size=0.3, random_state=42
        )
        self.scaler_ext = StandardScaler().fit(X_train_ext)
        X_train_scaled_ext = self.scaler_ext.transform(X_train_ext)
        X_test_scaled_ext = self.scaler_ext.transform(X_test_ext)

        x = X_train_scaled_ext if train else X_test_scaled_ext
        c = y_train_ext if train else y_test_ext

        self.x_all = torch.Tensor(X_imb_4_extreme)
        self.y_all = torch.Tensor(y_imb_4_extreme)
        
        self.x, self.c = torch.Tensor(x), torch.Tensor(c).long()


    def __len__(self) -> int:
        return self.x.shape[0]


    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        x = self.x[index]
        c = self.c[index]
        y = (c+2).float()
        c = F.one_hot(c, 4).float() if self.binarize_concepts else torch.tensor([c])
        return x, y, c, torch.tensor([]).float(), torch.tensor([])


def make_imbalanced_spiral_4class(n_points_major, n_points_minor, class_ratios):
    X = []
    y = []
    for j, ratio in enumerate(class_ratios):
        n_points = int(n_points_major * ratio) if j == 0 \
            else int(n_points_minor * ratio)
        for i in range(n_points):
            r = i / max(n_points, 1)
            t = j * 4 + 4 * r + np.random.randn() * 0.1
            X.append([r * np.sin(t), r * np.cos(t)])
            y.append(j)
    return np.array(X), np.array(y)