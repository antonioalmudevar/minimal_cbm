import random

import torch

from src.datasets import get_loader
from src.helpers import calc_accuracy
from .base import BaseExperiment


class InterveneExperiment(BaseExperiment):

    experiment_name = "intervene"
    wandb_offline = False

    def __init__(self, **kwargs) -> None:

        super().__init__(**kwargs)
        self.load_last_epoch()

    #==========Setters==========
    def _set_loaders(self):
        self.cfg['data']['batch_size'] = self.cfg['training']['batch_size']
        self.train_loader, self.model_kwargs, self.concepts_groups = get_loader(
            train=True, seed=self.seed, **self.cfg['data'])
        self.test_loader, _, _ = get_loader(
            train=False, seed=self.seed, intervention_stage=True, **self.cfg['data'])
        self.concepts = [c for group in self.concepts_groups for c in group]

    def _set_all(self):
        self._set_wandb()
        self._set_loaders()
        self._set_model()
        self.to(self.device)

    #==========Prepare intervention procedure==========
    def _prepare_representations_concepts(self, loader):
        cs, c_preds, zs = [], [], []
        for x, y, c in loader:
            with torch.no_grad():
                x, y, c = self._prepare_inputs(x, y, c)
                output = self.model(x, c)
                cs.extend(c.cpu())
                c_preds.extend(output['c_preds'].cpu())
                zs.extend(output['z'].cpu())
        return torch.stack(cs), torch.stack(c_preds), torch.stack(zs)
    

    def _prepare_interventions(self):
        cs, c_preds, zs = self._prepare_representations_concepts(self.train_loader)
        c_preds_groups = [[] for _ in range(len(self.concepts_groups))]
        for i, g in enumerate(self.concepts_groups):
            for c in g:
                c_preds_groups[i].extend(c_preds[:,self.concepts.index(c)])
        c_preds_groups = [torch.stack(g) for g in c_preds_groups]
        groups_conf = [torch.abs(0.5-i).mean() for i in c_preds_groups]
        self.groups_sorted = sorted(range(len(groups_conf)), key=lambda i: groups_conf[i], reverse=False)
        self.model.prepare_interventions(zs, cs)


    #==========Evaluation==========
    def _mask_concepts(self, c, n_int_groups, iter, random_mask=False):
        if random_mask:
            random.seed(self.seed+iter)
            int_groups = random.sample(range(0, len(self.concepts_groups)), n_int_groups)
        else:
            int_groups = self.groups_sorted[:n_int_groups]
        int_concepts = [concept for i, group in enumerate(self.concepts_groups) \
            for concept in group if i in int_groups]
        int_concepts_idxs = [self.concepts.index(c) for c in int_concepts]
        c_mod = torch.empty_like(c).fill_(torch.nan)
        c_mod[:,int_concepts_idxs] = c[:,int_concepts_idxs]
        return c_mod

    def _prepare_labels_preds(self, loader, n_int_groups, iter):
        labels_preds = {'y': [],  'y_preds': []}
        for x, y, c in loader:
            with torch.no_grad():
                x, y, c = self._prepare_inputs(x, y, c)
                c = self._mask_concepts(c, n_int_groups, iter)
                output_inter = self.model.intervene(x, c)
                labels_preds['y'].extend(y.cpu())
                labels_preds['y_preds'].extend(output_inter['y_preds'].cpu())
        return {k: torch.stack(v) for k, v in labels_preds.items()}
    
    def evaluate(self, n_int_groups, iter=0):
        test_set = self._prepare_labels_preds(self.test_loader, n_int_groups, iter)
        return 100-calc_accuracy(test_set['y_preds'], test_set['y'])

    #==========Run==========
    def run(self):
        self.eval()
        self._prepare_interventions()
        n_iters = 1
        for n_int_groups in range(0, len(self.concepts_groups), 1):
            with torch.no_grad():
                error = self.evaluate(n_int_groups) if n_int_groups==0 else \
                    sum([self.evaluate(n_int_groups, i) for i in range(n_iters)]) / n_iters
                print(error)
                self.wandb_run.log({'error': error})