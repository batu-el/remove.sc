
### Based on https://github.com/YyzHarry/SubpopBench ###

import torch
from reprshift.models.networks import Featurizer, Classifier
from reprshift.learning.optimization import get_bert_optim
from transformers import get_scheduler

class Algorithm(torch.nn.Module):
    def __init__(self, num_classes, num_attributes, hparams):
        super(Algorithm, self).__init__()
        self.hparams = hparams
        self.num_classes = num_classes
        self.num_attributes = num_attributes
    def _init_model(self):
        raise NotImplementedError
    def _compute_loss(self, i, x, y, a, step):
        raise NotImplementedError
    def update(self, minibatch, step):
        """Perform one update step."""
        raise NotImplementedError
    def return_feats(self, x):
        raise NotImplementedError
    def predict(self, x):
        raise NotImplementedError

### Algorithm 1: ERM ###
class ERM(Algorithm):
    """Empirical Risk Minimization (ERM)"""
    def __init__(self, num_classes, num_attributes, hparams):
        super(ERM, self).__init__(num_classes, num_attributes, hparams)
        last_layer_dropout = hparams['last_layer_dropout']
        self.featurizer = Featurizer(last_layer_dropout)
        cls_in_features = self.featurizer.featurizer.config.hidden_size # 768
        self.classifier = Classifier(cls_in_features, num_classes)
        self.network = torch.nn.Sequential(self.featurizer, self.classifier)
        self._init_model(lr=hparams['lr'], weight_decay=hparams['weight_decay'], num_warmup_steps=hparams['num_warmup_steps'], num_training_steps=hparams['num_training_steps'])
    def _init_model(self, lr, weight_decay, num_warmup_steps, num_training_steps):
        self.clip_grad = True
        self.network.zero_grad()
        self.optimizer = get_bert_optim(self.network, lr, weight_decay)
        self.lr_scheduler = get_scheduler("linear",optimizer=self.optimizer,num_warmup_steps=num_warmup_steps,num_training_steps=num_training_steps)
        self.loss = torch.nn.CrossEntropyLoss(reduction="none")
    def _compute_loss(self, i, x, y, a, step):
        return self.loss(self.predict(x), y).mean()
    def update(self, minibatch, step):
        all_i, all_x, all_y, all_a = minibatch
        loss = self._compute_loss(all_i, all_x, all_y, all_a, step)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
        self.optimizer.step()
        self.lr_scheduler.step()
        self.network.zero_grad()
        return {'loss': loss.item()}
    def return_feats(self, x):
        return self.featurizer(x)
    def predict(self, x):
        return self.network(x)


### Algorithm 3: GroupDRO ###
class GroupDRO(ERM):
    """
    Group DRO minimizes the error at the worst group [https://arxiv.org/pdf/1911.08731.pdf]
    """
    def __init__(self, num_classes, num_attributes, hparams):
        super(GroupDRO, self).__init__(num_classes, num_attributes, hparams)
        self.register_buffer("q", torch.ones(self.num_classes * self.num_attributes).cuda())
    def return_groups(self, y, a):
        """Given a list of (y, a) tuples, return indexes of samples belonging to each subgroup"""
        idx_g, idx_samples = [], []
        all_g = y * self.num_attributes + a
        for g in all_g.unique():
            idx_g.append(g)
            idx_samples.append(all_g == g)
        return zip(idx_g, idx_samples)
    def _compute_loss(self, i, x, y, a, step):
        losses = self.loss(self.predict(x), y)
        for idx_g, idx_samples in self.return_groups(y, a):
            self.q[idx_g] *= (self.hparams["groupdro_eta"] * losses[idx_samples].mean()).exp().item()
        self.q /= self.q.sum()
        loss_value = 0
        for idx_g, idx_samples in self.return_groups(y, a):
            loss_value += self.q[idx_g] * losses[idx_samples].mean()
        return loss_value

### Algorithm 15: Focal ###
class Focal(ERM):
    """Focal loss, https://arxiv.org/abs/1708.02002"""
    def __init__(self, num_classes, num_attributes, hparams):
        super(Focal, self).__init__(num_classes, num_attributes, hparams)

    @staticmethod
    def focal_loss(input_values, gamma):
        p = torch.exp(-input_values)
        loss = (1 - p) ** gamma * input_values
        return loss.mean()

    def _compute_loss(self, i, x, y, a, step):
        return self.focal_loss(self.loss(self.predict(x), y), self.hparams["gamma"])
    
