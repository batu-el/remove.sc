import torch
from transformers import BertModel

### Featurizer ###
class Featurizer(torch.nn.Module):
    def __init__(self, last_layer_dropout):
        super().__init__()
        self.featurizer = BertModel.from_pretrained('bert-base-uncased')
        self.last_layer_dropout = torch.nn.Dropout(last_layer_dropout)
    def forward(self, x):
        kwargs = {
            'input_ids': x[:, :, 0],
            'attention_mask': x[:, :, 1],
            'token_type_ids':  x[:, :, 2]
            }
        x = self.featurizer(**kwargs)
        x = self.last_layer_dropout(x.pooler_output)
        return x

### Classifier ###
class Classifier(torch.nn.Module):
    def __init__(self, cls_in_features, num_classes):
        super().__init__()
        self.classifier = torch.nn.Linear(cls_in_features, num_classes)
    def forward(self, x):
        x = self.classifier(x)
        return x