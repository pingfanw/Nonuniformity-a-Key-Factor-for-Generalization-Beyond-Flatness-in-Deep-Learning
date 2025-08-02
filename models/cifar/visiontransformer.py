import torch
import torch.nn as nn
import timm
__all__ = [
    'vit_small',
    'vit_base',
    'vit_large',
    'vit_huge'
]

def vit_small(pretrained=False, num_classes=100, **kwargs):
    model = timm.create_model('vit_small_patch16_224', pretrained=pretrained, **kwargs)
    model.head = nn.Linear(model.head.in_features, num_classes)
    return model

def vit_base(pretrained=False, num_classes=100, **kwargs):
    model = timm.create_model('vit_tiny_patch16_224', pretrained=pretrained, **kwargs)
    model.head = nn.Linear(model.head.in_features, num_classes)
    return model
def vit_large(pretrained=False, num_classes=100, **kwargs):
    model = timm.create_model('vit_tiny_patch16_224', pretrained=pretrained, **kwargs)
    model.head = nn.Linear(model.head.in_features, num_classes)
    return model
def vit_huge(pretrained=False, num_classes=100, **kwargs):
    model = timm.create_model('vit_tiny_patch16_224', pretrained=pretrained, **kwargs)
    model.head = nn.Linear(model.head.in_features, num_classes)
    return model