import torch
import torch.nn as nn
import torch.nn.functional as F

def get_param_norm(model,device='cpu'):
    f_norm = torch.tensor([0.0],dtype=torch.float32).to(device)
    for param in model.parameters():
        f_norm.add(torch.sum(param.square()))
    return f_norm.sqrt()


def get_fisher_trace(model,batch_size,device='cpu'):
    fisher_trace = torch.tensor([0.0],dtype=torch.float32).to(device)
    for param in model.parameters():
        fisher_trace.add(torch.sum(param.grad.square()))   
    return (fisher_trace.sqrt()) / batch_size

def try_contiguous(x):
    if not x.is_contiguous():
        x = x.contiguous()
    return x


def _extract_patches(x, kernel_size, stride, padding):
    if padding[0] + padding[1] > 0:         
        x = F.pad(x, (padding[1], padding[1], padding[0],
                      padding[0])).data       
    x = x.unfold(2, kernel_size[0], stride[0])          
    x = x.unfold(3, kernel_size[1], stride[1])          
    x = x.transpose_(1, 2).transpose_(2, 3).contiguous()            
    x = x.view(
        x.size(0), x.size(1), x.size(2),
        x.size(3) * x.size(4) * x.size(5))      
    return x

def update_running_stat(aa, m_aa, stat_decay):
    m_aa *= stat_decay / (1 - stat_decay)
    m_aa += aa
    m_aa *= (1 - stat_decay)

class ComputeCovA:
    
    @classmethod
    def compute_cov_a(cls, a, layer):
        return cls.__call__(a, layer)
    
    @classmethod
    def __call__(cls, a, layer):
        if isinstance(layer, nn.Conv2d):
            cov_a = cls.cova_conv2d(a, layer)
        elif isinstance(layer, nn.Linear):
            cov_a = cls.cova_linear(a, layer)
        else:
            cov_a = None
        return cov_a

    @staticmethod
    def cova_conv2d(a, layer):
        batch_size = a.size(0)
        a = _extract_patches(a, layer.kernel_size, layer.stride, layer.padding)     
        spatial_size = a.size(1) * a.size(2)       
        a = a.view(-1, a.size(-1))      
        if layer.bias is not None:              
            a = torch.cat([a, a.new(a.size(0), 1).fill_(1)], 1)
        a = a/spatial_size
        return a.t() @ (a / batch_size)     

    @staticmethod
    def cova_linear(a, layer):
        batch_size = a.size(0)
        if layer.bias is not None:              
            a = torch.cat([a, a.new(a.size(0), 1).fill_(1)], dim=1)     
        return a.t() @ (a / batch_size)       


class ComputeCovG:

    @classmethod
    def compute_cov_g(cls, g, layer, batch_averaged=False):
        return cls.__call__(g, layer, batch_averaged)

    @classmethod
    def __call__(cls, g, layer, batch_averaged):
        if isinstance(layer, nn.Conv2d):
            cov_g = cls.covg_conv2d(g, layer, batch_averaged)
        elif isinstance(layer, nn.Linear):
            cov_g = cls.covg_linear(g, layer, batch_averaged)
        else:
            cov_g = None
        return cov_g

    @staticmethod
    def covg_conv2d(g, layer, batch_averaged):
        spatial_size = g.size(2) * g.size(3)
        batch_size = g.shape[0]            
        g = g.transpose(1, 2).transpose(2, 3)       
        g = try_contiguous(g)   
        g = g.view(-1, g.size(-1))    

        if batch_averaged:
            g = g * batch_size    
        g = g * spatial_size
        cov_g = g.t() @ (g / g.size(0))
        return cov_g

    @staticmethod
    def covg_linear(g, layer, batch_averaged):
        batch_size = g.size(0)
        if batch_averaged:
            cov_g = g.t() @ (g * batch_size)      
        else:
            cov_g = g.t() @ (g / batch_size)       
        return cov_g

