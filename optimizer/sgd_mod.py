import torch
import torch.optim as optim
import torch.nn as nn
import torch.linalg as LA
from utils.gam_utils import ComputeCovA, ComputeCovG, update_running_stat
import math
import torch.nn.functional as F

class SGD(optim.Optimizer):
    def __init__(self,
                 model,
                 minimizer,
                 lr=0.001,
                 momentum=0.9,
                 weight_decay=0.01,
                 stat_decay=0.95,
                 damping=0.001,
                 TCov=5,
                 TInv=5,
                 batch_averaged=True):
        # legitimation check
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, momentum=momentum, damping=damping)

        # TODO (CW): SGD optimizer now only support model as input
        super(SGD, self).__init__(model.parameters(), defaults)

        self.CovAHandler = ComputeCovA()
        """ compute the covariance of the activation """

        self.CovGHandler = ComputeCovG()
        """ compute the covariance of the gradient """

        self.batch_averaged = batch_averaged
        """ bool markers for whether the gradient is batch averaged """
        
        self.known_modules = {'Linear', 'Conv2d'}
        """ dictionary for modules: {Linear,Conv2d} """

        self.modules = []
        """ list for saving modules temporarily """

        self.grad_outputs = {}
        """ buffer for saving the gradient output """

        self.model = model
        self.minimizer = minimizer
        print("=>We use ",self.minimizer)
        self.weight_decay = weight_decay
        if self.minimizer == 'GAM' or self.minimizer == 'CSAM':   # if u change the name of minimzer class, here must be changed
            self._prepare_model()           
            self.weight_decay = 0.0
        self.steps = 0

        self.m_aa, self.m_gg = {}, {}
        """ buffer for saving the running estimates of the covariance of the activation and gradient """

        self.Q_a, self.Q_g = {}, {}
        """ buffer for saving the eigenvectors of the covariance of the activation and gradient """

        self.d_a, self.d_g = {}, {}
        """ buffer for saving the eigenvalues of the covariance of the activation and gradient """

        self.stat_decay = stat_decay
        """ parameter determines the time scale for the moving average """

        self.TCov = TCov
        """ the period for computing the covariance of the activation and gradient """

        self.TInv = TInv
        """ the period for updating the inverse of the covariance """

        # self.acc_stats = True

    def _save_input(self, module, input):
        if self.acc_stats and torch.is_grad_enabled() and self.steps % self.TCov == 0:
            aa = self.CovAHandler(input[0].data, module)   
            if self.steps == 0:     
                self.m_aa[module] = torch.diag(aa.new(aa.size(0)).fill_(1))         
            update_running_stat(aa, self.m_aa[module], self.stat_decay)            


    def _save_grad_output(self, module, grad_input, grad_output):
        if self.acc_stats and self.steps % self.TCov == 0:
            gg = self.CovGHandler(grad_output[0].data, module, self.batch_averaged)     
            if self.steps == 0:
                self.m_gg[module] = torch.diag(gg.new(gg.size(0)).fill_(1))     
            update_running_stat(gg, self.m_gg[module], self.stat_decay)         
    

    def _prepare_model(self):
        count = 0
        
        print("=> We keep following layers in GAM. ")

        for module in self.model.modules():         
            classname = module.__class__.__name__      
            if classname in self.known_modules:    
                self.modules.append(module)         
                module.register_forward_pre_hook(self._save_input)  
                module.register_full_backward_hook(self._save_grad_output)
        
                print('=>(%s): %s' % (count, module))     
        
                count += 1


    def _update_inv(self, m):
        eps = 1e-10  
        self.d_a[m], self.Q_a[m] = LA.eigh(
            self.m_aa[m], UPLO='L')
        self.d_g[m], self.Q_g[m] = LA.eigh(
            self.m_gg[m], UPLO='U')
        self.d_a[m].mul_((self.d_a[m] > eps).float())
        self.d_g[m].mul_((self.d_g[m] > eps).float())


    @staticmethod
    def _get_matrix_form_grad(m, classname):
        """
        :param m: the mth layer
        :param classname: the class name of the layer
        :return: a matrix form of the gradient. it should be a [output_dim, input_dim] matrix.
        """
        if classname == 'Conv2d':
            param_grad_mat = m.weight.grad.data.view(m.weight.grad.data.size(0), -1) 
        else:
            param_grad_mat = m.weight.grad.data           
        if m.bias is not None:
            param_grad_mat = torch.cat([param_grad_mat, m.bias.grad.data.view(-1, 1)], 1)    
            # print("CAT param_grad_mat size:",param_grad_mat.size())
        return param_grad_mat


    def _get_natural_grad(self, m, param_grad_mat, damping=0.001):
        """
        :param m:  the mth layer
        :param param_grad_mat: the gradients in matrix form
        :return: a list of gradients w.r.t to the parameters in `m` th layer
        """
        v1 = self.Q_g[m].t() @ param_grad_mat @ self.Q_a[m]
        v2 = v1 / (self.d_g[m].unsqueeze(1) * self.d_a[m].unsqueeze(0) + damping)
        v = self.Q_g[m] @ v2 @ self.Q_a[m].t()                                      
        if m.bias is not None:
            # we always put gradient w.r.t weight in [0]
            # and w.r.t bias in [1]
            v = [v[:, :-1], v[:, -1:]]
            v[0] = v[0].view(m.weight.grad.data.size())
            v[1] = v[1].view(m.bias.grad.data.size())
        else:
            v = [v.view(m.weight.grad.data.size())]
        return v

    def _step(self, closure):
        for group in self.param_groups:
            momentum = group['momentum']
            for param in group['params']:
                if param.grad is None:
                    continue
                d_p = param.grad.data
                if self.weight_decay != 0:          # do regularization
                    d_p.add_(param.data, alpha=self.weight_decay)
                if momentum != 0:                                       # add momentum
                    param_state = self.state[param]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(param.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1)
                    d_p = buf

                param.data.add_(d_p, alpha=-group['lr'])  

    def step(self, closure=None):
        self._step(closure)
        self.steps += 1

class SGD_nonfull(SGD):
    def _prepare_model(self):
        count = 0
        print(self.model)
        print("=> We keep following layers in SGD. ")
        for module in self.model.modules():         
            classname = module.__class__.__name__      
            if classname in self.known_modules:    
                self.modules.append(module)         
                module.register_forward_pre_hook(self._save_input)  
                module.register_backward_hook(self._save_grad_output)
                print('=>(%s): %s' % (count, module))     
                count += 1

