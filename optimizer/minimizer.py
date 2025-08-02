import torch
from collections import defaultdict
import copy
class SAM:
    def __init__(self, optimizer, model, rho=0.1, eta=0.01):
        self.optimizer = optimizer
        self.model = model
        self.rho = rho
        self.eta = eta
        self.state = defaultdict(dict)
    @torch.no_grad()
    def ascent_step(self):
        grads = []
        for name, param in self.model.named_parameters():
            if param.grad is None:
                continue
            grads.append(torch.norm(param.grad, p=2))
        grad_norm = torch.norm(torch.stack(grads), p=2).item() + 1.e-16     
        for name, param in self.model.named_parameters():
            if param.grad is None:
                continue
            eps = self.state[param].get("eps")
            if eps is None:                 
                eps = torch.clone(param).detach()
                self.state[param]["eps"] = eps
            eps[...] = param.grad[...]     
            eps.mul_(self.rho / grad_norm)   
            param.data.add_(eps)             
        self.optimizer.zero_grad()
    @torch.no_grad()
    def descent_step(self):
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            p.sub_(self.state[p]["eps"])
        self.optimizer.step()
        self.optimizer.zero_grad()

class ASAM(SAM):
    @torch.no_grad()
    def ascent_step(self):
        wgrads = []
        for name, param in self.model.named_parameters():
            if param.grad is None:
                continue
            t_w = self.state[param].get("eps")
            if t_w is None:
                t_w = torch.clone(param).detach()
                self.state[param]["eps"] = t_w
            if 'weight' in name:
                t_w[...] = param[...]
                t_w.abs_().add_(self.eta)
                param.grad.mul_(t_w) 
            wgrads.append(torch.norm(param.grad, p=2))
        wgrad_norm = torch.norm(torch.stack(wgrads), p=2).item() + 1.e-16
        del wgrads
        for name, param in self.model.named_parameters():
            if param.grad is None:
                continue
            t_w = self.state[param].get("eps")
            if 'weight' in name:
                param.grad.mul_(t_w)
            eps = t_w   
            eps[...] = param.grad[...]
            eps.mul_(self.rho / wgrad_norm)
            param.data.add_(eps)
        self.optimizer.zero_grad()

class ACSAM_Identity(SAM):
    """
    modified versio without FIM
    """
    def __init__(self, optimizer, model, first_rho=0.1, second_rho=0.1, eta=0.01, consistent_momentum=0.9):
        self.optimizer = optimizer
        self.model = model
        self.first_rho = first_rho
        self.second_rho = second_rho
        self.eta = eta
        self.state = defaultdict(dict)
        self.alpha = consistent_momentum
        self.momentum = None
    @torch.no_grad()
    def first_ascent_step(self):
        grad = []
        grad_norm_buffer = []
        for name, param in self.model.named_parameters():
            if param.grad is None:
                continue
            grad_norm_buffer.append(torch.norm(param.grad, p=2))
            grad.append(param.grad)     # buffer for the original gradient
        grad_norm = torch.norm(torch.stack(grad_norm_buffer), p=2).item() + 1.e-16     
        for name, param in self.model.named_parameters():
            if param.grad is None:
                continue
            eps = self.state[param].get("eps")
            if eps is None:
                eps = torch.clone(param).detach()
                self.state[param]["eps"] = eps
            eps[...] = param.grad[...]     
            eps.mul_(self.first_rho / grad_norm)   
            param.data.add_(eps)     # \theta + eps
        self.optimizer.zero_grad()
        return grad
    @torch.no_grad()
    def second_ascent_step(self, original_grad):

        three_terms = []            # necessary for the three terms norm
        three_terms_buffer = []     # buffer for the three terms norm

        if self.momentum == None:
            self.momentum = copy.deepcopy(original_grad)
            initial_index = 0
            for name, param in self.model.named_parameters():
                self.momentum[initial_index] = torch.zeros_like(original_grad[initial_index])
                initial_index += 1

        for name, param in self.model.named_parameters():
            if param.grad is None:
                continue
            param.data.sub_(self.state[param]["eps"])
        
        index = 0
        for name, param in self.model.named_parameters():
            if param.grad is None:
                continue
            # t_w = self.state[param].get("eps")
            self.momentum[index] = self.alpha * self.momentum[index] + (1-self.alpha) * (param.grad - original_grad[index])      # accumulated self.momentum of CSAM
            three_terms.append(param.grad - original_grad[index] - self.momentum[index])        # three terms minus
            t_w = self.state[param].get("eps")
            if t_w is None:
                t_w = torch.clone(param).detach()
                self.state[param]["eps"] = t_w
            if 'weight' in name:
                t_w[...] = param[...]
                t_w.abs_().add_(self.eta)
                three_terms[index].mul_(t_w) 
            three_terms_buffer.append(torch.norm(three_terms[index], p=2))          # buffer for the three terms norm
            param.grad = three_terms[index]         # replacing the gradient of SAM disturbed with the three terms minus
            index += 1
        three_terms_norm = torch.norm(torch.stack(three_terms_buffer), p=2).item() + 1.e-16     # three terms TOTAL norm

        index = 0
        for name, param in self.model.named_parameters():
            if param.grad is None:
                continue
            t_w = self.state[param].get("eps")
            t_w = self.state[param].get("eps")
            if 'weight' in name:
                    param.grad.mul_(t_w)
            eps = t_w   
            eps[...] = param.grad[...]
            eps.mul_(self.second_rho / three_terms_norm)
            param.data.add_(eps)
            index += 1
        self.optimizer.zero_grad()
    @torch.no_grad()
    def descent_step(self):
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            p.sub_(self.state[p]["eps"])
        self.optimizer.step()
        self.optimizer.zero_grad()

class GAM(SAM):
    @torch.no_grad()
    def ascent_step(self):
        # compute the fvp w.r.t to the parameters
        fvp_eps = []
        fvp_list = []
        gradT_list = []
        all_bias = True
        for m in self.optimizer.modules:
            if self.optimizer.steps % self.optimizer.TInv ==0:
                self.optimizer._update_inv(m)
            param_grad_mat = self.optimizer._get_matrix_form_grad(m, m.__class__.__name__)
            gradT_list.append(param_grad_mat.view(1,-1))
            fvp_buffer = self.optimizer._get_natural_grad(m, param_grad_mat)
            fvp_eps.append((fvp_buffer[0]))
            fvp_buffer[0] = fvp_buffer[0].view(m.weight.grad.data.size(0), -1)
            if m.bias is not None:
                fvp_eps.append((fvp_buffer[1]))
                fvp_buffer[1] = fvp_buffer[1].view(m.bias.grad.data.size(0), 1)
                fvp_list.append(torch.cat(fvp_buffer,dim=1).view(-1, 1))
            else:
                fvp_list.append(fvp_buffer[0].view(-1, 1))
                all_bias = False
        num_modules = len(gradT_list)
        if all_bias == True:
            num_modules *= 2
        gradT = torch.cat(gradT_list,dim=1)
        fvp = torch.cat(fvp_list,dim=0)
        fvp_factor = gradT @ fvp
        fvp_norm = torch.sqrt_(fvp_factor).item() + 1.e-16
        if 1.0-fvp_factor < 0:
            scaling_factor = torch.sqrt_(1.0-torch.exp_(-fvp_factor)).item() + 1.e-16           # sqrt(1-x)
        else:
            scaling_factor = torch.sqrt_(1.0-fvp_factor).item() + 1.e-16
        index = 0
        for m in self.optimizer.modules:
            for param in m.parameters():
                if param.grad is None and index >= num_modules:
                    continue
                eps = self.state[param].get("eps")
                if eps is None:
                    eps = torch.clone(param).detach()
                    self.state[param]["eps"] = eps
                eps[...] = fvp_eps[index][...]
                eps.mul_(self.rho/(fvp_norm*scaling_factor))
                param.data.add_(eps)
                index += 1
        self.optimizer.zero_grad()
    @torch.no_grad()
    def descent_step(self):
        for m in self.optimizer.modules:
            for param in m.parameters():
                if param.grad is None:
                    continue
                param.data.sub_(self.state[param]["eps"])
        self.optimizer.step()
        self.optimizer.zero_grad()

class CSAM(SAM):    
    # global momentum
    # momentum = None
    def __init__(self, optimizer, model, first_rho=0.1, second_rho=0.1, eta=0.01, consistent_momentum=0.9):
        self.optimizer = optimizer
        self.model = model
        self.first_rho = first_rho
        self.second_rho = second_rho
        self.eta = eta
        self.state = defaultdict(dict)
        self.alpha = consistent_momentum
        self.momentum = None
    @torch.no_grad()
    def first_ascent_step(self):
        """ 
        1.store the original gradient and return it (g2)
        2.do the epsilon disturb of SAM 
        """
        grad = []
        grad_norm_buffer = []
        for m in self.optimizer.modules:
            for param in m.parameters():
                if param.grad is None:
                    continue
                grad_norm_buffer.append(torch.norm(param.grad, p=2))
                grad.append(param.grad)     # buffer for the original gradient
        grad_norm = torch.norm(torch.stack(grad_norm_buffer), p=2).item() + 1.e-16     
        for m in self.optimizer.modules:
            for param in m.parameters():
                if param.grad is None:
                    continue
                eps = self.state[param].get("eps")
                if eps is None:
                    eps = torch.clone(param).detach()
                    self.state[param]["eps"] = eps
                eps[...] = param.grad[...]     
                eps.mul_(self.first_rho / grad_norm)   
                param.data.add_(eps)     # \theta + eps
        self.optimizer.zero_grad()
        return grad
    @torch.no_grad()
    def second_ascent_step(self, original_grad):
        """
        1.minus the epsilon disturb of SAM during the first ascent step
        2.compute (g1t-g2t-mt), g1t=original_grad, g2t=param_grad, mt=lambda*m(t-1)
        """
        # compute the fvp w.r.t to the parameters
        fvp_eps = []
        fvp_list = []
        gradT_list = []
        all_bias = True
        for m in self.optimizer.modules:
            for param in m.parameters():
                if param.grad is None:
                    continue
                param.data.sub_(self.state[param]["eps"])
 
        index = 0
        for m in self.optimizer.modules:
            for param in m.parameters():
                if param.grad is None:
                    continue
                if self.momentum == None:
                    self.momentum = copy.deepcopy(original_grad)
                    initial_index = 0
                    for name, param in self.model.named_parameters():
                        self.momentum[index] =torch.zeros_like(original_grad[initial_index])
                        initial_index += 1
                self.momentum[index] = self.alpha * self.momentum[index] + (1-self.alpha) * (param.grad - original_grad[index])      # accumulated self.momentum of CSAM
                param.grad.sub_(original_grad[index] - self.momentum[index])            # replacing the gradient of SAM with the three terms minus
                index += 1

        for m in self.optimizer.modules:
            if self.optimizer.steps % self.optimizer.TInv ==0:
                self.optimizer._update_inv(m)
            param_grad_mat = self.optimizer._get_matrix_form_grad(m, m.__class__.__name__)
            gradT_list.append(param_grad_mat.view(1,-1))
            fvp_buffer = self.optimizer._get_natural_grad(m, param_grad_mat)
            fvp_eps.append((fvp_buffer[0]))
            fvp_buffer[0] = fvp_buffer[0].view(m.weight.grad.data.size(0), -1)
            if m.bias is not None:
                fvp_eps.append((fvp_buffer[1]))
                fvp_buffer[1] = fvp_buffer[1].view(m.bias.grad.data.size(0), 1)
                fvp_list.append(torch.cat(fvp_buffer,dim=1).view(-1, 1))
            else:
                fvp_list.append(fvp_buffer[0].view(-1, 1))
                all_bias = False
        num_modules = len(gradT_list)
        if all_bias == True:
            num_modules *= 2
        gradT = torch.cat(gradT_list,dim=1)
        fvp = torch.cat(fvp_list,dim=0)
        fvp_norm = torch.sqrt_(gradT @ fvp).item() + 1.e-16
        index = 0
        for m in self.optimizer.modules:
            for param in m.parameters():
                if param.grad is None and index >= num_modules:
                    continue
                eps = self.state[param].get("eps")
                if eps is None:
                    eps = torch.clone(param).detach()
                    self.state[param]["eps"] = eps
                eps[...] = fvp_eps[index][...]
                eps.mul_(self.second_rho/(fvp_norm))
                param.data.add_(eps)
                index += 1
        self.optimizer.zero_grad()
    @torch.no_grad()
    def descent_step(self):
        for m in self.optimizer.modules:
            for param in m.parameters():
                if param.grad is None:
                    continue
                param.data.sub_(self.state[param]["eps"])
        self.optimizer.step()
        self.optimizer.zero_grad()

class CSAM_Identity(SAM):
    """
    modified versio without FIM
    """
    def __init__(self, optimizer, model, first_rho=0.1, second_rho=0.1, eta=0.01, consistent_momentum=0.9):
        self.optimizer = optimizer
        self.model = model
        self.first_rho = first_rho
        self.second_rho = second_rho
        self.eta = eta
        self.state = defaultdict(dict)
        self.alpha = consistent_momentum
        self.momentum = None
    @torch.no_grad()
    def first_ascent_step(self):
        grad = []
        grad_norm_buffer = []
        for name, param in self.model.named_parameters():
            if param.grad is None:
                continue
            grad_norm_buffer.append(torch.norm(param.grad, p=2))
            grad.append(param.grad)     # buffer for the original gradient
        grad_norm = torch.norm(torch.stack(grad_norm_buffer), p=2).item() + 1.e-16   
        del grad_norm_buffer  
        for name, param in self.model.named_parameters():
            if param.grad is None:
                continue
            eps = self.state[param].get("eps")
            if eps is None:
                eps = torch.clone(param).detach()
                self.state[param]["eps"] = eps
            eps[...] = param.grad[...]     
            eps.mul_(self.first_rho / grad_norm)   
            param.data.add_(eps)     # \theta + eps
        self.optimizer.zero_grad()
        return grad
    @torch.no_grad()
    def second_ascent_step(self, original_grad):
        three_terms = []            # necessary for the three terms norm
        three_terms_buffer = []     # buffer for the three terms norm

        if self.momentum == None:
            self.momentum = copy.deepcopy(original_grad)
            initial_index = 0
            for name, param in self.model.named_parameters():
                self.momentum[initial_index] = torch.zeros_like(original_grad[initial_index])
                initial_index += 1
        # if self.momentum is None:
        #     self.momentum = [torch.zeros_like(g) for g in original_grad]

        for name, param in self.model.named_parameters():
            if param.grad is None:
                continue
            param.data.sub_(self.state[param]["eps"])
        index = 0
        for name, param in self.model.named_parameters():
            if param.grad is None:
                continue
            self.momentum[index] = self.alpha * self.momentum[index] + (1-self.alpha) * (param.grad - original_grad[index])      # accumulated self.momentum of CSAM
            three_terms.append(param.grad - original_grad[index] - self.momentum[index])        # three terms minus
            three_terms_buffer.append(torch.norm(three_terms[index], p=2))          # buffer for the three terms norm
            param.grad = three_terms[index]         # replacing the gradient of SAM disturbed with the three terms minus
            index += 1
        three_terms_norm = torch.norm(torch.stack(three_terms_buffer), p=2).item() + 1.e-16     # three terms TOTAL norm
        del three_terms, three_terms_buffer, original_grad
        for name, param in self.model.named_parameters():
            if param.grad is None:
                continue
            eps = self.state[param].get("eps")
            if eps is None:
                eps = torch.clone(param).detach()
                self.state[param]["eps"] = eps
            eps[...] = param.grad[...]     
            eps.mul_(self.second_rho / three_terms_norm)   
            param.data.add_(eps)     # \theta + eps
        self.optimizer.zero_grad()
    @torch.no_grad()
    def descent_step(self):
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            p.sub_(self.state[p]["eps"])
        self.optimizer.step()
        self.optimizer.zero_grad()