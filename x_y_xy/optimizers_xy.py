import torch
import torch.autograd as autograd

def zero_grad(params):
    for p in params:
        if p.grad is not None:
            p.grad.detach()
            p.grad.zero_()

# All updates with respect to player 1
# For player 2, we will initialize the class as linear_linear(theta2, theta1, lr2, lr1) and pass (eta2, eta1) to the step functions below

class linear_linear(object):
    def __init__(self, theta1, theta2, lr1, lr2, device=torch.device('cpu'), weight_decay=0, collect_info=True, exp=None, id=None):
        self.theta1 = list(theta1)
        self.theta2 = list(theta2)
        self.lr1 = lr1
        self.lr2 = lr2
        self.device = device
        self.weight_decay = weight_decay
        self.collect_info = collect_info
        self.factor = exp
        self.id = id

    def zero_grad(self):
        zero_grad(self.theta1)

    def get_info(self):
      if self.collect_info:
          return self.norms
      else:
          raise ValueError('No update information stored. Set collect_info=True before call this method')

    def update_parameters(self, delta):
        # parameter update
        index = 0
        for p in self.theta1:
            if self.weight_decay != 0:
                p.data.add_(- self.weight_decay * p)
            p.data.add_(delta[index: index + p.numel()].reshape(p.shape))
            index += p.numel()
        if index != delta.numel():
            raise ValueError('CG size mismatch')

    def level0_step(self, eta1, eta2):
        self.norms = []
        all_terms = []
        # level 0 terms
        n1_eta1 = autograd.grad(eta1, self.theta1)
        n1_eta1 = torch.cat([g.contiguous().view(-1, 1) for g in n1_eta1])
        all_terms.append(self.lr1 * n1_eta1)

        # parameter update
        self.update_parameters(sum(all_terms))

        if self.collect_info:
            for j in range(len(all_terms)):
                norm = torch.norm(all_terms[j], p=2)
                self.norms.append(norm)

    def level1_step(self, eta1, eta2):
        self.norms = []
        all_terms = []
        # level 0 terms
        n1_eta1 = autograd.grad(eta1, self.theta1, retain_graph=True) ### not switching on retain_graph here throws an error in the next line. Why? - because eta1 is used again
        n1_eta1 = torch.cat([g.contiguous().view(-1, 1) for g in n1_eta1])
        all_terms.append(self.factor*self.lr1 * n1_eta1)

        # level 1 terms
        n2_eta1 = autograd.grad(eta1, self.theta2, retain_graph=True)
        n2_eta1 = torch.cat([g.contiguous().view(-1, 1) for g in n2_eta1])
        n2_eta2 = autograd.grad(eta2, self.theta2, create_graph=True, retain_graph=True)
        n2_eta2 = torch.cat([g.contiguous().view(-1, 1) for g in n2_eta2])
        n12_eta2_n2_eta1 = autograd.grad(n2_eta2, self.theta1, grad_outputs=n2_eta1)
        n12_eta2_n2_eta1 = torch.cat([g.contiguous().view(-1, 1) for g in n12_eta2_n2_eta1])
        all_terms.append(self.lr1*self.lr2 * n12_eta2_n2_eta1)

        # parameter update
        self.update_parameters(sum(all_terms))

        if self.collect_info:
            for j in range(len(all_terms)):
                norm = torch.norm(all_terms[j], p=2)
                self.norms.append(norm)

    def level2_step(self, eta1, eta2):
        self.norms = []
        all_terms = []
        # level 0 terms
        n1_eta1 = autograd.grad(eta1, self.theta1, create_graph=True, retain_graph=True)
        n1_eta1 = torch.cat([g.contiguous().view(-1, 1) for g in n1_eta1])
        all_terms.append(self.factor*self.lr1 * n1_eta1)

        # level 1 terms
        n2_eta1 = autograd.grad(eta1, self.theta2, create_graph=True, retain_graph=True)
        n2_eta1 = torch.cat([g.contiguous().view(-1, 1) for g in n2_eta1])
        n2_eta2 = autograd.grad(eta2, self.theta2, create_graph=True, retain_graph=True)
        n2_eta2 = torch.cat([g.contiguous().view(-1, 1) for g in n2_eta2])
        n12_eta2_n2_eta1 = autograd.grad(n2_eta2, self.theta1, grad_outputs=n2_eta1, retain_graph=True)
        n12_eta2_n2_eta1 = torch.cat([g.contiguous().view(-1, 1) for g in n12_eta2_n2_eta1])
        all_terms.append(self.factor*self.lr1*self.lr2 * n12_eta2_n2_eta1)

        # parameter update
        self.update_parameters(sum(all_terms))

        if self.collect_info:
            for j in range(len(all_terms)):
                norm = torch.norm(all_terms[j], p=2)
                self.norms.append(norm)

    def level3_step(self, eta1, eta2):
        self.norms = []
        all_terms = []
        # level 0 terms
        n1_eta1 = autograd.grad(eta1, self.theta1, create_graph=True, retain_graph=True)
        n1_eta1 = torch.cat([g.contiguous().view(-1, 1) for g in n1_eta1])
        all_terms.append(self.factor*self.lr1 * n1_eta1)

        # level 1 terms
        n2_eta1 = autograd.grad(eta1, self.theta2, create_graph=True, retain_graph=True)
        n2_eta1 = torch.cat([g.contiguous().view(-1, 1) for g in n2_eta1])
        n2_eta2 = autograd.grad(eta2, self.theta2, create_graph=True, retain_graph=True)
        n2_eta2 = torch.cat([g.contiguous().view(-1, 1) for g in n2_eta2])
        n12_eta2_n2_eta1 = autograd.grad(n2_eta2, self.theta1, grad_outputs=n2_eta1, retain_graph=True)
        n12_eta2_n2_eta1 = torch.cat([g.contiguous().view(-1, 1) for g in n12_eta2_n2_eta1])
        all_terms.append(self.factor*self.lr1*self.lr2 * n12_eta2_n2_eta1)

        # parameter update
        self.update_parameters(sum(all_terms))

        if self.collect_info:
            for j in range(len(all_terms)):
                norm = torch.norm(all_terms[j], p=2)
                self.norms.append(norm)

class bilinear_bilinear(object):
    def __init__(self, theta1, theta2, lr1, lr2, device=torch.device('cpu'), weight_decay=0, collect_info=True, exp=None):
        self.theta1 = list(theta1)
        self.theta2 = list(theta2)
        self.lr1 = lr1
        self.lr2 = lr2
        self.device = device
        self.weight_decay = weight_decay
        self.collect_info = collect_info
        self.factor = exp

    def zero_grad(self):
        zero_grad(self.theta1)

    def get_info(self):
      if self.collect_info:
          return self.norms
      else:
          raise ValueError('No update information stored. Set collect_info=True before call this method')

    def update_parameters(self, delta):
        # parameter update
        index = 0
        for p in self.theta1:
            if self.weight_decay != 0:
                p.data.add_(- self.weight_decay * p)
            p.data.add_(delta[index: index + p.numel()].reshape(p.shape))
            index += p.numel()
        if index != delta.numel():
            raise ValueError('CG size mismatch')

    def level0_step(self, eta1, eta2):
        self.norms = []
        all_terms = []
        # level 0 terms
        n1_eta1 = autograd.grad(eta1, self.theta1)
        n1_eta1 = torch.cat([g.contiguous().view(-1, 1) for g in n1_eta1])
        all_terms.append(self.lr1 * n1_eta1)

        # parameter update
        self.update_parameters(sum(all_terms))

        if self.collect_info:
            for j in range(len(all_terms)):
                norm = torch.norm(all_terms[j], p=2)
                self.norms.append(norm)

    def level1_step(self, eta1, eta2):
        self.norms = []
        all_terms = []
        # level 0 terms
        n1_eta1 = autograd.grad(eta1, self.theta1, retain_graph=True) ### not switching on retain_graph here throws an error in the next line. Why? - because eta1 is used again
        n1_eta1 = torch.cat([g.contiguous().view(-1, 1) for g in n1_eta1])
        all_terms.append(self.factor*self.lr1 * n1_eta1)

        # level 1 terms
        n2_eta1 = autograd.grad(eta1, self.theta2, create_graph=True, retain_graph=True)
        n2_eta1 = torch.cat([g.contiguous().view(-1, 1) for g in n2_eta1])
        n2_eta2 = autograd.grad(eta2, self.theta2, create_graph=True, retain_graph=True)
        n2_eta2 = torch.cat([g.contiguous().view(-1, 1) for g in n2_eta2])
        n12_eta2_n2_eta1 = autograd.grad(n2_eta2, self.theta1, grad_outputs=n2_eta1)
        n12_eta2_n2_eta1 = torch.cat([g.contiguous().view(-1, 1) for g in n12_eta2_n2_eta1])
        all_terms.append(self.lr1*self.lr2 * n12_eta2_n2_eta1)

        n12_eta1_n2_eta2 = autograd.grad(n2_eta1, self.theta1, grad_outputs=n2_eta2)
        n12_eta1_n2_eta2 = torch.cat([g.contiguous().view(-1, 1) for g in n12_eta1_n2_eta2])
        all_terms.append(self.lr1*self.lr2 * n12_eta1_n2_eta2)

        # parameter update
        self.update_parameters(sum(all_terms))

        if self.collect_info:
            for j in range(len(all_terms)):
                norm = torch.norm(all_terms[j], p=2)
                self.norms.append(norm)

    def level2_step(self, eta1, eta2):
        self.norms = []
        all_terms = []
        # level 0 terms
        n1_eta1 = autograd.grad(eta1, self.theta1, create_graph=True, retain_graph=True)
        n1_eta1_ = torch.cat([g.contiguous().view(-1, 1) for g in n1_eta1])
        all_terms.append(self.factor*self.lr1 * n1_eta1_)

        # level 1 terms
        n2_eta1 = autograd.grad(eta1, self.theta2, create_graph=True, retain_graph=True)
        n2_eta1 = torch.cat([g.contiguous().view(-1, 1) for g in n2_eta1])
        n2_eta2 = autograd.grad(eta2, self.theta2, create_graph=True, retain_graph=True)
        n2_eta2 = torch.cat([g.contiguous().view(-1, 1) for g in n2_eta2])
        n12_eta2_n2_eta1 = autograd.grad(n2_eta2, self.theta1, grad_outputs=n2_eta1, retain_graph=True)
        all_terms.append(self.factor*self.lr1*self.lr2 * torch.cat([g.contiguous().view(-1, 1) for g in n12_eta2_n2_eta1]))

        n12_eta1_n2_eta2 = autograd.grad(n2_eta1, self.theta1, grad_outputs=n2_eta2, retain_graph=True)
        all_terms.append(self.factor*self.lr1*self.lr2 * torch.cat([g.contiguous().view(-1, 1) for g in n12_eta1_n2_eta2]))

        # level 2 terms

        n1_eta2 = autograd.grad(eta2, self.theta1, create_graph=True, retain_graph=True)
        n1_eta2 = torch.cat([g.contiguous().view(-1, 1) for g in n1_eta2])
        n21_eta1_n1_eta2 = autograd.grad(n1_eta1_, self.theta2, grad_outputs=n1_eta2, retain_graph=True)
        n21_eta1_n1_eta2 = torch.cat([g.contiguous().view(-1, 1) for g in n21_eta1_n1_eta2])
        # l2 term 1
        n12_eta1_n21_eta1_n1_eta2 = autograd.grad(n2_eta1, self.theta1, grad_outputs=n21_eta1_n1_eta2, retain_graph=True)
        all_terms.append(self.lr1*self.lr2*self.lr1 * torch.cat([g.contiguous().view(-1, 1) for g in n12_eta1_n21_eta1_n1_eta2]))

        n21_eta2_n1_eta1 = autograd.grad(n1_eta2, self.theta2, grad_outputs=n1_eta1_, retain_graph=True)
        n21_eta2_n1_eta1 = torch.cat([g.contiguous().view(-1, 1) for g in n21_eta2_n1_eta1])
        # l2 term 2
        n12_eta1_n21_eta2_n1_eta1 = autograd.grad(n2_eta1, self.theta1, grad_outputs=n21_eta2_n1_eta1, retain_graph=True)
        all_terms.append(self.lr1*self.lr2*self.lr1 * torch.cat([g.contiguous().view(-1, 1) for g in n12_eta1_n21_eta2_n1_eta1]))

        # parameter update
        self.update_parameters(sum(all_terms))

        if self.collect_info:
            for j in range(len(all_terms)):
                norm = torch.norm(all_terms[j], p=2)
                self.norms.append(norm)

    def level3_step(self, eta1, eta2):
        self.norms = []
        all_terms = []
        # level 0 terms
        n1_eta1 = autograd.grad(eta1, self.theta1, create_graph=True, retain_graph=True)
        n1_eta1_ = torch.cat([g.contiguous().view(-1, 1) for g in n1_eta1])
        all_terms.append(self.factor*self.lr1 * n1_eta1_)

        # level 1 terms
        n2_eta1 = autograd.grad(eta1, self.theta2, create_graph=True, retain_graph=True)
        n2_eta1 = torch.cat([g.contiguous().view(-1, 1) for g in n2_eta1])
        n2_eta2 = autograd.grad(eta2, self.theta2, create_graph=True, retain_graph=True)
        n2_eta2 = torch.cat([g.contiguous().view(-1, 1) for g in n2_eta2])
        n12_eta2_n2_eta1 = autograd.grad(n2_eta2, self.theta1, grad_outputs=n2_eta1, retain_graph=True)
        all_terms.append(self.factor*self.lr1*self.lr2 * torch.cat([g.contiguous().view(-1, 1) for g in n12_eta2_n2_eta1]))

        n12_eta1_n2_eta2 = autograd.grad(n2_eta1, self.theta1, grad_outputs=n2_eta2, retain_graph=True)
        all_terms.append(self.factor*self.lr1*self.lr2 * torch.cat([g.contiguous().view(-1, 1) for g in n12_eta1_n2_eta2]))

        # level 2 terms

        n1_eta2 = autograd.grad(eta2, self.theta1, create_graph=True, retain_graph=True)
        n1_eta2 = torch.cat([g.contiguous().view(-1, 1) for g in n1_eta2])
        n21_eta1_n1_eta2 = autograd.grad(n1_eta1_, self.theta2, grad_outputs=n1_eta2, retain_graph=True)
        n21_eta1_n1_eta2 = torch.cat([g.contiguous().view(-1, 1) for g in n21_eta1_n1_eta2])
        # l2 term 1
        n12_eta1_n21_eta1_n1_eta2 = autograd.grad(n2_eta1, self.theta1, grad_outputs=n21_eta1_n1_eta2, retain_graph=True)
        all_terms.append(self.factor*self.lr1*self.lr2*self.lr1 * torch.cat([g.contiguous().view(-1, 1) for g in n12_eta1_n21_eta1_n1_eta2]))

        n21_eta2_n1_eta1 = autograd.grad(n1_eta2, self.theta2, grad_outputs=n1_eta1_, retain_graph=True)
        n21_eta2_n1_eta1 = torch.cat([g.contiguous().view(-1, 1) for g in n21_eta2_n1_eta1])
        # l2 term 2
        n12_eta1_n21_eta2_n1_eta1 = autograd.grad(n2_eta1, self.theta1, grad_outputs=n21_eta2_n1_eta1, retain_graph=True)
        all_terms.append(self.factor*self.lr1*self.lr2*self.lr1 * torch.cat([g.contiguous().view(-1, 1) for g in n12_eta1_n21_eta2_n1_eta1]))

        # level 3 terms
        x = 3*n2_eta1 - n2_eta2
        all_terms.append(self.lr1*self.lr2*self.lr1*self.lr2 * x)
        
        # parameter update
        self.update_parameters(sum(all_terms))

        if self.collect_info:
            for j in range(len(all_terms)):
                norm = torch.norm(all_terms[j], p=2)
                self.norms.append(norm)

class bilinear_linear(object):
    def __init__(self, theta1, theta2, lr1, lr2, device=torch.device('cpu'), weight_decay=0, collect_info=True, exp=None):
        self.theta1 = list(theta1)
        self.theta2 = list(theta2)
        self.lr1 = lr1
        self.lr2 = lr2
        self.device = device
        self.weight_decay = weight_decay
        self.collect_info = collect_info
        self.factor = exp

    def zero_grad(self):
        zero_grad(self.theta1)

    def get_info(self):
      if self.collect_info:
          return self.norms
      else:
          raise ValueError('No update information stored. Set collect_info=True before call this method')

    def update_parameters(self, delta):
        # parameter update
        index = 0
        for p in self.theta1:
            if self.weight_decay != 0:
                p.data.add_(- self.weight_decay * p)
            p.data.add_(delta[index: index + p.numel()].reshape(p.shape))
            index += p.numel()
        if index != delta.numel():
            raise ValueError('CG size mismatch')

    def level0_step(self, eta1, eta2):
        self.norms = []
        all_terms = []
        # level 0 terms
        n1_eta1 = autograd.grad(eta1, self.theta1)
        n1_eta1 = torch.cat([g.contiguous().view(-1, 1) for g in n1_eta1])
        all_terms.append(self.lr1 * n1_eta1)

        # parameter update
        self.update_parameters(sum(all_terms))

        if self.collect_info:
            for j in range(len(all_terms)):
                norm = torch.norm(all_terms[j], p=2)
                self.norms.append(norm)

    def level1_step(self, eta1, eta2):
        self.norms = []
        all_terms = []
        # level 0 terms
        n1_eta1 = autograd.grad(eta1, self.theta1, retain_graph=True) ### not switching on retain_graph here throws an error in the next line. Why? - because eta1 is used again
        n1_eta1 = torch.cat([g.contiguous().view(-1, 1) for g in n1_eta1])
        all_terms.append(self.factor*self.lr1 * n1_eta1)

        # level 1 terms
        n2_eta1 = autograd.grad(eta1, self.theta2, create_graph=True, retain_graph=True)
        n2_eta1 = torch.cat([g.contiguous().view(-1, 1) for g in n2_eta1])
        n2_eta2 = autograd.grad(eta2, self.theta2, create_graph=True, retain_graph=True)
        n2_eta2 = torch.cat([g.contiguous().view(-1, 1) for g in n2_eta2])
        n12_eta2_n2_eta1 = autograd.grad(n2_eta2, self.theta1, grad_outputs=n2_eta1)
        n12_eta2_n2_eta1 = torch.cat([g.contiguous().view(-1, 1) for g in n12_eta2_n2_eta1])
        all_terms.append(self.lr1*self.lr2 * n12_eta2_n2_eta1)

        n12_eta1_n2_eta2 = autograd.grad(n2_eta1, self.theta1, grad_outputs=n2_eta2)
        n12_eta1_n2_eta2 = torch.cat([g.contiguous().view(-1, 1) for g in n12_eta1_n2_eta2])
        all_terms.append(self.lr1*self.lr2 * n12_eta1_n2_eta2)

        # parameter update
        self.update_parameters(sum(all_terms))

        if self.collect_info:
            for j in range(len(all_terms)):
                norm = torch.norm(all_terms[j], p=2)
                self.norms.append(norm)

    def level2_step(self, eta1, eta2):
        self.norms = []
        all_terms = []
        # level 0 terms
        n1_eta1 = autograd.grad(eta1, self.theta1, create_graph=True, retain_graph=True)
        n1_eta1_ = torch.cat([g.contiguous().view(-1, 1) for g in n1_eta1])
        all_terms.append(self.factor*self.lr1 * n1_eta1_)

        # level 1 terms
        n2_eta1 = autograd.grad(eta1, self.theta2, create_graph=True, retain_graph=True)
        n2_eta1 = torch.cat([g.contiguous().view(-1, 1) for g in n2_eta1])
        n2_eta2 = autograd.grad(eta2, self.theta2, create_graph=True, retain_graph=True)
        n2_eta2 = torch.cat([g.contiguous().view(-1, 1) for g in n2_eta2])
        n12_eta2_n2_eta1 = autograd.grad(n2_eta2, self.theta1, grad_outputs=n2_eta1, retain_graph=True)
        all_terms.append(self.factor*self.lr1*self.lr2 * torch.cat([g.contiguous().view(-1, 1) for g in n12_eta2_n2_eta1]))

        n12_eta1_n2_eta2 = autograd.grad(n2_eta1, self.theta1, grad_outputs=n2_eta2, retain_graph=True)
        all_terms.append(self.factor*self.lr1*self.lr2 * torch.cat([g.contiguous().view(-1, 1) for g in n12_eta1_n2_eta2]))

        # level 2 terms

        n1_eta2 = autograd.grad(eta2, self.theta1, create_graph=True, retain_graph=True)
        n1_eta2 = torch.cat([g.contiguous().view(-1, 1) for g in n1_eta2])
        n21_eta1_n1_eta2 = autograd.grad(n1_eta1_, self.theta2, grad_outputs=n1_eta2, retain_graph=True)
        n21_eta1_n1_eta2 = torch.cat([g.contiguous().view(-1, 1) for g in n21_eta1_n1_eta2])
        # l2 term 2
        n12_eta1_n21_eta1_n1_eta2 = autograd.grad(n2_eta1, self.theta1, grad_outputs=n21_eta1_n1_eta2, retain_graph=True)
        all_terms.append(self.lr1*self.lr2*self.lr1 * torch.cat([g.contiguous().view(-1, 1) for g in n12_eta1_n21_eta1_n1_eta2]))

        # parameter update
        self.update_parameters(sum(all_terms))

        if self.collect_info:
            for j in range(len(all_terms)):
                norm = torch.norm(all_terms[j], p=2)
                self.norms.append(norm)

    def level3_step(self, eta1, eta2):
        self.norms = []
        all_terms = []
        # level 0 terms
        n1_eta1 = autograd.grad(eta1, self.theta1, create_graph=True, retain_graph=True)
        n1_eta1_ = torch.cat([g.contiguous().view(-1, 1) for g in n1_eta1])
        all_terms.append(self.factor*self.lr1 * n1_eta1_)

        # level 1 terms
        n2_eta1 = autograd.grad(eta1, self.theta2, create_graph=True, retain_graph=True)
        n2_eta1 = torch.cat([g.contiguous().view(-1, 1) for g in n2_eta1])
        n2_eta2 = autograd.grad(eta2, self.theta2, create_graph=True, retain_graph=True)
        n2_eta2 = torch.cat([g.contiguous().view(-1, 1) for g in n2_eta2])
        n12_eta2_n2_eta1 = autograd.grad(n2_eta2, self.theta1, grad_outputs=n2_eta1, retain_graph=True)
        all_terms.append(self.factor*self.lr1*self.lr2 * torch.cat([g.contiguous().view(-1, 1) for g in n12_eta2_n2_eta1]))

        n12_eta1_n2_eta2 = autograd.grad(n2_eta1, self.theta1, grad_outputs=n2_eta2, retain_graph=True)
        all_terms.append(self.factor*self.lr1*self.lr2 * torch.cat([g.contiguous().view(-1, 1) for g in n12_eta1_n2_eta2]))

        # level 2 terms

        n1_eta2 = autograd.grad(eta2, self.theta1, create_graph=True, retain_graph=True)
        n1_eta2 = torch.cat([g.contiguous().view(-1, 1) for g in n1_eta2])
        n21_eta1_n1_eta2 = autograd.grad(n1_eta1_, self.theta2, grad_outputs=n1_eta2, retain_graph=True)
        n21_eta1_n1_eta2 = torch.cat([g.contiguous().view(-1, 1) for g in n21_eta1_n1_eta2])
        # l2 term 2
        n12_eta1_n21_eta1_n1_eta2 = autograd.grad(n2_eta1, self.theta1, grad_outputs=n21_eta1_n1_eta2, retain_graph=True)
        all_terms.append(self.factor*self.lr1*self.lr2*self.lr1 * torch.cat([g.contiguous().view(-1, 1) for g in n12_eta1_n21_eta1_n1_eta2]))
        
        # parameter update
        self.update_parameters(sum(all_terms))

        if self.collect_info:
            for j in range(len(all_terms)):
                norm = torch.norm(all_terms[j], p=2)
                self.norms.append(norm)

class linear_bilinear(object):
    def __init__(self, theta1, theta2, lr1, lr2, device=torch.device('cpu'), weight_decay=0, collect_info=True, exp=None):
        self.theta1 = list(theta1)
        self.theta2 = list(theta2)
        self.lr1 = lr1
        self.lr2 = lr2
        self.device = device
        self.weight_decay = weight_decay
        self.collect_info = collect_info
        self.factor = exp

    def zero_grad(self):
        zero_grad(self.theta1)

    def get_info(self):
      if self.collect_info:
          return self.norms
      else:
          raise ValueError('No update information stored. Set collect_info=True before call this method')

    def update_parameters(self, delta):
        # parameter update
        index = 0
        for p in self.theta1:
            if self.weight_decay != 0:
                p.data.add_(- self.weight_decay * p)
            p.data.add_(delta[index: index + p.numel()].reshape(p.shape))
            index += p.numel()
        if index != delta.numel():
            raise ValueError('CG size mismatch')

    def level0_step(self, eta1, eta2):
        self.norms = []
        all_terms = []
        # level 0 terms
        n1_eta1 = autograd.grad(eta1, self.theta1)
        n1_eta1 = torch.cat([g.contiguous().view(-1, 1) for g in n1_eta1])
        all_terms.append(self.lr1 * n1_eta1)

        # parameter update
        self.update_parameters(sum(all_terms))

        if self.collect_info:
            for j in range(len(all_terms)):
                norm = torch.norm(all_terms[j], p=2)
                self.norms.append(norm)

    def level1_step(self, eta1, eta2):
        self.norms = []
        all_terms = []
        # level 0 terms
        n1_eta1 = autograd.grad(eta1, self.theta1, retain_graph=True) ### not switching on retain_graph here throws an error in the next line. Why? - because eta1 is used again
        n1_eta1 = torch.cat([g.contiguous().view(-1, 1) for g in n1_eta1])
        all_terms.append(self.factor*self.lr1 * n1_eta1)

        # level 1 terms
        n2_eta1 = autograd.grad(eta1, self.theta2, create_graph=True, retain_graph=True)
        n2_eta1 = torch.cat([g.contiguous().view(-1, 1) for g in n2_eta1])
        n2_eta2 = autograd.grad(eta2, self.theta2, create_graph=True, retain_graph=True)
        n2_eta2 = torch.cat([g.contiguous().view(-1, 1) for g in n2_eta2])
        n12_eta2_n2_eta1 = autograd.grad(n2_eta2, self.theta1, grad_outputs=n2_eta1)
        n12_eta2_n2_eta1 = torch.cat([g.contiguous().view(-1, 1) for g in n12_eta2_n2_eta1])
        all_terms.append(self.lr1*self.lr2 * n12_eta2_n2_eta1)

        # parameter update
        self.update_parameters(sum(all_terms))

        if self.collect_info:
            for j in range(len(all_terms)):
                norm = torch.norm(all_terms[j], p=2)
                self.norms.append(norm)

    def level2_step(self, eta1, eta2):
        self.norms = []
        all_terms = []
        # level 0 terms
        n1_eta1 = autograd.grad(eta1, self.theta1, create_graph=True, retain_graph=True)
        n1_eta1_ = torch.cat([g.contiguous().view(-1, 1) for g in n1_eta1])
        all_terms.append(self.factor*self.lr1 * n1_eta1_)

        # level 1 terms
        n2_eta1 = autograd.grad(eta1, self.theta2, create_graph=True, retain_graph=True)
        n2_eta1 = torch.cat([g.contiguous().view(-1, 1) for g in n2_eta1])
        n2_eta2 = autograd.grad(eta2, self.theta2, create_graph=True, retain_graph=True)
        n2_eta2 = torch.cat([g.contiguous().view(-1, 1) for g in n2_eta2])
        n12_eta2_n2_eta1 = autograd.grad(n2_eta2, self.theta1, grad_outputs=n2_eta1, retain_graph=True)
        all_terms.append(self.factor*self.lr1*self.lr2 * torch.cat([g.contiguous().view(-1, 1) for g in n12_eta2_n2_eta1]))

        # parameter update
        self.update_parameters(sum(all_terms))

        if self.collect_info:
            for j in range(len(all_terms)):
                norm = torch.norm(all_terms[j], p=2)
                self.norms.append(norm)

    def level3_step(self, eta1, eta2):
        self.norms = []
        all_terms = []
        # level 0 terms
        n1_eta1 = autograd.grad(eta1, self.theta1, create_graph=True, retain_graph=True)
        n1_eta1_ = torch.cat([g.contiguous().view(-1, 1) for g in n1_eta1])
        all_terms.append(self.factor*self.lr1 * n1_eta1_)

        # level 1 terms
        n2_eta1 = autograd.grad(eta1, self.theta2, create_graph=True, retain_graph=True)
        n2_eta1 = torch.cat([g.contiguous().view(-1, 1) for g in n2_eta1])
        n2_eta2 = autograd.grad(eta2, self.theta2, create_graph=True, retain_graph=True)
        n2_eta2 = torch.cat([g.contiguous().view(-1, 1) for g in n2_eta2])
        n12_eta2_n2_eta1 = autograd.grad(n2_eta2, self.theta1, grad_outputs=n2_eta1, retain_graph=True)
        all_terms.append(self.factor*self.lr1*self.lr2 * torch.cat([g.contiguous().view(-1, 1) for g in n12_eta2_n2_eta1]))

        # level 3 terms
        x = n2_eta1
        all_terms.append(self.lr1*self.lr2*self.lr1*self.lr2 * x)
        
        # parameter update
        self.update_parameters(sum(all_terms))

        if self.collect_info:
            for j in range(len(all_terms)):
                norm = torch.norm(all_terms[j], p=2)
                self.norms.append(norm)