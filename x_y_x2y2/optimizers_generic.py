"""
This contains the code for a generic optimizer (linear-linear, linear-bilinear, bilinear-linear, bilinear-bilinear) that works
for any objectives eta1 and eta2
"""

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

        # level 2 terms
        n12_eta1_n2_eta1 = autograd.grad(n2_eta1, self.theta1, grad_outputs=n2_eta1, retain_graph=True)
        n12_eta1_n2_eta1 = torch.cat([g.contiguous().view(-1, 1) for g in n12_eta1_n2_eta1])
        n1_eta2 = autograd.grad(eta2, self.theta1, create_graph=True, retain_graph=True)
        n1_eta2 = torch.cat([g.contiguous().view(-1, 1) for g in n1_eta2])
        n11_eta2_n12_eta1_n2_eta1 = autograd.grad(n1_eta2, self.theta1, grad_outputs=n12_eta1_n2_eta1, retain_graph=True)
        n11_eta2_n12_eta1_n2_eta1 = torch.cat([g.contiguous().view(-1, 1) for g in n11_eta2_n12_eta1_n2_eta1])
        all_terms.append(self.lr1*self.lr2*self.lr1 * n11_eta2_n12_eta1_n2_eta1)

        n12_eta1_n2_eta1_ = autograd.grad(n2_eta1, self.theta1, grad_outputs=n2_eta1.detach(), create_graph=True, retain_graph=True)
        n12_eta1_n2_eta1_ = torch.cat([g.contiguous().view(-1, 1) for g in n12_eta1_n2_eta1_])
        scalar = torch.dot(n1_eta2.detach().view(-1,), n12_eta1_n2_eta1_.view(-1,))
        n121_eta1_n1_eta2_n2_eta1 = autograd.grad(scalar, self.theta1)
        n121_eta1_n1_eta2_n2_eta1 = torch.cat([g.contiguous().view(-1, 1) for g in n121_eta1_n1_eta2_n2_eta1])
        all_terms.append(self.lr1*self.lr2*self.lr1 * n121_eta1_n1_eta2_n2_eta1)

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

        # level 2 terms
        n12_eta1_n2_eta1 = autograd.grad(n2_eta1, self.theta1, grad_outputs=n2_eta1, retain_graph=True)
        n12_eta1_n2_eta1 = torch.cat([g.contiguous().view(-1, 1) for g in n12_eta1_n2_eta1])
        n1_eta2 = autograd.grad(eta2, self.theta1, create_graph=True, retain_graph=True)
        n1_eta2 = torch.cat([g.contiguous().view(-1, 1) for g in n1_eta2])
        n11_eta2_n12_eta1_n2_eta1 = autograd.grad(n1_eta2, self.theta1, grad_outputs=n12_eta1_n2_eta1, retain_graph=True)
        n11_eta2_n12_eta1_n2_eta1 = torch.cat([g.contiguous().view(-1, 1) for g in n11_eta2_n12_eta1_n2_eta1])
        all_terms.append(self.factor*self.lr1*self.lr2*self.lr1 * n11_eta2_n12_eta1_n2_eta1)

        n12_eta1_n2_eta1_ = autograd.grad(n2_eta1, self.theta1, grad_outputs=n2_eta1.detach(), create_graph=True, retain_graph=True)
        n12_eta1_n2_eta1_ = torch.cat([g.contiguous().view(-1, 1) for g in n12_eta1_n2_eta1_])
        scalar = torch.dot(n1_eta2.detach().view(-1,), n12_eta1_n2_eta1_.view(-1,))
        n121_eta1_n1_eta2_n2_eta1 = autograd.grad(scalar, self.theta1)
        n121_eta1_n1_eta2_n2_eta1 = torch.cat([g.contiguous().view(-1, 1) for g in n121_eta1_n1_eta2_n2_eta1])
        all_terms.append(self.factor*self.lr1*self.lr2*self.lr1 * n121_eta1_n1_eta2_n2_eta1)

        # level 3 terms

        # left side vector
        n21_eta2_n1_eta2 = autograd.grad(n1_eta2, self.theta2, grad_outputs=n1_eta2, create_graph=True, retain_graph=True)
        n21_eta2_n1_eta2 = torch.cat([g.contiguous().view(-1, 1) for g in n21_eta2_n1_eta2])
        # hessian * right side vector
        n22_eta1_n2_eta1 = autograd.grad(n2_eta1, self.theta2, grad_outputs=n2_eta1.detach(), create_graph=True, retain_graph=True)
        n22_eta1_n2_eta1 = torch.cat([g.contiguous().view(-1, 1) for g in n22_eta1_n2_eta1])
        # left vector^T * hessian * right vector == scalar
        scalar = torch.dot(n21_eta2_n1_eta2.detach().view(-1,), n22_eta1_n2_eta1.view(-1,))
        l3_term1 = autograd.grad(scalar, self.theta1, retain_graph=True)
        l3_term1 = torch.cat([g.contiguous().view(-1, 1) for g in l3_term1])
        all_terms.append(self.lr1*self.lr2*self.lr1*self.lr2 * l3_term1)

        # right side vector
        n22_eta1_n2_eta1 = autograd.grad(n2_eta1, self.theta2, grad_outputs=n2_eta1, create_graph=True, retain_graph=True)
        n22_eta1_n2_eta1 = torch.cat([g.contiguous().view(-1, 1) for g in n22_eta1_n2_eta1])
        # hessian * right side vector
        n12_eta2_n22_eta1_n2_eta1 = autograd.grad(n2_eta2, self.theta1, grad_outputs=n22_eta1_n2_eta1.detach(), create_graph=True, retain_graph=True)
        n12_eta2_n22_eta1_n2_eta1 = torch.cat([g.contiguous().view(-1, 1) for g in n12_eta2_n22_eta1_n2_eta1])
        # left vector^T * hessian * right vector == scalar
        scalar = torch.dot(n1_eta2.detach().view(-1,), n12_eta2_n22_eta1_n2_eta1.view(-1,))
        l3_term2 = autograd.grad(scalar, self.theta1)
        l3_term2 = torch.cat([g.contiguous().view(-1, 1) for g in l3_term2])
        all_terms.append(self.lr1*self.lr2*self.lr1*self.lr2 * l3_term2)

        l3_term3 = autograd.grad(n1_eta2, self.theta1, grad_outputs=n12_eta2_n22_eta1_n2_eta1, retain_graph=True)
        l3_term3 = torch.cat([g.contiguous().view(-1, 1) for g in l3_term3])
        all_terms.append(self.lr1*self.lr2*self.lr1*self.lr2 * l3_term3)

        n21_eta2_n1_eta2_ = autograd.grad(n1_eta2, self.theta2, grad_outputs=n1_eta2.detach(), create_graph=True, retain_graph=True)
        n21_eta2_n1_eta2_ = torch.cat([g.contiguous().view(-1, 1) for g in n21_eta2_n1_eta2_])
        # left vector^T * hessian * right vector
        scalar = torch.dot(n2_eta1.detach().view(-1,), n21_eta2_n1_eta2_.view(-1,))
        # new dimension vector
        n2_scalar = autograd.grad(scalar, self.theta2, create_graph=True, retain_graph=True)
        n2_scalar = torch.cat([g.contiguous().view(-1, 1) for g in n2_scalar])
        # n12_eta1 * new dimension vector
        l3_term4 = autograd.grad(n2_eta1, self.theta1, grad_outputs=n2_scalar)
        l3_term4 = torch.cat([g.contiguous().view(-1, 1) for g in l3_term4])
        all_terms.append(self.lr1*self.lr2*self.lr1*self.lr2 * l3_term4)

        # hessian * right vector
        n22_eta2_n2_eta1 = autograd.grad(n2_eta2, self.theta2, grad_outputs=n2_eta1.detach(), create_graph=True, retain_graph=True)
        n22_eta2_n2_eta1 = torch.cat([g.contiguous().view(-1, 1) for g in n22_eta2_n2_eta1])
        # left vector^T * hessian * right vector
        scalar = torch.dot(n2_eta1.detach().view(-1,), n22_eta2_n2_eta1.view(-1,))
        # new dimension vector
        n1_scalar = autograd.grad(scalar, self.theta1)
        n1_scalar = torch.cat([g.contiguous().view(-1, 1) for g in n1_scalar])
        # n11_eta2 * new dimension vector
        l3_term5 = autograd.grad(n1_eta2, self.theta1, grad_outputs=n1_scalar, retain_graph=True)
        l3_term5 = torch.cat([g.contiguous().view(-1, 1) for g in l3_term5])
        all_terms.append(self.lr1*self.lr2*self.lr1*self.lr2 * l3_term5)
        
        scalar = torch.dot(n2_eta1.detach().view(-1,), n2_scalar.view(-1,))
        l3_term6 = autograd.grad(scalar, self.theta1)
        l3_term6 = torch.cat([g.contiguous().view(-1, 1) for g in l3_term6])
        all_terms.append(self.lr1*self.lr2*self.lr1*self.lr2 * l3_term6)

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

        n12_eta1_n2_eta1 = autograd.grad(n2_eta1, self.theta1, grad_outputs=n2_eta1, retain_graph=True)
        n12_eta1_n2_eta1 = torch.cat([g.contiguous().view(-1, 1) for g in n12_eta1_n2_eta1])
        # l2 term 3
        n11_eta2_n12_eta1_n2_eta1 = autograd.grad(n1_eta2, self.theta1, grad_outputs=n12_eta1_n2_eta1)
        all_terms.append(self.lr1*self.lr2*self.lr1 * torch.cat([g.contiguous().view(-1, 1) for g in n11_eta2_n12_eta1_n2_eta1]))

        # l2 term 4
        n11_eta1_n12_eta2_n2_eta1 = autograd.grad(n1_eta1_, self.theta1, grad_outputs=n12_eta2_n2_eta1)
        all_terms.append(self.lr1*self.lr2*self.lr1 * torch.cat([g.contiguous().view(-1, 1) for g in n11_eta1_n12_eta2_n2_eta1]))
        
        n12_eta1_n2_eta1_ = autograd.grad(n2_eta1, self.theta1, grad_outputs=n2_eta1.detach(), create_graph=True, retain_graph=True)
        n12_eta1_n2_eta1_ = torch.cat([g.contiguous().view(-1, 1) for g in n12_eta1_n2_eta1_])
        scalar = torch.dot(n1_eta2.detach().view(-1,), n12_eta1_n2_eta1_.view(-1,))
        # l2 term 5
        n121_eta1_n1_eta2_n2_eta1 = autograd.grad(scalar, self.theta1)
        all_terms.append(self.lr1*self.lr2*self.lr1 * torch.cat([g.contiguous().view(-1, 1) for g in n121_eta1_n1_eta2_n2_eta1]))

        n12_eta2_n2_eta1_ = autograd.grad(n2_eta2, self.theta1, grad_outputs=n2_eta1.detach(), create_graph=True, retain_graph=True)
        n12_eta2_n2_eta1_ = torch.cat([g.contiguous().view(-1, 1) for g in n12_eta2_n2_eta1_])
        scalar = torch.dot(n1_eta1_.detach().view(-1,), n12_eta2_n2_eta1_.view(-1,))
        # l2 term 6
        n121_eta2_n1_eta1_n2_eta1 = autograd.grad(scalar, self.theta1)
        all_terms.append(self.lr1*self.lr2*self.lr1 * torch.cat([g.contiguous().view(-1, 1) for g in n121_eta2_n1_eta1_n2_eta1]))

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

        n12_eta1_n2_eta1 = autograd.grad(n2_eta1, self.theta1, grad_outputs=n2_eta1, retain_graph=True)
        n12_eta1_n2_eta1 = torch.cat([g.contiguous().view(-1, 1) for g in n12_eta1_n2_eta1])
        # l2 term 3
        n11_eta2_n12_eta1_n2_eta1 = autograd.grad(n1_eta2, self.theta1, grad_outputs=n12_eta1_n2_eta1, retain_graph=True)
        all_terms.append(self.factor*self.lr1*self.lr2*self.lr1 * torch.cat([g.contiguous().view(-1, 1) for g in n11_eta2_n12_eta1_n2_eta1]))

        # l2 term 4
        n11_eta1_n12_eta2_n2_eta1 = autograd.grad(n1_eta1_, self.theta1, grad_outputs=n12_eta2_n2_eta1, retain_graph=True)
        all_terms.append(self.factor*self.lr1*self.lr2*self.lr1 * torch.cat([g.contiguous().view(-1, 1) for g in n11_eta1_n12_eta2_n2_eta1]))
        
        n12_eta1_n2_eta1_ = autograd.grad(n2_eta1, self.theta1, grad_outputs=n2_eta1.detach(), create_graph=True, retain_graph=True)
        n12_eta1_n2_eta1_ = torch.cat([g.contiguous().view(-1, 1) for g in n12_eta1_n2_eta1_])
        scalar = torch.dot(self.factor*n1_eta2.detach().view(-1,), n12_eta1_n2_eta1_.view(-1,))
        # l2 term 5
        n121_eta1_n1_eta2_n2_eta1 = autograd.grad(scalar, self.theta1, retain_graph=True)
        all_terms.append(self.factor*self.lr1*self.lr2*self.lr1 * torch.cat([g.contiguous().view(-1, 1) for g in n121_eta1_n1_eta2_n2_eta1]))

        n12_eta2_n2_eta1_ = autograd.grad(n2_eta2, self.theta1, grad_outputs=n2_eta1.detach(), create_graph=True, retain_graph=True)
        n12_eta2_n2_eta1_ = torch.cat([g.contiguous().view(-1, 1) for g in n12_eta2_n2_eta1_])
        scalar = torch.dot(n1_eta1_.detach().view(-1,), n12_eta2_n2_eta1_.view(-1,))
        # l2 term 6
        n121_eta2_n1_eta1_n2_eta1 = autograd.grad(scalar, self.theta1, retain_graph=True)
        all_terms.append(self.factor*self.lr1*self.lr2*self.lr1 * torch.cat([g.contiguous().view(-1, 1) for g in n121_eta2_n1_eta1_n2_eta1]))

        # level 3 terms
        
        n12_eta2_n2_eta1_ = autograd.grad(n2_eta2, self.theta1, grad_outputs=n2_eta1.detach(), create_graph=True, retain_graph=True)
        n12_eta2_n2_eta1_ = torch.cat([g.contiguous().view(-1, 1) for g in n12_eta2_n2_eta1_])
        scalar = torch.dot(n12_eta2_n2_eta1_.detach().view(-1,), n12_eta2_n2_eta1_.view(-1,))
        # l3 term 1
        n121_eta2_n12_eta2_n2_eta1_n2_eta1 = autograd.grad(scalar, self.theta1, retain_graph=True)#
        all_terms.append(self.lr1*self.lr2*self.lr1*self.lr2 * torch.cat([g.contiguous().view(-1, 1) for g in n121_eta2_n12_eta2_n2_eta1_n2_eta1]))

        n21_eta2_n12_eta2_n2_eta1 = autograd.grad(n1_eta2, self.theta2, grad_outputs=n12_eta2_n2_eta1_.detach(), create_graph=True, retain_graph=True)
        n21_eta2_n12_eta2_n2_eta1 = torch.cat([g.contiguous().view(-1, 1) for g in n21_eta2_n12_eta2_n2_eta1])
        scalar = torch.dot(n2_eta1.detach().view(-1,), n21_eta2_n12_eta2_n2_eta1.view(-1,))
        # l3 term 2
        n211_eta2_n2_eta1_n12_eta2_n2_eta1 = autograd.grad(scalar, self.theta1, retain_graph=True)#
        all_terms.append(self.lr1*self.lr2*self.lr1*self.lr2 * torch.cat([g.contiguous().view(-1, 1) for g in n211_eta2_n2_eta1_n12_eta2_n2_eta1]))

        # l3 term 3
        n12_eta1_n21_eta2_n12_eta2_n2_eta1 = autograd.grad(n2_eta1, self.theta1, grad_outputs=n21_eta2_n12_eta2_n2_eta1, create_graph=True, retain_graph=True)
        all_terms.append(self.lr1*self.lr2*self.lr1*self.lr2 * torch.cat([g.contiguous().view(-1, 1) for g in n12_eta1_n21_eta2_n12_eta2_n2_eta1]))

        # l3 term 4
        n12_eta1_n2_eta2 = torch.cat([g.contiguous().view(-1, 1) for g in n12_eta1_n2_eta2])
        scalar = torch.dot(n12_eta1_n2_eta2.detach().view(-1,), n12_eta2_n2_eta1_.view(-1,))
        n121_eta2_n12_eta1_n2_eta2_n2_eta1 = autograd.grad(scalar, self.theta1, retain_graph=True)
        all_terms.append(self.lr1*self.lr2*self.lr1*self.lr2 * torch.cat([g.contiguous().view(-1, 1) for g in n121_eta2_n12_eta1_n2_eta2_n2_eta1]))

        # l3 term 5
        n21_eta1_n2_eta2 = autograd.grad(n1_eta1, self.theta2, grad_outputs=n2_eta2.detach(), create_graph=True, retain_graph=True)
        n21_eta1_n2_eta2 = torch.cat([g.contiguous().view(-1, 1) for g in n21_eta1_n2_eta2])
        scalar = torch.dot(n12_eta2_n2_eta1_.detach().view(-1,), n21_eta1_n2_eta2.view(-1,))
        n211_eta1_n2_eta2_n12_eta2_n2_eta1 = autograd.grad(scalar, self.theta1, retain_graph=True)#
        all_terms.append(self.lr1*self.lr2*self.lr1*self.lr2 * torch.cat([g.contiguous().view(-1, 1) for g in n211_eta1_n2_eta2_n12_eta2_n2_eta1]))

        # l3 term 6
        n21_eta1_n12_eta2_n2_eta1 = autograd.grad(n1_eta1, self.theta2, grad_outputs=n12_eta2_n2_eta1_.detach(), retain_graph=True)
        n21_eta1_n12_eta2_n2_eta1 = torch.cat([g.contiguous().view(-1, 1) for g in n21_eta1_n12_eta2_n2_eta1])
        n12_eta2_n21_eta1_n12_eta2_n2_eta1  = autograd.grad(n2_eta2, self.theta1, grad_outputs=n21_eta1_n12_eta2_n2_eta1, retain_graph=True)
        all_terms.append(self.lr1*self.lr2*self.lr1*self.lr2 * torch.cat([g.contiguous().view(-1, 1) for g in n12_eta2_n21_eta1_n12_eta2_n2_eta1]))

        # l3 term 7
        # n221_eta1_n21_eta2_n1_eta2_n2_eta1
        n22_eta1_n2_eta1 = autograd.grad(n2_eta1, self.theta2, grad_outputs=n2_eta1.detach(), create_graph=True, retain_graph=True)
        n22_eta1_n2_eta1 = torch.cat([g.contiguous().view(-1, 1) for g in n22_eta1_n2_eta1])
        n21_eta2_n1_eta2 = autograd.grad(n1_eta2, self.theta2, grad_outputs=n1_eta2, create_graph=True, retain_graph=True)
        n21_eta2_n1_eta2 = torch.cat([g.contiguous().view(-1, 1) for g in n21_eta2_n1_eta2])
        scalar = torch.dot(n21_eta2_n1_eta2.detach().view(-1,), n22_eta1_n2_eta1.view(-1,))
        n221_eta1_n21_eta2_n1_eta2_n2_eta1 = autograd.grad(scalar, self.theta1, retain_graph=True)
        all_terms.append(self.lr1*self.lr2*self.lr1*self.lr2 * torch.cat([g.contiguous().view(-1, 1) for g in n221_eta1_n21_eta2_n1_eta2_n2_eta1]))

        # l3 term 8
        #n121_eta2_n1_eta2_n22_eta1_n2_eta1
        n12_eta2_n22_eta1_n2_eta1 = autograd.grad(n2_eta2, self.theta1, grad_outputs=n22_eta1_n2_eta1.detach(), create_graph=True, retain_graph=True)
        n12_eta2_n22_eta1_n2_eta1 = torch.cat([g.contiguous().view(-1, 1) for g in n12_eta2_n22_eta1_n2_eta1])
        scalar = torch.dot(n1_eta2.detach().view(-1,), n12_eta2_n22_eta1_n2_eta1.view(-1))
        n121_eta2_n1_eta2_n22_eta1_n2_eta1 = autograd.grad(scalar, self.theta1, retain_graph=True)
        all_terms.append(self.lr1*self.lr2*self.lr1*self.lr2 * torch.cat([g.contiguous().view(-1, 1) for g in n121_eta2_n1_eta2_n22_eta1_n2_eta1]))

        # l3 term 9
        #n11_eta2_n12_eta2_n22_eta1_n2_eta1
        n11_eta2_n12_eta2_n22_eta1_n2_eta1 = autograd.grad(n1_eta2, self.theta1, grad_outputs=n12_eta2_n22_eta1_n2_eta1, retain_graph=True)
        all_terms.append(self.lr1*self.lr2*self.lr1*self.lr2 * torch.cat([g.contiguous().view(-1, 1) for g in n11_eta2_n12_eta2_n22_eta1_n2_eta1]))

        # l3 term 10
        #n221_eta2_n21_eta1_n1_eta2_n2_eta1
        n22_eta2_n2_eta1 = autograd.grad(n2_eta2, self.theta2, grad_outputs=n2_eta1.detach(), create_graph=True, retain_graph=True)
        n22_eta2_n2_eta1 = torch.cat([g.contiguous().view(-1, 1) for g in n22_eta2_n2_eta1])
        scalar = torch.dot(n21_eta1_n1_eta2.detach().view(-1,), n22_eta2_n2_eta1.view(-1,))
        n221_eta2_n21_eta1_n1_eta2_n2_eta1 = autograd.grad(scalar, self.theta1, retain_graph=True)
        all_terms.append(self.lr1*self.lr2*self.lr1*self.lr2 * torch.cat([g.contiguous().view(-1, 1) for g in n221_eta2_n21_eta1_n1_eta2_n2_eta1]))

        # l3 term 11
        # n121_eta1_n1_eta2_n22_eta2_n2_eta1
        n12_eta1_n22_eta2_n2_eta1 = autograd.grad(n2_eta1, self.theta1, grad_outputs=n22_eta2_n2_eta1.detach(), create_graph=True, retain_graph=True)
        n12_eta1_n22_eta2_n2_eta1 = torch.cat([g.contiguous().view(-1, 1) for g in n12_eta1_n22_eta2_n2_eta1])
        scalar = torch.dot(n1_eta2.detach().view(-1,), n12_eta1_n22_eta2_n2_eta1.view(-1,))
        n121_eta1_n1_eta2_n22_eta2_n2_eta1 = autograd.grad(scalar, self.theta1, retain_graph=True)
        all_terms.append(self.lr1*self.lr2*self.lr1*self.lr2 * torch.cat([g.contiguous().view(-1, 1) for g in n121_eta1_n1_eta2_n22_eta2_n2_eta1]))

        # l3 term 12
        # n11_eta2_n12_eta1_n22_eta2_n2_eta1
        n11_eta2_n12_eta1_n22_eta2_n2_eta1 = autograd.grad(n1_eta2, self.theta1, grad_outputs=n12_eta1_n22_eta2_n2_eta1, retain_graph=True)
        all_terms.append(self.lr1*self.lr2*self.lr1*self.lr2 * torch.cat([g.contiguous().view(-1, 1) for g in n11_eta2_n12_eta1_n22_eta2_n2_eta1]))

        # l3 term 13
        # n212_eta2_n2_eta1_n1_eta2_n21_eta1
        n21_eta2_n1_eta2 = autograd.grad(n1_eta2, self.theta2, grad_outputs=n1_eta2.detach(), create_graph=True, retain_graph=True)
        n21_eta2_n1_eta2 = torch.cat([g.contiguous().view(-1, 1) for g in n21_eta2_n1_eta2])
        scalar = torch.dot(n2_eta1.detach().view(-1,), n21_eta2_n1_eta2.view(-1))
        q = autograd.grad(scalar, self.theta2, retain_graph=True)
        n212_eta2_n2_eta1_n1_eta2_n21_eta1 = autograd.grad(n2_eta1, self.theta1, grad_outputs=q, retain_graph=True)
        all_terms.append(self.lr1*self.lr2*self.lr1*self.lr2 * torch.cat([g.contiguous().view(-1, 1) for g in n212_eta2_n2_eta1_n1_eta2_n21_eta1]))

        # l3 term 14
        # n212_eta2_n2_eta1_n11_eta2_n2_eta1
        n22_eta2_n1_eta2 = autograd.grad(n2_eta2, self.theta2, grad_outputs=n2_eta1.detach(), create_graph=True, retain_graph=True)
        n22_eta2_n1_eta2 = torch.cat([g.contiguous().view(-1, 1) for g in n22_eta2_n1_eta2])
        scalar = torch.dot(n2_eta1.detach().view(-1,), n22_eta2_n1_eta2.view(-1))
        q = autograd.grad(scalar, self.theta1, retain_graph=True)
        n212_eta2_n2_eta1_n11_eta2_n2_eta1 = autograd.grad(n1_eta2, self.theta1, grad_outputs=q, retain_graph=True)
        all_terms.append(self.lr1*self.lr2*self.lr1*self.lr2 * torch.cat([g.contiguous().view(-1, 1) for g in n212_eta2_n2_eta1_n11_eta2_n2_eta1]))

        # l3 term 15
        # n2121_eta2_n2_eta1_n1_eta2_n2_eta1
        scalar = torch.dot(n2_eta1.detach().view(-1,), n22_eta2_n2_eta1.view(-1,))
        q = autograd.grad(scalar, self.theta1, create_graph=True, retain_graph=True)
        q = torch.cat([g.contiguous().view(-1, 1) for g in q])
        scalar = torch.dot(n1_eta2.detach().view(-1,), q.view(-1,))
        n2121_eta2_n2_eta1_n1_eta2_n2_eta1 = autograd.grad(q, self.theta1, retain_graph=True)
        all_terms.append(self.lr1*self.lr2*self.lr1*self.lr2 * torch.cat([g.contiguous().view(-1, 1) for g in n2121_eta2_n2_eta1_n1_eta2_n2_eta1]))

        # l3 term 16
        # n212_eta1_n2_eta1_n1_eta2_n21_eta2
        n21_eta1_n1_eta2 = autograd.grad(n1_eta1_, self.theta2, grad_outputs=n1_eta2.detach(), create_graph=True, retain_graph=True)
        n21_eta1_n1_eta2 = torch.cat([g.contiguous().view(-1, 1) for g in n21_eta1_n1_eta2])
        scalar = torch.dot(n2_eta1.detach().view(-1,), n21_eta1_n1_eta2.view(-1))
        q = autograd.grad(scalar, self.theta2, retain_graph=True)
        n212_eta1_n2_eta1_n1_eta2_n21_eta2 = autograd.grad(n2_eta2, self.theta1, grad_outputs=q, retain_graph=True)
        all_terms.append(self.lr1*self.lr2*self.lr1*self.lr2 * torch.cat([g.contiguous().view(-1, 1) for g in n212_eta1_n2_eta1_n1_eta2_n21_eta2]))

        # l3 term 17
        # n212_eta1_n2_eta1_n11_eta2_n2_eta2
        n22_eta1_n2_eta2 = autograd.grad(n2_eta1, self.theta2, grad_outputs=n2_eta2.detach(), create_graph=True, retain_graph=True)
        n22_eta1_n2_eta2 = torch.cat([g.contiguous().view(-1, 1) for g in n22_eta1_n2_eta2])
        scalar = torch.dot(n2_eta1.detach().view(-1,), n22_eta1_n2_eta2.view(-1))
        q = autograd.grad(scalar, self.theta1, retain_graph=True)
        n212_eta1_n2_eta1_n11_eta2_n2_eta2 = autograd.grad(n1_eta2, self.theta1, grad_outputs=q, retain_graph=True)
        all_terms.append(self.lr1*self.lr2*self.lr1*self.lr2 * torch.cat([g.contiguous().view(-1, 1) for g in n212_eta1_n2_eta1_n11_eta2_n2_eta2]))

        # l3 term 18
        # n2121_eta1_n2_eta1_n1_eta2_n2_eta2
        scalar = torch.dot(n2_eta1.detach().view(-1,), n22_eta1_n2_eta2.view(-1,))
        q = autograd.grad(scalar, self.theta1, create_graph=True, retain_graph=True)
        q = torch.cat([g.contiguous().view(-1, 1) for g in q])
        scalar = torch.dot(n1_eta2.detach().view(-1,), q.view(-1,))
        n2121_eta1_n2_eta1_n1_eta2_n2_eta2 = autograd.grad(q, self.theta1, retain_graph=True)
        all_terms.append(self.lr1*self.lr2*self.lr1*self.lr2 * torch.cat([g.contiguous().view(-1, 1) for g in n2121_eta1_n2_eta1_n1_eta2_n2_eta2]))

        # l3 term 19
        #n12_eta1_n21_eta2_n12_eta2_n2_eta1
        all_terms.append(self.lr1*self.lr2*self.lr1*self.lr2 * torch.cat([g.contiguous().view(-1, 1) for g in n12_eta1_n21_eta2_n12_eta2_n2_eta1]))

        # l3 term 20
        #n12_eta1_n21_eta2_n12_eta1_n2_eta2
        n21_eta2_n12_eta1_n2_eta2 = autograd.grad(n1_eta2, self.theta2, grad_outputs=n12_eta1_n2_eta2.detach(), create_graph=True, retain_graph=True)
        n21_eta2_n12_eta1_n2_eta2 = torch.cat([g.contiguous().view(-1, 1) for g in n21_eta2_n12_eta1_n2_eta2])
        n12_eta1_n21_eta2_n12_eta1_n2_eta2 = autograd.grad(n2_eta1, self.theta1, grad_outputs=n21_eta2_n12_eta1_n2_eta2, retain_graph=True)
        all_terms.append(self.lr1*self.lr2*self.lr1*self.lr2 * torch.cat([g.contiguous().view(-1, 1) for g in n12_eta1_n21_eta2_n12_eta1_n2_eta2]))

        # l3 term 21
        #n12_eta1_n22_eta1_n21_eta2_n1_eta2
        n22_eta1_n21_eta2_n1_eta2 = autograd.grad(n2_eta1, self.theta2, grad_outputs=n21_eta2_n1_eta2.detach(), create_graph=True, retain_graph=True)
        n22_eta1_n21_eta2_n1_eta2 = torch.cat([g.contiguous().view(-1, 1) for g in n22_eta1_n21_eta2_n1_eta2])
        n12_eta1_n22_eta1_n21_eta2_n1_eta2 = autograd.grad(n2_eta1, self.theta1, grad_outputs=n22_eta1_n21_eta2_n1_eta2, retain_graph=True)
        all_terms.append(self.lr1*self.lr2*self.lr1*self.lr2 * torch.cat([g.contiguous().view(-1, 1) for g in n12_eta1_n22_eta1_n21_eta2_n1_eta2]))

        # l3 term 22
        #n12_eta1_n22_eta2_n21_eta1_n1_eta2
        n22_eta2_n21_eta1_n1_eta2 = autograd.grad(n2_eta2, self.theta2, grad_outputs=n21_eta1_n1_eta2.detach(), create_graph=True, retain_graph=True)
        n22_eta2_n21_eta1_n1_eta2 = torch.cat([g.contiguous().view(-1, 1) for g in n22_eta2_n21_eta1_n1_eta2])
        n12_eta1_n22_eta2_n21_eta1_n1_eta2 = autograd.grad(n2_eta1, self.theta1, grad_outputs=n22_eta2_n21_eta1_n1_eta2, retain_graph=True)
        all_terms.append(self.lr1*self.lr2*self.lr1*self.lr2 * torch.cat([g.contiguous().view(-1, 1) for g in n12_eta1_n22_eta2_n21_eta1_n1_eta2]))

        # l3 term 23
        #n12_eta1_n212_eta2_n2_eta1_n1_eta2
        scalar = torch.dot(n2_eta1.detach().view(-1,), n21_eta2_n1_eta2.view(-1,))
        q = autograd.grad(scalar, self.theta2, create_graph=True, retain_graph=True)
        q = torch.cat([g.contiguous().view(-1, 1) for g in q])
        n12_eta1_n212_eta2_n2_eta1_n1_eta2 = autograd.grad(n2_eta1, self.theta1, grad_outputs=q, retain_graph=True)
        all_terms.append(self.lr1*self.lr2*self.lr1*self.lr2 * torch.cat([g.contiguous().view(-1, 1) for g in n12_eta1_n212_eta2_n2_eta1_n1_eta2]))

        # l3 term 24
        #n12_eta1_n212_eta1_n2_eta2_n1_eta2
        scalar = torch.dot(n2_eta2.detach().view(-1,), n21_eta1_n1_eta2.view(-1,))
        q = autograd.grad(scalar, self.theta2, create_graph=True, retain_graph=True)
        q = torch.cat([g.contiguous().view(-1, 1) for g in q])
        n12_eta1_n212_eta1_n2_eta2_n1_eta2 = autograd.grad(n2_eta1, self.theta1, grad_outputs=q, retain_graph=True)
        all_terms.append(self.lr1*self.lr2*self.lr1*self.lr2 * torch.cat([g.contiguous().view(-1, 1) for g in n12_eta1_n212_eta1_n2_eta2_n1_eta2]))

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

        # l2 term 1
        n1_eta2 = autograd.grad(eta2, self.theta1, create_graph=True, retain_graph=True)
        n1_eta2 = torch.cat([g.contiguous().view(-1, 1) for g in n1_eta2])
        n12_eta1_n2_eta1 = autograd.grad(n2_eta1, self.theta1, grad_outputs=n2_eta1, retain_graph=True)
        n12_eta1_n2_eta1 = torch.cat([g.contiguous().view(-1, 1) for g in n12_eta1_n2_eta1])
        n11_eta2_n12_eta1_n2_eta1 = autograd.grad(n1_eta2, self.theta1, grad_outputs=n12_eta1_n2_eta1)
        all_terms.append(self.lr1*self.lr2*self.lr1 * torch.cat([g.contiguous().view(-1, 1) for g in n11_eta2_n12_eta1_n2_eta1]))

        # l2 term 2
        n1_eta2 = autograd.grad(eta2, self.theta1, create_graph=True, retain_graph=True)
        n1_eta2 = torch.cat([g.contiguous().view(-1, 1) for g in n1_eta2])
        n21_eta1_n1_eta2 = autograd.grad(n1_eta1_, self.theta2, grad_outputs=n1_eta2, retain_graph=True)
        n21_eta1_n1_eta2 = torch.cat([g.contiguous().view(-1, 1) for g in n21_eta1_n1_eta2])
        n12_eta1_n21_eta1_n1_eta2 = autograd.grad(n2_eta1, self.theta1, grad_outputs=n21_eta1_n1_eta2, retain_graph=True)
        all_terms.append(self.lr1*self.lr2*self.lr1 * torch.cat([g.contiguous().view(-1, 1) for g in n12_eta1_n21_eta1_n1_eta2]))
        
        # l2 term 3
        n12_eta1_n2_eta1_ = autograd.grad(n2_eta1, self.theta1, grad_outputs=n2_eta1.detach(), create_graph=True, retain_graph=True)
        n12_eta1_n2_eta1_ = torch.cat([g.contiguous().view(-1, 1) for g in n12_eta1_n2_eta1_])
        scalar = torch.dot(n1_eta2.detach().view(-1,), n12_eta1_n2_eta1_.view(-1,))
        n121_eta1_n1_eta2_n2_eta1 = autograd.grad(scalar, self.theta1)
        all_terms.append(self.lr1*self.lr2*self.lr1 * torch.cat([g.contiguous().view(-1, 1) for g in n121_eta1_n1_eta2_n2_eta1]))

        # parameter update
        self.update_parameters(sum(all_terms))

        if self.collect_info:
            for j in range(len(all_terms)):
                norm = torch.norm(all_terms[j], p=2)
                self.norms.append(norm)

    def level3_step(self, eta1, eta2):
        self.norms = []
        all_terms = []
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

        # l2 term 1
        n1_eta2 = autograd.grad(eta2, self.theta1, create_graph=True, retain_graph=True)
        n1_eta2 = torch.cat([g.contiguous().view(-1, 1) for g in n1_eta2])
        n12_eta1_n2_eta1 = autograd.grad(n2_eta1, self.theta1, grad_outputs=n2_eta1, retain_graph=True)
        n12_eta1_n2_eta1 = torch.cat([g.contiguous().view(-1, 1) for g in n12_eta1_n2_eta1])
        n11_eta2_n12_eta1_n2_eta1 = autograd.grad(n1_eta2, self.theta1, grad_outputs=n12_eta1_n2_eta1)
        all_terms.append(self.factor*self.lr1*self.lr2*self.lr1 * torch.cat([g.contiguous().view(-1, 1) for g in n11_eta2_n12_eta1_n2_eta1]))

        # l2 term 2
        n1_eta2 = autograd.grad(eta2, self.theta1, create_graph=True, retain_graph=True)
        n1_eta2 = torch.cat([g.contiguous().view(-1, 1) for g in n1_eta2])
        n21_eta1_n1_eta2 = autograd.grad(n1_eta1_, self.theta2, grad_outputs=n1_eta2, retain_graph=True)
        n21_eta1_n1_eta2 = torch.cat([g.contiguous().view(-1, 1) for g in n21_eta1_n1_eta2])
        n12_eta1_n21_eta1_n1_eta2 = autograd.grad(n2_eta1, self.theta1, grad_outputs=n21_eta1_n1_eta2, retain_graph=True)
        all_terms.append(self.factor*self.lr1*self.lr2*self.lr1 * torch.cat([g.contiguous().view(-1, 1) for g in n12_eta1_n21_eta1_n1_eta2]))
        
        # l2 term 3
        n12_eta1_n2_eta1_ = autograd.grad(n2_eta1, self.theta1, grad_outputs=n2_eta1.detach(), create_graph=True, retain_graph=True)
        n12_eta1_n2_eta1_ = torch.cat([g.contiguous().view(-1, 1) for g in n12_eta1_n2_eta1_])
        scalar = torch.dot(n1_eta2.detach().view(-1,), n12_eta1_n2_eta1_.view(-1,))
        n121_eta1_n1_eta2_n2_eta1 = autograd.grad(scalar, self.theta1)
        all_terms.append(self.factor*self.lr1*self.lr2*self.lr1 * torch.cat([g.contiguous().view(-1, 1) for g in n121_eta1_n1_eta2_n2_eta1]))
        
        # level 3 terms
        
        n12_eta2_n2_eta1_ = autograd.grad(n2_eta2, self.theta1, grad_outputs=n2_eta1.detach(), create_graph=True, retain_graph=True)
        n12_eta2_n2_eta1_ = torch.cat([g.contiguous().view(-1, 1) for g in n12_eta2_n2_eta1_])
        scalar = torch.dot(n12_eta2_n2_eta1_.detach().view(-1,), n12_eta2_n2_eta1_.view(-1,))
        # l3 term 1
        n121_eta2_n12_eta2_n2_eta1_n2_eta1 = autograd.grad(scalar, self.theta1, retain_graph=True)#
        #all_terms.append(self.lr1*self.lr2*self.lr1*self.lr2 * torch.cat([g.contiguous().view(-1, 1) for g in n121_eta2_n12_eta2_n2_eta1_n2_eta1]))

        n21_eta2_n12_eta2_n2_eta1 = autograd.grad(n1_eta2, self.theta2, grad_outputs=n12_eta2_n2_eta1_.detach(), create_graph=True, retain_graph=True)
        n21_eta2_n12_eta2_n2_eta1 = torch.cat([g.contiguous().view(-1, 1) for g in n21_eta2_n12_eta2_n2_eta1])
        scalar = torch.dot(n2_eta1.detach().view(-1,), n21_eta2_n12_eta2_n2_eta1.view(-1,))
        # l3 term 2
        n211_eta2_n2_eta1_n12_eta2_n2_eta1 = autograd.grad(scalar, self.theta1, retain_graph=True)#
        #all_terms.append(self.lr1*self.lr2*self.lr1*self.lr2 * torch.cat([g.contiguous().view(-1, 1) for g in n211_eta2_n2_eta1_n12_eta2_n2_eta1]))

        # l3 term 3
        n12_eta1_n21_eta2_n12_eta2_n2_eta1 = autograd.grad(n2_eta1, self.theta1, grad_outputs=n21_eta2_n12_eta2_n2_eta1, create_graph=True, retain_graph=True)
        #all_terms.append(self.lr1*self.lr2*self.lr1*self.lr2 * torch.cat([g.contiguous().view(-1, 1) for g in n12_eta1_n21_eta2_n12_eta2_n2_eta1]))

        # l3 term 4
        n12_eta1_n2_eta2 = torch.cat([g.contiguous().view(-1, 1) for g in n12_eta1_n2_eta2])
        scalar = torch.dot(n12_eta1_n2_eta2.detach().view(-1,), n12_eta2_n2_eta1_.view(-1,))
        n121_eta2_n12_eta1_n2_eta2_n2_eta1 = autograd.grad(scalar, self.theta1, retain_graph=True)
        #all_terms.append(self.lr1*self.lr2*self.lr1*self.lr2 * torch.cat([g.contiguous().view(-1, 1) for g in n121_eta2_n12_eta1_n2_eta2_n2_eta1]))

        # l3 term 5
        n21_eta1_n2_eta2 = autograd.grad(n1_eta1, self.theta2, grad_outputs=n2_eta2.detach(), create_graph=True, retain_graph=True)
        n21_eta1_n2_eta2 = torch.cat([g.contiguous().view(-1, 1) for g in n21_eta1_n2_eta2])
        scalar = torch.dot(n12_eta2_n2_eta1_.detach().view(-1,), n21_eta1_n2_eta2.view(-1,))
        n211_eta1_n2_eta2_n12_eta2_n2_eta1 = autograd.grad(scalar, self.theta1, retain_graph=True)#
        #all_terms.append(self.lr1*self.lr2*self.lr1*self.lr2 * torch.cat([g.contiguous().view(-1, 1) for g in n211_eta1_n2_eta2_n12_eta2_n2_eta1]))

        # l3 term 6
        n21_eta1_n12_eta2_n2_eta1 = autograd.grad(n1_eta1, self.theta2, grad_outputs=n12_eta2_n2_eta1_.detach(), retain_graph=True)
        n21_eta1_n12_eta2_n2_eta1 = torch.cat([g.contiguous().view(-1, 1) for g in n21_eta1_n12_eta2_n2_eta1])
        n12_eta2_n21_eta1_n12_eta2_n2_eta1  = autograd.grad(n2_eta2, self.theta1, grad_outputs=n21_eta1_n12_eta2_n2_eta1, retain_graph=True)
        #all_terms.append(self.lr1*self.lr2*self.lr1*self.lr2 * torch.cat([g.contiguous().view(-1, 1) for g in n12_eta2_n21_eta1_n12_eta2_n2_eta1]))

        """l3 term 7"""
        # n221_eta1_n21_eta2_n1_eta2_n2_eta1
        n22_eta1_n2_eta1 = autograd.grad(n2_eta1, self.theta2, grad_outputs=n2_eta1.detach(), create_graph=True, retain_graph=True)
        n22_eta1_n2_eta1 = torch.cat([g.contiguous().view(-1, 1) for g in n22_eta1_n2_eta1])
        n21_eta2_n1_eta2 = autograd.grad(n1_eta2, self.theta2, grad_outputs=n1_eta2, create_graph=True, retain_graph=True)
        n21_eta2_n1_eta2 = torch.cat([g.contiguous().view(-1, 1) for g in n21_eta2_n1_eta2])
        scalar = torch.dot(n21_eta2_n1_eta2.detach().view(-1,), n22_eta1_n2_eta1.view(-1,))
        n221_eta1_n21_eta2_n1_eta2_n2_eta1 = autograd.grad(scalar, self.theta1, retain_graph=True)
        all_terms.append(self.lr1*self.lr2*self.lr1*self.lr2 * torch.cat([g.contiguous().view(-1, 1) for g in n221_eta1_n21_eta2_n1_eta2_n2_eta1]))

        """l3 term 8"""
        #n121_eta2_n1_eta2_n22_eta1_n2_eta1
        n12_eta2_n22_eta1_n2_eta1 = autograd.grad(n2_eta2, self.theta1, grad_outputs=n22_eta1_n2_eta1.detach(), create_graph=True, retain_graph=True)
        n12_eta2_n22_eta1_n2_eta1 = torch.cat([g.contiguous().view(-1, 1) for g in n12_eta2_n22_eta1_n2_eta1])
        scalar = torch.dot(n1_eta2.detach().view(-1,), n12_eta2_n22_eta1_n2_eta1.view(-1))
        n121_eta2_n1_eta2_n22_eta1_n2_eta1 = autograd.grad(scalar, self.theta1, retain_graph=True)
        all_terms.append(self.lr1*self.lr2*self.lr1*self.lr2 * torch.cat([g.contiguous().view(-1, 1) for g in n121_eta2_n1_eta2_n22_eta1_n2_eta1]))

        """l3 term 9"""
        #n11_eta2_n12_eta2_n22_eta1_n2_eta1
        n11_eta2_n12_eta2_n22_eta1_n2_eta1 = autograd.grad(n1_eta2, self.theta1, grad_outputs=n12_eta2_n22_eta1_n2_eta1, retain_graph=True)
        all_terms.append(self.lr1*self.lr2*self.lr1*self.lr2 * torch.cat([g.contiguous().view(-1, 1) for g in n11_eta2_n12_eta2_n22_eta1_n2_eta1]))

        """l3 term 10"""
        #n221_eta2_n21_eta1_n1_eta2_n2_eta1
        n22_eta2_n2_eta1 = autograd.grad(n2_eta2, self.theta2, grad_outputs=n2_eta1.detach(), create_graph=True, retain_graph=True)
        n22_eta2_n2_eta1 = torch.cat([g.contiguous().view(-1, 1) for g in n22_eta2_n2_eta1])
        scalar = torch.dot(n21_eta1_n1_eta2.detach().view(-1,), n22_eta2_n2_eta1.view(-1,))
        n221_eta2_n21_eta1_n1_eta2_n2_eta1 = autograd.grad(scalar, self.theta1, retain_graph=True)
        all_terms.append(self.lr1*self.lr2*self.lr1*self.lr2 * torch.cat([g.contiguous().view(-1, 1) for g in n221_eta2_n21_eta1_n1_eta2_n2_eta1]))

        """l3 term 11"""
        # n121_eta1_n1_eta2_n22_eta2_n2_eta1
        n12_eta1_n22_eta2_n2_eta1 = autograd.grad(n2_eta1, self.theta1, grad_outputs=n22_eta2_n2_eta1.detach(), create_graph=True, retain_graph=True)
        n12_eta1_n22_eta2_n2_eta1 = torch.cat([g.contiguous().view(-1, 1) for g in n12_eta1_n22_eta2_n2_eta1])
        scalar = torch.dot(n1_eta2.detach().view(-1,), n12_eta1_n22_eta2_n2_eta1.view(-1,))
        n121_eta1_n1_eta2_n22_eta2_n2_eta1 = autograd.grad(scalar, self.theta1, retain_graph=True)
        all_terms.append(self.lr1*self.lr2*self.lr1*self.lr2 * torch.cat([g.contiguous().view(-1, 1) for g in n121_eta1_n1_eta2_n22_eta2_n2_eta1]))

        """l3 term 12"""
        # n11_eta2_n12_eta1_n22_eta2_n2_eta1
        n11_eta2_n12_eta1_n22_eta2_n2_eta1 = autograd.grad(n1_eta2, self.theta1, grad_outputs=n12_eta1_n22_eta2_n2_eta1, retain_graph=True)
        all_terms.append(self.lr1*self.lr2*self.lr1*self.lr2 * torch.cat([g.contiguous().view(-1, 1) for g in n11_eta2_n12_eta1_n22_eta2_n2_eta1]))

        """l3 term 13"""
        # n212_eta2_n2_eta1_n1_eta2_n21_eta1
        n21_eta2_n1_eta2 = autograd.grad(n1_eta2, self.theta2, grad_outputs=n1_eta2.detach(), create_graph=True, retain_graph=True)
        n21_eta2_n1_eta2 = torch.cat([g.contiguous().view(-1, 1) for g in n21_eta2_n1_eta2])
        scalar = torch.dot(n2_eta1.detach().view(-1,), n21_eta2_n1_eta2.view(-1))
        q = autograd.grad(scalar, self.theta2, retain_graph=True)
        n212_eta2_n2_eta1_n1_eta2_n21_eta1 = autograd.grad(n2_eta1, self.theta1, grad_outputs=q, retain_graph=True)
        all_terms.append(self.lr1*self.lr2*self.lr1*self.lr2 * torch.cat([g.contiguous().view(-1, 1) for g in n212_eta2_n2_eta1_n1_eta2_n21_eta1]))

        """l3 term 14"""
        # n212_eta2_n2_eta1_n11_eta2_n2_eta1
        n22_eta2_n1_eta2 = autograd.grad(n2_eta2, self.theta2, grad_outputs=n2_eta1.detach(), create_graph=True, retain_graph=True)
        n22_eta2_n1_eta2 = torch.cat([g.contiguous().view(-1, 1) for g in n22_eta2_n1_eta2])
        scalar = torch.dot(n2_eta1.detach().view(-1,), n22_eta2_n1_eta2.view(-1))
        q = autograd.grad(scalar, self.theta1, retain_graph=True)
        n212_eta2_n2_eta1_n11_eta2_n2_eta1 = autograd.grad(n1_eta2, self.theta1, grad_outputs=q, retain_graph=True)
        all_terms.append(self.lr1*self.lr2*self.lr1*self.lr2 * torch.cat([g.contiguous().view(-1, 1) for g in n212_eta2_n2_eta1_n11_eta2_n2_eta1]))

        """l3 term 15"""
        # n2121_eta2_n2_eta1_n1_eta2_n2_eta1
        scalar = torch.dot(n2_eta1.detach().view(-1,), n22_eta2_n2_eta1.view(-1,))
        q = autograd.grad(scalar, self.theta1, create_graph=True, retain_graph=True)
        q = torch.cat([g.contiguous().view(-1, 1) for g in q])
        scalar = torch.dot(n1_eta2.detach().view(-1,), q.view(-1,))
        n2121_eta2_n2_eta1_n1_eta2_n2_eta1 = autograd.grad(q, self.theta1, retain_graph=True)
        all_terms.append(self.lr1*self.lr2*self.lr1*self.lr2 * torch.cat([g.contiguous().view(-1, 1) for g in n2121_eta2_n2_eta1_n1_eta2_n2_eta1]))

        """l3 term 16"""
        # n212_eta1_n2_eta1_n1_eta2_n21_eta2
        n21_eta1_n1_eta2 = autograd.grad(n1_eta1_, self.theta2, grad_outputs=n1_eta2.detach(), create_graph=True, retain_graph=True)
        n21_eta1_n1_eta2 = torch.cat([g.contiguous().view(-1, 1) for g in n21_eta1_n1_eta2])
        scalar = torch.dot(n2_eta1.detach().view(-1,), n21_eta1_n1_eta2.view(-1))
        q = autograd.grad(scalar, self.theta2, retain_graph=True)
        n212_eta1_n2_eta1_n1_eta2_n21_eta2 = autograd.grad(n2_eta2, self.theta1, grad_outputs=q, retain_graph=True)
        all_terms.append(self.lr1*self.lr2*self.lr1*self.lr2 * torch.cat([g.contiguous().view(-1, 1) for g in n212_eta1_n2_eta1_n1_eta2_n21_eta2]))

        """l3 term 17"""
        # n212_eta1_n2_eta1_n11_eta2_n2_eta2
        n22_eta1_n2_eta2 = autograd.grad(n2_eta1, self.theta2, grad_outputs=n2_eta2.detach(), create_graph=True, retain_graph=True)
        n22_eta1_n2_eta2 = torch.cat([g.contiguous().view(-1, 1) for g in n22_eta1_n2_eta2])
        scalar = torch.dot(n2_eta1.detach().view(-1,), n22_eta1_n2_eta2.view(-1))
        q = autograd.grad(scalar, self.theta1, retain_graph=True)
        n212_eta1_n2_eta1_n11_eta2_n2_eta2 = autograd.grad(n1_eta2, self.theta1, grad_outputs=q, retain_graph=True)
        all_terms.append(self.lr1*self.lr2*self.lr1*self.lr2 * torch.cat([g.contiguous().view(-1, 1) for g in n212_eta1_n2_eta1_n11_eta2_n2_eta2]))

        """l3 term 18"""
        # n2121_eta1_n2_eta1_n1_eta2_n2_eta2
        scalar = torch.dot(n2_eta1.detach().view(-1,), n22_eta1_n2_eta2.view(-1,))
        q = autograd.grad(scalar, self.theta1, create_graph=True, retain_graph=True)
        q = torch.cat([g.contiguous().view(-1, 1) for g in q])
        scalar = torch.dot(n1_eta2.detach().view(-1,), q.view(-1,))
        n2121_eta1_n2_eta1_n1_eta2_n2_eta2 = autograd.grad(q, self.theta1, retain_graph=True)
        all_terms.append(self.lr1*self.lr2*self.lr1*self.lr2 * torch.cat([g.contiguous().view(-1, 1) for g in n2121_eta1_n2_eta1_n1_eta2_n2_eta2]))

        # l3 term 19
        #n12_eta1_n21_eta2_n12_eta2_n2_eta1
        #all_terms.append(self.lr1*self.lr2*self.lr1*self.lr2 * torch.cat([g.contiguous().view(-1, 1) for g in n12_eta1_n21_eta2_n12_eta2_n2_eta1]))

        # l3 term 20
        #n12_eta1_n21_eta2_n12_eta1_n2_eta2
        n21_eta2_n12_eta1_n2_eta2 = autograd.grad(n1_eta2, self.theta2, grad_outputs=n12_eta1_n2_eta2.detach(), create_graph=True, retain_graph=True)
        n21_eta2_n12_eta1_n2_eta2 = torch.cat([g.contiguous().view(-1, 1) for g in n21_eta2_n12_eta1_n2_eta2])
        n12_eta1_n21_eta2_n12_eta1_n2_eta2 = autograd.grad(n2_eta1, self.theta1, grad_outputs=n21_eta2_n12_eta1_n2_eta2, retain_graph=True)
        #all_terms.append(self.lr1*self.lr2*self.lr1*self.lr2 * torch.cat([g.contiguous().view(-1, 1) for g in n12_eta1_n21_eta2_n12_eta1_n2_eta2]))

        """l3 term 21""" 
        #n12_eta1_n22_eta1_n21_eta2_n1_eta2
        n22_eta1_n21_eta2_n1_eta2 = autograd.grad(n2_eta1, self.theta2, grad_outputs=n21_eta2_n1_eta2.detach(), create_graph=True, retain_graph=True)
        n22_eta1_n21_eta2_n1_eta2 = torch.cat([g.contiguous().view(-1, 1) for g in n22_eta1_n21_eta2_n1_eta2])
        n12_eta1_n22_eta1_n21_eta2_n1_eta2 = autograd.grad(n2_eta1, self.theta1, grad_outputs=n22_eta1_n21_eta2_n1_eta2, retain_graph=True)
        all_terms.append(self.lr1*self.lr2*self.lr1*self.lr2 * torch.cat([g.contiguous().view(-1, 1) for g in n12_eta1_n22_eta1_n21_eta2_n1_eta2]))

        """l3 term 22"""
        #n12_eta1_n22_eta2_n21_eta1_n1_eta2
        n22_eta2_n21_eta1_n1_eta2 = autograd.grad(n2_eta2, self.theta2, grad_outputs=n21_eta1_n1_eta2.detach(), create_graph=True, retain_graph=True)
        n22_eta2_n21_eta1_n1_eta2 = torch.cat([g.contiguous().view(-1, 1) for g in n22_eta2_n21_eta1_n1_eta2])
        n12_eta1_n22_eta2_n21_eta1_n1_eta2 = autograd.grad(n2_eta1, self.theta1, grad_outputs=n22_eta2_n21_eta1_n1_eta2, retain_graph=True)
        all_terms.append(self.lr1*self.lr2*self.lr1*self.lr2 * torch.cat([g.contiguous().view(-1, 1) for g in n12_eta1_n22_eta2_n21_eta1_n1_eta2]))

        """l3 term 23"""
        #n12_eta1_n212_eta2_n2_eta1_n1_eta2
        scalar = torch.dot(n2_eta1.detach().view(-1,), n21_eta2_n1_eta2.view(-1,))
        q = autograd.grad(scalar, self.theta2, create_graph=True, retain_graph=True)
        q = torch.cat([g.contiguous().view(-1, 1) for g in q])
        n12_eta1_n212_eta2_n2_eta1_n1_eta2 = autograd.grad(n2_eta1, self.theta1, grad_outputs=q, retain_graph=True)
        all_terms.append(self.lr1*self.lr2*self.lr1*self.lr2 * torch.cat([g.contiguous().view(-1, 1) for g in n12_eta1_n212_eta2_n2_eta1_n1_eta2]))

        """l3 term 24"""
        #n12_eta1_n212_eta1_n2_eta2_n1_eta2
        scalar = torch.dot(n2_eta2.detach().view(-1,), n21_eta1_n1_eta2.view(-1,))
        q = autograd.grad(scalar, self.theta2, create_graph=True, retain_graph=True)
        q = torch.cat([g.contiguous().view(-1, 1) for g in q])
        n12_eta1_n212_eta1_n2_eta2_n1_eta2 = autograd.grad(n2_eta1, self.theta1, grad_outputs=q, retain_graph=True)
        all_terms.append(self.lr1*self.lr2*self.lr1*self.lr2 * torch.cat([g.contiguous().view(-1, 1) for g in n12_eta1_n212_eta1_n2_eta2_n1_eta2]))

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

        n12_eta1_n2_eta2 = autograd.grad(n2_eta1, self.theta1, grad_outputs=n2_eta2, retain_graph=True)
        #all_terms.append(self.factor*self.lr1*self.lr2 * torch.cat([g.contiguous().view(-1, 1) for g in n12_eta1_n2_eta2]))

        # level 2 terms

        n1_eta2 = autograd.grad(eta2, self.theta1, create_graph=True, retain_graph=True)
        n1_eta2 = torch.cat([g.contiguous().view(-1, 1) for g in n1_eta2])
        n21_eta1_n1_eta2 = autograd.grad(n1_eta1_, self.theta2, grad_outputs=n1_eta2, retain_graph=True)
        n21_eta1_n1_eta2 = torch.cat([g.contiguous().view(-1, 1) for g in n21_eta1_n1_eta2])
        # l2 term 1
        n12_eta1_n21_eta1_n1_eta2 = autograd.grad(n2_eta1, self.theta1, grad_outputs=n21_eta1_n1_eta2, retain_graph=True)
        #all_terms.append(self.lr1*self.lr2*self.lr1 * torch.cat([g.contiguous().view(-1, 1) for g in n12_eta1_n21_eta1_n1_eta2]))

        n21_eta2_n1_eta1 = autograd.grad(n1_eta2, self.theta2, grad_outputs=n1_eta1_, retain_graph=True)
        n21_eta2_n1_eta1 = torch.cat([g.contiguous().view(-1, 1) for g in n21_eta2_n1_eta1])
        # l2 term 2
        n12_eta1_n21_eta2_n1_eta1 = autograd.grad(n2_eta1, self.theta1, grad_outputs=n21_eta2_n1_eta1, retain_graph=True)
        #all_terms.append(self.lr1*self.lr2*self.lr1 * torch.cat([g.contiguous().view(-1, 1) for g in n12_eta1_n21_eta2_n1_eta1]))

        n12_eta1_n2_eta1 = autograd.grad(n2_eta1, self.theta1, grad_outputs=n2_eta1, retain_graph=True)
        n12_eta1_n2_eta1 = torch.cat([g.contiguous().view(-1, 1) for g in n12_eta1_n2_eta1])
        # l2 term 3
        n11_eta2_n12_eta1_n2_eta1 = autograd.grad(n1_eta2, self.theta1, grad_outputs=n12_eta1_n2_eta1)
        all_terms.append(self.lr1*self.lr2*self.lr1 * torch.cat([g.contiguous().view(-1, 1) for g in n11_eta2_n12_eta1_n2_eta1]))

        # l2 term 4
        n11_eta1_n12_eta2_n2_eta1 = autograd.grad(n1_eta1_, self.theta1, grad_outputs=n12_eta2_n2_eta1)
        all_terms.append(self.lr1*self.lr2*self.lr1 * torch.cat([g.contiguous().view(-1, 1) for g in n11_eta1_n12_eta2_n2_eta1]))
        
        n12_eta1_n2_eta1_ = autograd.grad(n2_eta1, self.theta1, grad_outputs=n2_eta1.detach(), create_graph=True, retain_graph=True)
        n12_eta1_n2_eta1_ = torch.cat([g.contiguous().view(-1, 1) for g in n12_eta1_n2_eta1_])
        scalar = torch.dot(n1_eta2.detach().view(-1,), n12_eta1_n2_eta1_.view(-1,))
        # l2 term 5
        n121_eta1_n1_eta2_n2_eta1 = autograd.grad(scalar, self.theta1)
        all_terms.append(self.lr1*self.lr2*self.lr1 * torch.cat([g.contiguous().view(-1, 1) for g in n121_eta1_n1_eta2_n2_eta1]))

        n12_eta2_n2_eta1_ = autograd.grad(n2_eta2, self.theta1, grad_outputs=n2_eta1.detach(), create_graph=True, retain_graph=True)
        n12_eta2_n2_eta1_ = torch.cat([g.contiguous().view(-1, 1) for g in n12_eta2_n2_eta1_])
        scalar = torch.dot(n1_eta1_.detach().view(-1,), n12_eta2_n2_eta1_.view(-1,))
        # l2 term 6
        n121_eta2_n1_eta1_n2_eta1 = autograd.grad(scalar, self.theta1)
        all_terms.append(self.lr1*self.lr2*self.lr1 * torch.cat([g.contiguous().view(-1, 1) for g in n121_eta2_n1_eta1_n2_eta1]))

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
        #all_terms.append(self.factor*self.lr1*self.lr2 * torch.cat([g.contiguous().view(-1, 1) for g in n12_eta1_n2_eta2]))

        # level 2 terms

        n1_eta2 = autograd.grad(eta2, self.theta1, create_graph=True, retain_graph=True)
        n1_eta2 = torch.cat([g.contiguous().view(-1, 1) for g in n1_eta2])
        n21_eta1_n1_eta2 = autograd.grad(n1_eta1_, self.theta2, grad_outputs=n1_eta2, retain_graph=True)
        n21_eta1_n1_eta2 = torch.cat([g.contiguous().view(-1, 1) for g in n21_eta1_n1_eta2])
        # l2 term 1
        n12_eta1_n21_eta1_n1_eta2 = autograd.grad(n2_eta1, self.theta1, grad_outputs=n21_eta1_n1_eta2, retain_graph=True)
        #all_terms.append(self.factor*self.lr1*self.lr2*self.lr1 * torch.cat([g.contiguous().view(-1, 1) for g in n12_eta1_n21_eta1_n1_eta2]))

        n21_eta2_n1_eta1 = autograd.grad(n1_eta2, self.theta2, grad_outputs=n1_eta1_, retain_graph=True)
        n21_eta2_n1_eta1 = torch.cat([g.contiguous().view(-1, 1) for g in n21_eta2_n1_eta1])
        # l2 term 2
        n12_eta1_n21_eta2_n1_eta1 = autograd.grad(n2_eta1, self.theta1, grad_outputs=n21_eta2_n1_eta1, retain_graph=True)
        #all_terms.append(self.factor*self.lr1*self.lr2*self.lr1 * torch.cat([g.contiguous().view(-1, 1) for g in n12_eta1_n21_eta2_n1_eta1]))

        n12_eta1_n2_eta1 = autograd.grad(n2_eta1, self.theta1, grad_outputs=n2_eta1, retain_graph=True)
        n12_eta1_n2_eta1 = torch.cat([g.contiguous().view(-1, 1) for g in n12_eta1_n2_eta1])
        # l2 term 3
        n11_eta2_n12_eta1_n2_eta1 = autograd.grad(n1_eta2, self.theta1, grad_outputs=n12_eta1_n2_eta1, retain_graph=True)
        all_terms.append(self.factor*self.lr1*self.lr2*self.lr1 * torch.cat([g.contiguous().view(-1, 1) for g in n11_eta2_n12_eta1_n2_eta1]))

        # l2 term 4
        n11_eta1_n12_eta2_n2_eta1 = autograd.grad(n1_eta1_, self.theta1, grad_outputs=n12_eta2_n2_eta1, retain_graph=True)
        all_terms.append(self.factor*self.lr1*self.lr2*self.lr1 * torch.cat([g.contiguous().view(-1, 1) for g in n11_eta1_n12_eta2_n2_eta1]))
        
        n12_eta1_n2_eta1_ = autograd.grad(n2_eta1, self.theta1, grad_outputs=n2_eta1.detach(), create_graph=True, retain_graph=True)
        n12_eta1_n2_eta1_ = torch.cat([g.contiguous().view(-1, 1) for g in n12_eta1_n2_eta1_])
        scalar = torch.dot(self.factor*n1_eta2.detach().view(-1,), n12_eta1_n2_eta1_.view(-1,))
        # l2 term 5
        n121_eta1_n1_eta2_n2_eta1 = autograd.grad(scalar, self.theta1, retain_graph=True)
        all_terms.append(self.factor*self.lr1*self.lr2*self.lr1 * torch.cat([g.contiguous().view(-1, 1) for g in n121_eta1_n1_eta2_n2_eta1]))

        n12_eta2_n2_eta1_ = autograd.grad(n2_eta2, self.theta1, grad_outputs=n2_eta1.detach(), create_graph=True, retain_graph=True)
        n12_eta2_n2_eta1_ = torch.cat([g.contiguous().view(-1, 1) for g in n12_eta2_n2_eta1_])
        scalar = torch.dot(n1_eta1_.detach().view(-1,), n12_eta2_n2_eta1_.view(-1,))
        # l2 term 6
        n121_eta2_n1_eta1_n2_eta1 = autograd.grad(scalar, self.theta1, retain_graph=True)
        all_terms.append(self.factor*self.lr1*self.lr2*self.lr1 * torch.cat([g.contiguous().view(-1, 1) for g in n121_eta2_n1_eta1_n2_eta1]))

        # level 3 terms
        
        n12_eta2_n2_eta1_ = autograd.grad(n2_eta2, self.theta1, grad_outputs=n2_eta1.detach(), create_graph=True, retain_graph=True)
        n12_eta2_n2_eta1_ = torch.cat([g.contiguous().view(-1, 1) for g in n12_eta2_n2_eta1_])
        scalar = torch.dot(n12_eta2_n2_eta1_.detach().view(-1,), n12_eta2_n2_eta1_.view(-1,))
        """l3 term 1"""
        n121_eta2_n12_eta2_n2_eta1_n2_eta1 = autograd.grad(scalar, self.theta1, retain_graph=True)#
        all_terms.append(self.lr1*self.lr2*self.lr1*self.lr2 * torch.cat([g.contiguous().view(-1, 1) for g in n121_eta2_n12_eta2_n2_eta1_n2_eta1]))

        n21_eta2_n12_eta2_n2_eta1 = autograd.grad(n1_eta2, self.theta2, grad_outputs=n12_eta2_n2_eta1_.detach(), create_graph=True, retain_graph=True)
        n21_eta2_n12_eta2_n2_eta1 = torch.cat([g.contiguous().view(-1, 1) for g in n21_eta2_n12_eta2_n2_eta1])
        scalar = torch.dot(n2_eta1.detach().view(-1,), n21_eta2_n12_eta2_n2_eta1.view(-1,))
        """l3 term 2"""
        n211_eta2_n2_eta1_n12_eta2_n2_eta1 = autograd.grad(scalar, self.theta1, retain_graph=True)#
        all_terms.append(self.lr1*self.lr2*self.lr1*self.lr2 * torch.cat([g.contiguous().view(-1, 1) for g in n211_eta2_n2_eta1_n12_eta2_n2_eta1]))

        """l3 term 3"""
        n12_eta1_n21_eta2_n12_eta2_n2_eta1 = autograd.grad(n2_eta1, self.theta1, grad_outputs=n21_eta2_n12_eta2_n2_eta1, create_graph=True, retain_graph=True)
        all_terms.append(self.lr1*self.lr2*self.lr1*self.lr2 * torch.cat([g.contiguous().view(-1, 1) for g in n12_eta1_n21_eta2_n12_eta2_n2_eta1]))

        # l3 term 4
        n12_eta1_n2_eta2 = torch.cat([g.contiguous().view(-1, 1) for g in n12_eta1_n2_eta2])
        scalar = torch.dot(n12_eta1_n2_eta2.detach().view(-1,), n12_eta2_n2_eta1_.view(-1,))
        n121_eta2_n12_eta1_n2_eta2_n2_eta1 = autograd.grad(scalar, self.theta1, retain_graph=True)
        #all_terms.append(self.lr1*self.lr2*self.lr1*self.lr2 * torch.cat([g.contiguous().view(-1, 1) for g in n121_eta2_n12_eta1_n2_eta2_n2_eta1]))

        # l3 term 5
        n21_eta1_n2_eta2 = autograd.grad(n1_eta1, self.theta2, grad_outputs=n2_eta2.detach(), create_graph=True, retain_graph=True)
        n21_eta1_n2_eta2 = torch.cat([g.contiguous().view(-1, 1) for g in n21_eta1_n2_eta2])
        scalar = torch.dot(n12_eta2_n2_eta1_.detach().view(-1,), n21_eta1_n2_eta2.view(-1,))
        n211_eta1_n2_eta2_n12_eta2_n2_eta1 = autograd.grad(scalar, self.theta1, retain_graph=True)#
        #all_terms.append(self.lr1*self.lr2*self.lr1*self.lr2 * torch.cat([g.contiguous().view(-1, 1) for g in n211_eta1_n2_eta2_n12_eta2_n2_eta1]))

        # l3 term 6
        n21_eta1_n12_eta2_n2_eta1 = autograd.grad(n1_eta1, self.theta2, grad_outputs=n12_eta2_n2_eta1_.detach(), retain_graph=True)
        n21_eta1_n12_eta2_n2_eta1 = torch.cat([g.contiguous().view(-1, 1) for g in n21_eta1_n12_eta2_n2_eta1])
        n12_eta2_n21_eta1_n12_eta2_n2_eta1  = autograd.grad(n2_eta2, self.theta1, grad_outputs=n21_eta1_n12_eta2_n2_eta1, retain_graph=True)
        #all_terms.append(self.lr1*self.lr2*self.lr1*self.lr2 * torch.cat([g.contiguous().view(-1, 1) for g in n12_eta2_n21_eta1_n12_eta2_n2_eta1]))

        """l3 term 7"""
        # n221_eta1_n21_eta2_n1_eta2_n2_eta1
        n22_eta1_n2_eta1 = autograd.grad(n2_eta1, self.theta2, grad_outputs=n2_eta1.detach(), create_graph=True, retain_graph=True)
        n22_eta1_n2_eta1 = torch.cat([g.contiguous().view(-1, 1) for g in n22_eta1_n2_eta1])
        n21_eta2_n1_eta2 = autograd.grad(n1_eta2, self.theta2, grad_outputs=n1_eta2, create_graph=True, retain_graph=True)
        n21_eta2_n1_eta2 = torch.cat([g.contiguous().view(-1, 1) for g in n21_eta2_n1_eta2])
        scalar = torch.dot(n21_eta2_n1_eta2.detach().view(-1,), n22_eta1_n2_eta1.view(-1,))
        n221_eta1_n21_eta2_n1_eta2_n2_eta1 = autograd.grad(scalar, self.theta1, retain_graph=True)
        all_terms.append(self.lr1*self.lr2*self.lr1*self.lr2 * torch.cat([g.contiguous().view(-1, 1) for g in n221_eta1_n21_eta2_n1_eta2_n2_eta1]))

        """l3 term 8"""
        #n121_eta2_n1_eta2_n22_eta1_n2_eta1
        n12_eta2_n22_eta1_n2_eta1 = autograd.grad(n2_eta2, self.theta1, grad_outputs=n22_eta1_n2_eta1.detach(), create_graph=True, retain_graph=True)
        n12_eta2_n22_eta1_n2_eta1 = torch.cat([g.contiguous().view(-1, 1) for g in n12_eta2_n22_eta1_n2_eta1])
        scalar = torch.dot(n1_eta2.detach().view(-1,), n12_eta2_n22_eta1_n2_eta1.view(-1))
        n121_eta2_n1_eta2_n22_eta1_n2_eta1 = autograd.grad(scalar, self.theta1, retain_graph=True)
        all_terms.append(self.lr1*self.lr2*self.lr1*self.lr2 * torch.cat([g.contiguous().view(-1, 1) for g in n121_eta2_n1_eta2_n22_eta1_n2_eta1]))

        """l3 term 9"""
        #n11_eta2_n12_eta2_n22_eta1_n2_eta1
        n11_eta2_n12_eta2_n22_eta1_n2_eta1 = autograd.grad(n1_eta2, self.theta1, grad_outputs=n12_eta2_n22_eta1_n2_eta1, retain_graph=True)
        all_terms.append(self.lr1*self.lr2*self.lr1*self.lr2 * torch.cat([g.contiguous().view(-1, 1) for g in n11_eta2_n12_eta2_n22_eta1_n2_eta1]))

        # l3 term 10
        #n221_eta2_n21_eta1_n1_eta2_n2_eta1
        n22_eta2_n2_eta1 = autograd.grad(n2_eta2, self.theta2, grad_outputs=n2_eta1.detach(), create_graph=True, retain_graph=True)
        n22_eta2_n2_eta1 = torch.cat([g.contiguous().view(-1, 1) for g in n22_eta2_n2_eta1])
        scalar = torch.dot(n21_eta1_n1_eta2.detach().view(-1,), n22_eta2_n2_eta1.view(-1,))
        n221_eta2_n21_eta1_n1_eta2_n2_eta1 = autograd.grad(scalar, self.theta1, retain_graph=True)
        #all_terms.append(self.lr1*self.lr2*self.lr1*self.lr2 * torch.cat([g.contiguous().view(-1, 1) for g in n221_eta2_n21_eta1_n1_eta2_n2_eta1]))

        # l3 term 11
        # n121_eta1_n1_eta2_n22_eta2_n2_eta1
        n12_eta1_n22_eta2_n2_eta1 = autograd.grad(n2_eta1, self.theta1, grad_outputs=n22_eta2_n2_eta1.detach(), create_graph=True, retain_graph=True)
        n12_eta1_n22_eta2_n2_eta1 = torch.cat([g.contiguous().view(-1, 1) for g in n12_eta1_n22_eta2_n2_eta1])
        scalar = torch.dot(n1_eta2.detach().view(-1,), n12_eta1_n22_eta2_n2_eta1.view(-1,))
        n121_eta1_n1_eta2_n22_eta2_n2_eta1 = autograd.grad(scalar, self.theta1, retain_graph=True)
        #all_terms.append(self.lr1*self.lr2*self.lr1*self.lr2 * torch.cat([g.contiguous().view(-1, 1) for g in n121_eta1_n1_eta2_n22_eta2_n2_eta1]))

        # l3 term 12
        # n11_eta2_n12_eta1_n22_eta2_n2_eta1
        n11_eta2_n12_eta1_n22_eta2_n2_eta1 = autograd.grad(n1_eta2, self.theta1, grad_outputs=n12_eta1_n22_eta2_n2_eta1, retain_graph=True)
        #all_terms.append(self.lr1*self.lr2*self.lr1*self.lr2 * torch.cat([g.contiguous().view(-1, 1) for g in n11_eta2_n12_eta1_n22_eta2_n2_eta1]))

        """l3 term 13"""
        # n212_eta2_n2_eta1_n1_eta2_n21_eta1
        n21_eta2_n1_eta2 = autograd.grad(n1_eta2, self.theta2, grad_outputs=n1_eta2.detach(), create_graph=True, retain_graph=True)
        n21_eta2_n1_eta2 = torch.cat([g.contiguous().view(-1, 1) for g in n21_eta2_n1_eta2])
        scalar = torch.dot(n2_eta1.detach().view(-1,), n21_eta2_n1_eta2.view(-1))
        q = autograd.grad(scalar, self.theta2, retain_graph=True)
        n212_eta2_n2_eta1_n1_eta2_n21_eta1 = autograd.grad(n2_eta1, self.theta1, grad_outputs=q, retain_graph=True)
        all_terms.append(self.lr1*self.lr2*self.lr1*self.lr2 * torch.cat([g.contiguous().view(-1, 1) for g in n212_eta2_n2_eta1_n1_eta2_n21_eta1]))

        """l3 term 14"""
        # n212_eta2_n2_eta1_n11_eta2_n2_eta1
        n22_eta2_n1_eta2 = autograd.grad(n2_eta2, self.theta2, grad_outputs=n2_eta1.detach(), create_graph=True, retain_graph=True)
        n22_eta2_n1_eta2 = torch.cat([g.contiguous().view(-1, 1) for g in n22_eta2_n1_eta2])
        scalar = torch.dot(n2_eta1.detach().view(-1,), n22_eta2_n1_eta2.view(-1))
        q = autograd.grad(scalar, self.theta1, retain_graph=True)
        n212_eta2_n2_eta1_n11_eta2_n2_eta1 = autograd.grad(n1_eta2, self.theta1, grad_outputs=q, retain_graph=True)
        all_terms.append(self.lr1*self.lr2*self.lr1*self.lr2 * torch.cat([g.contiguous().view(-1, 1) for g in n212_eta2_n2_eta1_n11_eta2_n2_eta1]))

        """l3 term 15"""
        # n2121_eta2_n2_eta1_n1_eta2_n2_eta1
        scalar = torch.dot(n2_eta1.detach().view(-1,), n22_eta2_n2_eta1.view(-1,))
        q = autograd.grad(scalar, self.theta1, create_graph=True, retain_graph=True)
        q = torch.cat([g.contiguous().view(-1, 1) for g in q])
        scalar = torch.dot(n1_eta2.detach().view(-1,), q.view(-1,))
        n2121_eta2_n2_eta1_n1_eta2_n2_eta1 = autograd.grad(q, self.theta1, retain_graph=True)
        all_terms.append(self.lr1*self.lr2*self.lr1*self.lr2 * torch.cat([g.contiguous().view(-1, 1) for g in n2121_eta2_n2_eta1_n1_eta2_n2_eta1]))

        # l3 term 16
        # n212_eta1_n2_eta1_n1_eta2_n21_eta2
        n21_eta1_n1_eta2 = autograd.grad(n1_eta1_, self.theta2, grad_outputs=n1_eta2.detach(), create_graph=True, retain_graph=True)
        n21_eta1_n1_eta2 = torch.cat([g.contiguous().view(-1, 1) for g in n21_eta1_n1_eta2])
        scalar = torch.dot(n2_eta1.detach().view(-1,), n21_eta1_n1_eta2.view(-1))
        q = autograd.grad(scalar, self.theta2, retain_graph=True)
        n212_eta1_n2_eta1_n1_eta2_n21_eta2 = autograd.grad(n2_eta2, self.theta1, grad_outputs=q, retain_graph=True)
        #all_terms.append(self.lr1*self.lr2*self.lr1*self.lr2 * torch.cat([g.contiguous().view(-1, 1) for g in n212_eta1_n2_eta1_n1_eta2_n21_eta2]))

        # l3 term 17
        # n212_eta1_n2_eta1_n11_eta2_n2_eta2
        n22_eta1_n2_eta2 = autograd.grad(n2_eta1, self.theta2, grad_outputs=n2_eta2.detach(), create_graph=True, retain_graph=True)
        n22_eta1_n2_eta2 = torch.cat([g.contiguous().view(-1, 1) for g in n22_eta1_n2_eta2])
        scalar = torch.dot(n2_eta1.detach().view(-1,), n22_eta1_n2_eta2.view(-1))
        q = autograd.grad(scalar, self.theta1, retain_graph=True)
        n212_eta1_n2_eta1_n11_eta2_n2_eta2 = autograd.grad(n1_eta2, self.theta1, grad_outputs=q, retain_graph=True)
        #all_terms.append(self.lr1*self.lr2*self.lr1*self.lr2 * torch.cat([g.contiguous().view(-1, 1) for g in n212_eta1_n2_eta1_n11_eta2_n2_eta2]))

        # l3 term 18
        # n2121_eta1_n2_eta1_n1_eta2_n2_eta2
        scalar = torch.dot(n2_eta1.detach().view(-1,), n22_eta1_n2_eta2.view(-1,))
        q = autograd.grad(scalar, self.theta1, create_graph=True, retain_graph=True)
        q = torch.cat([g.contiguous().view(-1, 1) for g in q])
        scalar = torch.dot(n1_eta2.detach().view(-1,), q.view(-1,))
        n2121_eta1_n2_eta1_n1_eta2_n2_eta2 = autograd.grad(q, self.theta1, retain_graph=True)
        #all_terms.append(self.lr1*self.lr2*self.lr1*self.lr2 * torch.cat([g.contiguous().view(-1, 1) for g in n2121_eta1_n2_eta1_n1_eta2_n2_eta2]))

        # l3 term 19
        #n12_eta1_n21_eta2_n12_eta2_n2_eta1
        #all_terms.append(self.lr1*self.lr2*self.lr1*self.lr2 * torch.cat([g.contiguous().view(-1, 1) for g in n12_eta1_n21_eta2_n12_eta2_n2_eta1]))

        # l3 term 20
        #n12_eta1_n21_eta2_n12_eta1_n2_eta2
        n21_eta2_n12_eta1_n2_eta2 = autograd.grad(n1_eta2, self.theta2, grad_outputs=n12_eta1_n2_eta2.detach(), create_graph=True, retain_graph=True)
        n21_eta2_n12_eta1_n2_eta2 = torch.cat([g.contiguous().view(-1, 1) for g in n21_eta2_n12_eta1_n2_eta2])
        n12_eta1_n21_eta2_n12_eta1_n2_eta2 = autograd.grad(n2_eta1, self.theta1, grad_outputs=n21_eta2_n12_eta1_n2_eta2, retain_graph=True)
        #all_terms.append(self.lr1*self.lr2*self.lr1*self.lr2 * torch.cat([g.contiguous().view(-1, 1) for g in n12_eta1_n21_eta2_n12_eta1_n2_eta2]))

        # l3 term 21
        #n12_eta1_n22_eta1_n21_eta2_n1_eta2
        n22_eta1_n21_eta2_n1_eta2 = autograd.grad(n2_eta1, self.theta2, grad_outputs=n21_eta2_n1_eta2.detach(), create_graph=True, retain_graph=True)
        n22_eta1_n21_eta2_n1_eta2 = torch.cat([g.contiguous().view(-1, 1) for g in n22_eta1_n21_eta2_n1_eta2])
        n12_eta1_n22_eta1_n21_eta2_n1_eta2 = autograd.grad(n2_eta1, self.theta1, grad_outputs=n22_eta1_n21_eta2_n1_eta2, retain_graph=True)
        #all_terms.append(self.lr1*self.lr2*self.lr1*self.lr2 * torch.cat([g.contiguous().view(-1, 1) for g in n12_eta1_n22_eta1_n21_eta2_n1_eta2]))

        # l3 term 22
        #n12_eta1_n22_eta2_n21_eta1_n1_eta2
        n22_eta2_n21_eta1_n1_eta2 = autograd.grad(n2_eta2, self.theta2, grad_outputs=n21_eta1_n1_eta2.detach(), create_graph=True, retain_graph=True)
        n22_eta2_n21_eta1_n1_eta2 = torch.cat([g.contiguous().view(-1, 1) for g in n22_eta2_n21_eta1_n1_eta2])
        n12_eta1_n22_eta2_n21_eta1_n1_eta2 = autograd.grad(n2_eta1, self.theta1, grad_outputs=n22_eta2_n21_eta1_n1_eta2, retain_graph=True)
        #all_terms.append(self.lr1*self.lr2*self.lr1*self.lr2 * torch.cat([g.contiguous().view(-1, 1) for g in n12_eta1_n22_eta2_n21_eta1_n1_eta2]))

        # l3 term 23
        #n12_eta1_n212_eta2_n2_eta1_n1_eta2
        scalar = torch.dot(n2_eta1.detach().view(-1,), n21_eta2_n1_eta2.view(-1,))
        q = autograd.grad(scalar, self.theta2, create_graph=True, retain_graph=True)
        q = torch.cat([g.contiguous().view(-1, 1) for g in q])
        n12_eta1_n212_eta2_n2_eta1_n1_eta2 = autograd.grad(n2_eta1, self.theta1, grad_outputs=q, retain_graph=True)
        #all_terms.append(self.lr1*self.lr2*self.lr1*self.lr2 * torch.cat([g.contiguous().view(-1, 1) for g in n12_eta1_n212_eta2_n2_eta1_n1_eta2]))

        # l3 term 24
        #n12_eta1_n212_eta1_n2_eta2_n1_eta2
        scalar = torch.dot(n2_eta2.detach().view(-1,), n21_eta1_n1_eta2.view(-1,))
        q = autograd.grad(scalar, self.theta2, create_graph=True, retain_graph=True)
        q = torch.cat([g.contiguous().view(-1, 1) for g in q])
        n12_eta1_n212_eta1_n2_eta2_n1_eta2 = autograd.grad(n2_eta1, self.theta1, grad_outputs=q, retain_graph=True)
        #all_terms.append(self.lr1*self.lr2*self.lr1*self.lr2 * torch.cat([g.contiguous().view(-1, 1) for g in n12_eta1_n212_eta1_n2_eta2_n1_eta2]))
        
        # parameter update
        self.update_parameters(sum(all_terms))

        if self.collect_info:
            for j in range(len(all_terms)):
                norm = torch.norm(all_terms[j], p=2)
                self.norms.append(norm)