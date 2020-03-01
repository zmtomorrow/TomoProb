import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import grad

### see https://github.com/zmtomorrow/pytorch_probabilistic_tools

def jacobian(y, x, if_create_graph=True):
    B, N = x.shape
    _, M = y.shape
    jacobian = list()
    for i in range(M):
        v = torch.zeros_like(y)
        v[:, i] = 1.
        dy_i_dx = grad(y,x,grad_outputs=v,retain_graph=True,create_graph=if_create_graph,allow_unused=True)[0]
        jacobian.append(dy_i_dx)

    jacobian = torch.stack(jacobian, dim=1).requires_grad_()

    return jacobian


def batch_KL_diag_gaussian_std(mu_1,std_1,mu_2,std_2):
    diag_1=std_1**2
    diag_2=std_2**2
    ratio=diag_1/diag_2
    return 0.5*(torch.sum((mu_1-mu_2)**2/diag_2,dim=-1)+torch.sum(ratio,dim=-1)-torch.sum(torch.log(ratio),dim=-1)-mu_1.size(1))


def low_rank_cov_inverse(L,sigma):
    # L is D*R
    dim=L.size(0)
    rank=L.size(1)
    var=sigma**2
    inverse_var=1.0/var
    inner_inverse=torch.inverse(torch.diag(torch.ones([rank]))+inverse_var*(L.t())@L)
    return inverse_var*torch.diag(torch.ones([dim]))-inverse_var**2*L@inner_inverse@L.t()

def low_rank_gaussian_logdet(L,sigma):
    dim=L.size(0)
    rank=L.size(1)
    var=sigma**2
    inverse_var=1.0/var
    return torch.logdet(torch.diag(torch.ones([rank]))+inverse_var*(L.t())@L)+dim*tf.log(var)


def KL_low_rank_gaussian_with_diag_gaussian(mu_1,L_1,sigma_1,mu_2,sigma_2,cuda):
    dim_1=L_1.size(0)
    rank_1=L_1.size(1)
    var_1=sigma_1**2
    inverse_var_1=1.0/var_1
    if cuda:
        logdet_1=torch.logdet(torch.diag(torch.ones([rank_1]).cuda())+inverse_var_1*(L_1.t())@L_1)+dim_1*torch.log(var_1)
        cov_1=L_1@L_1.t()+torch.diag(torch.ones([dim_1]).cuda())*var_1
    else:
        logdet_1=torch.logdet(torch.diag(torch.ones([rank_1]))+inverse_var_1*(L_1.t())@L_1)+dim_1*torch.log(var_1)
        cov_1=L_1@L_1.t()+torch.diag(torch.ones([dim_1]))*var_1
    mu_diff=(mu_1-mu_2).view(-1,1)
    var_2=sigma_2**2
    return -0.5*(logdet_1-dim_1*torch.log(var_2)+dim_1-(1/var_2)*torch.trace(cov_1)-(1/var_2)*mu_diff.t()@mu_diff)



def KL_low_rank_gaussian_with_low_rank_gaussian(mu_1,L_1,sigma_1,mu_2,L_2,sigma_2,cuda):
    if cuda:
        dim_1=L_1.size(0)
        rank_1=L_1.size(1)
        var_1=sigma_1**2
        inverse_var_1=1.0/var_1
        logdet_1=torch.logdet(torch.diag(torch.ones([rank_1]).cuda())+inverse_var_1*(L_1.t())@L_1)+dim_1*torch.log(var_1)
        cov_1=L_1@L_1.t()+torch.diag(torch.ones([dim_1]).cuda())*var_1

        dim_2=L_2.size(0)
        rank_2=L_2.size(1)
        var_2=sigma_2**2
        inverse_var_2=1.0/var_2
        logdet_2=torch.logdet(torch.diag(torch.ones([rank_2]).cuda())+inverse_var_2*(L_2.t())@L_2)+dim_1*torch.log(var_2)

        inner_inverse_2=torch.inverse(torch.diag(torch.ones([rank_2]).cuda())+inverse_var_2*(L_2.t())@L_2)
        cov_inverse_2=inverse_var_2*torch.diag(torch.ones([dim_2]).cuda())-inverse_var_2**2*L_2@inner_inverse_2@L_2.t()

        mu_diff=(mu_1-mu_2).view(-1,1)
        return -0.5*(logdet_1-logdet_2+dim_1-torch.trace(cov_1@cov_inverse_2)-mu_diff.t()@ cov_inverse_2@mu_diff)
    else:
        dim_1=L_1.size(0)
        rank_1=L_1.size(1)
        var_1=sigma_1**2
        inverse_var_1=1.0/var_1
        logdet_1=torch.logdet(torch.diag(torch.ones([rank_1]))+inverse_var_1*(L_1.t())@L_1)+dim_1*torch.log(var_1)
        cov_1=L_1@L_1.t()+torch.diag(torch.ones([dim_1]))*var_1


        dim_2=L_2.size(0)
        rank_2=L_2.size(1)
        var_2=sigma_2**2
        inverse_var_2=1.0/var_2
        logdet_2=torch.logdet(torch.diag(torch.ones([rank_2]))+inverse_var_2*(L_2.t())@L_2)+dim_1*torch.log(var_2)

        inner_inverse_2=torch.inverse(torch.diag(torch.ones([rank_2]))+inverse_var_2*(L_2.t())@L_2)
        cov_inverse_2=inverse_var_2*torch.diag(torch.ones([dim_2]))-inverse_var_2**2*L_2@inner_inverse_2@L_2.t()

        mu_diff=(mu_1-mu_2).view(-1,1)
        return -0.5*(logdet_1-logdet_2+dim_1-torch.trace(cov_1@cov_inverse_2)-mu_diff.t()@ cov_inverse_2@mu_diff)


def general_kl_divergence(mu_1,cov_1,mu_2,cov_2):
    mu_diff=(mu_1-mu_2).view(-1,1)
    cov_2_inverse=torch.inverse(cov_2)
    return -0.5*(torch.logdet(cov_1)-torch.logdet(cov_2)+mu_1.size(0)-torch.trace(cov_1@cov_2_inverse)-mu_diff.t()@cov_2_inverse@mu_diff)

def low_rank_gaussian_one_sample(mu,L,sigma,cuda):
    # L is D*R
    dim=L.size(0)
    rank=L.size(1)
    if cuda:
        eps_z=torch.randn([rank]).cuda()
        eps=torch.randn([dim]).cuda()
    else:
        eps_z=torch.randn([rank])
        eps=torch.randn([dim])

    return eps_z@L.t()+eps*sigma+mu

def low_rank_gaussian_sample(mu,L,sigma,amount,cuda):
    # L is D*R
    dim=L.size(0)
    rank=L.size(1)
    if cuda:
        eps_z=torch.randn([amount,rank]).cuda()
        eps=torch.randn([amount,dim]).cuda()
    else:
        eps_z=torch.randn([amount,rank])
        eps=torch.randn([amount,dim])

    return eps_z@L.t()+eps*sigma+mu


def sample_from_batch_categorical(batch_logits,cuda):
    ### shape batch*dim
    ### gumbel max trick
    if cuda:
        noise = torch.rand(batch_logits.size()).cuda()
    else:
        noise = torch.rand(batch_logits.size())
    return torch.argmax(batch_logits - torch.log(-torch.log(noise)), dim=-1)


def sample_from_batch_categorical_multiple(batch_logits,sample_num,cuda):
    ### shape batch*dim
    ### gumbel max trick
    shape=list(batch_logits.size())
    shape.insert(-1, sample_num)
    if cuda:
        noise = torch.rand(shape).cuda()
    else:
        noise = torch.rand(shape)
    batch_logits_multiple=batch_logits.repeat(1,1,1,sample_num).view(shape)
    return torch.argmax(batch_logits_multiple - torch.log(-torch.log(noise)), dim=-1)


def one_hot_embedding(labels, num_classes,cuda):
    if cuda:
        y = torch.eye(num_classes).cuda()
    else:
        y = torch.eye(num_classes)
    return y[labels]
