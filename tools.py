import torch

def KL_diag_gaussian(mu_1,diag_1,mu_2,diag_2):
    ratio=diag_1/diag_2
    return torch.sum(0.5*(mu_1-mu_2)**2/diag_2)+0.5*torch.sum(ratio)-0.5*torch.sum(torch.log(ratio))-mu_1.size(0)/2


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
    

def low_rank_gaussian_one_sample(mu,L,sigma):
    # L is D*R
    dim=L.size(0)
    rank=L.size(1)
    eps_z=torch.randn([rank])
    eps=torch.randn([dim])
    
    return eps_z@L.t()+eps*sigma+mu

def low_rank_gaussian_sample(mu,L,sigma,amount):
    # L is D*R
    dim=L.size(0)
    rank=L.size(1)
    eps_z=torch.randn([amount,rank])
    eps=torch.randn([amount,dim])
    
    return eps_z@L.t()+eps*sigma+mu
    

def KL_low_rank_gaussian_with_diag_gaussian(mu_1,L_1,sigma_1,mu_2,sigma_2):
    dim_1=L_1.size(0)
    rank_1=L_1.size(1)
    var_1=sigma_1**2
    inverse_var_1=1.0/var_1
    logdet_1=torch.logdet(torch.diag(torch.ones([rank_1]))+inverse_var_1*(L_1.t())@L_1)+dim_1*torch.log(var_1)
    cov_1=L_1@L_1.t()+torch.diag(torch.ones([dim_1]))*var_1
    mu_diff=(mu_1-mu_2).view(-1,1)
    var_2=sigma_2**2
    return -0.5*(logdet_1-dim_1*torch.log(var_2)+dim_1-(1/var_2)*torch.trace(cov_1)-(1/var_2)*mu_diff.t()@mu_diff)
    


def KL_low_rank_gaussian_with_low_rank_gaussian(mu_1,L_1,sigma_1,mu_2,L_2,sigma_2):
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
   
    
def sample_from_batch_categorical(batch_logits):
    ### shape batch*dim
    ### gumbel max trick
    noise = torch.rand(batch_logits.size()).cuda()
    return torch.argmax(batch_logits - torch.log(-torch.log(noise)), dim=-1)


def sample_from_batch_categorical_multiple(batch_logits,sample_num=1):
    ### shape batch*dim
    ### gumbel max trick
    shape=list(batch_logits.size())
    shape.insert(-1, sample_num)
    noise = torch.rand(shape).cuda()
    batch_logits_multiple=batch_logits.repeat(1,1,1,sample_num).view(shape)
    return torch.argmax(batch_logits_multiple - torch.log(-torch.log(noise)), dim=-1)
   
