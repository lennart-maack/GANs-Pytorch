import torch

def get_gen_loss(crit_fake_pred):
    '''
    Return the loss of a generator given the critic's scores of the generator's fake images.
    Parameters:
        crit_fake_pred: the critic's scores of the fake images
    Returns:
        gen_loss: a scalar loss value for the current batch of the generator
    '''
    gen_loss = -torch.mean(crit_fake_pred)
    return gen_loss


def get_crit_loss(crit_fake_pred, crit_real_pred, gp, c_lambda):
    '''
    Return the loss of a critic given the critic's scores for fake and real images,
    the gradient penalty, and gradient penalty weight.
    Parameters:
        crit_fake_pred: the critic's scores of the fake images
        crit_real_pred: the critic's scores of the real images
        gp: the unweighted gradient penalty
        c_lambda: the current weight of the gradient penalty 
    Returns:
        crit_loss: a scalar for the critic's loss, accounting for the relevant factors
    '''
    crit_loss = -torch.mean(crit_real_pred) + torch.mean(crit_fake_pred) + c_lambda*gp
    return crit_loss


# Hinge Loss https://openaccess.thecvf.com/content/WACV2021/papers/Kavalerov_A_Multi-Class_Hinge_Loss_for_Conditional_GANs_WACV_2021_paper.pdf


def hinge_loss_dis(fake, real):
    assert fake.dim() == 2 and fake.shape[1] == 1 and real.shape == fake.shape, f'{fake.shape} {real.shape}'
    loss = torch.nn.functional.relu(1.0 - real).mean() + \
           torch.nn.functional.relu(1.0 + fake).mean()
    return loss


def hinge_loss_gen(fake):
    assert fake.dim() == 2 and fake.shape[1] == 1
    loss = -fake.mean()
    return loss