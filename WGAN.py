import Dataset
import FID
import gradientpenalty as g_p
import loss

import torch
from torch import nn
import torch.utils.data as data
from torchvision.utils import make_grid

import matplotlib.pyplot as plt
from tqdm.auto import tqdm

new_dataset_size = 0.5
n_epochs = 100


z_dim = 100 #size of the noise vector
display_step = 100 # Number of steps between displaying images 
batch_size = 128
num_workers = 2 # Setting the argument num_workers as a positive integer will turn on multi-process data loading with the specified number of loader worker processes.
lr = 0.0002
#Important for the Optimizer
beta_1 = 0.5 
beta_2 = 0.999

#important for Critic
c_lambda = 10
crit_repeats = 5


#important for scheduler
lr_decay_after = 10

device = 'cuda'


class Generator(nn.Module):
    '''
    Generator Class
    Values:
    z_dim: the dimension of the noise vector, a scalar
    im_chan: the number of channels of the output image, CIFAR-10 is 3 channel
    hidden_dim: the inner dimension, a scalar
    '''

    def __init__(self, z_dim=100, im_chan=3, hidden_dim=64):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        # Build the neural network
        self.gen = nn.Sequential(
            self.make_gen_block(z_dim, hidden_dim * 8, 4, 1, 0),
            self.make_gen_block(hidden_dim * 8, hidden_dim * 4, 4, 2, 1),
            self.make_gen_block(hidden_dim * 4, hidden_dim * 2, 4, 2, 1),
            self.make_gen_block(hidden_dim * 2, hidden_dim, 4, 2, 1),
            self.make_gen_block(hidden_dim, im_chan, 4, 2, 1, final_layer=True),
        )

    def make_gen_block(self, input_channels, output_channels, kernel_size, stride, padding, final_layer=False):
        '''
        Function to return a sequence of operations
        Values:
        input_channels: how many channels the input feature representation has
        output_channels: how many channels the output feature representation should have
        kernel_size: the size of each convolutional filter, equivalent to (kernel_size, kernel_size)
        stride: the stride of the convolution
        final_layer: whether we're on the final layer (affects activation and batchnorm)
        '''

        # Build the neural block
        if not final_layer:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride, padding, bias=False),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True),
            )
        else: # Final Layer
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride, padding, bias=False),
                nn.Tanh(),
            )

    def unsqueeze_noise(self, noise):
        '''
        Function for completing a forward pass of the Generator: Given a noise vector, 
        returns a copy of that noise with width and height = 1 and channels = z_dim.
        Parameters:
        noise: a noise tensor with dimensions (batch_size, z_dim)
        '''
        return noise.view(len(noise), self.z_dim, 1, 1)

    def forward(self, noise):
        '''
        Given a noise vector, returns a generated image.
        Parameters:
        noise: a noise tensor with dimensions (batch_size, z_dim)
        '''
        x = self.unsqueeze_noise(noise)
        return self.gen(x)




class Critic(nn.Module):
    '''
    Critic Class
    Values:
    im_chan: the number of channels of the output image, CIFAR-10 is 3 channel
    hidden_dim: the inner dimension, a scalar
    '''

    def __init__(self, im_chan=3, hidden_dim=64):
        super(Critic, self).__init__()
        self.crit = nn.Sequential(
            self.make_crit_block(im_chan, hidden_dim, 4, 2, 1),
            self.make_crit_block(hidden_dim, hidden_dim * 2, 4, 2, 1),
            self.make_crit_block(hidden_dim * 2, hidden_dim * 4, 4, 2, 1),
            self.make_crit_block(hidden_dim * 4, hidden_dim * 8, 4, 2, 1),
            self.make_crit_block(hidden_dim * 8, 1, 4, 1, 0, final_layer=True),
        )

    def make_crit_block(self, input_channels, output_channels, kernel_size, stride, padding, final_layer=False):
        '''
        Function to return a sequence of operations
        Parameters:
        input_channels: how many channels the input feature representation has
        output_channels: how many channels the output feature representation should have
        kernel_size: the size of each convolutional filter, equivalent to (kernel_size, kernel_size)
        stride: the stride of the convolution
        final_layer: whether we're on the final layer (affects activation and batchnorm)
        '''
        # Build the neural block
        if not final_layer:
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding, bias=False),
                nn.BatchNorm2d(output_channels),
                nn.LeakyReLU(0.2, inplace=True),
            )
        else: # Final Layer
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding, bias=False),
            )

    def forward(self, image):
        '''
        Function for completing a forward pass of the Critic: Given an image tensor, 
        returns a 1-dimension tensor representing fake/real.
        Parameters:
        image: a flattened image tensor with dimension (im_dim)
        '''
        crit_pred = self.crit(image)
        return crit_pred.view(len(crit_pred), -1)



dataset_class = Dataset.Dataset(dataset_type="CIFAR10", dataset_size=new_dataset_size)
dataset = dataset_class.get_dataset()

dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

gen = Generator(z_dim).to(device)
crit = Critic().to(device)

# Optimizers for the generator and critic
gen_opt = torch.optim.Adam(gen.parameters(), lr=lr, betas=(beta_1, beta_2))
crit_opt = torch.optim.Adam(crit.parameters(), lr=lr, betas=(beta_1, beta_2))

# We initialize the weights to be N(0, 0.02) here, since orthogonal
# initialization is a bit slower.
def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        # torch.nn.init.orthogonal_(m.weight)
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)

gen = gen.apply(weights_init)
crit = crit.apply(weights_init)


def get_noise(n_samples, z_dim, device='cpu'):
    '''
    Function for creating a noise vector: Given the dimensions (n_samples, z_dim)
    n_samples: the number of samples in the batch, a scalar
    z_dim: the dimension of the noise vector, a scalar
    device: the device type
    '''
    return torch.randn(n_samples, z_dim, device=device)


def lr_lambda(epoch):
    """ Function for scheduling learning """
    if epoch < lr_decay_after:
        return 1.
    else:
        return 1 - float(epoch - lr_decay_after) / (
            n_epochs - lr_decay_after + 1e-8)

# Learning rate schedulers for the generator and Critic
gen_scheduler = torch.optim.lr_scheduler.LambdaLR(gen_opt, lr_lambda=lr_lambda)
crit_scheduler = torch.optim.lr_scheduler.LambdaLR(crit_opt, lr_lambda=lr_lambda)


def show_tensor_images(image_tensor, num_images=10, size=(3, 32, 32)):
    image_tensor = (image_tensor + 1) / 2
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()

cur_step = 0

generator_losses = []
critic_losses = []
fretchet_dist = []

dataset_class.get_dataset_size(dataset)

block_idx = FID.InceptionV3.BLOCK_INDEX_BY_DIM[2048]
model = FID.InceptionV3([block_idx])
model = model.cuda()

for epoch in tqdm(range(n_epochs), position=0, leave=True):
    # Dataloader returns the batches
    print(epoch, " of ", n_epochs)
    for real, _ in tqdm(dataloader, position=0, leave=True):
        cur_batch_size = len(real)
        real = real.to(device)

        mean_iteration_critic_loss = 0

        for _ in range(crit_repeats):

          crit_opt.zero_grad()
          fake_noise = get_noise(cur_batch_size, z_dim, device=device)
          fake = gen(fake_noise)
          crit_fake_pred = crit(fake.detach())
          crit_real_pred = crit(real)

          epsilon = torch.rand(len(real), 1, 1, 1, device=device, requires_grad=True)
          gradient = g_p.get_gradient(crit, real, fake.detach(), epsilon)
          gp = g_p.gradient_penalty(gradient)
          crit_loss = loss.get_crit_loss(crit_fake_pred, crit_real_pred, gp, c_lambda)

          # Keep track of the average critic
          mean_iteration_critic_loss += crit_loss.item() / crit_repeats
          # Update gradients
          crit_loss.backward(retain_graph=True)
          # Update optimizer
          crit_opt.step()

        critic_losses += [mean_iteration_critic_loss]

        ## Update Generator ##
        gen_opt.zero_grad()
        fake_noise_2 = get_noise(cur_batch_size, z_dim, device=device)
        fake_2 = gen(fake_noise_2)
        crit_fake_pred = crit(fake_2)

        gen_loss = loss.get_gen_loss(crit_fake_pred)
        gen_loss.backward()
        gen_opt.step()

        # Keep track of the average generator loss
        generator_losses += [gen_loss.item()]

        ## Visualization code ##
        if cur_step % display_step == 0 and cur_step > 0:
            gen_mean = sum(generator_losses[-display_step:]) / display_step
            crit_mean = sum(critic_losses[-display_step:]) / display_step
            print(f"Step {cur_step}: Generator loss: {gen_mean}, critic loss: {crit_mean}")
            show_tensor_images(real)
            show_tensor_images(fake)
            step_bins = 20
            num_examples = (len(generator_losses) // step_bins) * step_bins
            plt.plot(
                range(num_examples // step_bins), 
                torch.Tensor(generator_losses[:num_examples]).view(-1, step_bins).mean(1),
                label="Generator Loss"
            )
            plt.plot(
                range(num_examples // step_bins), 
                torch.Tensor(critic_losses[:num_examples]).view(-1, step_bins).mean(1),
                label="Critic Loss"
            )
            plt.legend()
            plt.show()
        cur_step += 1

    fretchet_dist.append(FID.calculate_frechet(real, fake, model))

    crit_scheduler.step()
    gen_scheduler.step()


plot1 = plt.figure(figsize=(10,5))
plt.title("Generator Loss and Critic During Training")
plt.plot(generator_losses,label="gen")
plt.plot(critic_losses,label="Critic")
plt.xlabel("step")
plt.ylabel("Loss")
plt.legend()

plot2 = plt.figure(figsize=(10,5))
plt.title("FID Score")
plt.plot(fretchet_dist,label="FID")
plt.xlabel("epoch")
plt.ylabel("Score")
plt.legend()


plt.show()