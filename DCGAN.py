import loss
from Dataset import Dataset
import logs
import FID

import argparse
import os

import torch
from tqdm import tqdm

parser = argparse.ArgumentParser()

parser.add_argument("--new_dataset_size", type=int, default="1", help="value between 0 and 1")
parser.add_argument("--batch_size", type=int, default="128")
parser.add_argument("--num_total_steps", type=int, default="100000")
parser.add_argument("--num_epoch_steps", type=int, default="5000")
parser.add_argument("--num_dis_updates", type=int, default="5")
# parser.add_argument("--num_samples_for_metrics", type=int, default="50000")
parser.add_argument("--lr", type=float, default="2e-4")
parser.add_argument("--z_size", type=int, default="128", help = "size of noise vector")
parser.add_argument("--z_type", type=str, default="normal")
# parser.add_argument("--leading_metric", type=str, default="ISC")
parser.add_argument("--disable_sn", type=bool, default=False, help = "disable spectral normalization")
parser.add_argument("--conditional", type=bool, default=False, help = "conditional GAN")
# parser.add_argument("--dir_dataset", type=str, default="dataset")
parser.add_argument("--dir_logs", type=str, default="logs")

opt = parser.parse_args()
print(opt)

os.makedirs(opt.dir_logs + "/losses", exist_ok=True)
os.makedirs(opt.dir_logs + "/metrics", exist_ok=True)
os.makedirs(opt.dir_logs + "/images", exist_ok=True)
os.makedirs(opt.dir_logs + "/models", exist_ok=True)
os.makedirs(opt.dir_logs + "/settings", exist_ok=True)

logs.create_settings_file(opt, opt.dir_logs + "/settings")

class Generator(torch.nn.Module):
    # Adapted from https://github.com/christiancosgrove/pytorch-spectral-normalization-gan
    def __init__(self, z_size):
        super(Generator, self).__init__()
        self.z_size = z_size
        self.model = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(z_size, 512, 4, stride=1),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(512, 256, 4, stride=2, padding=(1,1)),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(256, 128, 4, stride=2, padding=(1,1)),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(128, 64, 4, stride=2, padding=(1,1)),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(64, 3, 3, stride=1, padding=(1,1)),
            torch.nn.Tanh()
        )

    def forward(self, z):
        fake = self.model(z.view(-1, self.z_size, 1, 1))
        if not self.training:
            fake = (255 * (fake.clamp(-1, 1) * 0.5 + 0.5))
            fake = fake.to(torch.uint8)
        return fake

class Discriminator(torch.nn.Module):
    # Adapted from https://github.com/christiancosgrove/pytorch-spectral-normalization-gan
    def __init__(self, sn):
        super(Discriminator, self).__init__()

        sn_fn = torch.nn.utils.spectral_norm if sn else lambda x: x #This line defines if Spectral Normalization is used or not!
        """
        normalizes the weight matrices in the discriminator by their corresponding spectral norm, which helps control the Lipschitz constant of the discriminator.
        Lipschitz continuity is important in ensuring the boundedness of the optimal discriminator.
        As a result spectral normalization helps improve stability and avoid vanishing gradient problems, such as mode collapse
        """

        self.conv1 = sn_fn(torch.nn.Conv2d(3, 64, 3, stride=1, padding=(1,1)))
        self.conv2 = sn_fn(torch.nn.Conv2d(64, 64, 4, stride=2, padding=(1,1)))
        self.conv3 = sn_fn(torch.nn.Conv2d(64, 128, 3, stride=1, padding=(1,1)))
        self.conv4 = sn_fn(torch.nn.Conv2d(128, 128, 4, stride=2, padding=(1,1)))
        self.conv5 = sn_fn(torch.nn.Conv2d(128, 256, 3, stride=1, padding=(1,1)))
        self.conv6 = sn_fn(torch.nn.Conv2d(256, 256, 4, stride=2, padding=(1,1)))
        self.conv7 = sn_fn(torch.nn.Conv2d(256, 512, 3, stride=1, padding=(1,1)))
        self.fc = sn_fn(torch.nn.Linear(4 * 4 * 512, 1))
        self.act = torch.nn.LeakyReLU(0.1)

    def forward(self, x):
        m = self.act(self.conv1(x))
        m = self.act(self.conv2(m))
        m = self.act(self.conv3(m))
        m = self.act(self.conv4(m))
        m = self.act(self.conv5(m))
        m = self.act(self.conv6(m))
        m = self.act(self.conv7(m))
        return self.fc(m.view(-1, 4 * 4 * 512))

dataset_class = Dataset(dataset_type="CIFAR10", dataset_size=opt.new_dataset_size)
dataset = dataset_class.get_dataset()

loader = torch.utils.data.DataLoader(
    dataset, batch_size=opt.batch_size, drop_last=True, shuffle=True, num_workers=4, pin_memory=True
)

loader_iter = iter(loader)


# reinterpret command line inputs
device = 'cuda' if torch.cuda.is_available() else 'cpu'

num_classes = 10 if opt.conditional else 0  # unconditional

# create Generator and Discriminator models
G = Generator(opt.z_size).to(device).train()
D = Discriminator(not opt.disable_sn).to(device).train()

# initialize persistent noise for observed samples
z_vis = torch.randn(64, opt.z_size, device=device)

# prepare optimizer and learning rate schedulers (linear decay)
optim_G = torch.optim.Adam(G.parameters(), lr=opt.lr, betas=(0.0, 0.9))
optim_D = torch.optim.Adam(D.parameters(), lr=opt.lr, betas=(0.0, 0.9))
scheduler_G = torch.optim.lr_scheduler.LambdaLR(optim_G, lambda step: 1. - step / opt.num_total_steps)
scheduler_D = torch.optim.lr_scheduler.LambdaLR(optim_D, lambda step: 1. - step / opt.num_total_steps)

block_idx = FID.InceptionV3.BLOCK_INDEX_BY_DIM[2048]
model = FID.InceptionV3([block_idx])
model = model.cuda()

losses_G = []
losses_D = []
num_steps_losses = []

fretchet_dist_list=[]
num_step_FID=[]

for step in tqdm(range(opt.num_total_steps), position=0, leave=True):
    # read next batch
    try:
        real_img, real_label = next(loader_iter)
    except StopIteration:
        loader_iter = iter(loader)
        real_img, real_label = next(loader_iter)
    real_img = real_img.to(device)
    real_label = real_label.to(device)

    # update Generator
    G.requires_grad_(True)
    D.requires_grad_(False)
    z = torch.randn(opt.batch_size, opt.z_size, device=device)
    optim_D.zero_grad()
    optim_G.zero_grad()
    fake = G(z)
    loss_G = loss.hinge_loss_gen(D(fake))
    loss_G.backward()
    optim_G.step()

    # update Discriminator
    G.requires_grad_(False)
    D.requires_grad_(True)
    for d_iter in range(opt.num_dis_updates):
        z = torch.randn(opt.batch_size, opt.z_size, device=device)
        optim_D.zero_grad()
        optim_G.zero_grad()
        fake = G(z)
        loss_D = loss.hinge_loss_dis(D(fake), D(real_img))
        loss_D.backward()
        optim_D.step()

    # logging the losses
    if (step + 1) % 100 == 0:
        print(f"\nloss_G: {loss_G.cpu().item()} loss_D: {loss_D.cpu().item()}")
        losses_G.append(loss_G.cpu().item())
        losses_D.append(loss_D.cpu().item())
        num_steps_losses.append(step)
        logs.print_losses(losses_G, losses_D, num_steps_losses, opt.dir_logs)
        

    # decay LR
    scheduler_G.step()
    scheduler_D.step()

    # check if it is validation time
    next_step = step + 1
    
    # Checking if validation is about to begin
    if next_step % opt.num_epoch_steps != 0:
        continue # if we are here, we return to the beginning of the loop

    print(f"\nEvaluating FID Score..")
    
    # Get FID Score (my way)
    FID_score = FID.calculate_frechet(real_img, fake, model)
    print(f"\nFID Score of step: {next_step} is {FID_score}")
    fretchet_dist_list.append(FID_score)
    num_step_FID.append(step)
    logs.print_FID(fretchet_dist_list, num_step_FID, opt.dir_logs)

    # Get FID Score (Barg.'s way - needs to be implemented)
    #################################################

    # Save images of the epoch
    samples_vis = G(z_vis)
    # print(type(samples_vis))
    # print(samples_vis.shape)
    # samples_vis_2 = torchvision.utils.make_grid(samples_vis).permute(1, 2, 0).numpy()
    # print(type(samples_vis_2))
    # print(samples_vis_2.shape)
    # samples_vis_3 = PIL.Image.fromarray(samples_vis_2)
    # samples_vis_3.save(os.path.join(opt.dir_logs + "/images", f'{next_step:06d}.png'))

    logs.save_tensor_images(samples_vis, opt.dir_logs, step)

    # Save the generator if it improved
    logs.save_gen(fretchet_dist_list, G, opt.z_size, opt.dir_logs, device)

    if next_step <= opt.num_total_steps:
        G.train()


print(f'Training finished')