import loss
from Dataset import Dataset

import argparse
import os

import PIL
import torch
import torchvision
from tqdm import tqdm
from torch.utils import tensorboard
from sklearn.model_selection import train_test_split
import torch.utils.data as data_utils


import torch_fidelity

parser = argparse.ArgumentParser()

parser.add_argument("--new_dataset_size", type=int, default="1", help="value between 0 and 1")
parser.add_argument("--batch_size", type=int, default="64")
parser.add_argument("--num_total_steps", type=int, default="100000")
parser.add_argument("--num_epoch_steps", type=int, default="5000")
parser.add_argument("--num_dis_updates", type=int, default="5")
parser.add_argument("--num_samples_for_metrics", type=int, default="50000")
parser.add_argument("--lr", type=float, default="2e-4")
parser.add_argument("--z_size", type=int, default="128")
parser.add_argument("--z_type", type=str, default="normal")
parser.add_argument("--leading_metric", type=str, default="FID")
parser.add_argument("--disable_sn", type=bool, default="False")
parser.add_argument("--conditional", type=bool, default="False")
parser.add_argument("--dir_dataset", type=str, default="dataset")
parser.add_argument("--dir_logs", type=str, default="logs")

opt = parser.parse_args()
print(opt)

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

leading_metric, last_best_metric, metric_greater_cmp = {
    'ISC': (torch_fidelity.KEY_METRIC_ISC_MEAN, 0.0, float.__gt__),
    'FID': (torch_fidelity.KEY_METRIC_FID, float('inf'), float.__lt__),
    'KID': (torch_fidelity.KEY_METRIC_KID_MEAN, float('inf'), float.__lt__),
    'PPL': (torch_fidelity.KEY_METRIC_PPL_MEAN, float('inf'), float.__lt__),
}[opt.leading_metric]

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

# initialize logging
tb = tensorboard.SummaryWriter(log_dir=opt.dir_logs)
pbar = tqdm(total=opt.num_total_steps, desc='Training', unit='batch')
os.makedirs(opt.dir_logs, exist_ok=True)



for step in range(opt.num_total_steps):
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

    # log
    if (step + 1) % 10 == 0:
        step_info = {'loss_G': loss_G.cpu().item(), 'loss_D': loss_D.cpu().item()}
        pbar.set_postfix(step_info)
        for k, v in step_info.items():
            tb.add_scalar(f'loss/{k}', v, global_step=step)
        tb.add_scalar(f'LR/lr', scheduler_G.get_last_lr()[0], global_step=step)
    pbar.update(1)

    # decay LR
    scheduler_G.step()
    scheduler_D.step()

    # check if it is validation time
    next_step = step + 1
    if next_step % opt.num_epoch_steps != 0:
        continue
    pbar.close()
    G.eval()
    print('Evaluating the generator...')

    # compute and log generative metrics
    metrics = torch_fidelity.calculate_metrics(
        input1=torch_fidelity.GenerativeModelModuleWrapper(G, opt.z_size, opt.z_type, num_classes),
        input1_model_num_samples=opt.num_samples_for_metrics,
        input2='cifar10-train',
        isc=True,
        fid=True,
        kid=True,
        ppl=True,
        ppl_epsilon=1e-2,
        ppl_sample_similarity_resize=64,
    )
    
    # log metrics
    for k, v in metrics.items():
        tb.add_scalar(f'metrics/{k}', v, global_step=next_step)

    # log observed images
    samples_vis = G(z_vis).detach().cpu()
    samples_vis = torchvision.utils.make_grid(samples_vis).permute(1, 2, 0).numpy()
    tb.add_image('observations', samples_vis, global_step=next_step, dataformats='HWC')
    samples_vis = PIL.Image.fromarray(samples_vis)
    samples_vis.save(os.path.join(opt.dir_logs, f'{next_step:06d}.png'))

    # save the generator if it improved
    if metric_greater_cmp(metrics[leading_metric], last_best_metric):
        print(f'Leading metric {leading_metric} improved from {last_best_metric} to {metrics[leading_metric]}')

        last_best_metric = metrics[leading_metric]

        dummy_input = torch.zeros(1, opt.z_size, device=device)
        torch.jit.save(torch.jit.trace(G, (dummy_input,)), os.path.join(opt.dir_logs, 'generator.pth'))
        torch.onnx.export(G, dummy_input, os.path.join(opt.dir_logs, 'generator.onnx'),
            opset_version=11, input_names=['z'], output_names=['rgb'],
            dynamic_axes={'z': {0: 'batch'}, 'rgb': {0: 'batch'}},
        )

    # resume training
    if next_step <= opt.num_total_steps:
        pbar = tqdm(total=opt.num_total_steps, initial=next_step, desc='Training', unit='batch')
        G.train()

tb.close()
print(f'Training finished; the model with best {leading_metric} value ({last_best_metric}) is saved as '
      f'{opt.dir_logs}/generator.onnx and {opt.dir_logs}/generator.pth')