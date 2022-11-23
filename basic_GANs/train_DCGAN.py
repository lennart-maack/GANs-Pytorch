from training import loss, logs
from data import Dataset
from metrics import FID

import argparse
import os

import torch
from tqdm import tqdm

from architectures import DCGAN


def main():

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
    G = DCGAN.Generator(opt.z_size).to(device).train()
    D = DCGAN.Discriminator(not opt.disable_sn).to(device).train()

    # initialize persistent noise for observed samples
    z_vis = torch.randn(64, opt.z_size, device=device)

    # prepare optimizer and learning rate schedulers (linear decay)
    optim_G = torch.optim.Adam(G.parameters(), lr=opt.lr, betas=(0.0, 0.9))
    optim_D = torch.optim.Adam(D.parameters(), lr=opt.lr, betas=(0.0, 0.9))
    scheduler_G = torch.optim.lr_scheduler.LambdaLR(optim_G, lambda step: 1. - step / opt.num_total_steps)
    scheduler_D = torch.optim.lr_scheduler.LambdaLR(optim_D, lambda step: 1. - step / opt.num_total_steps)


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

        G.eval()
        print(f"\nEvaluating FID Score..")

        # Get FID Score (my way)
        FID_score = FID.calculate_frechet(real_img, fake, opt.dims_inception, opt.device_inception)
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




if __name__ == "__main__":
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
    parser.add_argument("--device_inception", type=str, default="cuda", help="Device to run calculations")
    parser.add_argument("--dims_inception", type=int, default=2048, help="Dimensionality of features returned by Inception")
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