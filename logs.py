import matplotlib.pyplot as plt
import os
import torch
from torchvision.utils import make_grid


def print_losses(losses_G, losses_D, num_steps_losses, destination):
    """
    Prints the loss values of the Generator and Discriminator over the num of steps and saves it as pdf and png

    losses_G: list of loss values G (index 0: loss of first step)
    losses_D: lost of loss values D (index 0: loss of first step)
    num_steps_losses: list of values for num_steps at which the losses got saved
    destination: folder at which the plots get saved
    """

    plt.plot(num_steps_losses, losses_G)
    plt.xlabel("Number of steps")
    plt.ylabel("Generator loss")
    plt.savefig(os.path.join(destination + "/losses", 'G_losses.png'))
    plt.close()
    
    plt.plot(num_steps_losses, losses_D)
    plt.xlabel("Number of steps")
    plt.ylabel("Discriminator loss")
    plt.savefig(os.path.join(destination + "/losses", 'D_losses.png'))
    plt.close()

def print_FID(FID_score, num_steps_FID, destination):
    """
    Prints the FID score of the current model over the num of steps and saves it as pdf and png

    FID_score: List of FID scores of the current model
    num_steps_FID: list of values for num_steps at which the losses got saved
    destination: folder at which the plots get saved
    """

    plt.plot(num_steps_FID, FID_score)
    plt.xlabel("number of steps")
    plt.ylabel("FID Score")
    plt.savefig(os.path.join(destination + "/metrics", 'FID_score.png'))
    plt.close()


def save_gen(retchet_dist_list, G, z_size, destination, device):
    """
    Saves the generator model, if the FID score improved from last time.

    retchet_dist_list: List with the FID scores
    G: Instance of :class:`GenerativeModelBase`, implementing the generative model.
    z_size: size of the input noise vector
    destination: folder at which the plots get saved
    """

    destination = destination + "/models"
    if (len(retchet_dist_list) > 2):
        if(retchet_dist_list[-1] < retchet_dist_list[-2]):

            dummy_input = torch.zeros(1, z_size, device=device)
            torch.jit.save(torch.jit.trace(G, (dummy_input,)), os.path.join(destination, 'generator.pth'))
            torch.onnx.export(G, dummy_input, os.path.join(destination, 'generator.onnx'),
                opset_version=11, input_names=['z'], output_names=['rgb'],
                dynamic_axes={'z': {0: 'batch'}, 'rgb': {0: 'batch'}},
            )



def save_tensor_images(image_tensor, destination, step, num_images=25, size=(3, 32, 32)):
    '''
    Function for visualizing images
    image_tensor: image_tensor of type torch.tensor (output of G(z))
    destination: folder at which the plots get saved
    step: At what step the image gets saved
    '''
    image_tensor = (image_tensor + 1) / 2
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.savefig(os.path.join(destination + "/images", f'{step:06d}.png'))
    plt.close()


def create_settings_file(input, destination):
    
    complete_name = os.path.join(destination, "settings.txt")
    f = open(complete_name, "w+")

    f.write(str(input))

    f.close()

    print("created settings file")
