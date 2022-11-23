from . import metric_utils

import os
import copy
import dnnlib
import torch
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from skimage.metrics import structural_similarity as ssim
from scipy.spatial import distance



def get_ssim(**kwargs):
    """
    Calculates the mean and the std of the Structual similariry index

    **kwargs: See metric_utils.MetricOptions for the full list of arguments.
    """
    
    reals, fakes = get_datasets(dataset_size=None, **kwargs)

    assert len(reals) == len(fakes), 'imgs_1 and imgs_2 must have the same length.'

    ssim_array = np.empty(shape=(len(reals), ), dtype=np.float32)
    for i, (real, fake) in enumerate(zip(reals, fakes)):
        ssim_array[i] = ssim(real, fake, multichannel=True)
    print('' * 100, end='\r')

    ssim_mean = np.mean(ssim_array, axis=0)
    ssim_std = np.std(ssim_array, axis=0)

    #show(fakes, "fake_test2.png")
    #show(reals, "real_test2.png")

    return ssim_mean, ssim_std

def plot_hist(run_dir, cur_nimg, nimgs=None, **kwargs):
    """
    Plots three histograms of the pixel value distribution of two arrays of images (overlapping) and saves it to path
    First histogram with all 255 values, second with all values on log scale and the third histogram with value 30 cut out
    
    run_dir (string): path to the location where the histogram should be saved
    cur_nimg (int): Number of images already used for training the GAN - only needed for a proper file name of the histograms
    nimgs (int): number of images which are used per imgs array for displaying the histogram 
        - if None: the maximum images (all real images) are used
    **kwargs: See metric_utils.MetricOptions for the full list of arguments.
    """

    reals, fakes = get_datasets(dataset_size=nimgs, **kwargs)

    reals_flat = reals.flatten()
    fakes_flat = fakes.flatten()

    _ = plt.hist(reals_flat, bins=255, alpha=0.5, label="reals")
    _ = plt.hist(fakes_flat, bins=255, alpha=0.5, label="fakes")
    plt.legend()
    plt.title("Pixel value distribution all values")
    plt.xlabel("Pixel value")
    plt.ylabel("Count")
    plt.savefig(os.path.join(run_dir, f'color_hist_full_-{cur_nimg//1000:06d}.png'))
    plt.close()

    _ = plt.hist(reals_flat, log=True, bins=255, alpha=0.5, label="reals")
    _ = plt.hist(fakes_flat, log=True, bins=255, alpha=0.5, label="fakes")
    plt.legend()
    plt.title("Pixel value distribution all values log")
    plt.xlabel("Pixel value")
    plt.ylabel("Count")
    plt.savefig(os.path.join(run_dir, f'color_hist_log_-{cur_nimg//1000:06d}.png'))
    plt.close()

    reals_flat_del = np.delete(reals_flat, np.where(reals_flat == 30))
    
    fakes_flat_del = np.delete(fakes_flat, np.where(fakes_flat == 30))

    _ = plt.hist(reals_flat_del, bins=255, alpha=0.5, label="reals")
    _ = plt.hist(fakes_flat_del, bins=255, alpha=0.5, label="fakes")
    plt.legend()
    plt.title("Pixel value distribution w/o value=30")
    plt.xlabel("Pixel value")
    plt.ylabel("Count")
    plt.savefig(os.path.join(run_dir, f'color_hist_wo30_-{cur_nimg//1000:06d}.png'))
    plt.close()

def get_JS_divergence(**kwargs):
    """
    returns the Jensen shannon distance between the distribution of the pixel values
    of two set of images (np.array containing real images, np.array containing generated images)
    The value 30 is not cut off (in comparison to the plot_hist() function)
    **kwargs: See metric_utils.MetricOptions for the full list of arguments.
    """

    reals, fakes = get_datasets(dataset_size=None, **kwargs)

    reals = reals.flatten()
    fakes = fakes.flatten()

    hist_reals, _ = np.histogram(reals, bins=255)
    hist_fakes, _ = np.histogram(fakes, bins=255)

    return(distance.jensenshannon(hist_reals, hist_fakes))

def get_datasets(batch_size=64, dataset_size=None, **kwargs):
    """
    returns two np.arrays (real images from opts.dataset_kwargs,
    fake images created with Generator) of shape (n_img, w, h, n_channels)

    dataset_size (int): If None, the maximum length for the two np.arrays gets returned,
                    which is equal to the number of real images from the opts.dataset_kwargs
    **kwargs: See metric_utils.MetricOptions for the full list of arguments.
    """
    opts = metric_utils.MetricOptions(**kwargs)

    dataset = dnnlib.util.construct_class_by_name(**opts.dataset_kwargs)
    
    reals = get_reals(opts, dataset, batch_size=batch_size)
    fakes = get_fakes(opts, dataset, batch_size=batch_size)
    
    while len(fakes) <= len(reals):
        new_fakes = get_fakes(opts, dataset, batch_size=batch_size)
        fakes = torch.cat((fakes, new_fakes))

    reals = reals.numpy()
    fakes = fakes.numpy()
    
    if dataset_size is None:
        fakes = fakes[:len(reals)]
    else:
        fakes = fakes[:dataset_size]
        reals = reals[:dataset_size]

    assert len(fakes) == len(reals), 'fakes and reals must have the same length.'

    # transposes the tensor from torch.Size([1366, 3, 256, 256]) to torch.Size([1366, 256, 256, 3])
    reals = np.transpose(reals, (0,2,3,1))
    fakes = np.transpose(fakes, (0,2,3,1))

    return reals, fakes

def get_fakes(opts, dataset, batch_size):

    batch_gen = 4

    G = copy.deepcopy(opts.G).eval().requires_grad_(False).to(opts.device)

    def run_generator(z, c):
        img = G(z=z, c=c, **opts.G_kwargs)
        img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        return img

    images = []
    for _i in range(batch_size // batch_gen):
        z = torch.randn([batch_gen, G.z_dim], device=opts.device)
        c = [dataset.get_label(np.random.randint(len(dataset))) for _i in range(batch_gen)]
        c = torch.from_numpy(np.stack(c)).pin_memory().to(opts.device)
        images.append(run_generator(z, c))
    images = torch.cat(images)
    # if images.shape[1] == 1:
    #     images = images.repeat([1, 3, 1, 1])

    return images.cpu()

def get_reals(opts, dataset, batch_size):


    #item_subset = [(i * opts.num_gpus + opts.rank) % num_items for i in range((num_items - 1) // opts.num_gpus + 1)]
    item_subset = None
    data_loader_kwargs = dict(pin_memory=True, num_workers=3, prefetch_factor=2)
    image_list = []
    for images, _labels in torch.utils.data.DataLoader(dataset=dataset, sampler=item_subset, batch_size=batch_size, **data_loader_kwargs):
        image_list.append(images)
    
    images = torch.cat(image_list)
    # if images.shape[1] == 1:
    #     images = images.repeat([1, 3, 1, 1])

    return images.cpu()

def show(imgs, img_name):
    """
    img: pytorch tensor of multiple images
    img_name: e.g. "test.png"
    """
    img = imgs[0]
    print(img.shape)
    img_reshape = img.reshape((len(img[0]), len(img[1]))) # to get from shape [255, 255, 1] to [255, 255]
    im = Image.fromarray(img_reshape, mode="L")
    im.save(img_name)