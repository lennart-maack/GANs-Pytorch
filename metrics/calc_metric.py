"""
legacy is used for loading a pretrained network (Source: NVlabs stylegan2-ada-pytorch repository https://github.com/NVlabs/stylegan2-ada-pytorch)

Description:
Command Line Application which takes a path to images (test images) --data path to .pkl file of a pretrained Generator Network --network and

Calculates:

- Histogram of color values of test images and generated images plotted against each other
- SSIM mean, SSIm std
- JS divergence
- the FID with 50k generated images and maximum number of test images.
- the KID

"""

import click
import torch
import pandas as pd
import os

import legacy
import dnnlib
import metric_utils
import get_fid
import get_kid
import metric_extended


def calc_fid(**kwargs):
    
    opts = metric_utils.MetricOptions(**kwargs)

    opts.dataset_kwargs.update(max_size=None, xflip=False)

    fid = get_fid.compute_fid(opts, max_real=None, num_gen=50000)
    
    return fid


def calc_kid(**kwargs):

    opts = metric_utils.MetricOptions(**kwargs)

    kid = get_kid.compute_kid(opts, max_real=1000000, num_gen=50000, num_subsets=100, max_subset_size=1000)

    return kid

def set_up_data_kwargs(data):
    assert data is not None
    assert isinstance(data, str)
    test_set_kwargs = dnnlib.EasyDict(class_name='dataset.ImageFolderDataset', path=data, use_labels=True, max_size=None, xflip=False)
    data_loader_kwargs = dnnlib.EasyDict(pin_memory=True, num_workers=2, prefetch_factor=2)
    try:
        training_set = dnnlib.util.construct_class_by_name(**test_set_kwargs) # subclass of training.dataset.Dataset
        test_set_kwargs.resolution = training_set.resolution # be explicit about resolution
        test_set_kwargs.use_labels = training_set.has_labels # be explicit about labels
        test_set_kwargs.max_size = len(training_set) # be explicit about dataset size
        desc = training_set.name
        del training_set # conserve memory
    except IOError as err:
        raise UserError(f'--data: {err}')

    return test_set_kwargs



@click.command()

@click.option('--data', help='Test data for calculating FID(directory or zip)', metavar='PATH', required=True)
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--network_kimgs', help='Needed to have a unique name for the metric_on_test_data.csv - type in on how many kimgs the model has been trained', required=True)
@click.option('--run_dir', help='Path to the dictonary where metrics_on_test_data.csv should be saved', \
    metavar='PATH', required=True)

def main(data, network_pkl, network_kimgs, run_dir):

    num_gpus = 1
    rank = 0
    device="cuda"

    test_set_kwargs = set_up_data_kwargs(data)

    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore
    print("Loaded network..")

    print("Calculate SSIM..")
    ssim_mean, ssim_std = metric_extended.get_ssim(G=G, dataset_kwargs=test_set_kwargs,
        num_gpus=num_gpus, rank=rank, device=device)

    print("SSIM mean, SSIM std: ", ssim_mean, ssim_std)

    print("Print Histogram")

    metric_extended.plot_hist(run_dir=run_dir, G=G, dataset_kwargs=test_set_kwargs,
        num_gpus=num_gpus, rank=rank, device=device)


    print("Calculate JS divergence..")
    js_div = metric_extended.get_JS_divergence(G=G, dataset_kwargs=test_set_kwargs,
        num_gpus=num_gpus, rank=rank, device=device)

    print("JS divergence: ", js_div)

    fid = calc_fid(G=G, dataset_kwargs=test_set_kwargs, num_gpus=num_gpus, rank=rank, device=device)

    print("FID score is: ", fid)

    # kid = calc_kid(G=G, dataset_kwargs=test_set_kwargs, num_gpus=num_gpus, rank=rank, device=device)

    # print("KID score is: ", kid)

    metric_df = pd.DataFrame(columns=["SSIM mean", "SSIM std", "JS", "fid50k_full"])

    metric_df = metric_df.append({"SSIM mean" : ssim_mean}, ignore_index=True)

    metric_df.loc[metric_df["SSIM mean"]== ssim_mean, "SSIM std"] = ssim_std
    metric_df.loc[metric_df["SSIM mean"]== ssim_mean, "JS"] = js_div
    metric_df.loc[metric_df["SSIM mean"]== ssim_mean, "fid50k_full"] = fid
    # metric_df.loc[metric_df["SSIM mean"]== ssim_mean, "kid50k_full"] = kid

    metric_df.to_csv(os.path.join(run_dir, f'metrics_on_test_data_with_model_{network_kimgs}.csv'), index=False)

    print(f"Exported metrics_on_test_data_with_model_{network_kimgs}.csv inside the folder {run_dir}")

if __name__ == "__main__":
    main()