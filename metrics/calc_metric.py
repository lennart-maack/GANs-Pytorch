"""
legacy is used for loading a pretrained network (Source: NVlabs stylegan2-ada-pytorch repository https://github.com/NVlabs/stylegan2-ada-pytorch)



"""

import legacy
import dnnlib

import click



def set_up_data_kwargs(data):
    assert data is not None
    assert isinstance(data, str)
    test_set_kwargs = dnnlib.EasyDict(class_name='dataset.ImageFolderDataset', path=data, use_labels=True, max_size=None, xflip=False)
    data_loader_kwargs = dnnlib.EasyDict(pin_memory=True, num_workers=3, prefetch_factor=2)
    try:
        training_set = dnnlib.util.construct_class_by_name(**test_set_kwargs) # subclass of training.dataset.Dataset
        test_set_kwargs.resolution = training_set.resolution # be explicit about resolution
        test_set_kwargs.use_labels = training_set.has_labels # be explicit about labels
        test_set_kwargs.max_size = len(training_set) # be explicit about dataset size
        desc = training_set.name
        del training_set # conserve memory
    except IOError as err:
        raise UserError(f'--data: {err}')

    print(test_set_kwargs)
    print(data_loader_kwargs)



@click.command()

@click.option('--data', help='Test data for calculating FID(directory or zip)', metavar='PATH', required=True)

def main(data):
    set_up_data_kwargs(data)


if __name__ == "__main__":
    main()