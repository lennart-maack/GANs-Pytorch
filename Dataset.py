
import torchvision
from torchvision import transforms
import torch.utils.data as data_utils

from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm


class Dataset:

    def __init__(self, dataset_type, dataset_size):
        self.dataset_type = dataset_type
        self.dataset_size = dataset_size
        self.transform = transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
                               )

    def __choose_dataset(self):

        if (self.dataset_type=="CIFAR10"):
            
            dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                transform=self.transform,
                download=True)
            
            return dataset

    def __split_dataset(self, dataset):
        '''
        Function to return a split of dataset (new_dataset). This may take a while depending on the size of the dataset
        Parameters:
        dataset: dataset of class torch.utils.data.Dataset
        dataset_size: how big the new dataset size should be - in percent 0-1
        '''
        x = []
        y = []
        for i in tqdm(range(len(dataset))):
            y.append(dataset[i][1])
            x.append(i)

        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=self.dataset_size, stratify=y)
        new_dataset = data_utils.Subset(dataset, x_train)
        
        return new_dataset

    def get_dataset_size(self, dataset):
        """
        Works only for datasets with 10 classes - needs to get updated
        """
        appe_list = []
        for i in tqdm(range(len(dataset))):
            appe_list.append(dataset[i][1])

        print(len(appe_list))
        print(appe_list.count(0))
        print(appe_list.count(1))
        print(appe_list.count(2))
        print(appe_list.count(3))
        print(appe_list.count(4))
        print(appe_list.count(5))
        print(appe_list.count(6))
        print(appe_list.count(7))
        print(appe_list.count(8))
        print(appe_list.count(9))

    def get_dataset(self):

        if self.dataset_size != 1:
            dataset = self.__split_dataset(self.__choose_dataset())
        else:
            dataset = self.__choose_dataset()

        return dataset
            


