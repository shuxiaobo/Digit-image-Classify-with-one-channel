from torch.utils.data.dataset import Dataset
import numpy as np

class MyDataSet(Dataset):
    """Dataset wrapping images and target labels for Kaggle - Planet Amazon from Space competition.

    Arguments:
        A CSV file path
        Path to image folder
        Extension of images
        PIL transforms
    """

    def __init__(self, npz_path = 'data.npz', train=True, transform=None, target_transform=None):
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.npz_path = npz_path
        npz_file = np.load(npz_path)

        # if self.train:
        self.train_X = np.array(npz_file['train_X'].astype(np.uint8))
        self.train_y = np.array(npz_file['train_y'].astype(np.int))
        # self.train_y = self.train_y.reshape(self.train_y.shape[0], 1)

        self.train_X = self.train_X.reshape(self.train_X.shape[0],1, 64, 64).transpose((0, 2, 3, 1))
        # else:
        self.test_X = np.array(npz_file['test_X'].astype(np.uint8))
        self.test_X = self.test_X.reshape(self.test_X.shape[0], 1, 64, 64).transpose((0, 2, 3, 1))


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_X[index], self.train_y[index]
        else:
            img, target = self.test_X[index], self.train_y[index] # notice the test_y don't filled here ,you cannot use the to predict

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        # img = Image.fromarray(img)
        # 草拟吗啊，要转坐标轴啊 N * C * H * W


        if self.transform is not None:
            img = self.transform(img)
        else:
            img = img.transpose(2, 0, 1)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_X)
        else:
            return len(self.test_X)