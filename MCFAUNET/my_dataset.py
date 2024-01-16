import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

train_data_path = 'G:\\datas_all\\list\\data\\LiTS\\training'
val_data_path = 'G:\\datas_all\\list\\data\\LiTS\\test'


class DriveDataset(Dataset):
    def __init__(self, root: str, train: bool, transforms=None):
        super(DriveDataset, self).__init__()
        self.transforms = transforms
        self.flag = 'trainning' if train else 'test'
        data_root = os.path.join(root, self.flag)

        assert os.path.exists(data_root), f"path '{data_root}' does not exists."
        img_names = [i for i in os.listdir(os.path.join(data_root, "ct")) if i.endswith(".bmp")]
        self.img_list = [os.path.join(data_root, "ct", i) for i in img_names]  # 获取每个图片的路径
        self.manual = [os.path.join(data_root, "liver", i) for i in img_names]
        for i in self.manual:
            if os.path.exists(i) is False:
                raise FileNotFoundError(f"file {i} does not exists.")

        # self.flag = "training" if train else "test"
        # data_root = os.path.join(root, "DRIVE", self.flag)
        # assert os.path.exists(data_root), f"path '{data_root}' does not exists."
        # self.transforms = transforms
        # img_names = [i for i in os.listdir(os.path.join(data_root, "images")) if i.endswith(".tif")]  # 遍历当下目录
        # self.img_list = [os.path.join(data_root, "images", i) for i in img_names]  # 获取每个图片的路径
        # self.manual = [os.path.join(data_root, "1st_manual", i.split("_")[0] + "_manual1.gif")  #
        #                for i in img_names]
        # # check files
        # for i in self.manual:
        #     if os.path.exists(i) is False:
        #         raise FileNotFoundError(f"file {i} does not exists.")
        #
        # self.roi_mask = [os.path.join(data_root, "mask", i.split("_")[0] + f"_{self.flag}_mask.gif")
        #                  for i in img_names]
        # # check files
        # for i in self.roi_mask:
        #     if os.path.exists(i) is False:
        #         raise FileNotFoundError(f"file {i} does not exists.")

    def __getitem__(self, idx):
        img = Image.open(self.img_list[idx]).convert('RGB')
        manual = Image.open(self.manual[idx]).convert('L')  # L 变成灰度图片
        manual = np.array(manual) / 255
        # 这里转回PIL的原因是，transforms中是对PIL数据进行处理
        manual = Image.fromarray(manual)  # 很奇怪 源代码 这个mask 输出为全黑！！
        if self.transforms is not None:
            img, manual = self.transforms(img, manual)
        return img, manual

        # img = Image.open(self.img_list[idx]).convert('RGB')
        # manual = Image.open(self.manual[idx]).convert('L')  # L 变成灰度图片
        # manual = np.array(manual) / 255
        # roi_mask = Image.open(self.roi_mask[idx]).convert('L')
        # roi_mask = 255 - np.array(roi_mask)
        # mask = np.clip(manual + roi_mask, a_min=0, a_max=255)
        #
        # # 这里转回PIL的原因是，transforms中是对PIL数据进行处理
        # mask = Image.fromarray(mask)
        #
        # if self.transforms is not None:
        #     img, mask = self.transforms(img, mask)
        # return img, mask

    def __len__(self):
        return len(self.img_list)

    @staticmethod
    def collate_fn(batch):  ######  将图片和目标标签打包成一个批次（batch）
        images, targets = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=255)
        return batched_imgs, batched_targets


def cat_list(images, fill_value=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs


if __name__ == '__main__':
    root= 'G:\\datas_all\\list\\data\\LiTS'
    dr = DriveDataset('G:\\datas_all\\list\\data\\LiTS', True)
    print(dr.img_list)
    img, manual = dr[1]
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.subplot(1, 3, 2)
    plt.imshow(manual)
    plt.subplot(1, 3, 3)
    plt.imshow(manual, cmap='gray')
    plt.show()
