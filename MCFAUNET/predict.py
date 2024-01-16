import os
import time
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
import numpy as np
from PIL import Image

from Mynet.MCFAUNET.src.MCFAUnet import MCFAUnet
from image_segmentation_model.unet.src.unet import UNet
from Mynet.MCFAUNET.src.MCFUnet import MCFUnet


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def main():
    classes = 1  # exclude background
    weights_path = "save_weights/best_model1129modify_xiacaiyang.pth"
    img_path = "G:\\datas_all\\list\\data\\LiTS\\test\\ct\\0000_0000_0009.bmp"
    # img_path = "../../images/dong.png"
    # roi_mask_path = "./DRIVE/test/mask/01_test_mask.gif"
    assert os.path.exists(weights_path), f"weights {weights_path} not found."
    assert os.path.exists(img_path), f"image {img_path} not found."
    # assert os.path.exists(roi_mask_path), f"image {roi_mask_path} not found."

    mean = (0.709, 0.381, 0.224)
    std = (0.127, 0.079, 0.043)

    # get devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # create model
    # model = UNet(in_channels=3, num_classes=classes+1, base_c=32)
    # model = MCFUnet(in_channels=3, num_classes=classes+1, base_c=32)
    model = MCFAUnet(in_channels=3, num_classes=classes+1, base_c=32)

    # load weights
    model.load_state_dict(torch.load(weights_path, map_location='cpu')['model'])
    model.to(device)

    # load roi mask
    # roi_img = Image.open(roi_mask_path).convert('L')  # 转换为灰度图像
    # roi_img = np.array(roi_img)  # 转换为数组

    # load image
    original_img = Image.open(img_path).convert('RGB')
    # from pil image to tensor and normalize
    imgsize =(256,256)
    data_transform = transforms.Compose([transforms.Resize(imgsize),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=mean, std=std)])
    # transforms.Resize 是PyTorch中的一个图像变换操作，用于调整图像的大小。
    #   它可以按照指定的尺寸对图像进行缩放，可以是固定的尺寸，也可以是根据比例进行缩放。
    # torch.reshape 是PyTorch中的一个张量变换操作，用于改变张量的形状。它可以将张量重新排列成指定的形状，但不改变张量中元素的总数。
    img = data_transform(original_img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)  # 变成[1,C,H,W]具有批次的4维张量

    model.eval()  # 进入验证模式
    with torch.no_grad():
        # init model
        img_height, img_width = img.shape[-2:]
        init_img = torch.zeros((1, 3, img_height, img_width), device=device)
        model(init_img)

        t_start = time_synchronized()
        output = model(img.to(device))
        t_end = time_synchronized()
        print("inference time: {}".format(t_end - t_start))

        prediction = output['out'].argmax(1).squeeze(0)
        prediction = prediction.to("cpu").numpy().astype(np.uint8)
        # 将前景对应的像素值改成255(白色)
        prediction[prediction == 1] = 255
        # 将不敢兴趣的区域像素设置成0(黑色)
        # prediction[roi_img == 0] = 0
        mask = Image.fromarray(prediction)
        mask.save("test_result_unet.png")


if __name__ == '__main__':
    main()
    # roi_mask_path = "./DRIVE/test/mask/01_test_mask.gif"
    # roi_img = Image.open(roi_mask_path).convert('L')  # 转换为灰度图像
    # plt.imshow(roi_img)
    # plt.show()