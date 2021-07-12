import torchvision.transforms as transforms
import augly.image as imaugs
import os
import glob
from PIL import Image 
import PIL

CORE_PATH = '/mnt/batch/tasks/shared/LS_root/mounts/clusters/unet/code/Users/eskot/UNET-FUSION-CT-PET/train_unet_multiple_augmented'

AUGMENTATIONS_1 = [
        imaugs.Blur(),
        imaugs.PerspectiveTransform(sigma=20.0),
        imaugs.VFlip(),
        imaugs.HFlip(),
        imaugs.Rotate(degrees=15)
    ]

AUGMENTATIONS_2 = [
        imaugs.Blur(),
        imaugs.PerspectiveTransform(sigma=40.0),
        imaugs.VFlip(),
        imaugs.HFlip(),
        imaugs.Rotate(degrees=18)
    ]


def augment_dataset(dataset: list, is_second: bool = False):
    number = len(dataset)
    
    for no, img in enumerate(dataset):
        n = no + number + 1
        final_number = str(n).zfill(4)
        augment_slice(final_number, img, is_second)


def augment_slice(slice_number: str, img_path: str, is_second: bool = False):
    print(f"Processing number={slice_number} path={img_path}")

    if is_second:
        TRANSFORMS = imaugs.Compose(AUGMENTATIONS_2)
    else:
        TRANSFORMS = imaugs.Compose(AUGMENTATIONS_1)

    print(img_path)
    i = Image.open(img_path) 
    
    aug_img = TRANSFORMS(i) 
    
    mask_path = img_path.replace("img", "mask").replace(".png", "_mask.png")
    m = Image.open(mask_path)
    aug_mask = TRANSFORMS(m)

    os.mkdir(f"{CORE_PATH}/" + slice_number + "/")
    os.mkdir(f"{CORE_PATH}/" + slice_number + "/img/")
    os.mkdir(f"{CORE_PATH}/" + slice_number + "/mask/")
    
    aug_img.save(f"{CORE_PATH}/" + slice_number + "/img/" + slice_number + ".png")
    aug_mask.save(f"{CORE_PATH}/" + slice_number + "/mask/" + slice_number + "_mask.png")



def double_augmentation(path: str):
    first_aug_files = []
    for f in glob.glob(f'{path}/**/img/*.png', recursive=True):
        first_aug_files.append(f)

    print(len(first_aug_files))
    augment_dataset(first_aug_files)
    
    second_aug_files = []
    for f in glob.glob(f'{path}/**/img/*.png', recursive=True):
        second_aug_files.append(f)

    print(len(second_aug_files))
    augment_dataset(first_aug_files, True)

  
  
double_augmentation(CORE_PATH)
