import torch
import torchvision
torchvision.disable_beta_transforms_warning()
torchvision.disable_beta_transforms_warning()
from torchvision.transforms import v2
import torchvision.transforms.functional as tvf

from PIL import Image

import os
from tqdm import tqdm


def _calculate_img_normalize_values(loader, transform, device):
    print("Calculating image normalization constants...")

    with torch.no_grad():
        # Online mean and std via Welford's method
        # see: https://stackoverflow.com/a/15638726
        n = 0
        for i,(imgs_batch,labels_batch) in enumerate(tqdm(loader)):
            if i==0:
                img_mean = torch.zeros(size=imgs_batch[0].shape, dtype=torch.float32)
                img_M2 = torch.zeros(size=imgs_batch[0].shape, dtype=torch.float32)
            imgs_batch = imgs_batch.to(device)
            imgs_batch = transform(imgs_batch)
            imgs_batch = imgs_batch.cpu()
            
            for j,(img,label) in enumerate(zip(imgs_batch,labels_batch)):
                n += 1
                img_delta = img - img_mean
                img_mean += img_delta/n
                img_M2 += img_delta*(img-img_mean)
            torch.cuda.empty_cache()

        img_var = img_M2 / (n-1)
        img_std = torch.sqrt(img_var)
 
        print(f"Average img is: {img_mean}.")
        print(f"Img std is: {img_std}.")
        
    for i,channel in enumerate(img_mean.to(torch.uint8)):
        img = tvf.to_pil_image(channel)
        img.save(f"./img_mean_channel{i}.png")
        
    for i,channel in enumerate(img_std.to(torch.uint8)):
        img = tvf.to_pil_image(channel)
        img.save(f"./img_std_channel{i}.png")

    # display(tvf.to_pil_image(img_mean[0].to(torch.uint8)))
    # display(tvf.to_pil_image(img_mean[1].to(torch.uint8)))
    # display(tvf.to_pil_image(img_mean[2].to(torch.uint8)))
    # display(tvf.to_pil_image(img_std[0].to(torch.uint8)))
    # display(tvf.to_pil_image(img_std[1].to(torch.uint8)))
    # display(tvf.to_pil_image(img_std[2].to(torch.uint8)))

    # Convert float64 to float32 (was using 64 bits to avoid potential numerical issues)
    img_mean = img_mean.type(torch.float32)
    img_std = img_std.type(torch.float32)
    return img_mean, img_std


def _get_img_normalize_values(dataset_name, loader, img_transform, split_props, map_loc):
    # Get img normalization values if exist, else calculate from scratch
    if os.path.isfile(f"./normalize_{dataset_name}_imgs.pth"):
        normlz_dict = torch.load(f"./normalize_{dataset_name}_imgs.pth")
        IMG_MEAN = normlz_dict["IMG_MEAN"]
        IMG_STD = normlz_dict["IMG_STD"]
        assert dataset_name==normlz_dict["dataset_name"]
        print(f"Loading img normalization with data split = {normlz_dict['split_props']}")
    else:
        IMG_MEAN, IMG_STD = _calculate_img_normalize_values(loader, img_transform, map_loc)
    torch.save({
        "IMG_MEAN":IMG_MEAN,
        "IMG_STD":IMG_STD,
        "split_props":split_props,
        "dataset_name":dataset_name},
        f"./normalize_{dataset_name}_imgs.pth",
        _use_new_zipfile_serialization=False)
    return IMG_MEAN, IMG_STD



def _calculate_bbox_normalize_values(loader, bbox_transform):
    print("Calculating bbox normalization constants...")
    with torch.no_grad():
        collected_bboxes = []
        for (imgs_batch, bbox_batch) in tqdm(loader):
            collected_bboxes.append(bbox_batch.detach().clone())
        collected_bboxes = torch.cat(collected_bboxes, dim=0)
        print(f"{collected_bboxes.shape=}")
        if bbox_transform=="arctan":
            collected_bboxes = torch.arctan(collected_bboxes)
        elif bbox_transform=="none":
            pass
        # collected_bboxes = _apply_label_normalization(bbox_transform, collected_bboxes)
        bbox_mean = torch.mean(collected_bboxes, dim=0)
        bbox_std = torch.std(collected_bboxes, dim=0)
        print(f"{bbox_mean=}")
        print(f"{bbox_std=}")
    return bbox_mean, bbox_std


def _get_bbox_normalize_values(dataset_name, loader, bbox_transform, split_props):
    # Get bbox normalization values if exist, else calculate from scratch
    if os.path.isfile(f"./normalize_{dataset_name}_bboxes_{bbox_transform}.pth"):
        # Load up normalization constants
        normlz_dict = torch.load(f"./normalize_{dataset_name}_bboxes_{bbox_transform}.pth")
        BBOX_MEAN = normlz_dict["BBOX_MEAN"]
        BBOX_STD = normlz_dict["BBOX_STD"]
        assert bbox_transform==normlz_dict["bbox_transform"]
        assert dataset_name==normlz_dict["dataset_name"]
        print(f"Loading bbox normalization with data split = {normlz_dict['split_props']}")
    else:
        BBOX_MEAN, BBOX_STD = _calculate_bbox_normalize_values(loader, bbox_transform)

    torch.save({
        "BBOX_MEAN":BBOX_MEAN,
        "BBOX_STD":BBOX_STD,
        "split_props":split_props,
        "dataset_name":dataset_name,
        "bbox_transform":bbox_transform},
        f"./normalize_{dataset_name}_bboxes_{bbox_transform}.pth",
        _use_new_zipfile_serialization=False)
    return BBOX_MEAN, BBOX_STD

       