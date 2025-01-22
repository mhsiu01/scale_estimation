import os
# import argparse
import sys
import pdb
from tqdm import tqdm

import numpy as np
import json

# Import training libraries: torch, tqdm, tensorboard
import torch
import torchvision
torchvision.disable_beta_transforms_warning()
torchvision.disable_beta_transforms_warning()
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
import torchvision.transforms.functional as tvf

# fix RNG seed for reproducibility
import random
SEED = 123
random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)
"""
class ObjectsDataset(Dataset):
    def __init__(self,
                 file_dir="/home/jovyan/fast-vol/trimesh_rendering/renders",
                 asset_database_dir="/home/jovyan/fast-vol/ai2thorhab-uncompressed/assets/asset-database.json",
                 train_mode="per_img",
                 ):
        # Load object metadata from json
        with open(asset_database_dir) as f:
            d = json.load(f)

        # Flatten database to (objectId: object_bbox) pairs
        all_objects = {}
        # Loop through every object class...
        for objClass,objList in d.items():
            # ...and get object ids + bbox
            for objDict in objList:
                object_id = objDict['assetId']
                object_bbox = objDict['boundingBox']
                # Per object_id, store its bbox and object class
                all_objects[object_id] = {}
                all_objects[object_id]['bbox'] = object_bbox
                all_objects[object_id]['object_class'] = objClass
        
        self.asset_database = all_objects

        # Load all img file names
        self.file_dir = file_dir
        if train_mode=="per_img":
            imgs = os.listdir(file_dir)
            imgs = [img for img in imgs if img.endswith(".png")]  
            self.imgs = imgs

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        # Read img from file to CHW Tensor
        img = torchvision.io.read_image(
            os.path.join(self.file_dir, self.imgs[idx])
        ).float()
        # img = self.transform(img)

        # Scale to [-1,1] range
        # img = ((img.float()/255.) - 0.5) * 2
        
        # Get object size labels
        object_id, _, _ = self.imgs[idx].partition(".glb") # Keep object id
        bbox = self.asset_database[object_id]['bbox']
        bbox = torch.tensor([bbox['x'],bbox['y'],bbox['z']])

        # TODO: normalize bbox/scale values
        # bbox = (bbox - self.bbox_avg) / self.bbox_std
        return (img, bbox) #, object_id)
"""
# https://discuss.pytorch.org/t/torchvision-transforms-set-fillcolor-for-centercrop/98098/3
class PadCenterCrop(object):
    def __init__(self, final_size, pad_if_needed=True, fill=255, padding_mode='constant'):
        self.final_size = final_size
        self.pad_if_needed = pad_if_needed
        self.padding_mode = padding_mode
        self.fill = fill

    def _get_img_dims(self, img):
        if len(img.shape)==3:
            c,h,w = img.shape[0], img.shape[1], img.shape[2]
        elif len(img.shape)==4:
            c,h,w = img.shape[1], img.shape[2], img.shape[3]
        return c,h,w
        
    # Assume torch image of shape CHW
    def __call__(self, img):
        # Resize with longer edge being final_size
        c,h,w = self._get_img_dims(img)
        longer = float(max(h,w))
        ratio = self.final_size / longer
        if longer>self.final_size:
            img = tvf.resize(img, size=(int(h*ratio)+1,int(w*ratio)+1), antialias=True)
            # print(f"Resized by ratio {round(ratio, 2)}")
        # Check for grayscale, convert to 3-channel if so
        if c==1:
            img = img.expand(3,-1,-1) 
            # print(f"Expanded img with {c} channels.")
        # print(f"Resized image has size: {img.shape}")
        
        # Pad the shorter edge to make whole img square
        # H<W, ie. landscape
        c,h,w = self._get_img_dims(img)
        if self.pad_if_needed and h < w: 
            pad_amount = int((w - h)/2)
            # print(f"Padding {pad_amount} on top and bottom.")
            img = tvf.pad(img, (0, pad_amount+2), fill=self.fill, padding_mode=self.padding_mode)
        # or W<H, ie. portrait
        if self.pad_if_needed and w < h:
            pad_amount = int((h - w)/2)
            # print(f"Padding {pad_amount} on left and right.")
            img = tvf.pad(img, (pad_amount+2, 0), fill=self.fill, padding_mode=self.padding_mode)
        # print(f"Padded image has size: {img.shape}")
        
        return tvf.center_crop(img, self.final_size)
"""
class ABODataset(Dataset):
    def __init__(self,
                 # file_dir="./fast-vol/trimesh_rendering/renders",
                 # device,
                 final_size=256,
                 data_root = "/home/jovyan/fast-vol/ABO360/spins/original",
                 metadata_dir = "/home/jovyan/fast-vol/ABO360/clean_metadata",
                 mode="spins",
                 base_transform=v2.Compose([
                     # Resize and pad to standard-size square
                     PadCenterCrop(final_size=256), 
                     # v2.ToDtype(torch.float32),
                 ]),
                 sort_bbox=True,
                 use_subset=True
                 ):
        # Load product metadata and image database
        self.final_size=final_size
        self.data_root = data_root
        self.metadata_dir = metadata_dir
        self.mode = mode
        self.base_transform = base_transform
        self.sort_bbox = sort_bbox
        # self.device = device
        self.use_subset = use_subset
         
        # Pretraining
        if mode=="catalog":
            with open(os.path.join(metadata_dir, "images_csv_dict.json")) as f:
                self.csv_dict = json.load(f)
            with open(os.path.join(metadata_dir, "imgID_to_object.json")) as f:
                self.id_to_object = json.load(f)
            self.ids = list(self.id_to_object.keys()) # Image ids for individual catalog imgs
        # Finetuning
        elif mode=="spins":
            with open(os.path.join(metadata_dir, "spins_csv_dict.json")) as f:
                self.csv_dict = json.load(f)
            with open(os.path.join(metadata_dir, "spinID_to_object.json")) as f:
                self.id_to_object = json.load(f)
            # For spins csv data, need to remove/ignore rows of images whose
            # corresponding spinID/object does not have size labels, aka does not
            # appear in the id_to_object dict.
            imgIDs_without_sizes = []
            for i,(id,row) in enumerate(self.csv_dict.items()):
                if row[0] not in self.id_to_object:
                    imgIDs_without_sizes.append(id)
            for id in imgIDs_without_sizes:
                popped = self.csv_dict.pop(id, None)
            # Optionally take subset of 72 images, eg. every 3rd image
            if self.use_subset:
                self.ids = []
                for k,v in self.csv_dict.items():
                    # 2nd element of list corresponds to idx out of 72.
                    # Check if it's a multiple of 3.
                    if int(v[1])%3==0:
                        self.ids.append(k)
                print(f"Taking 1/3 subset of spins dataset, eg. 24 of 72 imgs per object.")
            else:
                self.ids = list(self.csv_dict.keys()) # Image ids for individual spin imgs
        else:
            raise ValueError("Invalid training mode.")
        
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        # Get one image id and its csv row
        id = self.ids[idx]
        csv_row = self.csv_dict[id]
        
        # Parse csv for img path and also load object metadata
        if self.mode=="catalog":    
            img_path = f"/home/jovyan/fast-vol/ABO360/images/small/{csv_row[3]}"
            obj_metadata = self.id_to_object[id] # Map img_id-->object
        elif self.mode=="spins":
            img_path = os.path.join(self.data_root, f"{csv_row[5]}")
            # img_path = f"/home/jovyan/fast-vol/ABO360/spins/original/{csv_row[5]}"
            obj_metadata = self.id_to_object[csv_row[0]] # Map img_id-->spin_id-->object
            
        # Load image
        img = torchvision.io.read_image(path=os.path.join(img_path)) #.to(self.device)
        img = self.base_transform(img)
        # Extract bbox
        bbox = list(obj_metadata["item_dimensions"])
        if self.sort_bbox:
            bbox = sorted(bbox)
        bbox = torch.tensor(bbox) # (w,h,l) tuple sorted by magnitude, then to tensor
        return (img, bbox)

"""


def partitionCatalogPerItem(
    split_props,
    metadata_dir = "/home/jovyan/fast-vol/ABO360/clean_metadata",
    use_subset=True
):  
    # Load metadata
    with open(os.path.join(metadata_dir, "images_csv_dict.json")) as f:
        csv_dict = json.load(f)
    with open(os.path.join(metadata_dir, "imgID_to_object.json")) as f:
        id2obj = json.load(f)
    for id,obj in tqdm(id2obj.items()):
        id2obj[id] = {
            "item_id": id2obj[id]["item_id"],
            "item_dimensions": id2obj[id]["item_dimensions"],
            # TODO: "main_and_other_image_ids"
        }
    
    # Get all item ids
    allItemIDs = []
    for imgID,prodDict in tqdm(id2obj.items()):
        assert 'item_dimensions' in prodDict.keys()
        allItemIDs.append(prodDict['item_id'])
    
    # Deduplicate, sort, shuffle
    allItemIDs = list(set(allItemIDs)) 
    allItemIDs.sort()
    random.shuffle(allItemIDs)

    # Split into train/val/test at item granularity
    N = len(allItemIDs)
    train_cutoff = int(N*split_props[0])
    val_cutoff = train_cutoff + int(N*split_props[1])
    train_itemIDs = allItemIDs[0:train_cutoff]
    val_itemIDs = allItemIDs[train_cutoff:val_cutoff]
    test_itemIDs = allItemIDs[val_cutoff:]
    assert (len(train_itemIDs) + len(val_itemIDs) + len(test_itemIDs))==N
    print(f"Total {N} item ID's partitioned into train/val/test of sizes: {len(train_itemIDs)} / {len(val_itemIDs)} / {len(test_itemIDs)}.")
    print(f"Reproducibility check -- first 10 itemIDs: {allItemIDs[:10]}")

    # Create datasets
    train_dataset = ABOCatalogPartition(item_ids=train_itemIDs,
                                      split="train", use_subset=use_subset)
    val_dataset = ABOCatalogPartition(item_ids=val_itemIDs,
                                    split="val", use_subset=use_subset)
    test_dataset = ABOCatalogPartition(item_ids=test_itemIDs,
                                     split="test", use_subset=use_subset)
    return train_dataset, val_dataset, test_dataset
    
class ABOCatalogPartition(Dataset):
    def __init__(self,
                 item_ids,
                 split,
                 final_size=256,
                 data_root = "/home/jovyan/fast-vol/ABO360/images/small",
                 metadata_dir = "/home/jovyan/fast-vol/ABO360/clean_metadata",
                 base_transform=v2.Compose([
                     # Resize and pad to standard-size square
                     PadCenterCrop(final_size=256), 
                 ]),
                 sort_bbox=True,
                 use_subset=None
                 ):
        # Load product metadata and image database
        self.item_ids = set(item_ids)
        self.split = split
        self.final_size=final_size
        self.data_root = data_root
        self.metadata_dir = metadata_dir
        self.base_transform = base_transform
        self.sort_bbox = sort_bbox
        self.use_subset = use_subset
         
        # Load metadata
        with open(os.path.join(metadata_dir, "images_csv_dict.json")) as f:
            self.imgs_csv = json.load(f)
        with open(os.path.join(metadata_dir, "imgID_to_object.json")) as f:
            self.img2obj = json.load(f)

        # Final list of valid image ids
        self.img_ids = []
        for imgID,objMeta in tqdm(self.img2obj.items()):
            if objMeta["item_id"] in self.item_ids:
                self.img_ids.append(imgID)
        
    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        # Get one image id and its csv row
        imgID = self.img_ids[idx]
        csv_row = self.imgs_csv[imgID] # Map imgID to csv row
        obj_metadata = self.img2obj[imgID] # Map imgID to its object metadata
        
        # Get image path
        img_path = os.path.join(self.data_root, f"{csv_row[3]}")            
        # Load image
        img = torchvision.io.read_image(path=os.path.join(img_path))
        img = self.base_transform(img)
        
        # Extract bbox
        bbox = list(obj_metadata["item_dimensions"])
        if self.sort_bbox:
            bbox = sorted(bbox)
        bbox = torch.tensor(bbox) # (w,h,l) tuple sorted by magnitude, then to tensor
        return (img, bbox)
    





def partitionSpinsPerObject(
    split_props,
    metadata_dir = "/home/jovyan/fast-vol/ABO360/clean_metadata",
    use_subset=True
):
    
    with open(os.path.join(metadata_dir, "spinID_to_object.json")) as f:
        spinID_to_object = json.load(f)
        
    # Sort and shuffle with fixed random seed for reproducible order
    allSpinIDs = sorted(list(spinID_to_object.keys()))
    random.shuffle(allSpinIDs)
    N = len(allSpinIDs)
    train_cutoff = int(N*split_props[0])
    val_cutoff = train_cutoff + int(N*split_props[1])
    train_spinIDs = allSpinIDs[0:train_cutoff]
    val_spinIDs = allSpinIDs[train_cutoff:val_cutoff]
    test_spinIDs = allSpinIDs[val_cutoff:]
    assert (len(train_spinIDs) + len(val_spinIDs) + len(test_spinIDs))==N
    print(f"Total {N} spin ID's partitioned into train/val/test of sizes: {len(train_spinIDs)} / {len(val_spinIDs)} / {len(test_spinIDs)}.")
    print(f"Reproducibility check -- first 10 spinIDs: {allSpinIDs[:10]}")

    # Create partition datasets
    train_dataset = ABOSpinsPartition(spin_ids=train_spinIDs,
                                      split="train", use_subset=use_subset)
    val_dataset = ABOSpinsPartition(spin_ids=val_spinIDs,
                                    split="val", use_subset=use_subset)
    test_dataset = ABOSpinsPartition(spin_ids=test_spinIDs,
                                     split="test", use_subset=use_subset)
    return train_dataset, val_dataset, test_dataset

class ABOSpinsPartition(Dataset):
    def __init__(self,
                 spin_ids,
                 split,
                 final_size=256,
                 data_root = "/home/jovyan/fast-vol/ABO360/spins/original",
                 metadata_dir = "/home/jovyan/fast-vol/ABO360/clean_metadata",
                 base_transform=v2.Compose([
                     # Resize and pad to standard-size square
                     PadCenterCrop(final_size=256), 
                 ]),
                 sort_bbox=True,
                 use_subset=True
                 ):
        # Load product metadata and image database
        self.spin_ids = spin_ids
        self.split = split
        self.final_size=final_size
        self.data_root = data_root
        self.metadata_dir = metadata_dir
        self.base_transform = base_transform
        self.sort_bbox = sort_bbox
        self.use_subset = use_subset
        if self.use_subset:
            print(f"On split {self.split}: Taking 1/3 subset of spins dataset, eg. 24 of 72 imgs per object.")
         
        # Load metadata
        with open(os.path.join(metadata_dir, "spins_csv_dict.json")) as f:
            self.imgID_to_spinID = json.load(f)
        with open(os.path.join(metadata_dir, "spinID_to_object.json")) as f:
            self.spinID_to_object = json.load(f)
            
        # For spins csv data, need to remove/ignore rows of images whose
        # corresponding spinID/object does not have size labels, aka does not
        # appear in the id_to_object dict.
        imgIDs_without_sizes = []
        for i,(imgID,row) in enumerate(self.imgID_to_spinID.items()):
            if row[0] not in self.spinID_to_object:
                imgIDs_without_sizes.append(imgID)
        for imgID in imgIDs_without_sizes:
            popped = self.imgID_to_spinID.pop(imgID, None)

        # Final list of valid image ids
        self.img_ids = []
        for imgID,row in self.imgID_to_spinID.items():
            # Optionally take subset of 72 images, eg. every 3rd image.
            if self.use_subset:
                if int(row[1])%3!=0:
                    continue
            # Include only imgID's belonging to given spin_ids.
            if row[0] in self.spin_ids:
                self.img_ids.append(imgID)
        
    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        # Get one image id and its csv row
        imgID = self.img_ids[idx]
        csv_row = self.imgID_to_spinID[imgID]
        spinID = csv_row[0]
        img_loc = csv_row[5]
        # Get image path
        img_path = os.path.join(self.data_root, f"{img_loc}")
        # Load object metadata: idx-->imgID-->spinID-->object
        obj_metadata = self.spinID_to_object[spinID] 
            
        # Load image
        img = torchvision.io.read_image(path=os.path.join(img_path))
        img = self.base_transform(img)
        # Extract bbox
        bbox = list(obj_metadata["item_dimensions"])
        if self.sort_bbox:
            bbox = sorted(bbox)
        bbox = torch.tensor(bbox) # (w,h,l) tuple sorted by magnitude, then to tensor
        return (img, bbox)
    