import numpy as np
import json
import csv

import os
from tqdm import tqdm
from pprint import pprint

import matplotlib.pyplot as plt
import pdb
import time

# Get global object metadata
prod_dicts = []
for f in tqdm(os.listdir("./fast-vol/ABO360/listings/metadata/")):
    with open(f"./fast-vol/ABO360/listings/metadata/{f}") as file:
        for line in file:
            prod_dicts.append(json.loads(line))

# Get catalog image metadata
with open(f"./fast-vol/ABO360/images/metadata/images.csv") as file:
    reader = csv.reader(file, delimiter=",", quotechar='"')
    next(reader, None)  # skip the headers
    images_csv = [row for row in reader]

# Get spins image metadata
with open(f"./fast-vol/ABO360/spins/metadata/spins.csv") as file:
    reader = csv.reader(file, delimiter=",", quotechar='"')
    next(reader, None)  # skip the headers
    spins_csv = [row for row in reader]

print(f"{len(prod_dicts)=}")
print(f"{len(images_csv)=}")
print(f"{len(spins_csv)=}")

from IPython.display import display
from PIL import Image

def get_image(csv_row):
    if len(csv_row)==4:
        img_path = csv_row[3]
        img_path = f"./fast-vol/ABO360/images/small/{img_path}"
    elif len(csv_row)==6:
        img_path = csv_row[5]
        img_path = f"./fast-vol/ABO360/spins/original/{img_path}"
    img = Image.open(img_path)
    return img

from matplotlib.pyplot import figure, imshow, axis
from matplotlib.image import imread

def showImagesHorizontally(list_of_imgs, title=None):
    fig = figure()
    number_of_imgs = len(list_of_imgs)
    if title is not None:
        fig.suptitle(title)
    for i in range(number_of_imgs):
        a=fig.add_subplot(1,number_of_imgs,i+1)
        image = list_of_imgs[i]
        imshow(image,cmap='Greys_r')
        axis('off')

# Create dict to map image id to its metadata (ie. csv row)
images_csv_dict = {}
for row in images_csv:
    id = row[0]
    images_csv_dict[id] = row

# Create dict to map spin image id to its metadata (ie. csv row)
spins_csv_dict = {}
for row in spins_csv:
    id = row[2]
    spins_csv_dict[id] = row



def validate_dims(dim_units, dim_values):
    # Note: feet, meters, micrometer almost never show up...
    accepted_units = ['centimeters', 'feet', 'inches', 'meters', 'micrometer', 'millimeters']
    valid = True
    # Should be a standard unit system (metric or imperial)
    for unit in dim_units:
        if unit not in accepted_units:
            valid = False
    # Should use the same unit system for length, width, height
    if not (dim_units[0]==dim_units[1] and dim_units[1]==dim_units[2]):
        valid = False
    # Should be positive values
    if not (dim_values[0]>0 and dim_values[1]>0 and dim_values[2]>0):
        valid = False
    return valid
        
def to_meters(dim_units, dim_values):
    dim_values = np.array(dim_values, dtype=np.float32)
    if dim_units[0]=='centimeters':
        dim_values*=0.01    
    elif dim_units[0]=='feet':
        dim_values*=0.3048
    elif dim_units[0]=='inches':
        dim_values*=0.0254
    elif dim_units[0]=='meters':
        dim_values
    elif dim_units[0]=='micrometer':
        dim_values*=1e-6
    elif dim_units[0]=='millimeters':
        dim_values*=0.001
    else:
        dim_values = None
    assert dim_values is not None
    return dim_values.tolist()
        
num_img_freqs = [] # Record distribution of number of images per product
imgID_to_object = {} # Map image id to its object's metadata
spinID_to_object = {} # Map spin id to its object's metadata

num_conflicts = 0 # Counts times one image used by two different products
has_spin_id = 0 # Counts size-labeled products that also appear in spins data

# Filter for products with labeled sizes
for idx, d in enumerate(tqdm(prod_dicts)):
    # Extract dimensions
    if "item_dimensions" in d:
        dim_units = (
            d["item_dimensions"]["width"]["unit"],
            d["item_dimensions"]["height"]["unit"],
            d["item_dimensions"]["length"]["unit"]
        )
        dim_values = (
            d["item_dimensions"]["width"]["value"],
            d["item_dimensions"]["height"]["value"],
            d["item_dimensions"]["length"]["value"]
        )
    else:
        continue
        
    # Validate, standardize to meters
    if validate_dims(dim_units, dim_values):
        meter_values = to_meters(dim_units, dim_values)
        w,h,l = meter_values
        meter_values = (round(w,5),round(h,5),round(l,5))
    else:
        continue
        
    # Enforce sane/realistic object size
    sane_size = (w>=0.01 and w<=5.0) and (h>=0.01 and h<=5.0) and (l>=0.01 and l<=5.0)
    if not sane_size:
        continue

    # Write relevant metadata
    subdict = {
        "item_dimensions": meter_values,
        "axes_order": ("width","height","length"),
        "dimensions_unit": "meters"
    }
    for k in ["item_id", "item_keywords", "item_name", "item_weight", "product_type", "main_image_id", "other_image_id", "spin_id", "product_type"]:
        if k in d:
            subdict[k] = d[k]

    # Get all image id's for this product
    ids = []
    num_ids = 0
    if "spin_id" in d:
        has_spin_id += 1
        if d["spin_id"] not in spinID_to_object:
            spinID_to_object[d["spin_id"]] = [subdict]
        else:
            spinID_to_object[d["spin_id"]] = spinID_to_object[d["spin_id"]] + [subdict]
            # print(d["spin_id"])
            # print()
    if "main_image_id" in d:
        ids += [d["main_image_id"]]
        num_ids += 1
    if "other_image_id" in d:
        ids += d["other_image_id"]
        num_ids += len(d["other_image_id"])
    assert num_ids==len(ids)
    if num_ids==0:
        continue
    else:
        num_img_freqs.append(num_ids)

    # Create mapping from image_id to index of its prod_dict in the filtered list
    conflict_ids = []
    for id in ids:
        if id not in imgID_to_object:
            imgID_to_object[id] = subdict
        else:
            num_conflicts += 1
            conflict_ids.append(id)
            
    # # (optionally) Visualize product images
    # list_of_imgs = [get_image(images_csv_dict[id]) for id in ids if id not in conflict_ids]
    # showImagesHorizontally(list_of_imgs)
            
print(f"{len(imgID_to_object)=}")
print(f"{num_conflicts=}")

print(f"{len(spinID_to_object)=}")
print(f"{has_spin_id=}")

# There's some item duplication in prod_dicts, so ~10 spin_id conflicts
# from the same product being listed twice. But it's negligible.

spin_id_dupes = 0
for k,dictlist in spinID_to_object.items():
    if len(dictlist)>1:
        spin_id_dupes += 1
        print(f"id {k} has {len(dictlist)} duplicates.")
        # for d in dictlist:
        #     print(f"{d['spin_id']=}")
        #     print(d)
        print()
        
    spinID_to_object[k] = dictlist[0]
print(f"{spin_id_dupes} total duplicates out of {len(spinID_to_object)} spin_id's.")


with open("./images_csv_dict.json", "w") as f:
    json.dump(images_csv_dict, f)

with open("./imgID_to_object.json", "w") as f:
    json.dump(imgID_to_object, f)

with open("./spins_csv_dict.json", "w") as f:
    json.dump(spins_csv_dict, f)

with open("./spinID_to_object.json", "w") as f:
    json.dump(spinID_to_object, f)
