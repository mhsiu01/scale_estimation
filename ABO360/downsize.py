import os
import shutil
from PIL import Image


def copy_downsized_img(src, dst, follow_symlinks=True):
    if str(src).endswith(".jpg"):
        img_num = int(str(src).removesuffix(".jpg")[-2:])
        if img_num%3==0:
            target = 288
            # Load, downsize, save with torch or PIL
            image = Image.open(src)
            
            width, height = image.size
            w_scale = width / target
            h_scale = height / target
            max_scale = max(w_scale, h_scale)
            if max_scale>1.0:
                image.thumbnail(size=(int(width/max_scale),int(height/max_scale)))
                
            image.save(dst)
    
    else:
        shutil.copy2(src, dst, follow_symlinks=follow_symlinks)

    return dst

def main():
    root = "/home/jovyan/fast-vol/ABO360/spins/"

    # Copy directory tree
    shutil.copytree(src=os.path.join(root),
                    dst="/home/jovyan/spins_small",
                    # ignore=shutil.ignore_patterns('*.jpg'),
                    copy_function=copy_downsized_img)

if __name__=='__main__':
    main()