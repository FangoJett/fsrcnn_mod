import argparse
import glob
import h5py
import numpy as np
import PIL.Image as pil_image
from utils import calc_patch_size, convert_rgb_to_y
import os


# python prepare.py --images-dir (HR_IMAGES_DIR) --low_res-dir (LR_IMAGES_DIR) --output-path (dataset_file_name.h5) --scale 4

@calc_patch_size
def train(args):
    h5_file = h5py.File(args.output_path, 'w')
    LR_FOLDER = args.low_res_path
    lr_patches = []
    hr_patches = []

    for image_path in sorted(glob.glob('{}/*'.format(args.images_dir))):
        image_name = os.path.basename(image_path)
        hr = pil_image.open(image_path).convert('RGB')
        hr_images = []

        if args.with_aug:
            for s in [1.0, 0.9, 0.8, 0.7, 0.6]:
                for r in [0, 90, 180, 270]:
                    tmp = hr.resize((int(hr.width * s), int(hr.height * s)), resample=pil_image.BICUBIC)
                    tmp = tmp.rotate(r, expand=True)
                    hr_images.append(tmp)
        else:
            hr_images.append(hr)

        for hr in hr_images:
            hr_width = (hr.width // args.scale) * args.scale
            hr_height = (hr.height // args.scale) * args.scale
            hr = hr.resize((hr_width, hr_height), resample=pil_image.BICUBIC)
            img_lr_path = os.path.join(LR_FOLDER, image_name)
            lr = pil_image.open(img_lr_path).convert('RGB')
            #lr = downscale_image_with_cubic_b_spline_color(hr,4)
            #hr.resize((hr.width // args.scale, hr_height // args.scale), resample=pil_image.BICUBIC)
            hr = np.array(hr).astype(np.float32)
            lr = np.array(lr).astype(np.float32)
            hr = convert_rgb_to_y(hr)
            lr = convert_rgb_to_y(lr)

            for i in range(0, lr.shape[0] - args.patch_size + 1, args.scale):
                for j in range(0, lr.shape[1] - args.patch_size + 1, args.scale):
                    lr_patches.append(lr[i:i+args.patch_size, j:j+args.patch_size])
                    hr_patches.append(hr[i*args.scale:i*args.scale+args.patch_size*args.scale, j*args.scale:j*args.scale+args.patch_size*args.scale])

    lr_patches = np.array(lr_patches)
    hr_patches = np.array(hr_patches)

    h5_file.create_dataset('lr', data=lr_patches)
    h5_file.create_dataset('hr', data=hr_patches)

    h5_file.close()


def eval(args):
    h5_file = h5py.File(args.output_path, 'w')

    lr_group = h5_file.create_group('lr')
    hr_group = h5_file.create_group('hr')

    for i, image_path in enumerate(sorted(glob.glob('{}/*'.format(args.images_dir)))):
        hr = pil_image.open(image_path).convert('RGB')
        hr_width = (hr.width // args.scale) * args.scale
        hr_height = (hr.height // args.scale) * args.scale
        hr = hr.resize((hr_width, hr_height), resample=pil_image.BICUBIC)
        lr = hr.resize((hr.width // args.scale, hr_height // args.scale), resample=pil_image.BICUBIC)
        hr = np.array(hr).astype(np.float32)
        lr = np.array(lr).astype(np.float32)
        hr = convert_rgb_to_y(hr)
        lr = convert_rgb_to_y(lr)

        lr_group.create_dataset(str(i), data=lr)
        hr_group.create_dataset(str(i), data=hr)

    h5_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images-dir', type=str, required=True)
    parser.add_argument('--low_res-dir', type=str, required=True)
    parser.add_argument('--output-path', type=str, required=True)
    parser.add_argument('--scale', type=int, default=2)
    parser.add_argument('--with-aug', action='store_true')
    parser.add_argument('--eval', action='store_true')
    args = parser.parse_args()

    if not args.eval:
        train(args)
    else:
        eval(args)