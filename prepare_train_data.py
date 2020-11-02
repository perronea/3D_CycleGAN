#! /usr/bin/env python3

import os
import nibabel as nib
import numpy as np


def get_max_crop(arr):
    """
    Return the dimensions of the array that would remove the maximum amount of blank space
    :param arr: 3D numpy array of the masked brain image
    :return: X, Y, and Z start and end dimensions to maximize blank space removal
    """

    x_mean = np.mean(arr, axis=0)
    y_mean = np.mean(arr, axis=1)

    y1 = 0
    while np.nonzero(x_mean[y1,:])[0].size == 0:
        y1 += 1
    y2 = x_mean.shape[0]
    while np.nonzero(x_mean[y2-1,:])[0].size == 0:
        y2 -= 1

    z1 = 0
    while np.nonzero(x_mean[:,z1])[0].size == 0:
        z1 += 1
    z2 = x_mean.shape[1]
    while np.nonzero(x_mean[:,z2-1])[0].size == 0:
        z2 -= 1

    x1 = 0
    while np.nonzero(y_mean[x1,:])[0].size == 0:
        x1 += 1
    x2 = y_mean.shape[1]
    while np.nonzero(y_mean[x2-1,:])[0].size == 0:
        x2 -= 1

    return(x1,x2,y1,y2,z1,z2)

def crop_paired_data(T1_path, T2_path, T1_out, T2_out):
    """
    Identify the minimum and maximum crop along each dimension for both paired images and then crops both to the smallest size sucj     that the same crop is applied. This ensures that both images still align even after cropping.
    :param T1_path, T2_path: Path to paired T1 and T2 images for a subject in training data.
    :return: Size of the final crop
    """
    T1_img = nib.load(T1_path)
    T2_img = nib.load(T2_path)
    T1_arr = T1_img.get_data()
    T2_arr = T2_img.get_data()
    # smallest upperbound on crop, start at zero and increase to the largest minimum crop
    xj, yj, zj = 0, 0, 0
    # smallest lowerbound on crop, start at original image size and decrease to the smallest minimum crop
    xi, yi, zi = T1_arr.shape
    for i in T1_arr,T2_arr:
        x1,x2,y1,y2,z1,z2 = get_max_crop(i)
        if x1 < xi:
            xi = x1
        if x2 > xj:
            xj = x2
        if y1 < yi:
            yi = y1
        if y2 > yj:
            yj = y2
        if z1 < zi:
            zi = z1
        if z2 > zj:
            zj = z2

    nib.save(nib.Nifti1Image(T1_arr[xi:xj, yi:yj, zi:zj], T1_img.affine), T1_out)
    nib.save(nib.Nifti1Image(T2_arr[xi:xj, yi:yj, zi:zj], T2_img.affine), T2_out)
    print(T1_arr[xi:xj, yi:yj, zi:zj].shape)
    return(T1_arr[xi:xj, yi:yj, zi:zj].shape)

def run_cropping(subfolder='MR'):
    train_A_path = os.path.join('data', subfolder, 'trainA')
    train_B_path = os.path.join('data', subfolder, 'trainB')
    test_A_path = os.path.join('data', subfolder, 'testA')
    test_B_path = os.path.join('data', subfolder, 'testB')
    train_A_crop_path = os.path.join('data', subfolder + '_crop', 'trainA')
    train_B_crop_path = os.path.join('data', subfolder + '_crop', 'trainB')
    test_A_crop_path = os.path.join('data', subfolder + '_crop', 'testA')
    test_B_crop_path = os.path.join('data', subfolder + '_crop', 'testB')
    os.makedirs(train_A_crop_path, exist_ok=True)
    os.makedirs(train_B_crop_path, exist_ok=True)
    os.makedirs(test_A_crop_path, exist_ok=True)
    os.makedirs(test_B_crop_path, exist_ok=True)

    train_A_image_names = sorted(os.listdir(train_A_path))
    train_B_image_names = sorted(os.listdir(train_B_path))
    test_A_image_names = sorted(os.listdir(test_A_path))
    test_B_image_names = sorted(os.listdir(test_B_path))

    assert(len(train_A_image_names) == len(train_B_image_names))
    assert(len(test_A_image_names) == len(test_B_image_names))

    cropped_shapes = []

    for i in range(0, len(train_A_image_names)):
        print("Cropping %s and %s" % (train_A_image_names[i], train_B_image_names[i]))
        new_shape = crop_paired_data(os.path.join(train_A_path, train_A_image_names[i]), os.path.join(train_B_path, train_B_image_names[i]), os.path.join(train_A_crop_path, train_A_image_names[i]), os.path.join(train_B_crop_path, train_B_image_names[i]))
        cropped_shapes.append(new_shape)
    for i in range(0, len(test_A_image_names)):
        new_shape = crop_paired_data(os.path.join(test_A_path, test_A_image_names[i]), os.path.join(test_B_path, test_B_image_names[i]), os.path.join(test_A_crop_path, test_A_image_names[i]), os.path.join(test_B_crop_path, test_B_image_names[i]))
        cropped_shapes.append(new_shape)

    for i in range(0, len(cropped_shapes)):
        print(cropped_shapes[i])
    print("Smallest recommended input image shape: ", np.amax(np.array(cropped_shapes), axis=0))
    
    return

def main():

    run_cropping(subfolder='MR')

if __name__ == '__main__':
    main()
