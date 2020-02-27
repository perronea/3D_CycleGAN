#!/usr/bin/env python3
import os
import sys
import argparse
import numpy as np
import nibabel as nib
from bids import BIDSLayout
from imageio import imwrite,imread

def main(args):
    # check if in compose mode
    if args.subcommand == 'compose':
        compose(
            args.input_dir,
            args.affine,
            args.orientation,
            args.pad_crop)
    # Check if being used in bids mode or single image mode
    elif args.subcommand == 'bids':
        bids_mode(
            args.bids_directory,
            args.output_directory,
            args.suffix,
            args.exclude,
            args.include,
            args.flat,
            args.orientation,
            args.scale,
            args.pad_crop)
    elif args.subcommand == 'single':
        output_dir = os.path.dirname(args.output_prefix)
        os.makedirs(output_dir,exist_ok=True) # make output directory if not exist
        img = nib.load(args.input_image) # load image
        decompose(
            img,
            args.output_prefix,
            args.orientation,
            args.scale,
            args.pad_crop)
    else: # We shouldn't get here???
        raise ValueError("Invalid inputs. Check that you are using the arguments properly.")

def compose(
        slice_dir: str,
        affine: str,
        orientation: str,
        pad_crop: list):
    """
        Composes nifti volume from slices found in slice directory
    """
    # get all slices and sort
    slice_list = [i for i in os.listdir(slice_dir) if '.png' in i]
    slice_list.sort()

    # get prefix of first entry, we'll use this for the basename
    name_prefix = get_prefix(slice_list[0]).split("_slice")[0]

    # read in images
    img_list = list()
    for s in slice_list:
        img = imread(os.path.join(slice_dir,s))
        if pad_crop: # do padding/crop if enabled
            img = resize_slice(img, pad_crop[::-1]) # nvert the pad_crop since resize expects WIDTH x HEIGHT
        if orientation == 'S':
            img = np.flip(img, axis=0).T[np.newaxis,:,:]
        elif orientation == 'C':
            img = np.flip(img, axis=0).T[:,np.newaxis,:]
        elif orientation == 'T':
            img = np.flip(img, axis=0).T[:,:,np.newaxis]
        img_list.append(img)

    # concatenate images
    orient_code = {'S': 0, 'C': 1, 'T': 2}[orientation]
    voldata = np.concatenate(img_list,axis=orient_code)

    # if affine defined, read in file and copy affine info
    try:
        if affine.shape == (4,4):
            affine_info = affine
    except:
        if affine:
            affine_base = nib.load(affine) # load img
            affine_base = nib.as_closest_canonical(affine_base) # Force RAS+ orientation
            affine_info = affine_base.affine
        # otherwise just use identity
        else:
            affine_info = np.eye(4)

    # writeout nifti image
    nib.Nifti1Image(voldata,affine_info).to_filename(os.path.join(slice_dir,name_prefix+".nii.gz"))

def decompose(
        img: nib.Nifti1Image,
        name: str,
        orientation: str,
        scale: int,
        pad_crop: list):
    """
        Splits a nibabel Nifti1Image into separate slices given orientation to make slices on
    """
    # Force RAS+ orientation
    #img = nib.as_closest_canonical(img)

    # get orient code
    orient_code = {'S': 0, 'C': 1, 'T': 2}[orientation]

    # get volume data
    data = img.get_fdata()
    dims = data.shape

    # scale data
    data = np.round((data/scale)*65535)

    # write images to disk
    for n in range(dims[orient_code]):
        output_filename = name+'_slice-{:0>4d}.png'.format(n)
        data_slice = np.flip(simple_slice(data,n,orient_code).T.astype('uint16'), axis=0)
        if pad_crop: # do padding/crop if enabled
            data_slice = resize_slice(data_slice, pad_crop)
        # write slice to file
        imwrite(output_filename, data_slice)

def bids_mode(
        input_dir: str,
        output_dir: str,
        suffix: str,
        exclude: str,
        include: str,
        flat: bool,
        orientation: str,
        scale: int,
        pad_crop: list):
    """
        Runs decompose on all images in a bids dataset
    """
    # make the output directory path
    os.makedirs(output_dir,exist_ok=True)

    # Create the layout object
    print('Loading BIDS Directory...')
    layout = BIDSLayout(input_dir)
    print(layout)

    # Get T1w images only
    files = layout.get(suffix=suffix,extension='nii.gz')

    # Get list of subjects to exclude if set
    exclude_list = [] # make exclude list
    if exclude:
        # read in exclude list
        with open(exclude,'r') as f:
            for line in f:
                exclude_list.append(line.rstrip())

    # Get list of subjects to include if set
    include_list = []
    if include:
        # read in include list
        with open(include,'r') as f:
            for line in f:
                include_list.append(line.rstrip())

    # loop over each file
    for f in files:
        print('Processing {}...'.format(f.filename))

        # get subject
        subject = f.entities['subject']

        # Check if subject in include list (only if not empty)
        if include_list:
            if not subject in include_list:
                print('sub-{} is not in include list. Skipping...'.format(subject))
                continue

        # Check if subject in exclude list
        if subject in exclude_list:
            print('sub-{} is in exclude list. Skipping...'.format(subject))
            # skip if in exclude list
            continue

        # create output file prefix and directory
        if args.flat: # places images directly in output dir
            name = os.path.join(output_dir, get_prefix(f.filename))
        else: # creates subfolder
            name = os.path.join(output_dir, get_prefix(f.filename), get_prefix(f.filename))
        file_dir = os.path.dirname(name)
        os.makedirs(file_dir,exist_ok=True)

        # decompose image
        decompose(f.get_image(),name,orientation,scale,pad_crop)

def resize_slice(data_slice: np.array, pad_crop: list):
    """
        Resizes the slice to given dimensions
    """
    # compare dims of slice to desired dims
    desired_dims = np.array(pad_crop).T # input is WIDTH x HEIGHT, but we need row x col
    current_dims = np.array(data_slice.shape)
    dim_diff = desired_dims - current_dims
    for i,d in np.ndenumerate(dim_diff):
        if d < 0: # need to shrink dim by d
            if d % 2 == 0: # difference even
                dim1 = -int(d/2) # find half of dim difference
                dim2 = -int(d/2)
            else: # difference odd
                dim1 = -int(np.floor(d/2))
                dim2 = -int(np.ceil(d/2))
            # now slice the dim of the data to the correct size
            data_slice = simple_slice(data_slice, slice(dim1,current_dims[i]-dim2,1), i[0])
        elif d > 0: # need to expand dim by d
            if d % 2 == 0: # difference even
                dim1 = int(d/2) # find half of dim difference
                dim2 = int(d/2)
            else: # difference odd
                dim1 = int(np.floor(d/2))
                dim2 = int(np.ceil(d/2))
            # constuct the padding list
            padder = list()
            for j in range(dim_diff.shape[0]):
                if j == i[0]: # expand this dim
                    padder.append([dim1,dim2])
                else: # don't expand this dim
                    padder.append([0,0])
            # now pad the slice with the padding list
            data_slice = np.pad(data_slice,padder, mode='constant')
    # return the modified slice
    return data_slice

def simple_slice(arr, inds, axis):
    """
        This does the same as np.take() except only supports simple slicing, not
        advanced indexing, and thus is much faster
    """
    sl = [slice(None)] * arr.ndim
    sl[axis] = inds
    return arr[tuple(sl)]

def get_prefix(filename):
    """
        Gets prefix of filename without extension
    """
    prefix,ext = os.path.splitext(filename)
    if '.' in prefix:
        return get_prefix(prefix)
    else:
        return prefix

if __name__ == '__main__':
    # Create command line parser
    parser = argparse.ArgumentParser(
        description='Splits volumes in a dataset to 2D. \
                    Useful for Deep Learning Applications.')
    if sys.version_info >= (3, 7): # required option only added in 3.7
        subparsers = parser.add_subparsers(
            title='subcommands',
            required=True,
            dest='subcommand',
            help='Use the -h/--help flag on each subcommand for more help.')
    else:
        subparsers = parser.add_subparsers(
            title='subcommands',
            dest='subcommand',
            help='Use the -h/--help flag on each subcommand for more help.')

    # BIDS mode arguments
    bids = subparsers.add_parser('bids', help='Splits an entire BIDS Dataset.')
    bids.add_argument('bids_directory', help='Path to BIDS directory to process.')
    bids.add_argument('output_directory', help='Path to dump image outputs.')
    bids.add_argument('-s', '--suffix', default='T1w', help="BIDS file suffix to process.")
    bids.add_argument('-n', '--include', help='Text file with list of subjects to include. Just the subject id (no sub-).')
    bids.add_argument('-e', '--exclude', help='Text file with list of subjects to exclude. Just the subject id (no sub-).')
    bids.add_argument('--flat', action="store_true", help='If used, write all images to the output directory directly without subfolders.')
    bids.add_argument('-r', '--orientation', choices=['T', 'S', 'C'], default='T',
        help='Orientation of output images (Volume is forced to RAS+ orientation on load). Can be T (Transverse), S (Sagittal), or C (Coronal). Default is T.')
    bids.add_argument('-p', '--pad_crop', nargs=2, type=int, metavar=('WIDTH','HEIGHT'),
        help='Specifies Width x Height to make each image. Will either center crop/pad image to reach desired dimensions.')
    bids.add_argument('--scale', type=int, default=4095, help='Data scale for write. Default is 4095.')

    # Single mode arguments
    single = subparsers.add_parser('single', help='Splits a Single Image. Use this if you want to process a single image, or non-BIDS data.')
    single.add_argument('input_image', help='Path to input image.')
    single.add_argument('output_prefix', help='Output prefix of split images.')
    single.add_argument('-r', '--orientation', choices=['T', 'S', 'C'], default='T',
        help='Orientation of output images (Volume is forced to RAS+ orientation on load). Can be T (Transverse), S (Sagittal), or C (Coronal). Default is T.')
    single.add_argument('-p', '--pad_crop', nargs=2, type=int, metavar=('WIDTH','HEIGHT'),
        help='Specifies Width x Height to make each image. Will either center crop/pad image to reach desired dimensions.')
    single.add_argument('--scale', type=int, default=4095, help='Data scale for write. Default is 4095.')

    # Compose mode arguments
    compose_mode = subparsers.add_parser('compose', help='Composes a volume from a folder of split images.')
    compose_mode.add_argument('input_dir', help='Composes nifti volume from slice directory.')
    compose_mode.add_argument('-a', '--affine', help='Use affine information from specified nifti file. Uses eye(4) otherwise.')
    compose_mode.add_argument('-r', '--orientation', choices=['T', 'S', 'C'], default='T',
        help='Orientation of output images (Volume is forced to RAS+ orientation on load). Can be T (Transverse), S (Sagittal), or C (Coronal). Default is T.')
    compose_mode.add_argument('-p', '--pad_crop', nargs=2, type=int, metavar=('WIDTH','HEIGHT'),
        help='Specifies Width x Height to make each image. Will either center crop/pad image to reach desired dimensions.')

    # parse the arguments
    args = parser.parse_args()

    # Pass parameters to main
    main(args)
