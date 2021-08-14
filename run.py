#!/usr/bin/env python3

import os
import sys
import argparse
import numpy as np
import nibabel as nib
from bids import BIDSLayout
import neurosplitter
import subprocess
import glob
import shutil
import GAN_generate

def main(args):

    if args.modality == 'T2w':
        gen_mod = 'T1w'
    elif args.modality == 'T1w':
        gen_mod = 'T2w'
    else:
        print('Input modality not recognized')
        exit

    # get metadata information including orig dimensions and affine matrix
    img = nib.load(args.input_img)
    img_aff = img.affine

    # Split the input NIfTI along each axis
    # (dim3 = dim1 x dim2) T padded to (dHCP=291 291 > 292 292) (EXITO=182 218)
    # (dim1 = dim2 x dim3) S padded to (dHCP=202 291 > 204 292) (EXITO=218 182)
    # (dim2 = dim1 x dim3) C padded to (dHCP=202 291 > 204 292) (EXITO=182 182)
    dim1 = img.header.get_data_shape()[0]
    dim2 = img.header.get_data_shape()[1]
    dim3 = img.header.get_data_shape()[2]
    
    T_out_dir = os.path.join(args.wd, args.modality, 'T')
    C_out_dir = os.path.join(args.wd, args.modality, 'C')
    S_out_dir = os.path.join(args.wd, args.modality, 'S')
    os.makedirs(T_out_dir, exist_ok=True)
    os.makedirs(C_out_dir, exist_ok=True)
    os.makedirs(S_out_dir, exist_ok=True)
    
    basename = os.path.basename(args.input_img).replace('.nii.gz','')
    neurosplitter.decompose(img, os.path.join(T_out_dir, basename), 'T', 4095, None)
    neurosplitter.decompose(img, os.path.join(C_out_dir, basename), 'C', 4095, None)
    neurosplitter.decompose(img, os.path.join(S_out_dir, basename), 'S', 4095, None)

    # Generate the corresponding modality of every slice
    GAN = GAN_generate.CycleGAN(args.wd, args.modality, gen_mod)

    # Recompose the generated NIfTI along each axis
    T_synth_dir = os.path.join(args.wd, 'synthetic_images', gen_mod, 'T')
    C_synth_dir = os.path.join(args.wd, 'synthetic_images', gen_mod, 'C')
    S_synth_dir = os.path.join(args.wd, 'synthetic_images', gen_mod, 'S')
    
    neurosplitter.compose(T_synth_dir, img_aff, 'T', (dim1,dim2))
    neurosplitter.compose(C_synth_dir, img_aff, 'C', (dim1,dim3))
    neurosplitter.compose(S_synth_dir, img_aff, 'S', (dim2,dim3))

    # Combining the generated NIfTIs and apply post-processing
    temp_dir = os.path.join(args.wd, 'temp')
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
        os.makedirs(temp_dir)
    else:
        os.makedirs(temp_dir)

    subprocess.call(['fslmaths', os.path.join(T_synth_dir, os.path.basename(args.input_img)), '-mul', args.mask, os.path.join(temp_dir, 'T_masked.nii.gz')])
    subprocess.call(['fslmaths', os.path.join(C_synth_dir, os.path.basename(args.input_img)), '-mul', args.mask, os.path.join(temp_dir, 'C_masked.nii.gz')])
    subprocess.call(['fslmaths', os.path.join(S_synth_dir, os.path.basename(args.input_img)), '-mul', args.mask, os.path.join(temp_dir, 'S_masked.nii.gz')])
    # Split each masked image and mean normalize all slices to 50
    #   T split along z
    #   S split along x
    #   C split along y
    FNULL = open(os.devnull, 'w')
    #T
    subprocess.call(['fslsplit', os.path.join(temp_dir, 'T_masked.nii.gz'), os.path.join(temp_dir, 'T_split_'), '-z'])
    for T_split in glob.glob(os.path.join(temp_dir, 'T_split_*')):
        output = T_split.replace('T_split', 'T_norm_split')
        return_code = subprocess.call(['fslmaths', T_split, '-inm', '50', output], stdout=FNULL, stderr=subprocess.STDOUT)
    subprocess.call(['fslmerge', '-z', os.path.join(temp_dir, 'T_masked_norm.nii.gz')] + glob.glob(os.path.join(temp_dir, 'T_norm_split*')))
    subprocess.call(['fslmaths', os.path.join(temp_dir, 'T_masked_norm.nii.gz'), '-nan', os.path.join(temp_dir, 'T_masked_norm.nii.gz')])
    #S
    subprocess.call(['fslsplit', os.path.join(temp_dir, 'S_masked.nii.gz'), os.path.join(temp_dir, 'S_split_'), '-x'])
    for S_split in glob.glob(os.path.join(temp_dir, 'S_split_*')):
        output = S_split.replace('S_split', 'S_norm_split')
        return_code = subprocess.call(['fslmaths', S_split, '-inm', '50', output], stdout=FNULL, stderr=subprocess.STDOUT)
    subprocess.call(['fslmerge', '-x', os.path.join(temp_dir, 'S_masked_norm.nii.gz')] + glob.glob(os.path.join(temp_dir, 'S_norm_split*')))
    subprocess.call(['fslmaths', os.path.join(temp_dir, 'S_masked_norm.nii.gz'), '-nan', os.path.join(temp_dir, 'S_masked_norm.nii.gz')])
    #C
    subprocess.call(['fslsplit', os.path.join(temp_dir, 'C_masked.nii.gz'), os.path.join(temp_dir, 'C_split_'), '-y'])
    for C_split in glob.glob(os.path.join(temp_dir, 'C_split_*')):
        output = C_split.replace('C_split', 'C_norm_split')
        return_code = subprocess.call(['fslmaths', C_split, '-inm', '50', output], stdout=FNULL, stderr=subprocess.STDOUT)
    subprocess.call(['fslmerge', '-y', os.path.join(temp_dir, 'C_masked_norm.nii.gz')] + glob.glob(os.path.join(temp_dir, 'C_norm_split*')))
    subprocess.call(['fslmaths', os.path.join(temp_dir, 'C_masked_norm.nii.gz'), '-nan', os.path.join(temp_dir, 'C_masked_norm.nii.gz')])


    subprocess.call(['fslmaths', os.path.join(temp_dir, 'T_masked_norm.nii.gz'), '-add', os.path.join(temp_dir, 'C_masked_norm.nii.gz'), '-add', os.path.join(temp_dir, 'S_masked_norm.nii.gz'), os.path.join(temp_dir, 'sum_masked_norm.nii.gz')])
    subprocess.call(['fslmaths', os.path.join(temp_dir, 'sum_masked_norm.nii.gz'), '-div', '3', os.path.join(temp_dir, 'avg_masked_norm.nii.gz')])
    output_synthetic_img = os.path.join(args.wd, basename + '_synthetic_' + gen_mod + '.nii.gz')
    subprocess.call(['DenoiseImage', '-i', os.path.join(temp_dir, 'avg_masked_norm.nii.gz'), '-o', output_synthetic_img])
   
    print('Complete: Synthetic {} of {} created here: {}'.format(gen_mod, args.input_img, output_synthetic_img))    
    


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Given a masked brain image generate the corresponding modality\
                    using a pre-trained GAN model.')
    parser.add_argument('-i', '--in',
                    dest='input_img',
                    help='Path to a masked brain image in NIfTI format.')
    parser.add_argument('-m', '--mask',
                    dest='mask',
                    help='Path to mask of the input brain.')
    parser.add_argument('-x', '--modality',
                    choices=['T1w', 'T2w'],
                    help='Modality of the input image.\
                            If "T1w" a T2w image will be generated.\
                            If "T2w" a T1w image will be generated.')
    parser.add_argument('-w', '--working-directory',
                    dest='wd',
                    help='Working directory into which all the temporary images will\
                    be created.')
    args = parser.parse_args()

    main(args)





