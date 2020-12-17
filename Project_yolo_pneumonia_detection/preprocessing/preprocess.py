#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 15:38:08 2020

@author: qhu
"""
from PIL import Image
from tqdm import tqdm
import argparse
import tensorflow as tf
import multiprocessing
import os
import pydicom
from functools import partial
import json

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', required=True)
    parser.add_argument('--dst', required=True)
    return parser.parse_args()

def dcm_to_png(patientId, src=None, dst=None):
    '''
    Convert dcm to png
    '''
    img = Image.fromarray(
        pydicom.dcmread(
            os.path.join(src, patientId + '.dcm')
        ).pixel_array
    )
    img.save(os.path.join(dst, patientId + '.png'))


def main(args):
    '''
    Main Method
    '''
    if not tf.io.gfile.exists(args.dst):
        tf.io.gfile.makedirs(args.dst)
    assert os.path.lexists(args.src)

    patient_ids = [f.split('.dcm')[0] for f in os.listdir(args.src)]

    with multiprocessing.Pool(multiprocessing.cpu_count()) as p:
        pbar = tqdm(
              total=len(patient_ids)
            , leave=False
            , desc="Converting"
        )
        func = partial(dcm_to_png, src = args.src, dst = args.dst)
        for _ in p.imap_unordered(func, patient_ids):
            pbar.update(1)
        pbar.close()

    print(f'Saved in {args.dst}')


if __name__ == '__main__':
    args = parse_args()
    main(args)