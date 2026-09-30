#!/bin/env python
from sklearn.decomposition import PCA
from PIL.Image import open, fromarray, Resampling, UnidentifiedImageError
from subprocess import call
from sys import argv
import numpy as np
N = 1.1
F = {}
for f in argv[1:]:
    try:
        img = open(f)
    except UnidentifiedImageError as e:
        print(e)
        continue
    if max(img.size) <= int(min(img.size)*N):
        continue
    F[f] = img.size
if F:
    call(['dinov2', *F])
for f, (w, h) in F.items():
    try:
        ax = np.argmin((h,w))
        mi = (h,w)[ax]
        ma = int(mi*N)
        img = np.asarray(fromarray(PCA(1).fit_transform(np.fromfile(f'{f}.{round(min(h,336)/14)},{round(min(w,336)/14)},384f','384f')).reshape(round(min(h,336)/14),-1)).resize((w,h),Resampling.LANCZOS))
        if img[:len(img)//2].sum() < img[-len(img)//2:].sum():
            img = -img
        img = img.sum(ax).cumsum()
        img[ma:] -= img[:-ma]
        img = img[ma-1:].argmin()
        fromarray(np.asarray(open(f))[*(slice(img,img+ma),slice(None))[::ax*2-1],:3]).save(f)
    except (FileNotFoundError, OSError) as e:
        print(e)
