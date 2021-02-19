import torch as t
import torchvision as tv
import oct2py as o
from oct2py import octave
import paths as p
import scipy.io as s

if __name__ == '__main__':
    oct = o.Oct2Py()
    dirc = p.SUN_RGBD_DIRC.joinpath('SUNRGBDtoolbox/traintestSUNRGBD/allsplit.mat')
    a = s.loadmat(dirc)
    a

