import torch as t
import torchvision as tv
import oct2py as o
from oct2py import octave
import paths as p
import scipy.io as s

if __name__ == '__main__':
    # with h.File(p.SUN_RGBD_DIRC.joinpath('SUNRGBDtoolbox/Metadata/SUNRGBD2Dseg.mat'), 'r') as f:
    #     segs_loc = f['SUNRGBD2Dseg']['seglabel']
    #     for i in range(segs_loc.shape[0]):
    #         a = np.array(f[segs_loc[i][0]])
    oct = o.Oct2Py()
    dirc = p.SUN_RGBD_DIRC.parent.joinpath('sun-rgbd-extra/SUNRGBDMeta2DBB_v2.mat')
    a = s.loadmat(dirc)
    a

