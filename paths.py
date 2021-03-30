import pathlib as pl

ROOT_DIRC = pl.Path(__file__).parent

# data dirc
DATA_DIRC = ROOT_DIRC.joinpath('data')
SUN_RGBD_EXTRA_DIRC = DATA_DIRC.joinpath('sun-rgbd-extra')
SUN_RGBD_DIRC = DATA_DIRC.joinpath('sun-rgbd')
SUN_RGBD_TRAIN_DIRC = SUN_RGBD_DIRC.joinpath('train')
SUN_RGBD_TEST_DIRC = SUN_RGBD_DIRC.joinpath('test')
SYNTHIA_DIRC = DATA_DIRC.joinpath('synthia-al')

RESULTS_DIRC = ROOT_DIRC.joinpath('results')

TRAINED_MODELS_DIRC = ROOT_DIRC.joinpath('trained-models')
