"""Process setup kept free of TensorFlow imports until CPU limits are set."""
import os


def initialize_worker():
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
    os.environ["TF_NUM_INTEROP_THREADS"] = "1"
    os.environ["MPLBACKEND"] = "Agg"


def train_iteration(*args):
    from model_shredski import _train_iteration
    return _train_iteration(*args)
