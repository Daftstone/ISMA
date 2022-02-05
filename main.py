import warnings

warnings.filterwarnings("ignore")
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np

from Dataset import Dataset
from model.weibo.MLP import MLP as weiboMLP
import utils

flags = tf.flags
FLAGS = flags.FLAGS
flags.DEFINE_string("dataset", "weibo", "Choose a dataset.")
flags.DEFINE_string("model", "mlp", "Choose a model.")
flags.DEFINE_string('gpu', '0', 'Input data path.')
flags.DEFINE_integer('batch_size', 4096, 'batch_size')
flags.DEFINE_integer('nb_epochs', 20, 'Number of epochs.')
flags.DEFINE_float("data_size", 1., "pass")
flags.DEFINE_integer('target_index', 0, 'select target items')
flags.DEFINE_integer('poison', 0, 'whether poison')
flags.DEFINE_integer('train', 1, 'whether train online')
flags.DEFINE_string("method", 'inf', "optimization")
flags.DEFINE_bool('cal_inf', False, "pass")
flags.DEFINE_bool("cal_cand", False, 'pass')
flags.DEFINE_bool("modify_user", False, '')
flags.DEFINE_bool("save", False, '')
flags.DEFINE_integer("number", 10000, 'pass')

os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu

if __name__ == '__main__':
    # initialize dataset
    dataset = Dataset()
    Model = {'weibomlp': weiboMLP}

    model = Model[FLAGS.model](dataset)
    model.train(10, 2048, FLAGS.poison)
