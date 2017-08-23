import tensorflow as tf
from configs import base_flags

import os
import numpy as np
FLAGS = tf.app.flags.FLAGS


def recreate_directory_structure():
    if not tf.gfile.Exists(FLAGS.checkpoint_dir):
        tf.gfile.MakeDirs(FLAGS.checkpoint_dir)
    if not tf.gfile.Exists(os.path.join(FLAGS.checkpoint_dir, FLAGS.algorithm)):
        tf.gfile.MakeDirs(os.path.join(FLAGS.checkpoint_dir, FLAGS.algorithm))
    else:
        if not FLAGS.resume and FLAGS.train:
            tf.gfile.DeleteRecursively(os.path.join(FLAGS.checkpoint_dir, FLAGS.algorithm))
            tf.gfile.MakeDirs(os.path.join(FLAGS.checkpoint_dir, FLAGS.algorithm))

    if not tf.gfile.Exists(FLAGS.experiments_dir):
        tf.gfile.MakeDirs(FLAGS.experiments_dir)
    if not tf.gfile.Exists(os.path.join(FLAGS.experiments_dir, FLAGS.algorithm)):
        tf.gfile.MakeDirs(os.path.join(FLAGS.experiments_dir, FLAGS.algorithm))
    else:
        if not FLAGS.resume and FLAGS.train:
            tf.gfile.DeleteRecursively(os.path.join(FLAGS.experiments_dir, FLAGS.algorithm))
            tf.gfile.MakeDirs(os.path.join(FLAGS.experiments_dir, FLAGS.algorithm))

    if not tf.gfile.Exists(FLAGS.summaries_dir):
        tf.gfile.MakeDirs(FLAGS.summaries_dir)
    if not tf.gfile.Exists(os.path.join(FLAGS.summaries_dir, FLAGS.algorithm)):
        tf.gfile.MakeDirs(os.path.join(FLAGS.summaries_dir, FLAGS.algorithm))
    else:
        if not FLAGS.resume and FLAGS.train:
            tf.gfile.DeleteRecursively(os.path.join(FLAGS.summaries_dir, FLAGS.algorithm))
            tf.gfile.MakeDirs(os.path.join(FLAGS.summaries_dir, FLAGS.algorithm))


def run():
    recreate_directory_structure()
    np.random.seed(FLAGS.seed)
    tf.reset_default_graph()

    if FLAGS.algorithm == "DQN":
        from configs import dqn_flags
        from baselines.dqn import DQN
        alg = DQN()
    elif FLAGS.algorithm == "CategoricalDQN":
        from configs import categorical_dqn_flags
        from baselines.categorical_dqn import CategoricalDQN
        alg = CategoricalDQN()
    else:
        print("Please specify a valid algorithm name")
        exit(0)

    if FLAGS.train:
        alg.train()
    else:
        alg.eval()


if __name__ == '__main__':
    run()
