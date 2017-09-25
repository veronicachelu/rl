import tensorflow as tf
from configs import base_flags

import os
import numpy as np
from utils import utility
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

    if not tf.gfile.Exists(FLAGS.draw_dir):
        tf.gfile.MakeDirs(FLAGS.draw_dir)
    if not tf.gfile.Exists(os.path.join(FLAGS.draw_dir, FLAGS.algorithm)):
        tf.gfile.MakeDirs(os.path.join(FLAGS.draw_dir, FLAGS.algorithm))
    else:
        if not FLAGS.resume and FLAGS.train:
            tf.gfile.DeleteRecursively(os.path.join(FLAGS.draw_dir, FLAGS.algorithm))
            tf.gfile.MakeDirs(os.path.join(FLAGS.draw_dir, FLAGS.algorithm))


def main():
    """Create or load configuration and launch the trainer."""
    utility.set_up_logging()
    if not FLAGS.config:
        raise KeyError('You must specify a configuration.')
    recreate_directory_structure()
    np.random.seed(FLAGS.seed)
    tf.reset_default_graph()
    logdir = os.path.join(FLAGS.experiments_dir, FLAGS.algorithm)
    try:
        config = utility.load_config(logdir)
    except IOError:
        config = tools.AttrDict(getattr(configs, FLAGS.config)())
        config = utility.save_config(config, logdir)

    with config.unlocked:
        alg = utility.define_alg(FLAGS.algorithm, config)
        if FLAGS.train:
            alg.train()
        else:
            alg.eval()

    # recreate_directory_structure()
    # np.random.seed(FLAGS.seed)
    # tf.reset_default_graph()
    #
    # if FLAGS.algorithm == "DQN":
    #     from configs import dqn_flags
    #     from baselines.dqn import DQN
    #     alg = DQN()
    # elif FLAGS.algorithm == "DQNLinear":
    #     from configs import dqn_linear_flags
    #     from baselines.dqn_linear import DQNLinear
    #     alg = DQNLinear()
    # elif FLAGS.algorithm == "SFLinear":
    #     from configs import sf_linear_flags
    #     from baselines.sf_linear import SFLinear
    #     alg = SFLinear()
    # elif FLAGS.algorithm == "CategoricalDQN":
    #     from configs import categorical_dqn_flags
    #     from baselines.categorical_dqn import CategoricalDQN
    #     alg = CategoricalDQN()
    # elif FLAGS.algorithm == "SuccessorFeatures":
    #     from configs import sf_flags
    #     from baselines.sf import SF
    #     alg = SF()
    # else:
    #     print("Please specify a valid algorithm name")
    #     exit(0)
    #
    # if FLAGS.train:
    #     alg.train()
    # else:
    #     alg.eval()


if __name__ == '__main__':
    tf.app.flags.DEFINE_string('algorithm', 'DQNLinear',
                               # tf.app.flags.DEFINE_string('algorithm', 'DQN',
                               """Algorithm name to train or test expriments""")
    tf.app.flags.DEFINE_string('config', 'dqnlinear',  """Config name to train or test expriments""")
    tf.app.flags.DEFINE_string('checkpoint_dir', './models',
                               """Directory where to save model checkpoints.""")
    tf.app.flags.DEFINE_string('summaries_dir', './summaries',
                               """Directory where to write event logs""")
    tf.app.flags.DEFINE_string('experiments_dir', './experiments',
                               """Directory where to write event experiments""")
    tf.app.flags.DEFINE_string('draw_dir', "./visuals", """Directory where to write visual plots""")
    tf.app.flags.DEFINE_boolean('resume', False,
                                """Resume training from latest checkpoint""")
    tf.app.flags.DEFINE_boolean('train', True,
                                """Whether to train or test""")
    tf.app.flags.DEFINE_float('TAO', 0.001, """""")
    tf.app.flags.DEFINE_integer('test_episodes', 150, """""")
    tf.app.flags.DEFINE_integer('eval_interval', 1200, """""")
    tf.app.flags.DEFINE_integer('seed', 99, """Seed value for numpy""")
    tf.app.run()
