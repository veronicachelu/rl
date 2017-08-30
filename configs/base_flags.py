import tensorflow as tf

tf.app.flags.DEFINE_string('algorithm', 'CategoricalDQN',
#tf.app.flags.DEFINE_string('algorithm', 'DQN',
                           """Algorithm name to train or test expriments""")
tf.app.flags.DEFINE_string('checkpoint_dir', './models',
                           """Directory where to save model checkpoints.""")
tf.app.flags.DEFINE_string('summaries_dir', './summaries',
                           """Directory where to write event logs""")
tf.app.flags.DEFINE_string('experiments_dir', './experiments',
                           """Directory where to write event experiments""")
tf.app.flags.DEFINE_boolean('resume', False,
                            """Resume training from latest checkpoint""")
tf.app.flags.DEFINE_boolean('train', True,
                            """Whether to train or test""")
tf.app.flags.DEFINE_float('TAO', 0.001, """""")
tf.app.flags.DEFINE_integer('test_episodes', 150, """""")
tf.app.flags.DEFINE_integer('eval_interval', 1200, """""")
tf.app.flags.DEFINE_integer('seed', 99, """Seed value for numpy""")
