import tensorflow as tf
# from configs import base_flags

# Basic model parameters.
tf.app.flags.DEFINE_string('game', 'Catcher-v0',
                           """Experiment name from Atari platform""")
tf.app.flags.DEFINE_boolean('monitor', False,
                            """Wrap env in monitor""")
tf.app.flags.DEFINE_boolean('layer_norm', True,
                            """Use layer normalization""")
tf.app.flags.DEFINE_boolean('show_training', True,
                            """Show windows with workers training""")
tf.app.flags.DEFINE_integer('summary_interval', 1000, """Number of episodes of interval between summary saves""")
tf.app.flags.DEFINE_integer('checkpoint_interval', 1000, """Number of episodes of interval between checkpoint saves""")
tf.app.flags.DEFINE_integer('agent_history_length', 1, """Number of frames that makes every state""")
tf.app.flags.DEFINE_integer('resized_width', 24, """Resized width of each frame""")
tf.app.flags.DEFINE_integer('resized_height', 24, """Resized height of each frame""")
tf.app.flags.DEFINE_float('gamma', 0.99, """Gamma value""")
tf.app.flags.DEFINE_float('lr', 0.000635, """Learning rate""")
tf.app.flags.DEFINE_string('optimizer', "Adam", """The type of optimizer to use""")
tf.app.flags.DEFINE_integer('gym_seed', 23, """Seed value for the gym env""")
tf.app.flags.DEFINE_integer('hidden_size', 128, """Hidden_size of FC layer""")
tf.app.flags.DEFINE_integer('batch_size', 96, """The size of the batch""")
tf.app.flags.DEFINE_integer('update_freq', 4, """Frequency of training updates""")
tf.app.flags.DEFINE_integer('target_update_freq', 24, """Frequency of target updates""")
tf.app.flags.DEFINE_integer('gradient_norm_clipping', 10, """Value to clip the gradient norm""")
tf.app.flags.DEFINE_integer('memory_size', 1e5, """The size of the memory""")
tf.app.flags.DEFINE_integer('explore_steps', 1e5, """Number of steps to use for epsilon decay, for exploration""")
tf.app.flags.DEFINE_integer('observation_steps', 96, """Number of steps to add entries in the experience replay before starting training""")
tf.app.flags.DEFINE_integer('max_total_steps', 1200000, """Total number of steps to use for training""")
tf.app.flags.DEFINE_float('initial_random_action_prob', 1.0, """Initial probability for epsilon greedy exploration""")
tf.app.flags.DEFINE_float('final_random_action_prob', 0.05, """Final probability for epsilon greedy exploration""")

tf.app.flags.FLAGS._parse_flags()

