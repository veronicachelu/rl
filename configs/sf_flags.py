import tensorflow as tf

tf.app.flags.DEFINE_integer('summary_interval', 10, """Number of episodes of interval between summary saves""")
tf.app.flags.DEFINE_integer('checkpoint_interval', 10, """Number of episodes of interval between checkpoint saves""")
tf.app.flags.DEFINE_integer('max_total_steps', 100000, """""")
tf.app.flags.DEFINE_string('mdp_path', './mdps/toy.mdp',
                           """Directory from where to load the environment geometry""")
tf.app.flags.DEFINE_integer('nb_steps_sf', 10000, """Number of steps to run fitted successor features learning""")
tf.app.flags.DEFINE_integer('sf_memory_size', 1000, """Number of steps to run fitted successor features learning""")
tf.app.flags.DEFINE_float('gamma', 0.99, """Gamma value""")
tf.app.flags.DEFINE_float('lr', 0.001, """Learning rate value""")
tf.app.flags.DEFINE_integer('batch_size', 96, """The size of the batch""")
tf.app.flags.DEFINE_integer('update_freq', 4, """Frequency of training updates""")
tf.app.flags.DEFINE_integer('target_update_freq', 24, """Frequency of target updates""")
tf.app.flags.DEFINE_integer('memory_size', 10000, """The size of the memory""")
tf.app.flags.DEFINE_integer('explore_steps', 10000, """Number of steps to use for epsilon decay, for exploration""")
tf.app.flags.DEFINE_integer('observation_steps', 96, """Number of steps to use for epsilon decay, for exploration""")
tf.app.flags.DEFINE_float('initial_random_action_prob', 1.0, """Initial probability for epsilon greedy exploration""")
tf.app.flags.DEFINE_float('final_random_action_prob', 0.05, """Final probability for epsilon greedy exploration""")
tf.app.flags.DEFINE_string('optimizer', "Adam", """The type of optimizer to use""")
tf.app.flags.DEFINE_integer('gradient_norm_clipping', 10, """Value to clip the gradient norm""")
tf.app.flags.DEFINE_string('task', "discover", """The task to perform""")


tf.app.flags.FLAGS._parse_flags()