import tensorflow as tf

tf.app.flags.DEFINE_integer('summary_interval', 1000, """Number of episodes of interval between summary saves""")
tf.app.flags.DEFINE_integer('checkpoint_interval', 1000, """Number of episodes of interval between checkpoint saves""")
tf.app.flags.DEFINE_integer('max_total_steps', 1000, """""")
tf.app.flags.DEFINE_string('mdp_path', '/Users/ioanaveronicachelu/RL/curiosity_options/mdps/toy.mdp',
                           """Directory from where to load the environment geometry""")
tf.app.flags.DEFINE_integer('nb_steps_sf', 20000, """Number of steps to run fitted successor features learning""")
tf.app.flags.DEFINE_float('gamma', 0.99, """Gamma value""")
tf.app.flags.DEFINE_float('lr', 0.01, """Learning rate value""")
tf.app.flags.DEFINE_integer('batch_size', 96, """The size of the batch""")
tf.app.flags.DEFINE_integer('update_freq', 4, """Frequency of training updates""")
tf.app.flags.DEFINE_integer('target_update_freq', 24, """Frequency of target updates""")
tf.app.flags.DEFINE_integer('memory_size', 100000, """The size of the memory""")
tf.app.flags.DEFINE_integer('explore_steps', 100000, """Number of steps to use for epsilon decay, for exploration""")
tf.app.flags.DEFINE_integer('observation_steps', 96, """Number of steps to use for epsilon decay, for exploration""")
