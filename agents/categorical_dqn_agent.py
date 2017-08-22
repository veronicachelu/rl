from threading import Lock

import numpy as np
import tensorflow as tf
from agents.base_agent import BaseAgent
from nets.categorical_dqn_network import CategoricalDQNetwork
from configs import categorical_dqn_flags
from collections import deque
from utils.schedules import LinearSchedule
from utils.timer import Timer
import os

FLAGS = tf.app.flags.FLAGS
import random

# Starting threadsv
main_lock = Lock()


class CategoricalDQNAgent(BaseAgent):
    def __init__(self, game, sess, nb_actions, global_step):
        BaseAgent.__init__(self, game, sess, nb_actions, global_step)
        self.name = "DQN_agent"
        self.model_path = os.path.join(FLAGS.checkpoint_dir, FLAGS.algorithm)
        self.support = np.linspace(FLAGS.v_min, FLAGS.v_max, FLAGS.nb_atoms)
        self.delta_z = (FLAGS.v_max - FLAGS.v_min) / (FLAGS.nb_atoms - 1)

        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_mean_values = []
        self.episode_max_values = []
        self.episode_min_values = []
        self.episode_mean_returns = []
        self.episode_max_returns = []
        self.episode_min_returns = []
        self.exploration = LinearSchedule(FLAGS.explore_steps, FLAGS.final_random_action_prob,
                                          FLAGS.initial_random_action_prob)
        self.summary_writer = tf.summary.FileWriter(os.path.join(FLAGS.summaries_dir, FLAGS.algorithm))
        self.summary = tf.Summary()

        self.q_net = CategoricalDQNetwork(nb_actions, 'orig')
        self.target_net = CategoricalDQNetwork(nb_actions, 'target')

        self.targetOps = self.update_target_graph('orig', 'target')

        self.probability_of_random_action = self.exploration.value(0)

    def train(self):
        minibatch = random.sample(self.episode_buffer, FLAGS.batch_size)
        rollout = np.array(minibatch)
        observations = rollout[:, 0]
        actions = rollout[:, 1]
        rewards = rollout[:, 2]
        next_observations = rollout[:, 3]
        done = rollout[:, 4]

        # Compute target distribution of Q(s_,a)
        target_actionv_values_evaled_new = self.get_target_distribution(rewards, done, next_observations, observations, actions)

        feed_dict = {self.q_net.target_q: target_actionv_values_evaled_new,
                     self.q_net.inputs: np.stack(observations, axis=0),
                     self.q_net.actions: actions}
        l, _, ms, img_summ, q, q_distrib = self.sess.run(
            [self.q_net.action_value_loss,
             self.q_net.train_op,
             self.q_net.merged_summary,
             self.q_net.image_summaries,
             self.q_net.action_value,
             self.q_net.action_values_soft],
            feed_dict=feed_dict)

        # self.updateTarget()

        return l / len(rollout), ms, img_summ

    def updateTarget(self):
        for op in self.targetOps:
            self.sess.run(op)

    def eval(self, saver):
        self.saver = saver
        total_steps = 0
        episode_rewards = []

        print("Starting eval agent")
        with self.sess.as_default(), self.graph.as_default():
            while total_steps < FLAGS.test_episodes:
                episode_reward = 0
                episode_step_count = 0
                d = False
                s = self.env.get_initial_state()

                while not d:
                    a = self.policy_evaluation_eval(s)

                    s1, r, d, info = self.env.step(a)

                    r = np.clip(r, -1, 1)
                    episode_reward += r
                    episode_step_count += 1

                    s = s1
                print("Episode reward was {}".format(episode_reward))
                episode_rewards.append(episode_reward)
                total_steps += 1
        print("Mean reward is {}".format(np.mean(np.asarray(episode_rewards))))

    def play(self, saver):
        self.saver = saver
        train_stats = None

        # self.episode_count = self.sess.run(self.global_episode)
        self.total_steps = self.sess.run(self.global_episode)
        if self.total_steps == 0:
            self.updateTarget()

        print("Starting agent")
        _t = {'episode': Timer(), "step": Timer()}
        with self.sess.as_default(), self.graph.as_default():
            while self.total_steps < FLAGS.max_total_steps:
                _t["episode"].tic()
                if self.total_steps % FLAGS.target_update_freq == 0:
                    self.updateTarget()
                episode_reward = 0
                episode_step_count = 0
                q_values = []
                d = False
                # self.probability_of_random_action = self.exploration.value(self.total_steps)
                s = self.env.get_initial_state()

                while not d:
                    _t["step"].tic()
                    a, max_action_values_evaled = self.policy_evaluation(s)

                    if max_action_values_evaled is not None:
                        q_values.append(max_action_values_evaled)

                    s1, r, d, info = self.env.step(a)

                    r = np.clip(r, -1, 1)
                    episode_reward += r
                    episode_step_count += 1
                    self.total_steps += 1
                    self.episode_buffer.append([s, a, r, s1, d])

                    s = s1

                    if len(self.episode_buffer) == FLAGS.memory_size:
                        self.episode_buffer.popleft()

                    if self.total_steps > FLAGS.observation_steps and len(
                            self.episode_buffer) > FLAGS.observation_steps and self.total_steps % FLAGS.update_freq == 0:
                        l, ms, img_summ = self.train()
                        train_stats = l, ms, img_summ

                    _t["step"].toc()


                self.add_summary(episode_reward, episode_step_count, q_values, train_stats)

                self.sess.run(self.increment_global_episode)

                _t["episode"].toc()

        print('Avg time per step is {:.3f}'.format(_t["step"].average_time()))
        print('Avg time per episode is {:.3f}'.format(_t["episode"].average_time()))

        # fps = self.total_steps / _t['Total'].duration
        # print('Average time per episod is {}'.format(_t['episode'].average_time))

    def add_summary(self, episode_reward, episode_step_count, q_values, train_stats):
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_step_count)
        if len(q_values):
            self.episode_mean_values.append(np.mean(np.asarray(q_values)))
            self.episode_max_values.append(np.max(np.asarray(q_values)))
            self.episode_min_values.append(np.min(np.asarray(q_values)))

        if self.total_steps % FLAGS.summary_interval == 0 and self.total_steps != 0 and self.total_steps > FLAGS.observation_steps:
            if self.total_steps % FLAGS.checkpoint_interval == 0:
                self.save_model(self.saver, self.total_steps)

            l, ms, img_summ = train_stats

            # self.episode_mean_returns.append(np.mean(np.asarray(returns)))
            # self.episode_max_returns.append(np.max(np.asarray(returns)))
            # self.episode_min_returns.append(np.min(np.asarray(returns)))

            mean_reward = np.mean(self.episode_rewards[-FLAGS.summary_interval:])
            mean_length = np.mean(self.episode_lengths[-FLAGS.summary_interval:])
            mean_value = np.mean(self.episode_mean_values[-FLAGS.summary_interval:])
            max_value = np.mean(self.episode_max_values[-FLAGS.summary_interval:])
            min_value = np.mean(self.episode_min_values[-FLAGS.summary_interval:])

            # if episode_count % FLAGS.test_performance_interval == 0:
            #     won_games = self.episode_rewards[-FLAGS.test_performance_interval:].count(1)
            #     self.summary.value.add(tag='Perf/Won Games/1000', simple_value=float(won_games))


            self.summary.value.add(tag='Perf/Reward', simple_value=float(mean_reward))
            self.summary.value.add(tag='Perf/Length', simple_value=float(mean_length))
            self.summary.value.add(tag='Perf/Value_Mean', simple_value=float(mean_value))
            self.summary.value.add(tag='Perf/Value_Max', simple_value=float(max_value))
            self.summary.value.add(tag='Perf/Value_Min', simple_value=float(min_value))
            # self.summary.value.add(tag='Perf/Return_Mean', simple_value=float(mean_return))
            # self.summary.value.add(tag='Perf/Return_Max', simple_value=float(max_return))
            # self.summary.value.add(tag='Perf/Return_Min', simple_value=float(min_return))
            self.summary.value.add(tag='Perf/Probability_random_action',
                                   simple_value=float(self.probability_of_random_action))
            self.summary.value.add(tag='Losses/Loss', simple_value=float(l))

            self.write_summary(ms, img_summ)

    def policy_evaluation(self, s):
        action_values_q = None
        self.probability_of_random_action = self.exploration.value(self.total_steps)

        if random.random() <= self.probability_of_random_action:
            a = np.random.choice(range(len(self.env.gym_actions)))
        else:
            feed_dict = {self.q_net.inputs: [s]}
            action_values_evaled = self.sess.run(self.q_net.action_values_soft, feed_dict=feed_dict)[0]

            action_values_q = np.sum(
                np.multiply(action_values_evaled, np.tile(np.expand_dims(self.support, 0), [self.nb_actions, 1])), 1)
            a = np.argmax(action_values_q)
            a_one_hot = np.zeros(shape=(self.q_net.nb_actions, FLAGS.nb_atoms), dtype=np.int32)
            a_one_hot[a] = 1
            p_a_star = np.sum(np.multiply(action_values_evaled, a_one_hot), 0)

            # import matplotlib.pyplot as plt
            # ax = plt.subplot(111)
            # p1 = ax.step(self.support, p_a_star, color='blue')
            # # p2 = ax.step(skewed_support[0], p_a_star[0], color='magenta')
            # # p3 = ax.step(bellman[0], p_a_star[0], color='green')
            # # p4 = ax.step(self.support, m[0], color='red')
            # ax.autoscale(tight=True)
            #
            # plt.show()

        return a, np.max(action_values_q)

    def policy_evaluation_eval(self, s):
        feed_dict = {self.q_net.inputs: [s]}
        action_values_evaled = self.sess.run(self.q_net.action_values_soft, feed_dict=feed_dict)[0]

        action_values_q = np.sum(
            np.multiply(action_values_evaled, np.tile(np.expand_dims(self.support, 0), [self.nb_actions, 1])), 1)
        a = np.argmax(action_values_q)
        a_one_hot = np.zeros(shape=(self.q_net.nb_actions, FLAGS.nb_atoms), dtype=np.int32)
        a_one_hot[a] = 1
        p_a_star = np.sum(np.multiply(action_values_evaled, a_one_hot), 0)

        # import matplotlib.pyplot as plt
        # ax = plt.subplot(111)
        # p1 = ax.step(self.support, p_a_star, color='blue')
        # # p2 = ax.step(skewed_support[0], p_a_star[0], color='magenta')
        # # p3 = ax.step(bellman[0], p_a_star[0], color='green')
        # # p4 = ax.step(self.support, m[0], color='red')
        # ax.autoscale(tight=True)
        #
        # plt.show()

        return a


    def get_target_distribution(self, rewards, done, next_observations, observations, actions):
        target_actionv_values_evaled, action_values_evaled = self.sess.run([self.target_net.action_values_soft, self.q_net.action_values_soft],
                                                     feed_dict={
                                                         self.target_net.inputs: np.stack(next_observations, axis=0),
                                                         self.q_net.inputs: np.stack(observations, axis=0)
                                                     })
        a_one_hot = np.zeros(shape=(FLAGS.batch_size, self.q_net.nb_actions, FLAGS.nb_atoms), dtype=np.int32)
        a_one_hot[np.arange(FLAGS.batch_size), np.asarray(actions, dtype=np.int32)] = 1
        pt_a_star = np.sum(np.multiply(action_values_evaled, a_one_hot), axis=1)

        p = np.sum(
            target_actionv_values_evaled * np.tile(np.expand_dims(np.expand_dims(self.support, 0), 0),
                                                   [FLAGS.batch_size, self.q_net.nb_actions, 1]), 2)

        a = np.argmax(p, axis=1)

        a_one_hot = np.zeros(shape=(FLAGS.batch_size, self.q_net.nb_actions, FLAGS.nb_atoms), dtype=np.int32)
        # a_one_hot[:, a, :] = 1
        a_one_hot[np.arange(FLAGS.batch_size), a] = 1
        # a_one_hot = np.tile(np.expand_dims(a_one_hot, 2), [1, 1, FLAGS.nb_atoms])
        # a_one_hot = np.reshape(a_one_hot, (FLAGS.batch_size, self.q_net.nb_actions, FLAGS.nb_atoms))
        p_a_star = np.sum(np.multiply(target_actionv_values_evaled, a_one_hot), axis=1)

        rewards = np.tile(np.expand_dims(np.asarray(rewards, dtype=np.float32), 1), [1, FLAGS.nb_atoms])
        gamma = np.tile(np.expand_dims(np.logical_not(np.asarray(done, dtype=np.int32)) * FLAGS.gamma, 1),
                        [1, FLAGS.nb_atoms])
        # Compute projection of the application of the Bellman operator.
        skewed_support = gamma * np.tile(np.expand_dims(self.support, 0), [FLAGS.batch_size, 1])
        bellman = rewards + skewed_support
        bellman = np.clip(bellman, FLAGS.v_min, FLAGS.v_max)

        # Compute categorical indices for distributing the probability
        m = np.zeros(shape=(FLAGS.batch_size, FLAGS.nb_atoms))
        b = (bellman - FLAGS.v_min) / self.delta_z
        l = np.asarray(np.floor(b), dtype=np.int32)
        u = np.asarray(np.ceil(b), dtype=np.int32)

        # Distribute probability
        # for j in range(FLAGS.nb_atoms):
        #     m[:, l[:, j]] += target_actionv_values_evaled_max[:, j] * (u[:, j] - b[:, j])
        #     m[:, u[:, j]] += target_actionv_values_evaled_max[:, j] * (b[:, j] - l[:, j])

        for i in range(FLAGS.batch_size):
            for j in range(FLAGS.nb_atoms):
                lidx = l[i][j]
                uidx = u[i][j]
                m[i, lidx] += p_a_star[i, j] * (uidx - b[i, j])
                m[i, uidx] += p_a_star[i, j] * (b[i, j] - lidx)

        # if self.total_steps > FLAGS.explore_steps:
        #     import matplotlib.pyplot as plt
        #     ax = plt.subplot(111)
        #     # p1 = ax.step(self.support, p_a_star[0], color='blue')
        #     # p2 = ax.step(skewed_support[0], p_a_star[0], color='magenta')
        #     # p3 = ax.step(bellman[0], p_a_star[0], color='green')
        #     # p4 = ax.step(self.support, m[0], color='red')
        #     p4 = ax.step(self.support, pt_a_star[1], color='cyan')
        #     ax.autoscale(tight=True)
        #
        #     plt.show()
        return m



