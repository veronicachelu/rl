from threading import Lock

import numpy as np
import tensorflow as tf
from agents.base_agent import BaseAgent
from nets.sf_network import SFNetwork
from configs import sf_flags
from collections import deque
from utils.schedules import LinearSchedule
from utils.timer import Timer
import os
FLAGS = tf.app.flags.FLAGS
import random

# Starting threads
main_lock = Lock()

class SFAgent(BaseAgent):
    def __init__(self, game, sess, nb_actions, global_step):
        BaseAgent.__init__(self, game, sess, nb_actions, global_step)
        self.name = "SF_agent"
        self.model_path = os.path.join(FLAGS.checkpoint_dir, FLAGS.algorithm)

        self.nb_states = self.env.nb_states

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

        self.q_net = SFNetwork(self.nb_actions, self.nb_states, 'orig')
        self.target_net = SFNetwork(self.nb_actions, self.nb_states, 'target')

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

        state_features = np.identity(self.nb_states)

        target_sf_evaled = self.sess.run(self.target_net.sf,
                                                     feed_dict={self.target_net.features: state_features[next_observations]})
        target_sf_evaled_exp = np.mean(target_sf_evaled, axis=1)

        target_sf_evaled_new = []

        for i in range(FLAGS.batch_size):
            if done[i]:
                target_sf_evaled_new.append(rewards[i])
            else:
                target_sf_evaled_new.append(
                    rewards[i] + FLAGS.gamma * target_sf_evaled_exp[i])

        feed_dict = {self.q_net.target_sf: target_sf_evaled_new,
                     self.q_net.features: state_features[observations],
                     self.q_net.actions: actions}
        l, _, ms, img_summ = self.sess.run(
            [self.q_net.sf_loss,
             self.q_net.train_op,
             self.q_net.merged_summary],
            feed_dict=feed_dict)

        # self.updateTarget()

        return l / len(rollout), ms

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
        self.total_steps = self.sess.run(self.global_step)
        if self.total_steps == 0:
            self.updateTarget()

        print("Starting agent")
        _t = {'episode': Timer(), "step": Timer()}
        with self.sess.as_default(), self.graph.as_default():
            while self.total_steps < FLAGS.max_total_steps:

                if self.total_steps == 0 or d:
                    _t["episode"].tic()
                    if self.total_steps % FLAGS.target_update_freq == 0:
                        self.updateTarget()
                    episode_reward = 0
                    episode_step_count = 0
                    q_values = []

                    d = False
                    # self.probability_of_random_action = self.exploration.value(self.total_steps)
                    s = self.env.get_initial_state()

                _t["step"].tic()
                a = self.policy_evaluation(s)


                s1, r, d, info = self.env.step(a)

                r = np.clip(r, -1, 1)
                # episode_reward += r
                # episode_step_count += 1
                self.total_steps += 1
                self.episode_buffer.append([s, a, r, s1, d])

                s = s1

                if len(self.episode_buffer) == FLAGS.memory_size:
                    self.episode_buffer.popleft()

                if self.total_steps > FLAGS.observation_steps and len(
                        self.episode_buffer) > FLAGS.observation_steps and self.total_steps % FLAGS.update_freq == 0:
                    l, ms = self.train()
                    train_stats = l, ms

                _t["step"].toc()

                self.sess.run(self.increment_global_step)


                self.add_summary(r, train_stats)


                _t["episode"].toc()

        # print('Avg time per step is {:.3f}'.format(_t["step"].average_time()))
        # print('Avg time per episode is {:.3f}'.format(_t["episode"].average_time()))

        # fps = self.total_steps / _t['Total'].duration
        # print('Average time per episod is {}'.format(_t['episode'].average_time))

    def add_summary(self, reward, train_stats):
        if self.total_steps % FLAGS.summary_interval == 0 and self.total_steps != 0 and self.total_steps > FLAGS.observation_steps:
            if self.total_steps % FLAGS.checkpoint_interval == 0:
                self.save_model(self.saver, self.total_steps)

            l, ms = train_stats

            self.summary.value.add(tag='Perf/Reward', simple_value=float(reward))
            self.summary.value.add(tag='Losses/Loss', simple_value=float(l))

            self.write_summary(ms)

    def policy_evaluation(self, s):
        # action_values_evaled = None
        # self.probability_of_random_action = self.exploration.value(self.total_steps)
        # if random.random() <= self.probability_of_random_action:
        a = np.random.choice(range(len(self.env.gym_actions)))
        # else:
        #     feed_dict = {self.q_net.inputs: [s]}
        #     action_values_evaled = self.sess.run(self.q_net.action_values, feed_dict=feed_dict)[0]
        #
        #     a = np.argmax(action_values_evaled)

        return a

    def policy_evaluation_eval(self, s):
        feed_dict = {self.q_net.inputs: [s]}
        action_values_evaled = self.sess.run(self.q_net.action_values, feed_dict=feed_dict)[0]

        a = np.argmax(action_values_evaled)

        return a
