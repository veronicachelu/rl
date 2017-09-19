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
from tools.policy_iteration import PolicyIteration
FLAGS = tf.app.flags.FLAGS
import random
from utils.visualizer import Visualizer

# Starting threads
main_lock = Lock()

class SFAgent(BaseAgent):
    def __init__(self, game, sess, nb_actions, global_step):
        BaseAgent.__init__(self, game, sess, nb_actions, global_step)
        self.name = "SF_agent"
        self.model_path = os.path.join(FLAGS.checkpoint_dir, FLAGS.algorithm)

        self.nb_states = self.env.nb_states
        self.sf_buffer = np.zeros([25, 25])
        self.seen_states = set()
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
        # target_sf_evaled_exp = np.mean(target_sf_evaled, axis=1)

        # gamma = np.tile(np.expand_dims(np.asarray(np.logical_not(done), dtype=np.int32) * FLAGS.gamma, 1),
        #                 [1, self.nb_states])
        #
        # target_sf_evaled_new = state_features[next_observations] + gamma * target_sf_evaled_exp
        #
        feed_dict = {self.q_net.target_sf: target_sf_evaled,
                     # self.q_net.target_reward: np.stack(rewards, axis=0),
                     self.q_net.features: state_features[observations]}
                     # self.q_net.actions: actions}
        sf_l, _, ms = self.sess.run(
            [self.q_net.sf_loss,
             # self.q_net.reward_loss,
             # self.q_net.total_loss,
             self.q_net.train_op,
             self.q_net.merged_summary],
            feed_dict=feed_dict)

        # self.updateTarget()

        return sf_l / len(rollout), ms

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
        d = True
        # self.episode_count = self.sess.run(self.global_episode)
        self.total_steps = self.sess.run(self.global_step)
        if self.total_steps == 0:
            self.updateTarget()
        self.nb_episodes = 0
        episode_reward = 0
        episode_step_count = 0
        q_values = []
        print("Starting agent")
        _t = {'episode': Timer(), "step": Timer()}
        with self.sess.as_default(), self.graph.as_default():
            while self.total_steps < FLAGS.max_total_steps:

                if self.total_steps == 0 or d or episode_step_count % 30 == 0:
                    _t["episode"].tic()
                    if self.total_steps % FLAGS.target_update_freq == 0:
                        self.updateTarget()
                    self.add_summary(episode_reward, episode_step_count, q_values, train_stats)
                    episode_reward = 0
                    episode_step_count = 0
                    q_values = []
                    if self.total_steps != 0:
                        self.nb_episodes += 1
                    d = False
                    # self.probability_of_random_action = self.exploration.value(self.total_steps)
                    s = self.env.get_initial_state()

                _t["step"].tic()
                a, max_action_values_evaled = self.policy_evaluation(s)

                if max_action_values_evaled is None:
                    q_values.append(0)
                else:
                    q_values.append(max_action_values_evaled)

                s1, r, d = self.env.step(a)
                # self.env.render()

                r = np.clip(r, -1, 1)
                episode_reward += r
                episode_step_count += 1
                self.total_steps += 1
                self.episode_buffer.append([s, a, r, s1, d])

                s = s1

                if len(self.episode_buffer) == FLAGS.memory_size:
                    self.episode_buffer.popleft()

                if self.total_steps > FLAGS.observation_steps and len(
                        self.episode_buffer) > FLAGS.observation_steps and self.total_steps % FLAGS.update_freq == 0:# and FLAGS.task != "discover":
                    sf_l, ms = self.train()
                    train_stats = sf_l, ms

                if len(self.seen_states) == self.nb_states:
                    s, v = self.discover_options()
                    # self.sf_buffer.popleft()

                if self.total_steps > FLAGS.nb_steps_sf:
                    self.construct_successive_matrix()
                    # self.add_successive_feature(s, a)


                _t["step"].toc()

                self.sess.run(self.increment_global_step)


            _t["episode"].toc()
        # print('Avg time per step is {:.3f}'.format(_t["step"].average_time()))
        # print('Avg time per episode is {:.3f}'.format(_t["episode"].average_time()))

        # fps = self.total_steps / _t['Total'].duration
        # print('Average time per episod is {}'.format(_t['episode'].average_time))

    def construct_successive_matrix(self):
        for s in range(self.nb_states):
            state_features = np.identity(self.nb_states)
            sf_feat = self.sess.run(self.q_net.sf,
                                    feed_dict={self.q_net.features: state_features[s:s + 1]})
            a = np.random.choice(range(self.nb_actions))
            a_one_hot = np.zeros(shape=(1, self.nb_actions, self.nb_states), dtype=np.int32)
            a_one_hot[0, a] = 1
            sf_feat_a = np.sum(np.multiply(sf_feat, a_one_hot), axis=1)
            self.sf_buffer[s] = sf_feat_a
            if s not in self.seen_states:
                self.seen_states.add(s)

    def add_successive_feature(self, s, a):
        state_features = np.identity(self.nb_states)
        sf_feat = self.sess.run(self.q_net.sf,
                                         feed_dict={self.q_net.features: state_features[s:s+1]})
        a_one_hot = np.zeros(shape=(1, self.nb_actions, self.nb_states), dtype=np.int32)
        a_one_hot[0, a] = 1
        sf_feat_a = np.sum(np.multiply(sf_feat, a_one_hot), axis=1)
        if s not in self.seen_states:
            self.seen_states.add(s)
        self.sf_buffer[s] = sf_feat_a

    def discover_options(self):
        sf_matrix = tf.convert_to_tensor(np.squeeze(np.array(self.sf_buffer)), dtype=tf.float32)
        s, u, v = tf.svd(sf_matrix)

        # discard noise, get first 10
        # s = s[:10]
        # v = v[:10]

        if FLAGS.task == "discover":
            # Plotting all the basis
            plot = Visualizer(self.env)
            s_evaled , v_evaled = self.sess.run([s, v])
            idx = s_evaled.argsort()[::-1]
            s_evaled = s_evaled[idx]
            v_evaled = v_evaled[:, idx]
            plot.plotBasisFunctions(s_evaled , v_evaled)

            guard = len(s_evaled)
            epsilon = 0.001
            options = []
            actionSetPerOption = []
            for i in range(guard):
                idx = guard - i - 1
                print('Solving for eigenvector #' + str(idx))
                polIter = PolicyIteration(0.9, self.env, augmentActionSet=True)
                self.env.define_reward_function(v_evaled[:, idx])
                V, pi = polIter.solvePolicyIteration()

                # Now I will eliminate any actions that may give us a small improvement.
                # This is where the epsilon parameter is important. If it is not set all
                # it will never be considered, since I set it to a very small value
                for j in range(len(V)):
                    if V[j] < epsilon:
                        pi[j] = len(self.env.get_action_set())

                # if plotGraphs:
                plot.plotValueFunction(V[0:self.nb_states], str(idx) + '_')
                plot.plotPolicy(pi[0:self.nb_states], str(idx) + '_')

                options.append(pi[0:self.nb_states])
                optionsActionSet = self.env.get_action_set()
                np.append(optionsActionSet, ['terminate'])
                actionSetPerOption.append(optionsActionSet)

        return s, v


    def add_summary(self, episode_reward, episode_step_count, q_values, train_stats):
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_step_count)
        if len(q_values):
            self.episode_mean_values.append(np.mean(np.asarray(q_values)))
            self.episode_max_values.append(np.max(np.asarray(q_values)))
            self.episode_min_values.append(np.min(np.asarray(q_values)))

        if self.nb_episodes % FLAGS.summary_interval == 0 and self.nb_episodes != 0 and self.total_steps > FLAGS.observation_steps:
            if self.nb_episodes % FLAGS.checkpoint_interval == 0:
                self.save_model(self.saver, self.total_steps)
            if train_stats is not None:
                sf_l, ms = train_stats
    
                mean_reward = np.mean(self.episode_rewards[-FLAGS.summary_interval:])
                mean_length = np.mean(self.episode_lengths[-FLAGS.summary_interval:])
                mean_value = np.mean(self.episode_mean_values[-FLAGS.summary_interval:])
                max_value = np.mean(self.episode_max_values[-FLAGS.summary_interval:])
                min_value = np.mean(self.episode_min_values[-FLAGS.summary_interval:])
    
    
                self.summary.value.add(tag='Perf/Reward', simple_value=float(mean_reward))
                self.summary.value.add(tag='Perf/Length', simple_value=float(mean_length))
                self.summary.value.add(tag='Perf/Value_Mean', simple_value=float(mean_value))
                self.summary.value.add(tag='Perf/Value_Max', simple_value=float(max_value))
                self.summary.value.add(tag='Perf/Value_Min', simple_value=float(min_value))
                self.summary.value.add(tag='Perf/Probability_random_action',
                                       simple_value=float(self.probability_of_random_action))
    
                self.summary.value.add(tag='Losses/SF_Loss', simple_value=float(sf_l))
                # self.summary.value.add(tag='Losses/R_Loss', simple_value=float(r_l))
                # self.summary.value.add(tag='Losses/T_Loss', simple_value=float(t_l))
    
                self.write_summary(ms)

    def policy_evaluation(self, s):
        action_values_evaled = None
        self.probability_of_random_action = self.exploration.value(self.total_steps)
        # if random.random() <= self.probability_of_random_action:
        a = np.random.choice(range(self.nb_actions))
        # else:
        #     state_features = np.identity(self.nb_states)
        #     feed_dict = {self.q_net.features: state_features[s:s+1]}
        #     action_values_evaled = self.sess.run(self.q_net.q, feed_dict=feed_dict)[0]
        #
        #     a = np.argmax(action_values_evaled)

        return a, 0

    def policy_evaluation_eval(self, s):
        feed_dict = {self.q_net.inputs: [s]}
        action_values_evaled = self.sess.run(self.q_net.action_values, feed_dict=feed_dict)[0]

        a = np.argmax(action_values_evaled)

        return a
