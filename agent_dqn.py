import time
import numpy as np
import random
import tensorflow as tf
import tensorflow.contrib.slim as slim
import os
from threading import Thread
from tensorboardX import SummaryWriter
from envs.Harvest import HarvestGame

logs_dir = "./logs/logs_drqnd"

batch_size = 4  # How many experience traces to use for each training step.
trace_length = 8  # How long each experience trace will be when training
update_freq = 4  # How often to perform a training step.
gamma = .99  # Discount factor on the target Q-values
init_epsilon = 1  # Starting chance of random action
final_epsilon = 0.1  # Final chance of random action
annealing_steps = 500000  # How many steps of training to reduce startE to endE.
max_steps = 5000000  # How many episodes of game environment to train network with.
pre_train_timesteps = 50000  # How many steps of random actions before training begins.
max_epLength = 1000  # The max allowed length of our episode.
load_model = False  # Whether to load a saved model.
train = True  # Whether to train or test
models_dir = "./models_drqn"  # The path to save our model to.
h_size = 32  # The size of the final hidden layer before splitting it into Advantage and Value streams.
tau = 0.001  # Rate to update target network toward primary network

n_agents = 2  # Number of agents
agent_lat_view = 3
agent_front_view = 10
input_h = agent_lat_view * 2 + 1
input_w = agent_front_view

hidden_units = 128

env = HarvestGame(n_agents=n_agents, agent_view_h=agent_front_view, agent_view_w=agent_lat_view)
action_dim = env.action_dim


class DQN:
    def __init__(self, scope, rnn_cell):
        with tf.variable_scope(scope):
            # The network receives a frame from the game, flattened into an array.
            # scalarinput is the state
            self.scalarInput = tf.placeholder(shape=[None, input_h * input_w * 3], dtype=tf.float32)
            self.fc1 = slim.fully_connected(self.scalarInput, 256)
            self.trainLength = tf.placeholder(dtype=tf.int32)
            self.batch_size = tf.placeholder(dtype=tf.int32, shape=[])
            self.rnn_in = tf.reshape(self.fc1, [self.batch_size, self.trainLength, 256])
            # how many episode in one training
            self.state_in = rnn_cell.zero_state(self.batch_size, tf.float32)
            self.rnn, self.rnn_state = tf.nn.dynamic_rnn(inputs=self.rnn_in, cell=rnn_cell, dtype=tf.float32,
                                                         initial_state=self.state_in)
            self.rnn = tf.reshape(self.rnn, shape=[-1, 32])
            self.streamA, self.streamV = tf.split(self.rnn, 2, 1)
            xavier_init = tf.contrib.layers.xavier_initializer()
            self.AW = tf.Variable(xavier_init([h_size // 2, action_dim]))
            self.VW = tf.Variable(xavier_init([h_size // 2, 1]))
            self.Advantage = tf.matmul(self.streamA, self.AW)
            self.Value = tf.matmul(self.streamV, self.VW)
        # Q_eval
        self.Qout = self.Value + tf.subtract(self.Advantage, tf.reduce_mean(self.Advantage, axis=1, keepdims=True))
        self.predict = tf.argmax(self.Qout, 1)
        # Q_target
        self.targetQ = tf.placeholder(shape=[None], dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
        self.actions_onehot = tf.one_hot(self.actions, action_dim, dtype=tf.float32)

        self.Q = tf.reduce_sum(tf.multiply(self.Qout, self.actions_onehot), axis=1)
        self.td_error = tf.square(self.targetQ - self.Q)
        # loss and gradient descent
        self.loss = tf.reduce_mean(self.td_error)
        self.trainer = tf.train.AdamOptimizer(learning_rate=0.0001)

        self.updateModel = self.trainer.minimize(self.loss)


class Qnetwork():
    def __init__(self, scope):
        # num_actions = action_dim
        with tf.variable_scope(scope):
            # The network recieves a frame from the game, flattened into an array.
            # It then resizes it and processes it through four convolutional layers.
            self.scalarInput = tf.placeholder(shape=[None, input_h, input_w, 3], dtype=tf.float32)

            self.conv1 = slim.conv2d(self.scalarInput, 32,
                                     kernel_size=[3, 3], stride=[2, 2],
                                     biases_initializer=None,
                                     activation_fn=tf.nn.elu)
            self.conv2 = slim.conv2d(self.conv1, 64,
                                     kernel_size=[3, 3],
                                     stride=[2, 2],
                                     biases_initializer=None,
                                     activation_fn=tf.nn.elu)

            # We take the output from the final convolutional layer
            # and split it into separate advantage and value streams.
            self.hidden = slim.fully_connected(slim.flatten(self.conv2),
                                               hidden_units, activation_fn=tf.nn.elu)
            self.advantage = slim.fully_connected(self.hidden, action_dim, activation_fn=None,
                                                  biases_initializer=None)
            self.value = slim.fully_connected(self.hidden, 1, activation_fn=None,
                                              biases_initializer=None)

            # Then combine them together to get our final Q-values.
            self.q_out = self.value + tf.subtract(self.advantage,
                                                  tf.reduce_mean(self.advantage, axis=1, keep_dims=True))
            self.predict = tf.argmax(self.q_out, 1)

            # Below we obtain the loss by taking the sum of squares difference
            # between the target and prediction Q values.
            self.targetQ = tf.placeholder(shape=[None], dtype=tf.float32)
            self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
            self.actions_onehot = tf.one_hot(self.actions, action_dim, dtype=tf.float32)

            self.Q = tf.reduce_sum(tf.multiply(self.q_out, self.actions_onehot), axis=1)

            self.td_error = tf.square(self.targetQ - self.Q)
            self.loss = tf.reduce_mean(self.td_error)
            self.trainer = tf.train.AdamOptimizer(learning_rate=0.001)
            self.updateModel = self.trainer.minimize(self.loss)


# experience replay
class experience_buffer:
    def __init__(self, buffer_size=500):
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self, experience):
        if len(self.buffer) + 1 >= self.buffer_size:
            self.buffer [0:(1 + len(self.buffer)) - self.buffer_size] = []
        self.buffer.append(experience)

    def sample(self, sample_size, seq_length):
        sampled_episodes = random.sample(self.buffer, sample_size)
        sampledTraces = []
        for episode in sampled_episodes:
            point = np.random.randint(0, len(episode) + 1 - seq_length)
            sampledTraces.append(episode [point:point + seq_length])
        sampledTraces = np.array(sampledTraces)
        return np.reshape(sampledTraces, [sample_size * seq_length, 5])


def processState(x):
    return np.reshape(x, [n_agents, input_h * input_w * 3])


def updateTargetGraph(main_vars, target_vars):
    op_holder = []
    for idx, var in enumerate(main_vars):
        # update slowly
        if idx % 2 == 0:
            op_holder.append(target_vars [idx].assign(
                (var.value() * tau) + ((1 - tau) * target_vars [idx].value())))
        else:
            # directly replace the target network
            op_holder.append(target_vars [idx].assign(var.value()))
    return op_holder


def updateTarget(op_holder):
    for op in op_holder:
        sess.run(op)


# choose actions
def act(policy, observation, actions_batch, states_batch, idx, ac_type):
    if (np.random.rand(1) < epsilon or total_steps < pre_train_timesteps) and train:
        actions_batch [idx] = np.random.randint(0, action_dim)
        ac_type [idx] = False
    else:
        a, new_cell_state = sess.run([policy.predict, policy.rnn_state],
                                     feed_dict={policy.scalarInput: [observation / 255.0],
                                                policy.trainLength: 1,
                                                policy.state_in: states_batch [idx],
                                                policy.batch_size: 1})
        actions_batch [idx] = a [0]
        ac_type [idx] = True
        states_batch [idx] = new_cell_state


def act_dqn(policy, observation, actions_batch, idx, ac_type):
    if (np.random.rand(1) < epsilon or total_steps < pre_train_timesteps) and train:
        actions_batch [idx] = np.random.randint(0, action_dim)
        ac_type [idx] = False
    else:
        a = sess.run(policy.predict,feed_dict={policy.scalarInput: np.array([observation / 255.0]).reshape((1,input_h,input_w,3))})
        actions_batch [idx] = a [0]
        ac_type [idx] = True


def learn(main_policy, target_policy, target_ops, my_buffer, idx):
    updateTarget(target_ops)
    # Reset the recurrent layer's hidden state
    state_train = (np.zeros([batch_size, h_size]), np.zeros([batch_size, h_size]))
    trainBatch = my_buffer.sample(batch_size, trace_length)  # Get a random batch of experiences.
    # Below we perform the Double-DQN update to the target Q-values
    if idx % 2 == 0:
        # argmax of Q2
        Q1 = sess.run(main_policy.predict,
                  feed_dict={main_policy.scalarInput: np.vstack(trainBatch [:, 3] / 255.0),
                             main_policy.trainLength: trace_length,
                             main_policy.state_in: state_train,
                             main_policy.batch_size: batch_size})
        # Q_eval
        Q2 = sess.run(target_policy.Qout,
                  feed_dict={target_policy.scalarInput: np.vstack(trainBatch [:, 3] / 255.0),
                             target_policy.trainLength: trace_length,
                             target_policy.state_in: state_train,
                             target_policy.batch_size: batch_size})
        end_multiplier = -(trainBatch [:, 4] - 1)
    # double q_next?
        doubleQ = Q2 [range(batch_size * trace_length), Q1]

        targetQ = trainBatch [:, 2] + (gamma * doubleQ * end_multiplier)
        p_loss, _ = sess.run([main_policy.loss, main_policy.updateModel],
                             feed_dict={main_policy.scalarInput: np.vstack(trainBatch [:, 0] / 255.0),
                                        main_policy.targetQ: targetQ,
                                        main_policy.actions: trainBatch [:, 1],
                                        main_policy.trainLength: trace_length,
                                        main_policy.state_in: state_train,
                                        main_policy.batch_size: batch_size})
    else:
        # vanilla dqn
        Q1 = sess.run(main_policy.predict,
                      feed_dict={main_policy.scalarInput: np.vstack(trainBatch [:, 3] / 255.0).reshape((32,input_h,input_w,3))})

        Q2 = sess.run(target_policy.q_out,
                      feed_dict={target_policy.scalarInput: np.vstack(trainBatch [:, 3] / 255.0).reshape((32,input_h,input_w,3))})
        end_multiplier = -(trainBatch [:, 4] - 1)
        doubleQ = Q2 [range(batch_size * trace_length), Q1]
        targetQ = trainBatch [:, 2] + (gamma * doubleQ * end_multiplier)
    # Update the network with our target values.
        p_loss, _ = sess.run([main_policy.loss, main_policy.updateModel],
                             feed_dict={main_policy.scalarInput: np.vstack(trainBatch [:, 0] / 255.0).reshape((32,input_h,input_w,3)),
                                        main_policy.targetQ: targetQ,
                                        main_policy.actions: trainBatch [:, 1]})
    policy_losses [idx] += p_loss


if __name__ == '__main__':
    tf.reset_default_graph()
    agents = {}
    with tf.device("/cpu:0"):
        for i in range(n_agents):
            if i%2 == 0:
                cell = tf.contrib.rnn.BasicLSTMCell(num_units=32, state_is_tuple=True)
                cellT = tf.contrib.rnn.BasicLSTMCell(num_units=32, state_is_tuple=True)
                agents [str(i)] = {'mainNet': DQN("agent_" + str(i) + "/main", cell),
                                   'targetNet': DQN("agent_" + str(i) + "/target", cellT),
                                   'my_buffer': experience_buffer(),
                                   'episode_buffer': [],
                                   'targetOps': []}
            else:
                agents [str(i)] = {'mainNet': Qnetwork("agent_" + str(i) + "/main"),
                                   'targetNet': Qnetwork("agent_" + str(i) + "/target"),
                                   'my_buffer': experience_buffer(),
                                   'episode_buffer': [],
                                   'targetOps': []}
        init = tf.global_variables_initializer()
        for i in range(n_agents):
            main_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                            scope="agent_" + str(i) + "/main")
            target_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                              scope="agent_" + str(i) + "/target")
            targetOps = updateTargetGraph(main_params, target_params)
            agents [str(i)] ['targetOps'] = targetOps

    saver = tf.train.Saver()
    writer = SummaryWriter(log_dir=logs_dir)

    # Set the rate of random action decrease.
    epsilon = init_epsilon
    stepDrop = (init_epsilon - final_epsilon) / annealing_steps
    # Losses tracking
    policy_losses = np.zeros((n_agents,))
    # create lists to contain statistics
    R_series = []
    U_series = []
    E_series = []
    S_series = []
    V_series = []
    total_steps = 0
    num_episodes = 0
    # Make a path for our model to be saved in.
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    acs = [{} for i in range(n_agents)]
    ac_class = [{} for i in range(n_agents)]
    cell_states = [{} for i in range(n_agents)]
    timeout_obs = np.ones((input_h, input_w, 3), dtype=np.uint8) * 255

    with tf.Session() as sess:
        sess.run(init)
        # Copy main to target params
        for i in range(n_agents):
            updateTarget(agents [str(i)] ['targetOps'])
        if load_model:
            print('Loading Model...')
            ckpt = tf.train.get_checkpoint_state(models_dir)
            saver.restore(sess, ckpt.model_checkpoint_path)
        while total_steps < max_steps:
            for k in range(n_agents):
                agents [str(k)] ['episode_buffer'] = []
            # Reset environment and get first new observation
            s = env.reset()
            s = processState(s)
            # d = False
            # r_t = np.zeros((1, n_agents))
            r_t = np.zeros((1, n_agents))
            stock = np.zeros((1,))
            shots = np.zeros((1,))
            T = 0
            # Reset the recurrent layer's hidden state
            for k in range(n_agents):
                cell_states [k] = (np.zeros([1, h_size]), np.zeros([1, h_size]))
            # The Q-Network
            t0 = time.time()
            while T < max_epLength:
                T += 1
                threads = []
                for k in range(n_agents):
                    # act(policy, observation, actions_batch, states_batch, idx, ac_type)
                    if k%2==0:
                        thread = Thread(target=act,
                                    args=[agents [str(k)] ['mainNet'], s [k], acs, cell_states, k, ac_class])
                    else:
                        thread = Thread(target=act_dqn,
                                        args=[agents [str(k)] ['mainNet'], s [k], acs, k, ac_class])
                    thread.start()
                    threads.append(thread)
                for thread in threads:
                    thread.join()
                s1, r, d, stock_t = env.step(acs)

                if num_episodes % 500 == 0 and (num_episodes > 0 or not train):
                    if not train:
                        env.render(save_snapshots=True)
                    else:
                        env.render()

                s1 = processState(s1)
                total_steps += 1
                for k in range(n_agents):
                    if s [k] != timeout_obs:
                        agents [str(k)] ['episode_buffer'].append(np.reshape(np.array(
                            [s [k], acs [k], r [k], s1 [k], d]), [1, 5]))  # Save the experience to our episode buffer.
                if total_steps > pre_train_timesteps and train:
                    if epsilon > final_epsilon:
                        epsilon -= stepDrop
                    if total_steps % update_freq == 0:
                        threads = []
                        for k in range(n_agents):
                            thread = Thread(target=learn,
                                            args=[agents [str(k)] ['mainNet'], agents [str(k)] ['targetNet'],
                                                  agents [str(k)] ['targetOps'],
                                                  agents [str(k)] ['my_buffer'], k])
                            thread.start()
                            threads.append(thread)
                        for thread in threads:
                            thread.join()

                # # share the rewards
                # for idx,value in enumerate(r):
                #     if idx%2 == 0:
                #         temp_r = np.add(temp_r,i)
                #     else:
                #         temp_r2 = np.add(temp_r2,i)
                #
                # r_t = np.append(r_t, [np.array(temp_r)], 0)
                # r_t = np.append(r_t, [np.array(temp_r2)], 0)
                # r_t = np.append(r_t, [np.array(temp_r)], 0)
                # r_t = np.append(r_t, [np.array(temp_r2)], 0)

                # temp_r2,temp_r = 0,0
                # for idx,value in enumerate(r):
                #     if idx%2 == 0:
                #         temp_r = np.add(temp_r, value)
                #     else:
                #         temp_r2 = np.add(temp_r2, value)

                # r_t = np.append(r_t, [np.array([temp_r,temp_r2,temp_r,temp_r2])], 0)
                ## r_t = np.append(r_t, [np.array(temp_r2)], 0)
                ## r_t = np.append(r_t, [np.array(temp_r)], 0)
                ## r_t = np.append(r_t, [np.array(temp_r2)], 0)

                r_t = np.append(r_t, [np.array(r)], 0)
                stock = np.append(stock, [np.array(stock_t)], 0)
                shots = np.append(shots, [np.sum((np.array(acs) == 7))], 0)
                s = s1
                if d:
                    break
            t1 = time.time()
            num_episodes += 1

            for k in range(n_agents):
                bufferArray = np.array(agents [str(k)] ['episode_buffer'])
                episodeBuffer = list(zip(bufferArray))
                agents [str(k)] ['my_buffer'].add(episodeBuffer)
            R = np.sum(r_t, 0)
            U = np.sum(R)
            if U > 0:
                E = 1 - 0.5 * np.abs(np.subtract.outer(R, R)).mean() / U
            else:
                E = 1
            R_series.append(R)
            S = np.sum(stock [1:])
            P = np.mean(shots [1:])
            U_series.append(U)
            E_series.append(E)
            S_series.append(S)
            V_series.append(P)
            # Periodically save the model.
            if num_episodes % 100 == 0:
                saver.save(sess, models_dir + '/model-' + str(num_episodes) + '.ckpt')
                print("Saved Model")
            # Print and plot the statistics
            if len(R_series) % 10 == 0 and total_steps > pre_train_timesteps:
                individual_rewards = {}
                losses = {}
                for k in range(n_agents):
                    individual_rewards ['Agent {}'.format(k)] = np.mean(R_series [-10:], 0) [k]
                    losses ['Agent {}'.format(k)] = policy_losses [k] / (
                            total_steps - pre_train_timesteps)
                writer.add_scalars('Individual/Rewards', individual_rewards,
                                   total_steps - pre_train_timesteps)
                writer.add_scalars('Losses/Policy Loss', losses,
                                   total_steps - pre_train_timesteps)
                writer.add_scalar('Social/Utilities', np.mean(U_series [-10:]),
                                  total_steps - pre_train_timesteps)
                writer.add_scalar('Social/Equality', np.mean(E_series [-10:]),
                                  total_steps - pre_train_timesteps)
                writer.add_scalar('Social/Violence', np.mean(V_series [-10:]),
                                  total_steps - pre_train_timesteps)
                writer.add_scalar('Social/Sustainability', np.mean(S_series [-10:]),
                                  total_steps - pre_train_timesteps)
                print('------------------STATISTICS------------------')
                print('Steps so far: {}'.format(total_steps - pre_train_timesteps))
                print('Epsilon: {}'.format(epsilon))
                print('Episode time length: {}'.format(t1 - t0))
                print('Mean Policy Loss: ',
                      np.mean(policy_losses) / (total_steps - pre_train_timesteps))
                for k in range(n_agents):
                    print('Reward agent {}: {}'.format(k, np.mean(R_series [-10:], 0) [k]))
                print('Utilities ', str(np.mean(U_series [-10:])))
                print('Equality ', str(np.mean(E_series [-10:])))
                print('Violence ', str(np.mean(V_series [-10:])))
                print('Sustainability ', str(np.mean(S_series [-10:])))
        saver.save(sess, models_dir + '/model-final.ckpt')
        print("Interaction ended")
