import time
import numpy as np
import random
import tensorflow as tf
import tensorflow.contrib.slim as slim
import os
from threading import Thread
from tensorboardX import SummaryWriter
from envs.Harvest import HarvestGame

# logs_dir = "./logs_0"
# logs_dir = "./logs/logs_dqna2cteam"
logs_dir = "./logs/logs_dudqna2cteam"

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
models_dir = "./models_duqn"  # The path to save our model to.
h_size = 32  # The size of the final hidden layer before splitting it into Advantage and Value streams.
tau = 0.001  # Rate to update target network toward primary network

n_agents = 4  # Number of agents
agent_lat_view = 3
agent_front_view = 10
input_h = agent_lat_view * 2 + 1
input_w = agent_front_view

env = HarvestGame(n_agents=n_agents, agent_view_h=agent_front_view, agent_view_w=agent_lat_view)
action_dim = env.action_dim

# a2c
# y = 0.99 # Discount rate.
# update_frequency = 10 # How many episodes before updating model.
learning_rate = 1e-4 # Agent learning rate.
hidden_units = 32 # Number of units in hidden layer.
# model_dir_a2c = "./models_a2c" # The path to save our model to.
# summary_dir_a2c = "./summaries_a2c" # The path to save our model to.


class DQN:
    def __init__(self, scope, rnn_cell):
        with tf.variable_scope(scope):
            # The network receives a frame from the game, flattened into an array.
            self.scalarInput = tf.placeholder(shape=[None, input_h * input_w * 3], dtype=tf.float32)
            self.fc1 = slim.fully_connected(self.scalarInput, 256)
            self.trainLength = tf.placeholder(dtype=tf.int32)
            self.batch_size = tf.placeholder(dtype=tf.int32, shape=[])
            self.rnn_in = tf.reshape(self.fc1, [self.batch_size, self.trainLength, 256])
            self.state_in = rnn_cell.zero_state(self.batch_size, tf.float32)
            # dynamic_rnn returns [output, final_output_state]
            self.rnn, self.rnn_state = tf.nn.dynamic_rnn(inputs=self.rnn_in, cell=rnn_cell, dtype=tf.float32,
                                                         initial_state=self.state_in)
            self.rnn = tf.reshape(self.rnn, shape=[-1, 32])
            # We take the output from the final convolutional layer and split it into separate advantage and value streams.
            self.streamA, self.streamV = tf.split(self.rnn, 2, 1)
            xavier_init = tf.contrib.layers.xavier_initializer()
            self.AW = tf.Variable(xavier_init([h_size // 2, action_dim]))
            self.VW = tf.Variable(xavier_init([h_size // 2, 1]))
            self.Advantage = tf.matmul(self.streamA, self.AW)
            self.Value = tf.matmul(self.streamV, self.VW)

        # #Then combine them together to get our final Q-value function
        self.Qout = self.Value + tf.subtract(self.Advantage, tf.reduce_mean(self.Advantage, axis=1, keepdims=True))
        self.predict = tf.argmax(self.Qout, 1)

        # Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
        self.targetQ = tf.placeholder(shape=[None], dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
        self.actions_onehot = tf.one_hot(self.actions, action_dim, dtype=tf.float32)

        self.Q = tf.reduce_sum(tf.multiply(self.Qout, self.actions_onehot), axis=1)

        self.td_error = tf.square(self.targetQ - self.Q)
        self.loss = tf.reduce_mean(self.td_error)
        self.trainer = tf.train.AdamOptimizer(learning_rate)
        self.updateModel = self.trainer.minimize(self.loss)

# experience replay: to store episodes and sample then randomly to train the network
class experience_buffer:
    def __init__(self, buffer_size=500):
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self, experience):
        # if butter overflows, deleting the older episode
        if len(self.buffer) + 1 >= self.buffer_size:
            self.buffer[0:(1 + len(self.buffer)) - self.buffer_size] = []
        self.buffer.append(experience)

    def sample(self, sample_size, seq_length):
        sampled_episodes = random.sample(self.buffer, sample_size)
        sampledTraces = []
        for episode in sampled_episodes:
            point = np.random.randint(0, len(episode) + 1 - seq_length)
            sampledTraces.append(episode[point:point + seq_length])
        sampledTraces = np.array(sampledTraces)
        return np.reshape(sampledTraces, [sample_size * seq_length, 5])


class A2CAgent:
    def __init__(self, scope, hidden_size):
        with tf.variable_scope(scope):
            # These lines established the feed-forward part of the network. The agent takes a state and produces an action.
            # self.scalarInput = tf.placeholder(shape=[None, input_h * input_w * 3], dtype=tf.float32)
            self.batch_size = tf.placeholder(dtype=tf.int32, shape=[])  # shape=none
            self.state_in = tf.placeholder(shape=[None,input_h * input_w * 3], dtype=tf.float32)
            # self.state_in = tf.placeholder(shape=[None, s_size], dtype=tf.float32)
            # self.fc1 = slim.fully_connected(self.scalarInput, 256)
            hidden = slim.fully_connected(self.state_in, hidden_size, biases_initializer=None, activation_fn=tf.nn.elu)
            self.out = slim.fully_connected(hidden, action_dim, activation_fn=tf.nn.softmax, biases_initializer=None)
            self.value = slim.fully_connected(hidden, 1, activation_fn=None, biases_initializer=None)
            self.output = self.out * (0.9) + 0.1 / action_dim
            self.chosen_action = tf.argmax(self.output, 1)

            # The next six lines establish the training proceedure. We feed the reward and chosen action into the network
            # to compute the loss, and use it to update the network.
            self.reward_holder = tf.placeholder(shape=[None], dtype=tf.float32)
            self.action_holder = tf.placeholder(shape=[None], dtype=tf.int32)
            self.actions = slim.one_hot_encoding(self.action_holder, action_dim)

            self.responsible_outputs = tf.reduce_sum(self.output * self.actions, axis=1)
            self.advantage = self.reward_holder - tf.stop_gradient(tf.reduce_sum(self.value, axis=1))

            self.policy_loss = -tf.reduce_mean(tf.log(self.responsible_outputs) * self.advantage)
            self.value_loss = tf.reduce_mean(
                tf.squared_difference(self.reward_holder, tf.reduce_sum(self.value, axis=1)))
            self.loss = self.policy_loss + self.value_loss

            tvars = tf.trainable_variables(scope)
            self.gradient_holders = []
            for idx, var in enumerate(tvars):
                placeholder = tf.placeholder(tf.float32, name=str(idx) + '_holder')
                self.gradient_holders.append(placeholder)

            self.gradients = tf.gradients(self.loss, tvars)
            optimizer = tf.train.AdamOptimizer(learning_rate)
            self.update_batch = optimizer.apply_gradients(zip(self.gradient_holders, tvars))


def discount_rewards(r, gamma):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        running_add = running_add * gamma + r [t]
        discounted_r [t] = running_add
    return discounted_r


def processState(x):
    return np.reshape(x, [n_agents, input_h * input_w * 3])


def updateTargetGraph(from_scope, to_scope):
    main_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope) #from_scope="agent_" + str(i) + "/main"
    target_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope) #to_scope="agent_" + str(i) + "/target"
    op_holder = []
    for idx, var in enumerate(main_params):
        op_holder.append(target_params[idx].assign(
            (var.value() * tau) + (target_params[idx].value() * (1 - tau))))
    return op_holder


def updateTarget(op_holder):
    for op in op_holder:
        sess.run(op)


def act(policy, observation, actions_batch, states_batch, idx, ac_type):
    # choose the action by greedily from the Q-network
    if (np.random.rand(1) < epsilon or total_steps < pre_train_timesteps) and train:
        actions_batch[idx] = np.random.randint(0, action_dim)
        ac_type[idx] = False
    else:

        a, new_cell_state = sess.run([policy.predict, policy.rnn_state], feed_dict={policy.scalarInput: [observation / 255.0],
                                  policy.trainLength: 1, policy.state_in: states_batch[idx], policy.batch_size: 1})
        actions_batch[idx] = a[0]
        ac_type[idx] = True
        states_batch[idx] = new_cell_state


def learn(main_policy, target_policy, target_ops, my_buffer, idx):
    updateTarget(target_ops)
    # Reset the recurrent layer's hidden state
    state_train = (np.zeros([batch_size, h_size]), np.zeros([batch_size, h_size]))
    trainBatch = my_buffer.sample(batch_size, trace_length)  # Get a random batch of experiences.
    # Below we perform the Double-DQN update to the target Q-values
    Q1 = sess.run(main_policy.predict,
                  feed_dict={main_policy.scalarInput: np.vstack(trainBatch[:, 3] / 255.0),
                             main_policy.trainLength: trace_length,
                             main_policy.state_in: state_train,
                             main_policy.batch_size: batch_size})
    Q2 = sess.run(target_policy.Qout,
                  feed_dict={target_policy.scalarInput: np.vstack(trainBatch[:, 3] / 255.0),
                             target_policy.trainLength: trace_length,
                             target_policy.state_in: state_train,
                             target_policy.batch_size: batch_size})
    end_multiplier = -(trainBatch[:, 4] - 1)
    doubleQ = Q2[range(batch_size * trace_length), Q1]
    targetQ = trainBatch[:, 2] + (gamma * doubleQ * end_multiplier)
    # Update the network with our target values.
    p_loss, _ = sess.run([main_policy.loss, main_policy.updateModel],
                         feed_dict={main_policy.scalarInput: np.vstack(trainBatch[:, 0] / 255.0),
                                    main_policy.targetQ: targetQ,
                                    main_policy.actions: trainBatch[:, 1],
                                    main_policy.trainLength: trace_length,
                                    main_policy.state_in: state_train,
                                    main_policy.batch_size: batch_size})
    policy_losses[idx] += p_loss


# a2c methods
def init_gradbuffer(gradBuffer):
    for ix, grad in enumerate(gradBuffer):
        gradBuffer [ix] = grad * 0


def update_gradBuffer(gradBuffer, grads):
    for idx, grad in enumerate(grads):
        gradBuffer[idx] += grad


def act_a2c(myAgent, actions_batch, observation, idx, ac_type):
    # Probabilistically pick an action given our network outputs.
    a_dist = sess.run(myAgent.output,
                      feed_dict={myAgent.state_in: [observation / 255.0], myAgent.batch_size: 1})
    a = np.random.choice(a_dist[0], p=a_dist[0])
    action = np.argmax(a_dist == a)
    actions_batch[idx] = action
    ac_type[idx] = True
#
#
# def learn_a2c(myAgent, gradBuffer):
#     bufferArray = np.array(gradBuffer)
#     # feed_dict = dict(zip(myAgent.gradient_holders, bufferArray))
#     _ = sess.run(myAgent.update_batch, feed_dict={myAgent.gradient_holders: bufferArray})



if __name__ == '__main__':
    tf.reset_default_graph()
    agents = {}
    with tf.device("/cpu:0"):
        # how to define the agents
        for i in range(n_agents):
            if i%2 == 0:
                cell = tf.contrib.rnn.BasicLSTMCell(num_units=32, state_is_tuple=True)
                cellT = tf.contrib.rnn.BasicLSTMCell(num_units=32, state_is_tuple=True)
                agents[str(i)] = {'mainNet': DQN("agent_" + str(i) + "/main", cell),
                                  'targetNet': DQN("agent_" + str(i) + "/target", cellT),
                                  'my_buffer': experience_buffer(),
                                  'episode_buffer': [],
                                  'targetOps': []}
            else:
                agents [str(i)] = {'mainNet': A2CAgent("agent_" + str(i) + "/main", hidden_size=h_size),
                                   'grad_buffer': [],
                                   'my_buffer': experience_buffer(),
                                   'episode_buffer': []}

        init = tf.global_variables_initializer()
        # for i in range(n_agents):
        #     if i==0:
        #         # main_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope="agent_" + str(i) + "/main")
        #         # target_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope="agent_" + str(i) + "/target")
        #         targetOps = updateTargetGraph("agent_" + str(i)  + "/main", "agent_" + str(i) + "/target")
        #         agents[str(i)]['targetOps'] = targetOps
        for i in range(n_agents):
            if i % 2 == 0:
                targetOps = updateTargetGraph("agent_" + str(i)  + "/main", "agent_" + str(i) + "/target")
                agents[str(i)]['targetOps'] = targetOps


    saver = tf.train.Saver()
    writer = SummaryWriter(log_dir=logs_dir)

    # Set the rate of random action decrease.
    epsilon = init_epsilon
    stepDrop = (init_epsilon - final_epsilon) / annealing_steps
    # Losses tracking
    # total_reward = []
    # total_length = []
    value_losses = np.zeros((n_agents,))
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
            if i%2 == 0:
                updateTarget(agents[str(i)]['targetOps'])
            else:
                gradBuffer = sess.run(tf.trainable_variables("agent_" + str(i) + "/main"))
                for ix, grad in enumerate(gradBuffer):
                    gradBuffer [ix] = grad * 0
                agents[str(i)]['grad_buffer'] =gradBuffer

        if load_model:
            print('Loading Model...')
            ckpt = tf.train.get_checkpoint_state(models_dir)
            saver.restore(sess, ckpt.model_checkpoint_path)
        while total_steps < max_steps:
            for k in range(n_agents):
                agents[str(k)]['episode_buffer'] = []
            # Reset environment and get first new observation
            s = env.reset()
            s = processState(s)
            # d = False
            r_t = np.zeros((1, n_agents))
            stock = np.zeros((1,))
            shots = np.zeros((1,))
            T = 0
            # Reset the recurrent layer's hidden state
            for k in range(n_agents):
                cell_states[k] = (np.zeros([1, h_size]), np.zeros([1, h_size]))
            # The Q-Network
            t0 = time.time()
            while T < max_epLength:
                # running_rewards = np.zeros((n_agents,))
                T += 1
                # act
                threads = []
                for k in range(n_agents):
                    if k%2 == 0:
                        thread = Thread(target=act, args=[agents[str(k)]['mainNet'], s[k], acs, cell_states, k, ac_class])
                        thread.start()
                        threads.append(thread)
                    else:
                        thread = Thread(target=act_a2c, args=[agents[str(k)]['mainNet'], acs, s[k], k, ac_class])
                        thread.start()
                        threads.append(thread)
                for thread in threads:
                    thread.join()
                s1, r, d, stock_t = env.step(acs)

                # save some screen shot
                if num_episodes % 500 == 0 and (num_episodes > 0 or not train):
                    if not train:
                        env.render(save_snapshots=True)
                    else:
                        env.render()

                s1 = processState(s1)
                total_steps += 1


                for k in range(n_agents):
                    if s[k] != timeout_obs:
                        # ep_history.append([state, action, reward, state_1])
                        # if k==0:
                        agents[str(k)]['episode_buffer'].append(np.reshape(np.array(
                                [s[k], acs[k], r[k], s1[k], d]), [1, 5]))  # Save the experience to our episode buffer.
                        # else:
                        #     agents [str(k)] ['episode_buffer'].append(np.reshape(np.array(
                        #         [s[k], acs [k], r [k], s1 [k], d]), [1, 5]))
                if total_steps > pre_train_timesteps and train:
                    # learn a2c
                    if d:
                        for i in range(n_agents):
                            if i % 2 != 0:
                                myAgent = agents[str(1)]['mainNet']
                                ep_history = np.array(agents[str(1)]['episode_buffer'])
                                length = np.shape(ep_history)[0]
                        # ep_history = agents [str(1)] ['episode_buffer']
                                ep_history[:,:,2] = discount_rewards(ep_history [:,:,2].reshape(length), gamma).reshape(length,1)
                                feed_dict = {myAgent.reward_holder: ep_history[:,:,2].reshape(length),
                                              myAgent.action_holder: ep_history[:,:,1].reshape(length),
                                              myAgent.state_in: np.vstack(ep_history[:,:,0].tolist()),
                                              myAgent.batch_size: len(ep_history)}
                                v_loss, p_loss, grads = sess.run([myAgent.value_loss,
                                                                  myAgent.policy_loss,
                                                                  myAgent.gradients],
                                                                  feed_dict=feed_dict)
                                value_losses[1] += v_loss
                                policy_losses[1]+= p_loss
                                update_gradBuffer(agents[str(1)]['grad_buffer'], grads)

                    if epsilon > final_epsilon:
                        epsilon -= stepDrop
                    if total_steps % update_freq == 0:
                        threads = []
                        for k in range(n_agents):
                            if k %2 == 0:
                                thread = Thread(target=learn,
                                                args=[agents [str(k)] ['mainNet'], agents [str(k)] ['targetNet'],
                                                      agents [str(k)] ['targetOps'],
                                                      agents [str(k)] ['my_buffer'], k])
                                thread.start()
                                threads.append(thread)
                            else:
                                myAgent = agents[str(k)]['mainNet']
                                feed_dict = dictionary = dict(zip(myAgent.gradient_holders, agents [str(k)] ['grad_buffer']))
                                _ = sess.run(myAgent.update_batch, feed_dict=feed_dict)
                                init_gradbuffer(agents [str(k)] ['grad_buffer'])
                                # thread = Thread(target=learn_a2c, args=[agents [str(k)] ['mainNet'],agents [str(k)] ['grad_buffer']])

                        for thread in threads:
                            thread.join()

                # four agent shared rewards
                temp_r2,temp_r = 0,0
                for idx,value in enumerate(r):
                    if idx%2 == 0:
                        temp_r = np.add(temp_r, value)
                    else:
                        temp_r2 = np.add(temp_r2, value)
                r_t = np.append(r_t, [np.array([temp_r, temp_r2, temp_r, temp_r2])], 0)
                # two agent rewards
                # r_t = np.append(r_t, [np.array(r)], 0)
                stock = np.append(stock, [np.array(stock_t)], 0)
                shots = np.append(shots, [np.sum((np.array(acs) == 7))], 0)
                s = s1
                if d:
                    break
            t1 = time.time()
            num_episodes += 1
            # total_reward.append(running_rewards)
            # total_length.append(T)

            for k in range(n_agents):
                if k%2==0:
                    bufferArray = np.array(agents[str(k)]['episode_buffer'])
                    episodeBuffer = list(zip(bufferArray))
                    agents[str(k)]['my_buffer'].add(episodeBuffer)


            R = np.sum(r_t, 0)
            U = np.sum(R)/2
            # U = np.sum(R)
            if U > 0:
                E = 1 - 0.5 * np.abs(np.subtract.outer(R, R)).mean() / U
            else:
                E = 1
            R_series.append(R)
            S = np.sum(stock[1:])
            P = np.mean(shots[1:])
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
                writer.add_scalar('Social/Utilities', np.mean(U_series[-10:]),
                                  total_steps - pre_train_timesteps)
                writer.add_scalar('Social/Equality', np.mean(E_series[-10:]),
                                  total_steps - pre_train_timesteps)
                writer.add_scalar('Social/Violence', np.mean(V_series[-10:]),
                                  total_steps - pre_train_timesteps)
                writer.add_scalar('Social/Sustainability', np.mean(S_series[-10:]),
                                  total_steps - pre_train_timesteps)
                individual_rewards = {}
                losses = {}
                losses_v = {}

                for k in range(n_agents):
                    individual_rewards['Agent {}'.format(k)] = np.mean(R_series[-10:], 0)[k]
                    losses['Agent {}'.format(k)] = policy_losses[k] / (
                            total_steps - pre_train_timesteps)
                    losses_v['Agent {}'.format(k)] = value_losses[k] / (total_steps-pre_train_timesteps)
                    # EPlength['Agent {}'.format(k)] = np.mean(total_length [-10:],0)
                writer.add_scalars('Individual/Rewards', individual_rewards,
                                   total_steps - pre_train_timesteps)
                writer.add_scalars('Losses/Policy Loss', losses,
                                   total_steps - pre_train_timesteps)
                writer.add_scalars('Losses/Value Loss', losses_v,
                                   total_steps - pre_train_timesteps)
                # writer.add_scalars('Losses/EPlength', EPlength,
                #                    total_steps - pre_train_timesteps)
                print('------------------STATISTICS------------------')
                print('Steps so far: {}'.format(total_steps - pre_train_timesteps))
                print('Epsilon: {}'.format(epsilon))
                print('Episode time length: {}'.format(t1 - t0))
                print('Mean Policy Loss: ',
                      np.mean(policy_losses) / (total_steps - pre_train_timesteps))
                for k in range(n_agents):
                    print('Reward agent {}: {}'.format(k, np.mean(R_series[-10:], 0)[k]))
                print('Utilities ', str(np.mean(U_series[-10:])))
                print('Equality ', str(np.mean(E_series[-10:])))
                print('Violence ', str(np.mean(V_series[-10:])))
                print('Sustainability ', str(np.mean(S_series[-10:])))
        saver.save(sess, models_dir + '/model-final.ckpt')
        print("Interaction ended")
