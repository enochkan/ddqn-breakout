# coding=utf-8

# {
#     EECE5870 project 1:
#     Team: Chi Nok Enoch Kan, Zachary Nordgren, Jin Xiang
# }

# {
#     Please install keras==2.1.2 or above
# }

import cv2
import os.path
import gym
import random
import numpy as np
from keras import layers
from keras import backend as K
from keras.models import Model
from keras.models import load_model
from keras.models import clone_model
from keras.optimizers import Adam, RMSprop
from keras.utils import multi_gpu_model

from skimage.color import rgb2gray
from collections import deque
from datetime import datetime

#custom huber loss function 
def clipping(y, q_value):
    quadratic_part = K.clip(K.abs(y - q_value), 0.0, 1.0)
    linear_part = K.abs(y - q_value) - quadratic_part
    loss = K.mean(0.5 * K.square(quadratic_part) + linear_part)
    return loss

class AtariBot:
    def __init__(self, game='Breakout-v0', max_replay_memory=500000, initial_e=1, final_e=.1, e_step=50000,
                 epochs=90000, learn_rate=0.00025, no_op_max=20, steps_observe_before_train=5000, refresh=10000,
                 batch_size=32, imsize=84, action_size=3, mod_name='ddqn', optimizer='rmsprop', trained_model='breakout_model_20191022160932.h5'):

        #which game to play
        self.game = game

        #which model to use
        self.mod_name = mod_name

        #initialize replay memory D to capacity N
        #size of memory to be stored, will purge once max_replay_memory has been reached
        self.memory = deque(maxlen=max_replay_memory)

        #exploration factor epsilon
        self.epsilon = initial_e
        self.final_e = final_e
        self.e_step = e_step

        #rate at which epsilon should drop
        self.e_decay = (self.epsilon - self.final_e) / self.e_step

        #num epochs, same as episodes but I don't like the term episodes ¯\_(ツ)_/¯
        self.epochs = epochs

        #learn rate for optimizor
        self.lr = learn_rate

        #max frames that the agent will do nothing
        self.no_op_max = no_op_max

        self.steps_observe_before_train = steps_observe_before_train
        self.refresh = refresh
        self.bs = batch_size
        self.imsize = imsize
        self.frame_size = (self.imsize, self.imsize, 4)
        self.action_size = action_size
        self.train_dir = 'atari_bot_models'
        self.optimizer = optimizer

        #discount factor gamma used in q learning update
        self.gamma = 0.99

        #testing
        self.test_epsilon = 0.001
        self.trained_model = trained_model
        self.test_epochs = 80
    
    def build_model(self):
        if self.mod_name == 'ddqn':
            #dueling DQN implementation
            frames_input, actions_input = layers.Input(self.frame_size, name='frames'), layers.Input((self.action_size,), name='action')
            #normalization layer
            normalized = layers.Lambda(lambda x: x / 255.0, name='normalization')(frames_input)
            #same conv as dqn
            conv_1 = layers.convolutional.Conv2D(16, (8, 8), strides=(4, 4), activation='relu')(normalized)
            conv_2 = layers.convolutional.Conv2D(32, (4, 4), strides=(2, 2), activation='relu')(conv_1)
            flattened = layers.core.Flatten()(conv_2)
            #advantage and value layers
            fullycon1 = layers.Dense(256)(flattened)
            advantage = layers.Dense(self.action_size)(fullycon1)
            fullycon2 = layers.Dense(256)(flattened)
            value = layers.Dense(1)(fullycon2)
            # print(value)
            output = layers.merge([advantage, value], mode = lambda x: x[0]-K.mean(x[0])+x[1], output_shape = (self.action_size,))
            #filter outputs for q values
            model = Model(inputs=[frames_input, actions_input], outputs=layers.Multiply(name='Q')([output, actions_input]))
            model.summary()

            #allow the selection of adam as optimizer
            if self.optimizer == 'adam':
                optimizer = Adam(lr=self.lr, beta_1=0.9, beta_2=0.999)
            elif self.optimizer == 'rmsprop':
                optimizer = RMSprop(lr=self.lr, rho=0.95, epsilon=0.01)

            #compiling model with customized huber loss
            model.compile(optimizer, loss=clipping)
            return model

    def downsize_input(self, input):
        return np.uint8(cv2.resize(rgb2gray(input), (self.imsize, self.imsize), interpolation = cv2.INTER_CUBIC) * 255)

    def action(self, history, epsilon, step, model):
        #random selection of action if not train stage
        if np.random.rand() <= epsilon:
            return random.randrange(self.action_size)
        elif step <= self.steps_observe_before_train:
            return random.randrange(self.action_size)
        else:
            return np.argmax(model.predict([history, np.ones(self.action_size).reshape(1, self.action_size)])[0])

    def decay(self, steps):
        if self.epsilon > self.final_e and steps > self.steps_observe_before_train:
            # decay epsilon at a fixed rate
            self.epsilon -= self.e_decay

    def train_minibatch(self, memory, model):
        #initialize containers
        action, reward, dead, history, next_history, target = [], [], [], np.zeros((self.bs, self.frame_size[0], self.frame_size[1], self.frame_size[2])), \
                  np.zeros((self.bs, self.frame_size[0], self.frame_size[1], self.frame_size[2])), np.zeros((self.bs,))
        #sample minibatches from memory
        mb = random.sample(memory, self.bs)

        #append values
        for idx, val in enumerate(mb):
            history[idx] = val[0]
            next_history[idx] = val[3]
        for _, val in enumerate(mb):
            action.append(val[1])
            reward.append(val[2])
            dead.append(val[4])

        q_vals = model.predict([next_history, np.ones((self.bs, self.action_size))])

        for i in range(self.bs):
            #rj = gamma*amax(Q(j+1))
            target[i] = -1 if dead[i] else target[i] = reward[i] + np.amax(q_vals[i])*self.gamma
                
        # print([np.array(action).reshape(-1)])
        hist = model.fit(
            [history, np.eye(self.action_size)[np.array(action).reshape(-1)]], np.eye(self.action_size)[np.array(action).reshape(-1)] * target[:, None], epochs=1,
            batch_size=self.bs, verbose=0)

        return hist.history['loss'][0]

    def set_phase(self, total_steps):
        return "observation" if total_steps <= self.steps_observe_before_train else "annealing" if total_steps <= self.steps_observe_before_train + self.e_step else "training"

    def calculate_loss(self, loss, memory, model):
        return loss + self.train_minibatch(memory, model)

    def no_op_fire(self, env):
        # observe for no_op_max steps before running
        for _ in range(self.no_op_max):
            observe, _, _, _ = env.step(1)

    def save_model(self, epoch, total_steps):
        # save model every 200 epochs
        if epoch % 200 == 0:
            model_path = os.path.join(self.train_dir, "breakout_model_{}.h5".format(total_steps))
            self.model.save(model_path)

    def stack_history(self, gs):
        self.test_history = np.reshape([np.stack((gs, gs, gs, gs), axis=2)], (1, self.imsize, self.imsize, 4))

    ########################################################
    ###                      train                       ###
    ########################################################
    def train(self):
        #initialize variables
        env = gym.make(self.game)
        total_steps = 0

        #construct model
        self.model = self.build_model()
        model_target = clone_model(self.model)
        model_target.set_weights(self.model.get_weights())

        #constructing save directory if not exist
        if not os.path.exists('atari_bot_models'):
            os.makedirs('atari_bot_models')

        #start training
        for epoch in range(0, self.epochs):

            #initializing variables
            done, dead, loss, step, score, start_life = False, False, 0.0, 0, 0, 4
            #reset environment
            observe = env.reset()

            #implementing no_op_max
            self.no_op_fire(env)
            #convert to grayscale
            gs = self.downsize_input(observe)
            #stacking initial 4 frames as input
            self.stack_history(gs)

            #play game
            while not done:
                action = self.action(self.history, self.epsilon, total_steps, model_target)
                self.decay(total_steps)
                #+1 to map to actual action space [1,2,3] -> fire,right,left
                observe, reward, done, info = env.step(action+1)

                next_state = self.downsize_input(observe)

                next_state = np.reshape([next_state], (1, self.imsize, self.imsize, 1))
                self.next_history = np.append(next_state, self.history[:, :, :, :3], axis=3)

                start_life = info['ale.lives'] if start_life > info['ale.lives'] else pass
                dead = True if start_life > info['ale.lives'] else pass

                #et = (st, at, rt+1, st+1)
                self.memory.append((self.history, action, reward, self.next_history, dead)) 

                if total_steps > self.steps_observe_before_train:
                    if total_steps % self.refresh == 0:
                        model_target.set_weights(self.model.get_weights())
                    loss = self.calculate_loss(loss, self.memory, self.model)

                score += reward
                self.history = self.next_history if not dead else dead = False
                #adding total steps and current step
                total_steps += 1; step += 1
                if done:
                    self.save_model(epoch, total_steps)
                    print('phase: {}, epoch: {}, score: {}, avg loss: {}'
                        .format(self.set_phase(total_steps), epoch, score, loss / float(step)))

    ########################################################
    ###                      test                       ###
    ########################################################
    def test(self):
        # initialize variables
        env = gym.make(self.game)
        high_score, total_steps = 0, self.steps_observe_before_train+1

        #load model for testing
        self.test_model = load_model(self.trained_model, custom_objects={'huber_loss': clipping})

        for epoch in range(0,self.test_epochs):

            score, start_life, done, dead = 0, 4, False, False

            observe = env.reset()

            self.no_op_fire(env)
            gs = self.downsize_input(observe)
            #stack 4 frames as input
            self.stack_history(gs)

            #play game
            while not done:
                env.render()
                # get action for the current history and go one step in environment
                action = self.action(self.test_history, self.test_epsilon, total_steps, self.test_model)
                observe, reward, done, info = env.step(action+1)

                self.test_next_history = np.append(np.reshape([self.downsize_input(observe)], (1, self.imsize, self.imsize, 1)), self.test_history[:, :, :, :3], axis=3)

                start_life = info['ale.lives'] if start_life > info['ale.lives'] else pass
                dead = True if start_life > info['ale.lives'] else pass

                #add up reward points as total score
                score += reward
                self.test_history = self.test_next_history if not dead else dead = False

                total_steps += 1
                if done:
                    high_score = score if score > high_score else pass
                    print('epoch: {}, current score: {}, high score: {}'.format(epoch+1, score, high_score))


def run_main():
    bot = AtariBot()
    bot.train()
    # bot.test()

run_main()