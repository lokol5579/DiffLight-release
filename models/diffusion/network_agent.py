import numpy as np
import torch
import random
import os
import copy
from models.diffusion.agent import Agent
import traceback


class NetworkAgent(Agent):
    def __init__(self, dic_agent_conf, dic_traffic_env_conf, dic_path, cnt_round, intersection_id="0"):
        super(NetworkAgent, self).__init__(
            dic_agent_conf, dic_traffic_env_conf, dic_path, intersection_id=intersection_id)

        # ===== check num actions == num phases ============
        self.num_actions = len(dic_traffic_env_conf["PHASE"])
        self.num_phases = len(dic_traffic_env_conf["PHASE"])
        self.lane_num = 12
        self.observation_dim = 2

        self.phase_lane_index = []
        for (k, v) in self.dic_traffic_env_conf["PHASE"].items():
            self.phase_lane_index.append(np.nonzero(v)[0])
        
        # self.num_lanes = np.sum(np.array(list(self.dic_traffic_env_conf["LANE_NUM"].values())))

        self.memory = self.build_memory()
        self.cnt_round = cnt_round

        self.Xs, self.Y = None, None

        if cnt_round == 0:
            if os.listdir(self.dic_path["PATH_TO_MODEL"]):
                self.load_network("round_0_inter_{0}".format(intersection_id))
            else:
                self.model = self.build_network()
        else:
            try:
                self.load_network("round_{0}_inter_{1}".format(cnt_round-1, self.intersection_id))
            except Exception:
                print('traceback.format_exc():\n%s' % traceback.format_exc())

        # decay the epsilon
        decayed_epsilon = self.dic_agent_conf["EPSILON"] * pow(self.dic_agent_conf["EPSILON_DECAY"], cnt_round)
        self.dic_agent_conf["EPSILON"] = max(decayed_epsilon, self.dic_agent_conf["MIN_EPSILON"])

    def load_network(self, file_name, file_path=None):
        if file_path is None:
            file_path = self.dic_path["PATH_TO_MODEL"]

        # model
        self.model = self.build_network()
        state_dict = torch.load(os.path.join(file_path, "%s.pth" % (file_name)))
        self.model.load_state_dict(state_dict)

        print("succeed in loading model %s" % file_name)

    def load_network_transfer(self, file_name, file_path=None):
        if file_path is None:
            file_path = self.dic_path["PATH_TO_MODEL"]

        # model
        self.model = self.build_network()
        state_dict = torch.load(os.path.join(file_path, "%s.pth" % (file_name)))
        self.model.load_state_dict(state_dict)

        print("succeed in loading model %s" % file_name)

    def save_network(self, file_name):
        model_path = os.path.join(self.dic_path["PATH_TO_MODEL"], "%s.pth" % (file_name))
        torch.save(self.model.state_dict(), model_path)

    def build_network(self):
        raise NotImplementedError

    def build_memory(self):
        return []

    def build_network_from_copy(self, network_copy):
        """Initialize a Q network from a copy"""
        network = copy.deepcopy(network_copy)
        return network

    def prepare_Xs_Y(self, memory):

        ind_end = len(memory)
        print("memory size before forget: {0}".format(ind_end))
        # use all the samples to pretrain, i.e., without forgetting
        ind_sta = max(0, ind_end - self.dic_agent_conf["MAX_MEMORY_LEN"])
        memory_after_forget = memory[ind_sta: ind_end]
        print("memory size after forget:", len(memory_after_forget))

        # sample the memory
        sample_size = min(self.dic_agent_conf["SAMPLE_SIZE"], len(memory_after_forget))
        sample_slice = random.sample(memory_after_forget, sample_size)
        print("memory samples number:", sample_size)

        dic_state_feature_arrays = {}
        for feature_name in self.dic_traffic_env_conf["LIST_STATE_FEATURE"]:
            dic_state_feature_arrays[feature_name] = []
        Y = []

        for i in range(len(sample_slice)):
            state, action, next_state, reward, instant_reward, _, _ = sample_slice[i]
            for feature_name in self.dic_traffic_env_conf["LIST_STATE_FEATURE"]:
                dic_state_feature_arrays[feature_name].append(state[feature_name])
            _state = []
            _next_state = []
            for feature_name in self.dic_traffic_env_conf["LIST_STATE_FEATURE"]:
                _state.append(np.array([state[feature_name]]))
                _next_state.append(np.array([next_state[feature_name]]))

            target = self.model.predict(_state)

            next_state_qvalues = self.model_bar.predict(_next_state)

            if self.dic_agent_conf["LOSS_FUNCTION"] == "mean_squared_error":
                final_target = np.copy(target[0])
                final_target[action] = reward / self.dic_agent_conf["NORMAL_FACTOR"] + self.dic_agent_conf["GAMMA"] * \
                                       np.max(next_state_qvalues[0])
            elif self.dic_agent_conf["LOSS_FUNCTION"] == "categorical_crossentropy":
                raise NotImplementedError

            Y.append(final_target)

        self.Xs = [np.array(dic_state_feature_arrays[feature_name]) for feature_name in
                   self.dic_traffic_env_conf["LIST_STATE_FEATURE"]]
        self.Y = np.array(Y)

    def convert_state_to_input(self, s):
        if self.dic_traffic_env_conf["BINARY_PHASE_EXPANSION"]:
            inputs = []
            for feature in self.dic_traffic_env_conf["LIST_STATE_FEATURE"]:
                if "cur_phase" in feature:
                    inputs.append(np.array([self.dic_traffic_env_conf['PHASE'][s[feature][0]]]))
                else:
                    inputs.append(np.array([s[feature]]))
            return inputs
        else:
            return [np.array([s[feature]]) for feature in self.dic_traffic_env_conf["LIST_STATE_FEATURE"]]

    def choose_action(self, count, state):
        """choose the best action for current state """
        state_input = self.convert_state_to_input(state)

        self.model.eval()
        q_values = self.model(state_input)
        if random.random() <= self.dic_agent_conf["EPSILON"]:  # continue explore new Random Action
            action = random.randrange(len(q_values[0]))
        else:  # exploitation
            action = np.argmax(q_values[0])
        return action

    def train_epoch(self, batch_size, loss_fun, optim):
        raise NotImplementedError
    
    def val_epoch(self, batch_size, loss_fun):
        raise NotImplementedError

    def train_network(self):
        epochs = self.dic_agent_conf["EPOCHS"]
        train_ratio = 0.7

        loss_fun = torch.nn.MSELoss()
        optim = torch.optim.Adam(params=self.model.parameters(), lr=self.dic_agent_conf["LEARNING_RATE"], eps=1.0e-8,
                                        weight_decay=0, amsgrad=False)

        best_loss = float('inf')
        not_improved_count = 0
        best_model_state_dict = None

        train_loss_list = []
        val_loss_list = []
        
        for epoch in range(epochs):
            # get sample data
            _, _, _, _, _, batches = self.memory.sample()

            train_batches = batches[:int(len(batches) * train_ratio)]
            val_batches = batches[int(len(batches) * train_ratio):]

            train_epoch_loss = self.train_epoch(train_batches, loss_fun, optim)
            val_epoch_loss = self.val_epoch(val_batches, loss_fun)
            
            train_loss_list.append(train_epoch_loss)
            val_loss_list.append(val_epoch_loss)

            print(f"epoch:{epoch}, train_epoch_loss:{train_epoch_loss} ,val_epoch_loss:{val_epoch_loss}")
            
            # val loss
            if val_epoch_loss < best_loss:
                best_loss = val_epoch_loss
                best_model_state_dict = copy.deepcopy(self.model.state_dict())
                not_improved_count = 0
            else:
                not_improved_count += 1
            
            # early stop
            if not_improved_count == self.dic_agent_conf["PATIENCE"]:
                self.model.load_state_dict(best_model_state_dict)
                print('Early stopping triggered at epoch', epoch + 1)
                break

        self.memory.clear()


                