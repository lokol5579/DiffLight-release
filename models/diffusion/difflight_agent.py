import numpy as np
import os
import copy
import heapq
import matplotlib.pyplot as plt

import torch

from models.diffusion.network_agent import NetworkAgent
from models.diffusion.replay_buffer import ReplayBuffer
from models.diffusion.diffusion import GaussianInvDynDiffusion
from models.diffusion.spatial_temoral import STFormer
from models.diffusion.temporal import TemporalUnet
from models.diffusion.helpers import max_min_norm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
step_cnt = 0
loss_dict = {}

class EMA():
    '''
        empirical moving average
    '''
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

class DiffLightAgent(NetworkAgent):
    def __init__(self, dic_agent_conf, dic_traffic_env_conf, dic_path, cnt_round, intersection_id="0"):
        super(DiffLightAgent, self).__init__(dic_agent_conf, dic_traffic_env_conf, dic_path, cnt_round, intersection_id=intersection_id)
        
        # ema
        self.gradient_accumulate_every = 2
        self.ema = EMA(beta=0.995)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = 10
        self.log_freq = 10
        self.step_start_ema=2000
        self.lane_neighbors_idx, self.lane_neighbors_idx_mask = self.get_lane_neighbors(self.dic_traffic_env_conf['NUM_INTERSECTIONS'], self.dic_traffic_env_conf["NUM_ROW"], self.dic_traffic_env_conf["NUM_COL"])
        self.topk_nodes = []
        for inter_i in range(self.dic_traffic_env_conf['NUM_INTERSECTIONS']):
            for lane_j in range(12):
                self.topk_nodes.append(self.get_closest_nodes(inter_i * 12 + lane_j, self.lane_neighbors_idx, self.lane_neighbors_idx_mask))
        normal_factor = self.dic_agent_conf["NORMAL_FACTOR"]

        self.batch_idx = 0

        self.bound = {
            "lane_num_vehicle_in": [0, -normal_factor],
            "lane_queue_vehicle_in": [0, -normal_factor],
            "lane_flow_in": [0, -normal_factor],
            "lane_run_in_part": [0, 30],
            "lane_queue_in_part": [0, 30],
            "reward": [normal_factor, 0], 
        }

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def build_network(self):
        horizon = self.dic_traffic_env_conf["HORIZON"]
        cond_step = self.dic_traffic_env_conf["COND_STEP"]
        block_depth = self.dic_agent_conf["BLOCK_DEPTH"]
        drop_neighbor = self.dic_agent_conf["DROP_NEIGHBOR"]
        if "SAMPLE_STEP" in self.dic_agent_conf.keys():
            sample_steps = self.dic_agent_conf["SAMPLE_STEP"]
        else:
            sample_steps = 1

        if "USE_UNET" in self.dic_agent_conf.keys():
            use_unet = self.dic_agent_conf["USE_UNET"]
        else:
            use_unet = False

        if not use_unet:
            model = STFormer(
                horizon=horizon,
                cond_step=cond_step,
                transition_dim=2, 
                hidden_dim=64, 
                block_depth=block_depth, 
                reward_condition=True, 
                condition_dropout=0.25,
                drop_neighbor=drop_neighbor
            ).to(device)
        else:
            model = TemporalUnet(
                horizon=horizon,
                transition_dim=12 * 2,
                cond_dim=cond_step,
                dim=64, 
                dim_mults=(1, 2, 4),
                returns_condition=True, 
                condition_dropout=0.25,
            ).to(device)

        network = GaussianInvDynDiffusion(
            model=model,
            horizon=horizon,
            cond_step=cond_step,
            lane_num=12,
            observation_dim=2,
            action_dim=1,
            eta=0,
            n_timesteps = 100,
            sample_steps=sample_steps,
            loss_type = 'l2',
            clip_denoised = True,
            predict_epsilon = True,
            hidden_dim = 64,
            action_weight = 10,
            loss_discount = 1,
            loss_weights = None,
            returns_condition = True,
            condition_guidance_w = 1.2,
            ar_inv = False,
            train_only_inv = False,
            use_unet = use_unet
        ).to(device)

        # print('================= Model Parameter =================')
        # total_num = sum(param.numel() for param in network.parameters() if param.requires_grad)
        # print('Total params num: {}'.format(total_num))
        # print('================= Finish Parameter ================')

        return network
    
    def load_network(self, file_name, file_path=None):
        if file_path is None:
            file_path = self.dic_path["PATH_TO_MODEL"]

        # model
        self.model = self.build_network()
        state_dict = torch.load(os.path.join(file_path, "%s.pth" % (file_name)), map_location=device)
        self.model.load_state_dict(state_dict)

        # ema_model
        self.ema_model = self.build_network()
        state_dict = torch.load(os.path.join(file_path, "%s_ema.pth" % (file_name)), map_location=device)
        self.ema_model.load_state_dict(state_dict)

        # print("succeed in loading model %s" % file_name)

    def save_network(self, file_name):
        model_path = os.path.join(self.dic_path["PATH_TO_MODEL"], "%s.pth" % (file_name))
        torch.save(self.model.state_dict(), model_path)

        model_path = os.path.join(self.dic_path["PATH_TO_MODEL"], "%s_ema.pth" % (file_name))
        torch.save(self.ema_model.state_dict(), model_path)

    def build_memory(self):
        num_agents = self.dic_traffic_env_conf['NUM_INTERSECTIONS']
        batch_size = self.dic_agent_conf['BATCH_SIZE']
        gamma = self.dic_agent_conf['GAMMA']
        reward_norm_factor = self.dic_agent_conf['NORMAL_FACTOR']

        return ReplayBuffer(num_agents, batch_size, gamma, reward_norm_factor)
    
    def get_lane_neighbors(self, inter_num, num_row, num_col):
        edge_index = np.zeros((inter_num * 12, 3), dtype=np.int32)
        edge_index_mask = np.zeros((inter_num * 12, 3), dtype=np.int32)
        edge_index_pos = np.zeros((inter_num * 12,), dtype=np.int32)

        for inter_i in range(inter_num):
            inter_i_col = inter_i % num_col
            inter_i_row = inter_i // num_col

            for lane_j in range(12):
                # to_east
                if inter_i_col - 1 >= 0 and lane_j // 3 == 0:
                    for offset in [1, 5, 9]:
                        index1 = inter_i * 12 + lane_j
                        index2 = (inter_i - 1) * 12 + offset

                        edge_index[index1][edge_index_pos[index1]] = index2
                        edge_index_mask[index1][edge_index_pos[index1]] = 1
                        edge_index_pos[index1] += 1

                        # edge_index[index2][edge_index_pos[index2]] = index1
                        # edge_index_mask[index2][edge_index_pos[index2]] = 1
                        # edge_index_pos[index2] += 1


                # to_west
                if inter_i_col + 1 < num_col and lane_j // 3 == 2:
                    for offset in [3, 7, 11]:
                        index1 = inter_i * 12 + lane_j
                        index2 = (inter_i + 1) * 12 + offset

                        edge_index[index1][edge_index_pos[index1]] = index2
                        edge_index_mask[index1][edge_index_pos[index1]] = 1
                        edge_index_pos[index1] += 1

                        # edge_index[index2][edge_index_pos[index2]] = index1
                        # edge_index_mask[index2][edge_index_pos[index2]] = 1
                        # edge_index_pos[index2] += 1


                # to_north
                if inter_i_row - 1 >= 0 and lane_j // 3 == 1:
                    for offset in [0, 4, 8]:
                        index1 = inter_i * 12 + lane_j
                        index2 = (inter_i - num_col) * 12 + offset

                        edge_index[index1][edge_index_pos[index1]] = index2
                        edge_index_mask[index1][edge_index_pos[index1]] = 1
                        edge_index_pos[index1] += 1

                        # edge_index[index2][edge_index_pos[index2]] = index1
                        # edge_index_mask[index2][edge_index_pos[index2]] = 1
                        # edge_index_pos[index2] += 1


                # to_south
                if inter_i_row + 1 < num_row and lane_j // 3 == 3:
                    for offset in [2, 6, 10]:
                        index1 = inter_i * 12 + lane_j
                        index2 = (inter_i + num_col) * 12 + offset

                        edge_index[index1][edge_index_pos[index1]] = index2
                        edge_index_mask[index1][edge_index_pos[index1]] = 1
                        edge_index_pos[index1] += 1

                        # edge_index[index2][edge_index_pos[index2]] = index1
                        # edge_index_mask[index2][edge_index_pos[index2]] = 1
                        # edge_index_pos[index2] += 1


        return edge_index, edge_index_mask

    def get_closest_nodes(self, node, edge_index, edge_index_mask, num_closest=12):
        distances = {i: float('inf') for i in range(len(edge_index))}
        distances[node] = 0

        queue = [(0, node)]

        while queue:
            current_distance, current_node = heapq.heappop(queue)

            for neighbor, neighbor_mask in zip(edge_index[current_node], edge_index_mask[current_node]):
                if neighbor_mask == 0:
                    continue
                distance = current_distance + 1

                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    heapq.heappush(queue, (distance, neighbor))

        closest_nodes = sorted(distances, key=distances.get)[:num_closest]
        return closest_nodes


    def convert_state_to_input(self, s, choosing_action=False):
        inter_num = self.dic_traffic_env_conf['NUM_INTERSECTIONS']
        features = self.dic_traffic_env_conf["LIST_STATE_FEATURE"]

        state = {}

        for inter_i in range(inter_num):
            inter_state = s[inter_i]
            for feat in features:
                if 'new_phase' in feat:
                    continue
                if feat not in state:
                    state[feat] = []

                state[feat].append(inter_state[feat])

        # normalize
        for feat in features:
            if feat not in state:
                continue
            state[feat] = max_min_norm(np.array(state[feat]), self.bound[feat][0], self.bound[feat][1])

        state = np.array(list(state.values()))
        if choosing_action:
            state = np.expand_dims(state, axis=2)

        return np.array(state)

    @torch.no_grad()
    def choose_action(self, count, sim_info):
        """
        choose action for each traffic light

        Args:
            count: int, the number of current iteration
            sim_info: dict, the information of current simulation

        Returns:
            action: list, the action for each traffic light
        """
        self.ema_model.eval()

        horizon = self.dic_traffic_env_conf["HORIZON"]
        cond_step = self.dic_traffic_env_conf["COND_STEP"]
        reward = self.dic_traffic_env_conf["REWARD"]
        missing_pattern = self.dic_traffic_env_conf["MISSING_PATTERN"]
        drop_dcm = self.dic_agent_conf["DROP_DCM"]
        if "DROP_PRCD" not in self.dic_agent_conf:
            drop_prcd = False
        else:
            drop_prcd = self.dic_agent_conf["DROP_PRCD"]

        states, rewards, _, obs_mask = sim_info
        states = self.convert_state_to_input(states, choosing_action=True)
        rewards = max_min_norm(np.array(rewards), self.bound["reward"][0], self.bound["reward"][1])

        states = torch.tensor(states, dtype=torch.float32, device=device).permute(2, 1, 3, 4, 0)
        obs_mask = torch.tensor(obs_mask, dtype=torch.float32, device=device)[None]
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device)[None]
        lane_neighbors_idx = torch.tensor(self.lane_neighbors_idx, dtype=torch.long, requires_grad=False).to(device)
        lane_neighbors_idx_mask = torch.tensor(self.lane_neighbors_idx_mask, dtype=torch.long, requires_grad=False).to(device)

        states = states * obs_mask[..., None, None]
        rewards = rewards * obs_mask

        input_states = states[..., -cond_step:, :, :]
        if missing_pattern is not None and "km" in missing_pattern:
            input_obs_mask = obs_mask
        else:
            input_obs_mask = torch.cat((obs_mask[..., -cond_step:], torch.ones((*obs_mask.shape[:2], horizon - cond_step), dtype=torch.float32, device=obs_mask.device)), dim=-1)
        input_rewards = torch.cat((rewards[..., -cond_step:], reward * torch.ones((*input_states.shape[:2], horizon - cond_step), dtype=torch.float32, device=input_states.device)), dim=-1)
        input_neighbor_info = [lane_neighbors_idx, lane_neighbors_idx_mask]

        output_states = self.ema_model(cond=input_states, returns=input_rewards, neighbor_info=input_neighbor_info, obs_mask=input_obs_mask, drop_dcm=drop_dcm, drop_prcd=drop_prcd)
        x_comb_t = torch.cat([output_states[..., cond_step - 1, :, :], output_states[..., cond_step, :, :]], dim=-1).reshape(input_states.shape[0] * input_states.shape[1], -1)
        values = self.ema_model.inv_model(x_comb_t)

        action = np.argmax(values.cpu().detach().numpy(), axis=1)

        return action


    def prepare_Xs_Y(self, memo, mask):
        """
        prepare Xs and Y for training

        Args:
            memo: dict, the memory for each algorithm
            mask: dict, the mask for each algorithm
        """

        horizon = self.dic_traffic_env_conf["HORIZON"]
        missing_pattern = self.dic_traffic_env_conf["MISSING_PATTERN"]
        
        states_list, actions_list, next_states_list, rewards_list = [], [], [], []
        obs_mask_list = None if missing_pattern is None else []
        
        for algo_i in memo.keys():
            states = self.convert_state_to_input(memo[algo_i]["state"])
            actions = np.array(memo[algo_i]["action"])
            # next_states, neighbor_next_state_idx = self.convert_state_to_input(memo[algo_i]["next_state"])
            rewards = np.array(memo[algo_i]["reward"])

            states = states.transpose((2, 1, 3, 4, 0))
            actions = actions.transpose((1, 0, 2))
            # without next_states
            # next_states = next_states.transpose((2, 1, 3, 4, 0))
            rewards = rewards.transpose((1, 0, 2))

            if missing_pattern is not None:
                obs_mask = np.array(mask[algo_i][missing_pattern]).transpose((1, 0, 2))
                next_obs_mask = np.concatenate((obs_mask[..., 1:], np.zeros((*obs_mask.shape[:2], 1))), axis=-1)

                states = states * obs_mask[:, :, :, None, None]
                rewards = rewards * next_obs_mask

                obs_mask_list.append(obs_mask[:, :, 1:horizon+1])


            states_list.append(states[:, :, 1:horizon+1])
            actions_list.append(actions[:, :, 1:horizon+1])
            # next_states_list.append(next_states[:, :, :horizon])
            rewards_list.append(rewards[:, :, :horizon])

        states_list = np.concatenate(states_list, axis=0)
        actions_list = np.concatenate(actions_list, axis=0)
        # next_states_list = np.concatenate(next_states_list, axis=0)
        rewards_list = np.concatenate(rewards_list, axis=0)

        if missing_pattern is not None:
            obs_mask_list = np.concatenate(obs_mask_list, axis=0)
        else:
            obs_mask_list = np.ones_like(rewards_list, dtype=np.int32)

        rewards_list = max_min_norm(rewards_list, self.bound["reward"][0], self.bound["reward"][1])

        self.memory.push(states_list, actions_list, None, rewards_list, obs_mask_list)


    def train_epoch(self, optim):
        n_steps_per_epoch = self.dic_agent_conf["N_STEPS_PER_EPOCH"]
        drop_dcm = self.dic_agent_conf["DROP_DCM"]
        if "DROP_PRCD" not in self.dic_agent_conf:
            drop_prcd = False
        else:
            drop_prcd = self.dic_agent_conf["DROP_PRCD"]

        # cond_step = self.dic_traffic_env_conf["COND_STEP"]
        
        missing_pattern = self.dic_traffic_env_conf["MISSING_PATTERN"]
        pattern, rate = missing_pattern.rsplit("_", 1) if missing_pattern is not None else (None, 0.0)
        rate = float(rate)

        self.model.train()
        self.model.train_only_inv = False

        # get sample data
        states, actions, _, rewards, obs_mask, train_batches = self.memory.sample()

        global step_cnt
        train_loss_list = []
        for step in range(n_steps_per_epoch):
            step_loss = []
            for i in range(self.gradient_accumulate_every):
                batch = train_batches[self.batch_idx]
                self.batch_idx = (self.batch_idx + 1) % len(train_batches)

                batch_states = torch.tensor(states[batch], dtype=torch.float32, requires_grad=False).to(device)
                batch_rewards = torch.tensor(rewards[batch], dtype=torch.float32, requires_grad=False).to(device)
                batch_actions = torch.tensor(actions[batch], dtype=torch.float32, requires_grad=False).to(device)
                batch_obs_mask = torch.tensor(obs_mask[batch], dtype=torch.long, requires_grad=False).to(device)

                lane_neighbors_idx = torch.tensor(self.lane_neighbors_idx, dtype=torch.long, requires_grad=False).to(device)
                lane_neighbors_idx_mask = torch.tensor(self.lane_neighbors_idx_mask, dtype=torch.long, requires_grad=False).to(device)

                if pattern is not None:
                    if "rm" in pattern:
                        batch_cond_mask = (torch.rand_like(batch_rewards) > rate / (2 * 10)).type(torch.long).to(device)
                        batch_cond_mask[batch_obs_mask == 0] = 1
                    elif "km" in pattern:
                        batch_cond_mask_index = torch.randint(states.shape[1], (batch_rewards.shape[0],))
                        batch_cond_mask = torch.ones_like(batch_rewards, dtype=torch.long, requires_grad=False).to(device)
                        batch_cond_mask[torch.arange(batch_rewards.shape[0]), batch_cond_mask_index] = 0
                        batch_cond_mask[batch_obs_mask == 0] = 1
                    else:
                        ValueError("pattern not supported")
                else:
                    batch_cond_mask = torch.ones_like(batch_rewards, dtype=torch.long, requires_grad=False).to(device)

                neighbor_info = [lane_neighbors_idx, lane_neighbors_idx_mask]
                batch_traj_mask = [batch_states, batch_actions, batch_obs_mask, batch_cond_mask]

                loss, infos = self.model.loss(batch_traj_mask, returns=batch_rewards, neighbor_info=neighbor_info, drop_dcm=drop_dcm, drop_prcd=drop_prcd)
                loss = loss / self.gradient_accumulate_every
                
                loss.backward()
                step_loss.append(loss.item())
                train_loss_list.append(loss.item())
                

            optim.step()
            optim.zero_grad()

            if step_cnt % self.update_ema_every == 0:
                self.step_ema(self.ema_model, self.model)

            if step_cnt % self.log_freq == 0:
                infos_str = ' | '.join([f'{key}: {val:.4f}' for key, val in infos.items()])
                print(f'Step {step_cnt}: {loss:.4f} | {infos_str}')

                loss_dict[step_cnt] = np.mean(step_loss)
                # plot loss_dict
                plt.figure()
                plt.plot(loss_dict.keys(), loss_dict.values())
                plt.title("loss")
                plt.savefig(os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"], "loss.png"))
                plt.close()

            step_cnt += 1

        return np.mean(train_loss_list)
    
    def train_network(self):
        epochs = self.dic_agent_conf["EPOCHS"]
        optim = torch.optim.Adam(params=self.model.parameters(), lr=self.dic_agent_conf["LEARNING_RATE"])

        for epoch in range(epochs):
            # get sample data
            self.train_epoch(optim)

        self.memory.clear()

    def step_ema(self, ema_model, model):
        global step_cnt
        if step_cnt < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(ema_model, model)