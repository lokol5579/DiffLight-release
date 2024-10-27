from utils.config import DIC_AGENTS
from copy import deepcopy
from utils.cityflow_env import CityFlowEnv
import json
import os
import random
from tqdm import tqdm
import pickle
import numpy as np
import time


def test(model_dir, memo_dir, cnt_round, run_cnt, _dic_traffic_env_conf, agent_test_conf=None):
    def init_buffer(state, horizon, missing_pattern=None, missing_pattern_file_path=None):
        state_buffer = deepcopy(state)
        obs_mask_buffer = [[] * horizon] * dic_traffic_env_conf["NUM_INTERSECTIONS"]
        reward_buffer = [[] * horizon] * dic_traffic_env_conf["NUM_INTERSECTIONS"]
        action_buffer = [[] * horizon] * dic_traffic_env_conf["NUM_INTERSECTIONS"]
        if missing_pattern is not None:
            pattern, rate = missing_pattern

        if missing_pattern is not None:
            if "km" in pattern:
                missing_pattern_file = open(missing_pattern_file_path, "rb")
                mask_set = pickle.load(missing_pattern_file)
                mask_set = np.array(next(iter(mask_set.values()))["%s_%s" % (pattern, str(rate))], dtype=int).transpose((1, 0, 2))[0, :, :horizon]
            elif "rm" in pattern:
                rate = float(rate) / 10

        for inter_i in range(len(state_buffer)):
            if missing_pattern is not None:
                if "rm" in pattern and rate > 0:
                    obs_mask_buffer[inter_i] = [0] * (horizon - 1)
                    if random.random() < rate:
                        obs_mask_buffer[inter_i].append(0)
                    else:
                        obs_mask_buffer[inter_i].append(1)
                elif "km" in pattern:
                    obs_mask_buffer[inter_i] = list(mask_set[inter_i])
                else:
                    ValueError("missing pattern not supported")
            else:
                obs_mask_buffer[inter_i] = [0] * (horizon - 1) + [1]

            for feat in state_buffer[inter_i].keys():
                state_buffer[inter_i][feat] = [state[inter_i][feat]] * horizon

            reward_buffer[inter_i] = [0] * horizon
            action_buffer[inter_i] = [0] * horizon

        return state_buffer, reward_buffer, action_buffer, obs_mask_buffer

    def update_buffer(state_buffer, reward_buffer, action_buffer, obs_mask_buffer, state, reward, action, missing_pattern=None):
        if missing_pattern is not None:
            pattern, rate = missing_pattern

        if missing_pattern is not None and "rm" in pattern:
            rate = float(rate) / 10

        for inter_i in range(len(state_buffer)):
            if missing_pattern is not None:
                if "rm" in pattern and rate > 0:
                    obs_mask_buffer[inter_i] = obs_mask_buffer[inter_i][1:]
                    if random.random() < rate:
                        obs_mask_buffer[inter_i].append(0)
                    else:
                        obs_mask_buffer[inter_i].append(1)
                elif "km" in pattern:
                    pass
                else:
                    ValueError("missing pattern not supported")
            else:
                obs_mask_buffer[inter_i] = obs_mask_buffer[inter_i][1:]
                obs_mask_buffer[inter_i].append(1)

            for feat in state_buffer[inter_i].keys():
                state_buffer[inter_i][feat] = state_buffer[inter_i][feat][1:]
                state_buffer[inter_i][feat].append(state[inter_i][feat])

            reward_buffer[inter_i] = reward_buffer[inter_i][1:]
            reward_buffer[inter_i].append(reward[inter_i])

            action_buffer[inter_i] = action_buffer[inter_i][1:]
            action_buffer[inter_i].append(action[inter_i])

        return state_buffer, reward_buffer, action_buffer, obs_mask_buffer

    dic_traffic_env_conf = deepcopy(_dic_traffic_env_conf)
    records_dir = model_dir.replace("checkpoints", "records")
    model_round = "round_%d" % cnt_round
    dic_path = {"PATH_TO_MODEL": model_dir, "PATH_TO_WORK_DIRECTORY": records_dir, "PATH_TO_MEMO": memo_dir}
    with open(os.path.join(records_dir, "agent.conf"), "r") as f:
        dic_agent_conf = json.load(f)
    if os.path.exists(os.path.join(records_dir, "anon_env.conf")):
        with open(os.path.join(records_dir, "anon_env.conf"), "r") as f:
            dic_traffic_env_conf = json.load(f)
    dic_traffic_env_conf["RUN_COUNTS"] = run_cnt

    if dic_traffic_env_conf["MODEL_NAME"] in dic_traffic_env_conf["LIST_MODEL_NEED_TO_UPDATE"]:
        dic_agent_conf["EPSILON"] = 0
        dic_agent_conf["MIN_EPSILON"] = 0

    if agent_test_conf is not None:
        dic_agent_conf["SAMPLE_STEP"] = agent_test_conf["SAMPLE_STEP"]
        dic_agent_conf["DROP_DCM"] = agent_test_conf["DROP_DCM"]

    agents = []
    for i in range(dic_traffic_env_conf['NUM_AGENTS']):
        agent_name = dic_traffic_env_conf["MODEL_NAME"]
        agent = DIC_AGENTS[agent_name](
            dic_agent_conf=dic_agent_conf,
            dic_traffic_env_conf=dic_traffic_env_conf,
            dic_path=dic_path,
            cnt_round=0,
            intersection_id=str(i)
        )
        agents.append(agent)
    
    for i in range(dic_traffic_env_conf['NUM_AGENTS']):
        agents[i].load_network("{0}_inter_{1}".format(model_round, agents[i].intersection_id))
    path_to_log = os.path.join(dic_path["PATH_TO_WORK_DIRECTORY"], "test_round", model_round)
    if not os.path.exists(path_to_log):
        os.makedirs(path_to_log)
    env = CityFlowEnv(
        path_to_log=path_to_log,
        path_to_work_directory=dic_path["PATH_TO_WORK_DIRECTORY"],
        dic_traffic_env_conf=dic_traffic_env_conf
    )

    done = False

    step_num = 0

    total_time = dic_traffic_env_conf["RUN_COUNTS"]
    if dic_traffic_env_conf["MISSING_PATTERN"] is None:
        missing_pattern = None
        missing_pattern_file_path = None
    else:
        pattern, rate = dic_traffic_env_conf["MISSING_PATTERN"].rsplit("_", 1)
        missing_pattern_file_path = os.path.join(dic_path["PATH_TO_MEMO"], pattern, "%s_%s.pkl" % (dic_traffic_env_conf["TRAFFIC_FILE"][:-5], pattern))
        missing_pattern = [pattern, rate]

    state = env.reset()

    horizon = dic_traffic_env_conf["HORIZON"]
    
    state_buffer, reward_buffer, action_buffer, obs_mask_buffer = init_buffer(state, horizon, missing_pattern, missing_pattern_file_path)
    test_time = 0
    with tqdm(total=int(total_time / dic_traffic_env_conf["MIN_ACTION_TIME"]), desc="test_simulate", leave=False) as pbar:
        while not done and step_num < int(total_time / dic_traffic_env_conf["MIN_ACTION_TIME"]):
            action_list = []

            for i in range(dic_traffic_env_conf["NUM_AGENTS"]):

                t1 = time.time()
                if dic_traffic_env_conf["MODEL_NAME"] in ["DiffLight", "BC", "BEAR", "CQL", "TD3_BC", "DT", "DD", "Diffuser"]:
                    # one_state = state
                    action_list = agents[i].choose_action(step_num, [state_buffer, reward_buffer, action_buffer, obs_mask_buffer])
                else:
                    one_state = state[i]
                    action = agents[i].choose_action(step_num, one_state)
                    action_list.append(action)
                t2 = time.time()
                test_time += (t2 - t1)

            next_state, reward, done, _ = env.step(action_list)

            state = next_state
            state_buffer, reward_buffer, action_buffer, obs_mask_buffer = update_buffer(state_buffer, reward_buffer, action_buffer, obs_mask_buffer, state, reward, action_list, missing_pattern)
            step_num += 1

            pbar.update(1)

    print("test_time: %fs" % test_time)

    env.batch_log_2()
    env.end_cityflow()
