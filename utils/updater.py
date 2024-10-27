from utils.config import DIC_AGENTS
import pickle
import os
import time

class Updater:

    def __init__(self, cnt_round, dic_agent_conf, dic_traffic_env_conf, dic_path):

        self.cnt_round = cnt_round
        self.dic_path = dic_path
        self.dic_traffic_env_conf = dic_traffic_env_conf
        self.dic_agent_conf = dic_agent_conf
        self.agents = []
        self.sample_set_list = []
        self.sample_indexes = None

        for i in range(dic_traffic_env_conf['NUM_AGENTS']):
            agent_name = self.dic_traffic_env_conf["MODEL_NAME"]
            agent= DIC_AGENTS[agent_name](
                self.dic_agent_conf, self.dic_traffic_env_conf,
                self.dic_path, self.cnt_round, intersection_id=str(i))
            self.agents.append(agent)

    def load_sample_for_agents(self):
        start_time = time.time()
        missing_pattern = self.dic_traffic_env_conf["MISSING_PATTERN"]

        sample_file = open(os.path.join(self.dic_path["PATH_TO_MEMO"], "%s.pkl" % self.dic_traffic_env_conf["TRAFFIC_FILE"][:-5]), "rb")
        if missing_pattern is not None:
            pattern, rate = missing_pattern.rsplit("_", 1)
            missing_pattern_file = open(os.path.join(self.dic_path["PATH_TO_MEMO"], pattern, "%s_%s.pkl" % (self.dic_traffic_env_conf["TRAFFIC_FILE"][:-5], pattern)), "rb")

        try:
            while True:
                sample_set = pickle.load(sample_file)
                if missing_pattern is not None:
                    mask_set = pickle.load(missing_pattern_file)
                else:
                    mask_set = None
                print("Samples loaded!")
                self.agents[0].prepare_Xs_Y(sample_set, mask_set)
        except EOFError:
            sample_file.close()

    def update_network(self, i):
        self.agents[i].train_network()
        self.agents[i].save_network("round_{0}_inter_{1}".format(self.cnt_round, self.agents[i].intersection_id))
        if self.dic_traffic_env_conf["MODEL_NAME"] in ["BEAR", "TD3_BC"]:
            self.agents[i].save_network_bar("round_{0}_inter_{1}".format(self.cnt_round, self.agents[i].intersection_id))

    def update_network_for_agents(self):
        for i in range(self.dic_traffic_env_conf['NUM_AGENTS']):
            self.update_network(i)
