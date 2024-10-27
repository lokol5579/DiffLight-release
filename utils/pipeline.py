from utils.updater import Updater
from utils import model_test
import json
import shutil
import os
import time
import pickle
import torch
from torch.multiprocessing import Process


def path_check(dic_path):
    if os.path.exists(dic_path["PATH_TO_WORK_DIRECTORY"]):
        if dic_path["PATH_TO_WORK_DIRECTORY"] != "records/default":
            raise FileExistsError
        else:
            pass
    else:
        os.makedirs(dic_path["PATH_TO_WORK_DIRECTORY"])
    if os.path.exists(dic_path["PATH_TO_MODEL"]):
        if dic_path["PATH_TO_MODEL"] != "checkpoints/default":
            raise FileExistsError
        else:
            pass
    else:
        os.makedirs(dic_path["PATH_TO_MODEL"])


def copy_conf_file(dic_path, dic_agent_conf, dic_traffic_env_conf, path=None):
    if path is None:
        path = dic_path["PATH_TO_WORK_DIRECTORY"]
    json.dump(dic_agent_conf, open(os.path.join(path, "agent.conf"), "w"), indent=4)
    json.dump(dic_traffic_env_conf, open(os.path.join(path, "traffic_env.conf"), "w"), indent=4)


def copy_cityflow_file(dic_path, dic_traffic_env_conf, path=None):
    if path is None:
        path = dic_path["PATH_TO_WORK_DIRECTORY"]
    shutil.copy(os.path.join(dic_path["PATH_TO_DATA"], dic_traffic_env_conf["TRAFFIC_FILE"]),
                os.path.join(path, dic_traffic_env_conf["TRAFFIC_FILE"]))
    shutil.copy(os.path.join(dic_path["PATH_TO_DATA"], dic_traffic_env_conf["ROADNET_FILE"]),
                os.path.join(path, dic_traffic_env_conf["ROADNET_FILE"]))


def updater_wrapper(cnt_round, dic_agent_conf, dic_traffic_env_conf, dic_path):

    updater = Updater(
        cnt_round=cnt_round,
        dic_agent_conf=dic_agent_conf,
        dic_traffic_env_conf=dic_traffic_env_conf,
        dic_path=dic_path
    )
    updater.load_sample_for_agents()
    updater.update_network_for_agents()
    # print("updater_wrapper end")
    return

class Pipeline:

    def __init__(self, dic_agent_conf, dic_traffic_env_conf, dic_path):
        self.dic_agent_conf = dic_agent_conf
        self.dic_traffic_env_conf = dic_traffic_env_conf
        self.dic_path = dic_path
        if "SAMPLE_STEP" in dic_agent_conf.keys():
            self.sample_step = dic_agent_conf["SAMPLE_STEP"]
        else:
            self.sample_step = -1

        if self.dic_traffic_env_conf["TEST_MODE"] is False:
            self.initialize()
            self.agent_test_conf = None
        else:
            if dic_traffic_env_conf["MODEL_NAME"] == "DiffLight":
                self.agent_test_conf = {
                    "SAMPLE_STEP": dic_agent_conf["SAMPLE_STEP"],
                    "DROP_DCM": dic_agent_conf["DROP_DCM"],
                }
            else:
                self.agent_test_conf = None

    def initialize(self):
        path_check(self.dic_path)
        copy_conf_file(self.dic_path, self.dic_agent_conf, self.dic_traffic_env_conf)
        copy_cityflow_file(self.dic_path, self.dic_traffic_env_conf)

    def run(self, multi_process=False):
        f_time = open(os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"], "running_time.csv"), "w")
        f_time.write("generator_time\tmaking_samples_time\tupdate_network_time\ttest_evaluation_times\tall_times\n")
        f_time.close()

        if self.dic_traffic_env_conf["TEST_MODE"] is False:
            for cnt_round in range(self.dic_traffic_env_conf["NUM_ROUNDS"]):
                print("round %d starts" % cnt_round)
                round_start_time = time.time()
                
                print("==============  update network =============")
                update_network_start_time = time.time()
                if self.dic_traffic_env_conf["MODEL_NAME"] in self.dic_traffic_env_conf["LIST_MODEL_NEED_TO_UPDATE"]:
                    if multi_process:
                        p = Process(target=updater_wrapper,
                                    args=(cnt_round,
                                        self.dic_agent_conf,
                                        self.dic_traffic_env_conf,
                                        self.dic_path))
                        p.start()
                        print("update to join")
                        p.join()
                        print("update finish join")
                    else:
                        updater_wrapper(cnt_round=cnt_round,
                                        dic_agent_conf=self.dic_agent_conf,
                                        dic_traffic_env_conf=self.dic_traffic_env_conf,
                                        dic_path=self.dic_path)

                update_network_end_time = time.time()
                update_network_total_time = update_network_end_time - update_network_start_time

                torch.cuda.empty_cache()

                # print("==============  test evaluation =============")
                test_evaluation_start_time = time.time()
                model_test.test(self.dic_path["PATH_TO_MODEL"], self.dic_path["PATH_TO_MEMO"], cnt_round,
                                self.dic_traffic_env_conf["RUN_COUNTS"], self.dic_traffic_env_conf, self.agent_test_conf)

                test_evaluation_end_time = time.time()
                test_evaluation_total_time = test_evaluation_end_time - test_evaluation_start_time

                print("update_network time:", update_network_total_time)
                print("test_evaluation time:", test_evaluation_total_time)

                print("round {0} ends, total_time: {1}".format(cnt_round, time.time()-round_start_time))
                f_time = open(os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"], "running_time.csv"), "a")
                f_time.write("{0}\t{1}\t{2}\n".format(update_network_total_time, test_evaluation_total_time,
                                                                time.time()-round_start_time))
                f_time.close()

        else:
            # print("==============  test evaluation =============")
            test_evaluation_start_time = time.time()
            cnt_round = self.dic_traffic_env_conf["TEST_ROUND"]
            model_test.test(self.dic_path["PATH_TO_MODEL"], self.dic_path["PATH_TO_MEMO"], cnt_round,
                            self.dic_traffic_env_conf["RUN_COUNTS"], self.dic_traffic_env_conf, self.agent_test_conf)

            test_evaluation_end_time = time.time()
            test_evaluation_total_time = test_evaluation_end_time - test_evaluation_start_time

            print("test_evaluation time:", test_evaluation_total_time)

            f_time = open(os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"], "running_time.csv"), "a")
            f_time.close()

    def load_data(self, data_file):
        with open(data_file, "rb") as f:
            data = pickle.load(f)

        
        return data