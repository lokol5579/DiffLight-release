from utils.utils import pipeline_wrapper, merge
from utils import config
import time
from torch.multiprocessing import Process
import argparse
import os
from summary import summary_detail_RL

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def parse_args():
    parser = argparse.ArgumentParser()

    # experiment
    parser.add_argument("-memo",       type=str,           default='DiffLight')
    parser.add_argument("-mod",        type=str,           default="DiffLight")
    parser.add_argument("-eightphase", action="store_true", default=False)
    parser.add_argument("-gen",        type=int,            default=1)
    parser.add_argument("-multi_process", action="store_true", default=False)
    parser.add_argument("-workers",    type=int,            default=3)

    # evaluation
    parser.add_argument(
        "-test",
        action="store_true",
        default=False,
        help="evalate the model in test mode, test_time and test_round should be specified",
    )
    parser.add_argument(
        "-test_time",
        type=str,
        default=None,
        help="the timestamp of the experiment to be tested",
    )
    parser.add_argument(
        "-test_round",
        type=int,
        default=0,
        help="the round of the experiment to be tested",
    )

    # model configuration
    parser.add_argument(
        "-reward",
        type=float,
        default=1.0,
        help="the reward for the model to be achieved, 0.0 means the best reward",
    )
    parser.add_argument(
        "-horizon", type=int, default=8, help="the history horizon of the model"
    )
    parser.add_argument(
        "-drop_dcm",
        action="store_true",
        default=False,
        help="drop the diffusion communication mechanism",
    )
    parser.add_argument(
        "-drop_prcd",
        action="store_true",
        default=False,
        help="drop the partial reward conditioned diffusion",
    )
    parser.add_argument(
        "-drop_neighbor", action="store_true", default=False, help="drop the neighbors"
    )
    parser.add_argument(
        "-unet", action="store_true", default=False, help="use the unet"
    )
    parser.add_argument("-sample_step", type=int, default=1, help="sample step")

    # missing pattern & rate
    parser.add_argument("-missing_pattern", type=str, default=None)

    # datasets
    parser.add_argument("-hangzhou_1",    action="store_true", default=False)
    parser.add_argument("-hangzhou_2",    action="store_true", default=False)
    parser.add_argument("-jinan_1",       action="store_true", default=False)
    parser.add_argument("-jinan_2",       action="store_true", default=False)
    parser.add_argument("-jinan_3",       action="store_true", default=False)
    parser.add_argument("-newyork",     action="store_true", default=False)
    return parser.parse_args()


def main(in_args=None):    
    if in_args.hangzhou_1:
        count = 3600
        road_net = "4_4"
        traffic_file_list = ["anon_4_4_hangzhou_real.json"]
        num_rounds = 15
        template = "Hangzhou"
    elif in_args.hangzhou_2:
        count = 3600
        road_net = "4_4"
        traffic_file_list = ["anon_4_4_hangzhou_real_5816.json"]
        num_rounds = 15
        template = "Hangzhou"
    elif in_args.jinan_1:
        count = 3600
        road_net = "3_4"
        traffic_file_list = ["anon_3_4_jinan_real.json"]
        num_rounds = 15
        template = "Jinan"
    elif in_args.jinan_2:
        count = 3600
        road_net = "3_4"
        traffic_file_list = ["anon_3_4_jinan_real_2000.json"]
        num_rounds = 15
        template = "Jinan"
    elif in_args.jinan_3:
        count = 3600
        road_net = "3_4"
        traffic_file_list = ["anon_3_4_jinan_real_2500.json"]
        num_rounds = 15
        template = "Jinan"
    elif in_args.newyork:
        count = 3600
        road_net = "16_3"
        traffic_file_list = ["anon_16_3_newyork_real.json"]
        num_rounds = 15
        template = "newyork_16_3"

    NUM_COL = int(road_net.split('_')[1])
    NUM_ROW = int(road_net.split('_')[0])
    num_intersections = NUM_ROW * NUM_COL
    print('num_intersections:', num_intersections)
    print(traffic_file_list)
    process_list = []
    for traffic_file in traffic_file_list:
        dic_traffic_env_conf_extra = {

            "NUM_ROUNDS": num_rounds,
            "NUM_GENERATORS": in_args.gen,
            "NUM_AGENTS": 1,
            "NUM_INTERSECTIONS": num_intersections,
            "RUN_COUNTS": count,

            "MISSING_PATTERN": in_args.missing_pattern,

            "TEST_MODE": in_args.test,
            "TEST_ROUND": in_args.test_round,

            "MODEL_NAME": in_args.mod,
            "NUM_ROW": NUM_ROW,
            "NUM_COL": NUM_COL,

            "TRAFFIC_FILE": traffic_file,
            "ROADNET_FILE": "roadnet_{0}.json".format(road_net),
            "TRAFFIC_SEPARATE": traffic_file,
            "LIST_STATE_FEATURE": [
                "lane_num_vehicle_in",
                "lane_queue_vehicle_in",
            ],

            "DIC_REWARD_INFO": {
                "queue_length": -0.25,
            },

            "HORIZON": in_args.horizon,
            "COND_STEP": in_args.horizon - 3,
            "REWARD": in_args.reward,
        }

        if in_args.eightphase:
            dic_traffic_env_conf_extra["PHASE"] = {
                    1: [0, 1, 0, 1, 0, 0, 0, 0],
                    2: [0, 0, 0, 0, 0, 1, 0, 1],
                    3: [1, 0, 1, 0, 0, 0, 0, 0],
                    4: [0, 0, 0, 0, 1, 0, 1, 0],
                    5: [1, 1, 0, 0, 0, 0, 0, 0],
                    6: [0, 0, 1, 1, 0, 0, 0, 0],
                    7: [0, 0, 0, 0, 0, 0, 1, 1],
                    8: [0, 0, 0, 0, 1, 1, 0, 0]
                }
            dic_traffic_env_conf_extra["PHASE_LIST"] = ['WT_ET', 'NT_ST', 'WL_EL', 'NL_SL',
                                                        'WL_WT', 'EL_ET', 'SL_ST', 'NL_NT']

        if in_args.test:
             print("test_mode")
             if in_args.test_time is not None:
                 time_str = in_args.test_time
        else:
            time_str = time.strftime('%m_%d_%H_%M_%S', time.localtime(time.time()))
        print(time_str)

        dic_path_extra = {
            "PATH_TO_MODEL": os.path.join("checkpoints", in_args.memo, template, traffic_file, time_str),
            "PATH_TO_WORK_DIRECTORY": os.path.join("records", in_args.memo, template, traffic_file, time_str),
            "PATH_TO_DATA": os.path.join("data", template, str(road_net)),
            "PATH_TO_ERROR": os.path.join("errors", in_args.memo, template),
            "PATH_TO_MEMO": os.path.join("memory", "eightphase" if args.eightphase else "fourphase"),
        }

        deploy_dic_agent_conf_extra = {
            "LEARNING_RATE": 2e-4,
            "BATCH_SIZE": 64 if template != "newyork_16_3" else 24,
            "NORMAL_FACTOR": -100 if template != "newyork_16_3" else -50,
            "EPOCHS": 1,
            "BLOCK_DEPTH": 1,
            "DROP_DCM": in_args.drop_dcm,
            "DROP_NEIGHBOR": in_args.drop_neighbor,
            "DROP_PRCD": in_args.drop_prcd,
            "USE_UNET": in_args.unet,
            "SAMPLE_STEP": in_args.sample_step
        }

        deploy_dic_traffic_env_conf = merge(config.dic_traffic_env_conf, dic_traffic_env_conf_extra)
        deploy_dic_path = merge(config.DIC_PATH, dic_path_extra)
        deploy_dic_agent_conf = merge(config.DIC_BASE_AGENT_CONF, deploy_dic_agent_conf_extra)

        if in_args.multi_process:
            ppl = Process(target=pipeline_wrapper,
                          args=(deploy_dic_agent_conf,
                                deploy_dic_traffic_env_conf,
                                deploy_dic_path))
            process_list.append(ppl)
        else:
            pipeline_wrapper(dic_agent_conf=deploy_dic_agent_conf,
                             dic_traffic_env_conf=deploy_dic_traffic_env_conf,
                             dic_path=deploy_dic_path)

    if in_args.multi_process:
        for i in range(0, len(process_list), in_args.workers):
            i_max = min(len(process_list), i + in_args.workers)
            for j in range(i, i_max):
                print(j)
                print("start_traffic")
                process_list[j].start()
                print("after_traffic")
            for k in range(i, i_max):
                print("traffic to join", k)
                process_list[k].join()
                print("traffic finish join", k)

    summary_detail_RL(dic_path_extra["PATH_TO_WORK_DIRECTORY"])
    return in_args.memo


if __name__ == "__main__":
    args = parse_args()

    main(args)

