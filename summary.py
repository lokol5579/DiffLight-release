import os
import pandas as pd
import numpy as np
import json
import shutil
import copy
from matplotlib import pyplot as plt

def get_metrics(duration_list, traffic_name, num_of_out):
    # calculate the mean final 5 rounds
    validation_duration_length = 5
    total_summary_metrics = {
    "traffic": [],
    "final_duration_avg": [],
    "final_duration_std": [],
    "final_through_avg": [],
    }
    duration_list = np.array(duration_list)
    validation_duration = duration_list[-validation_duration_length:]
    validation_through = num_of_out[-validation_duration_length:]
    final_through = np.round(np.mean(validation_through), decimals=2)
    final_duration = np.round(np.mean(validation_duration[validation_duration > 0]), decimals=2)
    final_duration_std = np.round(np.std(validation_duration[validation_duration > 0]), decimals=2)

    total_summary_metrics["traffic"].append(traffic_name)
    total_summary_metrics["final_duration_avg"].append(final_duration)
    total_summary_metrics["final_duration_std"].append(final_duration_std)
    total_summary_metrics["final_through_avg"].append(final_through)

    return total_summary_metrics


def summary_detail_RL(path_to_record):
    """
    Used for test RL results
    """
    temp = path_to_record.split('/')
    model_name = temp[1]
    road = temp[2]
    flow_name = temp[3]
    timestamp = temp[4]
    print("############################### Summary ##############################")
    print("model name: ", model_name)
    print("road net: ", road)
    print("flow: ", flow_name)
    print("timestamp: ", timestamp)

    traffic_env_conf = open(os.path.join(path_to_record, "traffic_env.conf"), 'r')
    dic_traffic_env_conf = json.load(traffic_env_conf)
    run_counts = dic_traffic_env_conf["RUN_COUNTS"]
    num_intersection = dic_traffic_env_conf['NUM_INTERSECTIONS']
    duration_each_round_list = []
    num_of_vehicle_in = []
    num_of_vehicle_out = []
    test_round_dir = os.path.join(path_to_record, "test_round")
    try:
        round_files = os.listdir(test_round_dir)
    except:
        print("no test round in {}".format(path_to_record))
        exit(-1)
    round_files = [f for f in round_files if "round" in f]
    round_files.sort(key=lambda x: int(x[6:]))
    for round_rl in round_files:
        print("============================================================")
        print("round:", round_rl)
        df_vehicle_all = []
        for inter_index in range(num_intersection):
            try:
                round_dir = os.path.join(test_round_dir, round_rl)  # , "generator_0"
                df_vehicle_inter = pd.read_csv(os.path.join(round_dir, "vehicle_inter_{0}.csv".format(inter_index)),
                                                sep=',', header=0, dtype={0: str, 1: float, 2: float},
                                                names=["vehicle_id", "enter_time", "leave_time"])
                # [leave_time_origin, leave_time, enter_time, duration]
                df_vehicle_inter['leave_time_origin'] = df_vehicle_inter['leave_time']
                df_vehicle_inter['leave_time'].fillna(run_counts, inplace=True)  # Replace all nan values with the maximum number of steps per round
                df_vehicle_inter['duration'] = df_vehicle_inter["leave_time"].values - \
                                                df_vehicle_inter["enter_time"].values
                tmp_idx = []
                for i, v in enumerate(df_vehicle_inter["vehicle_id"]):
                    if "shadow" in v:
                        tmp_idx.append(i)
                df_vehicle_inter.drop(df_vehicle_inter.index[tmp_idx], inplace=True)

                ave_duration = df_vehicle_inter['duration'].mean(skipna=True)
                print("-------- inter_index: {0}\tave_duration: {1}".format(inter_index, ave_duration))
                df_vehicle_all.append(df_vehicle_inter)
            except:
                print("======= Error occured during reading vehicle_inter_{}.csv")

        if len(df_vehicle_all) == 0:
            print("============== There are no vehicles in the road network ==============")
            continue

        df_vehicle_all = pd.concat(df_vehicle_all)
        # calculate the duration through the entire network
        vehicle_duration = df_vehicle_all.groupby(by=['vehicle_id'])['duration'].sum()
        ave_duration = vehicle_duration.mean()  # mean amomng all the vehicle

        duration_each_round_list.append(ave_duration)

        num_of_vehicle_in.append(len(df_vehicle_all['vehicle_id'].unique()))
        num_of_vehicle_out.append(len(df_vehicle_all.dropna()['vehicle_id'].unique()))

        print("@@@@@@@@ round: {0}\t ave_duration: {1}\t num_of_vehicle_in: {2}\t num_of_vehicle_out: {3}"
                .format(round_rl, ave_duration, num_of_vehicle_in[-1], num_of_vehicle_out[-1]))

    result_dir = os.path.join("summary", model_name, road, flow_name, timestamp)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    _res = {
        "duration": duration_each_round_list,
        "vehicle_in": num_of_vehicle_in,
        "vehicle_out": num_of_vehicle_out
    }
    result = pd.DataFrame(_res)
    result.to_csv(os.path.join(result_dir, "test_results.csv"))
    
    fig = plt.figure()
    ax1 = fig.subplots()
    ax2 = ax1.twinx()

    ax1.plot(range(len(duration_each_round_list)), duration_each_round_list, label="duration", color='red')
    ax2.plot(range(len(num_of_vehicle_in)), num_of_vehicle_in, label="vehicle_in", color='blue')
    ax2.plot(range(len(num_of_vehicle_out)), num_of_vehicle_out, label="vehicle_out", color='green')

    ax1.set_xlabel('Round')
    ax1.set_ylabel('ATT')
    ax2.set_ylabel('NV')

    plt.title(model_name + '_' + '_' + road + '_' + flow_name)
    plt.legend()
    plt.savefig(os.path.join(result_dir, "ave_duration_each_round.png"))
    plt.close()

    total_summary_rl = get_metrics(duration_each_round_list, model_name + '_' + '_' + road + '_' + flow_name, num_of_vehicle_out)
    total_result = pd.DataFrame(total_summary_rl)
    total_result.to_csv(os.path.join(result_dir, "final_ten_rounds_test_results.csv"))
    print()
    print(total_result)


def summary_detail_conventional(memo_cv):
    """
    Used for test conventional results.
    """
    total_summary_cv = []
    records_dir = os.path.join("records", memo_cv)
    for traffic_file in os.listdir(records_dir):
        if "anon" not in traffic_file:
            continue
        traffic_conf = open(os.path.join(records_dir, traffic_file, "traffic_env.conf"), 'r')

        dic_traffic_env_conf = json.load(traffic_conf)
        run_counts = dic_traffic_env_conf["RUN_COUNTS"]

        print(traffic_file)
        train_dir = os.path.join(records_dir, traffic_file)
        use_all = True
        if use_all:
            with open(os.path.join(records_dir, traffic_file, 'agent.conf'), 'r') as agent_conf:
                dic_agent_conf = json.load(agent_conf)

            df_vehicle_all = []
            NUM_OF_INTERSECTIONS = int(traffic_file.split('_')[1]) * int(traffic_file.split('_')[2])

            for inter_id in range(int(NUM_OF_INTERSECTIONS)):
                vehicle_csv = "vehicle_inter_{0}.csv".format(inter_id)

                df_vehicle_inter_0 = pd.read_csv(os.path.join(train_dir, vehicle_csv),
                                                 sep=',', header=0, dtype={0: str, 1: float, 2: float},
                                                 names=["vehicle_id", "enter_time", "leave_time"])

                # [leave_time_origin, leave_time, enter_time, duration]
                df_vehicle_inter_0['leave_time_origin'] = df_vehicle_inter_0['leave_time']
                df_vehicle_inter_0['leave_time'].fillna(run_counts, inplace=True)
                df_vehicle_inter_0['duration'] = df_vehicle_inter_0["leave_time"].values - df_vehicle_inter_0[
                    "enter_time"].values

                tmp_idx = []
                for i, v in enumerate(df_vehicle_inter_0["vehicle_id"]):
                    if "shadow" in v:
                        tmp_idx.append(i)
                df_vehicle_inter_0.drop(df_vehicle_inter_0.index[tmp_idx], inplace=True)

                ave_duration = df_vehicle_inter_0['duration'].mean(skipna=True)
                print("------------- inter_index: {0}\tave_duration: {1}".format(inter_id, ave_duration))
                df_vehicle_all.append(df_vehicle_inter_0)

            df_vehicle_all = pd.concat(df_vehicle_all, axis=0)
            vehicle_duration = df_vehicle_all.groupby(by=['vehicle_id'])['duration'].sum()
            ave_duration = vehicle_duration.mean()
            num_of_vehicle_in = len(df_vehicle_all['vehicle_id'].unique())
            num_of_vehicle_out = len(df_vehicle_all.dropna()['vehicle_id'].unique())
            save_path = os.path.join('records', memo_cv, traffic_file).replace("records", "summary")
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            # duration.to_csv(os.path.join(save_path, 'flow.csv'))
            total_summary_cv.append(
                [traffic_file, ave_duration, num_of_vehicle_in, num_of_vehicle_out, dic_agent_conf["FIXED_TIME"]])
        else:
            shutil.rmtree(train_dir)
    total_summary_cv = pd.DataFrame(total_summary_cv)
    total_summary_cv.sort_values([0], ascending=[True], inplace=True)
    total_summary_cv.columns = ['TRAFFIC', 'DURATION', 'CAR_NUMBER_in', 'CAR_NUMBER_out', 'CONFIG']
    total_summary_cv.to_csv(os.path.join("records", memo_cv,
                                         "total_baseline_results.csv").replace("records", "summary"),
                            sep='\t', index=False)


if __name__ == "__main__":
    """Only use these data"""

    path_to_record = "records/DiffLight/Jinan/anon_3_4_jinan_real_2000.json/10_10_04_12_28"
    summary_detail_RL(path_to_record)
    # summary_detail_conventional(memo)

