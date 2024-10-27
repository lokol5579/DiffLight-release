from models.diffusion.difflight_agent import DiffLightAgent
# from models.baseline.bc_agent import BCAgent
# from models.baseline.cql_agent import CQLAgent
# from models.baseline.td3_bc_agent import TD3BCAgent
# from models.baseline.DT_agent import DTAgent
# from models.baseline.DD_agent import DDAgent
# from models.baseline.diffuser_agent import DiffuserAgent


DIC_AGENTS = {
    "DiffLight": DiffLightAgent,
    # "BC": BCAgent,
    # "CQL": CQLAgent,
    # "TD3_BC": TD3BCAgent,
    # "DT": DTAgent,
    # "DD": DDAgent,
    # "Diffuser": DiffuserAgent,
}

DIC_PATH = {
    "PATH_TO_MODEL": "checkpoints/default",
    "PATH_TO_WORK_DIRECTORY": "records/default",
    "PATH_TO_DATA": "data/template",
    "PATH_TO_ERROR": "errors/default",
}

dic_traffic_env_conf = {
    "MIN_Q_W": 0.005,
    "THRESHOLD": 0.3,

    "LIST_MODEL": ["DiffLight", "BC", "BEAR", "CQL", "TD3_BC", "DT", "DD", "Diffuser"],
    "LIST_MODEL_NEED_TO_UPDATE": ["DiffLight", "BC", "BEAR", "CQL", "TD3_BC", "DT", "DD", "Diffuser"],

    "FORGET_ROUND": 20,
    "RUN_COUNTS": 3600,
    "MODEL_NAME": None,
    "TOP_K_ADJACENCY": 5,

    "ACTION_PATTERN": "set",
    "NUM_INTERSECTIONS": 1,

    "OBS_LENGTH": 167,
    "MIN_ACTION_TIME": 15,
    "MEASURE_TIME": 15,

    "BINARY_PHASE_EXPANSION": True,

    "YELLOW_TIME": 5,
    "ALL_RED_TIME": 0,
    "NUM_PHASES": 4,
    "NUM_LANES": [3, 3, 3, 3],

    "INTERVAL": 1,

    "LIST_STATE_FEATURE": [
        "cur_phase",
        "pressure",
        "adjacency_matrix"
    ],
    "DIC_REWARD_INFO": {
        "queue_length": 0,
        "pressure": 0,
    },
    "PHASE": {
        1: [0, 1, 0, 1, 0, 0, 0, 0],
        2: [0, 0, 0, 0, 0, 1, 0, 1],
        3: [1, 0, 1, 0, 0, 0, 0, 0],
        4: [0, 0, 0, 0, 1, 0, 1, 0]
        },
    "list_lane_order": ["WL", "WT", "EL", "ET", "NL", "NT", "SL", "ST"],
    "PHASE_LIST": ['WT_ET', 'NT_ST', 'WL_EL', 'NL_SL'],
    "PHASE_MAP": [[1, 4], [7, 10], [0, 3], [6, 9]],

    "HORIZON": 9,
    "COND_STEP": 6,
}

DIC_BASE_AGENT_CONF = {
    "D_DENSE": 20,
    "LEARNING_RATE": 0.001,
    "PATIENCE": 10,
    "BATCH_SIZE": 20,
    "EPOCHS": 100,
    "N_STEPS_PER_EPOCH": 10000,
    "SAMPLE_SIZE": 3000,
    "MAX_MEMORY_LEN": 12000,

    "UPDATE_Q_BAR_FREQ": 5,
    "UPDATE_Q_BAR_EVERY_C_ROUND": False,

    "GAMMA": 0.8,
    "LMBDA": 0.8,
    "EPS_CLIP": 0.2,
    "NORMAL_FACTOR": -100,

    "EPSILON": 0.8,
    "EPSILON_DECAY": 0.95,
    "MIN_EPSILON": 0.2,
    "LOSS_FUNCTION": "mean_squared_error",
}

DIC_FIXEDTIME_AGENT_CONF = {
    "FIXED_TIME": [15, 15, 15, 15]
}

DIC_MAXPRESSURE_AGENT_CONF = {
    "FIXED_TIME": [15, 15, 15, 15]
}
