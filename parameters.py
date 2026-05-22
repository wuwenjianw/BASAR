from project_paths import gifs_dir, model_dir, train_dir


class EnvParams:
    SPECIES_AGENTS_RANGE = (4, 4)
    SPECIES_RANGE = (3, 5)
    TASKS_RANGE = (15, 50)
    MAX_TIME = 150  # 200
    TRAIT_DIM = 5
    DECISION_DIM = 30


class TrainParams:
    USE_GPU = False
    USE_GPU_GLOBAL = True
    NUM_GPU = 1
    NUM_META_AGENT = 24
    WORKER_TORCH_NUM_THREADS = 1
    LR = 1e-5
    GAMMA = 1
    DECAY_STEP = 2e3
    RESET_OPT = False
    EVALUATE = True
    EVALUATION_SAMPLES = 64
    RESET_RAY = False
    INCREASE_DIFFICULTY = 20000
    SUMMARY_WINDOW = 8
    DEMON_RATE = 0.5
    IL_DECAY = -1e-5  # -1e-6 700k decay 0.5, -1e-5 70k decay 0.5, -1e-4 7k decay 0.5
    BATCH_SIZE = 256 # 2048
    AGENT_INPUT_DIM = 6 + EnvParams.TRAIT_DIM
    TASK_INPUT_DIM = 6 + 2 * EnvParams.TRAIT_DIM  # 从5改为6，增加了deadline维度
    EMBEDDING_DIM = 128
    SAMPLE_SIZE = 200
    PADDING_SIZE = 50
    POMO_SIZE = 10
    FORCE_MAX_OPEN_TASK = False
    MODEL_NAME = 'myself'  # 可选: 'attention' / 'myself' / 'capam'
    CROSS_ATTENTION_MODE = 'dual_cross'  # 可选: 'dual_cross' / 'shared_self'
    GLOBAL_DECODER_MODE = 'attention'  # 可选: 'attention' / 'mlp'
    GLOBAL_MLP_HIDDEN_DIM = EMBEDDING_DIM * 2
    IMPRO_BOTTLENECK_DIM = 16
    IMPRO_L1_LAMBDA = 1e-5


class SaverParams:
    FOLDER_NAME = 'SAVE_5'  # 
    MODEL_PATH = str(model_dir(FOLDER_NAME))
    TRAIN_PATH = str(train_dir(FOLDER_NAME))
    GIFS_PATH = str(gifs_dir(FOLDER_NAME))
    LOAD_MODEL = False
    LOAD_FROM = 'current'  # 'best'
    SAVE = True
    SAVE_IMG = False
    SAVE_IMG_GAP = 200
