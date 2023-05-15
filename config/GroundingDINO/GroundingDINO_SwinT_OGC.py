# model

MODEL = dict(
    WEIGHT = "groundingdino_swint_ogc.pth",
    modelname = "groundingdino",
    # batch_size = 1,
    backbone = "swin_T_224_1k",
    position_embedding = "sine",
    pe_temperatureH = 20,
    pe_temperatureW = 20,
    return_interm_indices = [1, 2, 3],
    backbone_freeze_keywords = None,
    enc_layers = 6,
    dec_layers = 6,
    pre_norm = False,
    dim_feedforward = 2048,
    hidden_dim = 256,
    dropout = 0.0,
    nheads = 8,
    num_queries = 900,
    query_dim = 4,
    num_patterns = 0,
    num_feature_levels = 4,
    enc_n_points = 4,
    dec_n_points = 4,
    two_stage_type = "standard",
    two_stage_bbox_embed_share = False,
    two_stage_class_embed_share = False,
    transformer_activation = "relu",
    dec_pred_bbox_embed_share = True,           # decoder box embedding is shared across 6 layers
    dn_box_noise_scale = 1.0,
    dn_label_noise_ratio = 0.5,
    dn_label_coef = 1.0,
    dn_bbox_coef = 1.0,
    embed_init_tgt = True,
    dn_labelbook_size = 2000,
    max_text_len = 256,
    text_encoder_type = "bert-base-uncased",
    tokenizer_type = "bert-base-uncased",
    use_text_enhancer = True,
    use_fusion_layer = True,
    use_checkpoint = True,
    use_transformer_ckpt = True,
    use_text_cross_attention = True,
    text_dropout = 0.0,
    fusion_dropout = 0.0,
    fusion_droppath = 0.1,
    sub_sentence_present = True,
)

# solver
SOLVER = dict(
    IMS_PER_BATCH = 4,
    FIND_UNUSED_PARAMETERS = False,
    USE_AMP = True,
    MAX_NEG_PER_BATCH = 0.1,
    TEST_WITH_INFERENCE = False,

    # Gradient clipping 
    CLIP_GRADIENTS_ENABLED = True,
    CLIP_GRADIENTS_TYPE = "full_model",
    CLIP_GRADIENTS_VALUE = 1.0,
    CLIP_GRADIENTS_NORM_TYPE = 2.0,

    # learning rates
    BASE_LR = 0.001,
    LANG_LR = 0.00001,
    BACKBONE_BODY_LR_FACTOR = 0.1,
    BIAS_LR_FACTOR = 2.0,
    MIN_LR = 0.000001,

    # weight decay
    WEIGHT_DECAY_SCHEDULE = False, 
    WEIGHT_DECAY_SCHEDULE_RATIO = 0.667,
    WEIGHT_DECAY = 0.0005,
    WEIGHT_DECAY_NORM_FACTOR = 1.0,
    WEIGHT_DECAY_BIAS = 1.0,

    # optimizer & warm up schedule
    MOMENTUM = 0.9,
    OPTIMIZER = "ADAMW",         # "SGD", "ADAMW"
    USE_COSINE = False,
    GAMMA = 0.1,
    WARMUP_FACTOR = 1.0 / 3,
    WARMUP_ITERS = 500,
    WARMUP_METHOD = "linear",

    # MODEL EMA
    MODEL_EMA = 0,
    USE_EMA_FOR_MONITOR = False,

    # checkpoints
    MULTI_MAX_EPOCH = (),
    MAX_EPOCH = 0,
    MULTI_MAX_ITER = (),
    MAX_ITER = 40000,
    STEPS = (30000, ),
    CHECKPOINT_PER_EPOCH = -1,
    AUTO_TERMINATE_PATIENCE = -1,
    CHECKPOINT_PERIOD = 10000,
)

TEST = dict(
    EXPECTED_RESULTS=[],
    EXPECTED_RESULTS_SIGMA_TOL=4,
    DURING_TRAINING = True,
    EVAL_TASK = "detection",
    CHUNKED_EVALUATION = -1,    # for datasets like lvis
    IMS_PER_BATCH = 8,
    USE_MULTISCALE = False,
    SUBSET = -1
)

DATASETS = dict(
    TRAIN = ("object365_dt_train", ),
    TEST = ("coco_2017_val", ),
    DISABLE_SHUFFLE = False,
    USE_CROWD = False,
    CLASS_AGNOSTIC = False,
    CLASS_CONCAT = False,
    MULTISTAGE_TRAINING = False,
    ALTERNATIVE_TRAINING = False,
    CONTROL_PROB = (0.0, 0.0, 0.5, 0.0),
    RANDOM_SAMPLE_NEG = 85,
    ADD_DET_PROMPT = False,
    SEPARATION_TOKENS = ". ",
    CAPTION_NMS = 0.9,
    SAFEGUARD_POSITIVE_CAPTION = True,
    CAPTION_FORMAT_VERSION = "v1",
    USE_CAPTION_PROMPT = False,
    CAPTION_PROMPT = None,
    USE_SUPRESS_QUERY = False,
    SUPRESS_QUERY = None
)


DATALOADER = dict(
    NUM_WORKERS = 4,
    SIZE_DIVISIBILITY = 32,
    ASPECT_RATIO_GROUPING = True,
    USE_RANDOM_SEED = False,
    DISTRIBUTE_CHUNK_AMONG_NODE = False,
)

# data
AUGMENT = dict(
    MULT_MIN_SIZE_TRAIN = (480, 560, 640, 720, 800),
    MIN_SIZE_TRAIN = 800,
    MAX_SIZE_TRAIN = 1333,
    MIN_SIZE_TEST = 800,
    MAX_SIZE_TEST = 1333,
    FLIP_PROB_TRAIN = 0.5,
    BRIGHTNESS = 0.0,
    CONTRAST = 0.0,
    SATURATION = 0.0,
    HUE = 0.0,
    CROP_PROB = 0.5,
    CROP_MIN_IOUS = (0.1, 0.3, 0.5, 0.7, 0.9),
    CROP_MIN_SIZE = 0.3,
    INPUT_TO_BGR255 = True,
    INPUT_FORMAT = '',
    INPUT_FIX_RES = False,
    PIXEL_MEAN = [0.485, 0.456, 0.406],
    PIXEL_STD = [0.229, 0.224, 0.225],
)
