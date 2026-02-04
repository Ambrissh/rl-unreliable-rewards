class Config:
    # Environment
    ENV_NAME = 'CartPole-v1'
    
    # Training
    BATCH_SIZE = 128
    GAMMA = 0.99
    EPS_START = 0.9
    EPS_END = 0.01
    EPS_DECAY = 2500
    TAU = 0.005
    LR = 3e-4
    NUM_EPISODES = 600
    
    # Model
    HIDDEN_SIZE = 128
    
    # Checkpoint
    CHECKPOINT_DIR = 'checkpoints'
    SAVE_FREQUENCY = 50
    
    # Seed
    SEED = 42
