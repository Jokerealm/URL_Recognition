class Config:
    # 数据配置
    DATA_PATH = 'data/train.csv'
    TEST_DATA_PATH = 'data/test.csv'
    
    # 模型配置
    EMBED_DIM = 16
    NUM_EPOCHS = 20
    BATCH_SIZE = 32
    
    # 训练配置
    LEARNING_RATE = 0.001
    EARLY_STOPPING_PATIENCE = 5
    
    # 路径配置
    MODEL_SAVE_PATH = 'models/best_model.pth'