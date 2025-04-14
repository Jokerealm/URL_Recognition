import torch
from torch.utils.data import Dataset
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

class URLDataset(Dataset):
    def __init__(self, urls, labels, feature_extractor):
        self.feature_extractor = feature_extractor
        self.labels = [int(label == 1) for label in labels]  # 转换为0/1
        self.urls = urls  # 添加这行，保存URL列表
        
        # 提取所有特征
        features = [feature_extractor.extract_features(url) for url in urls]
        
        # 数值特征处理
        self.scaler = StandardScaler()
        numeric_data = np.array([f['numeric'] for f in features])
        self.numeric_data = self.scaler.fit_transform(numeric_data)
        
        # 类别特征处理
        self.label_encoder = LabelEncoder()
        categorical_data = [f['categorical'] for f in features]
        self.categorical_data = self.label_encoder.fit_transform(categorical_data)
        
        # URL字符处理
        self.max_url_length = max(len(url) for url in urls)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        # 获取URL字符编码 (ASCII)
        url = self.urls[idx]  # 使用self.urls访问URL
        url_chars = torch.zeros(self.max_url_length, dtype=torch.long)
        for i, c in enumerate(url[:self.max_url_length]):
            url_chars[i] = min(ord(c), 255)
        
        return {
            'url_chars': url_chars,
            'numeric': torch.FloatTensor(self.numeric_data[idx]),
            'categorical': torch.LongTensor([self.categorical_data[idx]]),
            'label': torch.FloatTensor([self.labels[idx]])
        }