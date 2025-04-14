import torch
import numpy as np
from feature_extractor import URLFeatureExtractor
from model import URLClassifier
import os

class URLPredictor:
    def __init__(self, model_path=None):
        self.feature_extractor = URLFeatureExtractor()
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.max_url_len = None
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
            
        checkpoint = torch.load(model_path)
        
        self.model = URLClassifier(
            num_numeric=len(self.feature_extractor.numeric_features),
            num_categories=len(checkpoint['label_encoder'].classes_)
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        self.scaler = checkpoint['scaler']
        self.label_encoder = checkpoint['label_encoder']
        self.max_url_len = checkpoint['max_url_len']
    
    def predict(self, url):
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # 特征提取
        features = self.feature_extractor.extract_features(url)
        
        # 预处理数值特征
        numeric = torch.FloatTensor(
            self.scaler.transform([features['numeric']])
        )
        
        # 预处理类别特征
        categorical = torch.LongTensor(
            self.label_encoder.transform([features['categorical']])
        )
        
        # URL编码
        url_chars = torch.zeros(self.max_url_len, dtype=torch.long)
        for i, c in enumerate(url[:self.max_url_len]):
            url_chars[i] = min(ord(c), 255)
        url_chars = url_chars.unsqueeze(0)
        
        # 预测
        with torch.no_grad():
            prob = self.model(url_chars, numeric, categorical)
            confidence = prob.item()
            prediction = "合法" if confidence > 0.5 else "不合法"
            
        return {
            'prediction': prediction,
            'confidence': confidence,
            'is_legal': confidence > 0.5
        }

# 使用示例
if __name__ == '__main__':
    predictor = URLPredictor()
    print(predictor.predict("https://www.baidu.com"))