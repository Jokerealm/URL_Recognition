import torch
import torch.nn as nn

class URLClassifier(nn.Module):
    def __init__(self, num_numeric, num_categories, embed_dim=16, max_url_len=100):
        super().__init__()
        
        # URL文本处理
        self.url_embed = nn.Embedding(256, 8)  # ASCII字符嵌入
        self.url_cnn = nn.Sequential(
            nn.Conv1d(8, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
        
        # 结构化特征处理
        self.num_fc = nn.Sequential(
            nn.Linear(num_numeric, 32),
            nn.ReLU()
        )
        self.cat_embed = nn.Embedding(num_categories, embed_dim)
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(64 + 32 + embed_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, url_chars, numeric, categorical):
        # URL处理
        url_emb = self.url_embed(url_chars).permute(0, 2, 1)  # [batch, channels, seq_len]
        url_feat = self.url_cnn(url_emb)
        
        # 数值特征
        num_feat = self.num_fc(numeric)
        
        # 类别特征
        cat_feat = self.cat_embed(categorical).squeeze(1)
        
        # 合并特征
        combined = torch.cat([url_feat, num_feat, cat_feat], dim=1)
        return self.classifier(combined)