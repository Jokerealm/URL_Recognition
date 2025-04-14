import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
from feature_extractor import URLFeatureExtractor
from dataset import URLDataset
from model import URLClassifier
from tqdm import tqdm  # 导入tqdm库

def train():
    # 1. 加载数据
    df = pd.read_csv('data/train.csv')
    urls, labels = df['URL'].values, df['label'].values
    
    # 2. 准备数据集
    feature_extractor = URLFeatureExtractor()
    train_urls, val_urls, train_labels, val_labels = train_test_split(urls, labels, test_size=0.2)
    
    train_dataset = URLDataset(train_urls, train_labels, feature_extractor)
    val_dataset = URLDataset(val_urls, val_labels, feature_extractor)
    
    # 3. 初始化模型
    model = URLClassifier(
        num_numeric=len(feature_extractor.numeric_features),
        num_categories=len(train_dataset.label_encoder.classes_)
    )
    
    # 4. 训练配置
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.BCELoss()
    
    # 5. 训练循环
    for epoch in range(10):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        # 添加训练进度条
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/10 [Training]', leave=False)
        
        for batch in train_bar:
            optimizer.zero_grad()
            outputs = model(batch['url_chars'], batch['numeric'], batch['categorical'])
            loss = criterion(outputs, batch['label'])
            loss.backward()
            optimizer.step()
            
            # 计算训练统计量
            train_loss += loss.item()
            predicted = (outputs > 0.5).float()
            total += batch['label'].size(0)
            correct += (predicted == batch['label']).sum().item()
            
            # 更新进度条显示
            train_bar.set_postfix({
                'loss': f"{train_loss/(train_bar.n+1):.4f}",
                'acc': f"{correct/total:.2%}"
            })
        
        # 验证循环
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
            val_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}/10 [Validation]', leave=False)
            
            for batch in val_bar:
                outputs = model(batch['url_chars'], batch['numeric'], batch['categorical'])
                loss = criterion(outputs, batch['label'])
                
                val_loss += loss.item()
                predicted = (outputs > 0.5).float()
                val_total += batch['label'].size(0)
                val_correct += (predicted == batch['label']).sum().item()
                
                val_bar.set_postfix({
                    'val_loss': f"{val_loss/(val_bar.n+1):.4f}",
                    'val_acc': f"{val_correct/val_total:.2%}"
                })
        
        # 打印每个epoch的总结
        print(f"\nEpoch {epoch+1}/10 Summary:")
        print(f"Train Loss: {train_loss/len(train_loader):.4f} | Train Acc: {correct/total:.2%}")
        print(f"Val Loss: {val_loss/len(val_loader):.4f} | Val Acc: {val_correct/val_total:.2%}")
        
        # 保存模型
        torch.save({
            'model_state_dict': model.state_dict(),
            'scaler': train_dataset.scaler,
            'label_encoder': train_dataset.label_encoder,
            'max_url_len': train_dataset.max_url_length,
            'epoch': epoch,
            'train_loss': train_loss/len(train_loader),
            'val_loss': val_loss/len(val_loader)
        }, 'models/best_model.pth')

if __name__ == '__main__':
    train()