import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                             QWidget, QLabel, QLineEdit, QPushButton, 
                             QFileDialog, QMessageBox, QGroupBox)
from PyQt5.QtCore import Qt
from predictor import URLPredictor

class URLClassifierGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.predictor = None
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle('URL合法性分类器')
        self.setGeometry(300, 300, 500, 300)
        
        # 主布局
        main_widget = QWidget()
        main_layout = QVBoxLayout()
        
        # 模型选择部分
        model_group = QGroupBox("模型选择")
        model_layout = QVBoxLayout()
        
        self.model_path_label = QLabel("未选择模型")
        self.model_path_label.setWordWrap(True)
        
        browse_btn = QPushButton("选择模型文件")
        browse_btn.clicked.connect(self.browse_model)
        
        model_layout.addWidget(self.model_path_label)
        model_layout.addWidget(browse_btn)
        model_group.setLayout(model_layout)
        
        # URL输入部分
        url_group = QGroupBox("URL分类")
        url_layout = QVBoxLayout()
        
        self.url_input = QLineEdit()
        self.url_input.setPlaceholderText("输入要分类的URL...")
        
        predict_btn = QPushButton("分类")
        predict_btn.clicked.connect(self.predict_url)
        
        self.result_label = QLabel("结果: ")
        self.confidence_label = QLabel("置信度: ")
        
        url_layout.addWidget(self.url_input)
        url_layout.addWidget(predict_btn)
        url_layout.addWidget(self.result_label)
        url_layout.addWidget(self.confidence_label)
        url_group.setLayout(url_layout)
        
        # 添加到主布局
        main_layout.addWidget(model_group)
        main_layout.addWidget(url_group)
        
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
        
    def browse_model(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择模型文件", "", "PyTorch模型 (*.pth);;所有文件 (*)", options=options)
            
        if file_path:
            try:
                self.predictor = URLPredictor(file_path)
                self.model_path_label.setText(f"已加载模型: {file_path}")
                QMessageBox.information(self, "成功", "模型加载成功！")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"加载模型失败: {str(e)}")
    
    def predict_url(self):
        if not self.predictor:
            QMessageBox.warning(self, "警告", "请先选择模型文件！")
            return
            
        url = "https://" + self.url_input.text().strip()
        print(url)
        if not url:
            QMessageBox.warning(self, "警告", "请输入URL！")
            return
            
        try:
            result = self.predictor.predict(url)
            
            # 设置结果样式
            if result['is_legal']:
                self.result_label.setStyleSheet("color: green; font-weight: bold;")
            else:
                self.result_label.setStyleSheet("color: red; font-weight: bold;")
                
            self.result_label.setText(f"结果: {result['prediction']}")
            
            # 格式化置信度
            confidence_percent = result['confidence'] * 100 if result['is_legal'] else (1 - result['confidence']) * 100
            self.confidence_label.setText(f"置信度: {confidence_percent:.2f}%")
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"预测失败: {str(e)}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = URLClassifierGUI()
    gui.show()
    sys.exit(app.exec_())