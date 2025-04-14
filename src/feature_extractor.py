import re
from urllib.parse import urlparse
import tldextract

class URLFeatureExtractor:
    def __init__(self):
        self.numeric_features = [
            'URLLength', 'DomainLength', 'IsDomainIP', 
            'NoOfSubDomain', 'HasObfuscation', 'NoOfObfuscatedChar',
            'ObfuscationRatio', 'IsHTTPS'
        ]
        self.categorical_features = ['TLD']
    
    def extract_features(self, url):
        features = {}
        
        # 基础URL特征
        features['URLLength'] = len(url)
        
        # 解析URL
        parsed = urlparse(url)
        ext = tldextract.extract(url)
        
        # 域名特征
        domain = f"{ext.domain}.{ext.suffix}" if ext.suffix else ext.domain
        features['Domain'] = domain
        features['DomainLength'] = len(domain)
        features['IsDomainIP'] = int(bool(re.match(r'^\d+\.\d+\.\d+\.\d+$', domain)))
        features['TLD'] = ext.suffix if ext.suffix else ''
        
        # 子域名特征
        subdomains = ext.subdomain.split('.') if ext.subdomain else []
        features['NoOfSubDomain'] = len(subdomains)
        
        # 混淆特征
        obfuscation_chars = ['@', '//', '..']
        features['HasObfuscation'] = int(any(c in url for c in obfuscation_chars))
        features['NoOfObfuscatedChar'] = sum(url.count(c) for c in obfuscation_chars)
        features['ObfuscationRatio'] = features['NoOfObfuscatedChar'] / max(1, len(url))
        
        # HTTPS特征
        features['IsHTTPS'] = int(parsed.scheme == 'https')
        
        return {
            'numeric': [features[k] for k in self.numeric_features],
            'categorical': features['TLD'],
            'url': url
        }