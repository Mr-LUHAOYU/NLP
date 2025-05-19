"""
依存分析前馈网络完整实现
文件：第二章/model.py
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np


class DependencyDataset(Dataset):
    def __init__(self, data_path, emb_path, max_len=100):
        self.embeddings = self.load_embeddings(emb_path)
        self.data = self.process_data(data_path)
        self.max_len = max_len
        self.rel2idx = self.build_rel2idx(data_path)

    def build_rel2idx(self, conll_path):
        """从CoNLL文件构建依存关系标签字典"""
        rel_set = set()

        with open(conll_path, 'r', encoding='utf-8') as f:
            for line in f:
                # 跳过注释行和空行
                if line.startswith('#') or line.strip() == '':
                    continue

                parts = line.strip().split()
                if len(parts) >= 8:  # 确保有足够列
                    deprel = parts[7]  # 第8列为依存关系
                    rel_set.add(deprel)

        # 排序并创建字典（按字母顺序）
        sorted_rels = sorted(rel_set)
        return {rel: idx for idx, rel in enumerate(sorted_rels)}

    def load_embeddings(self, path):
        """加载预训练词向量"""
        embeddings = {}
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                word = parts[0]
                vec = np.array([float(x) for x in parts[1:301]])
                embeddings[word] = vec
        return embeddings

    def process_data(self, data_path):
        """处理CoNLL格式数据"""
        processed = []
        with open(data_path, 'r', encoding='utf-8') as f:
            sentence = []
            for line in f:
                if line.startswith('#'):
                    continue
                if line == '\n':
                    # 处理完整句子
                    for i in range(len(sentence)):
                        head_idx = int(sentence[i][6]) - 1
                        if head_idx >= 0:
                            dep_rel = sentence[i][7]
                            processed.append((
                                sentence[i][1],  # 当前词
                                sentence[head_idx][1],  # 中心词
                                dep_rel
                            ))
                    sentence = []
                else:
                    parts = line.strip().split('\t')
                    sentence.append(parts)
        return processed

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        child, head, rel = self.data[idx]

        # 获取词向量（处理OOV）
        child_vec = self.embeddings.get(child, np.zeros(300))
        head_vec = self.embeddings.get(head, np.zeros(300))

        features = np.concatenate([child_vec, head_vec])
        label = self.rel2idx.get(rel, len(self.rel2idx))  # 未知关系归为最后一类

        return torch.FloatTensor(features), torch.tensor(label)


class DependencyParser(nn.Module):
    def __init__(self, input_dim=600, hidden_dim1=512, hidden_dim2=256, output_dim=61):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, output_dim),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.net(x)


def train(BATCH_SIZE, EPOCHS, LR):
    # 初始化
    dataset = DependencyDataset('text.train.conll', 'word_emb300.txt')
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    # print(len(dataset.rel2idx) + 1)
    model = DependencyParser(output_dim=len(dataset.rel2idx) + 1)  # +1 for unknown
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # 训练循环
    for epoch in range(EPOCHS):
        total_loss = 0
        correct = 0
        total = 0

        for features, labels in dataloader:
            optimizer.zero_grad()

            outputs = model(features)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f'Epoch {epoch + 1}/{EPOCHS}')
        print(f'Loss: {total_loss / len(dataloader):.4f} | Acc: {correct / total:.4f}')

    print('正在保存参数到 model.pt')

    torch.save({
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'rel2idx': dataset.rel2idx
    }, 'model.pt')

    print('参数保存完成')


if __name__ == '__main__':
    train(64, 10, 1e-3)
