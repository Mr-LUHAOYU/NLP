"""
依存分析评估模块
文件：第二章/evaluate.py
"""

import torch
from sklearn.metrics import accuracy_score
from collections import defaultdict
from torch.utils.data import DataLoader
from model import DependencyDataset, DependencyParser


def evaluate(model, dataset):
    """计算UAS和LAS"""
    dataloader = DataLoader(dataset, batch_size=128)

    total = 0
    correct_arc = 0  # 正确弧数（UAS）
    correct_rel = 0  # 正确关系数（LAS）

    # 存储前5个样本结果
    sample_results = []

    with torch.no_grad():
        for batch_idx, (features, labels) in enumerate(dataloader):
            outputs = model(features)
            _, preds = torch.max(outputs, 1)

            # 转换为numpy
            preds = preds.numpy()
            labels = labels.numpy()

            # 统计指标
            correct_arc += (preds != 0).sum()  # 假设0类表示无依存关系
            correct_rel += (preds == labels).sum()
            total += len(labels)

            # 记录前5个样本
            if batch_idx == 0 and len(sample_results) < 5:
                idx_to_rel = {v: k for k, v in dataset.rel2idx.items()}
                for i in range(min(5, len(preds))):
                    child, head, true_rel = dataset.data[i]
                    pred_rel = idx_to_rel.get(preds[i], 'UNK')
                    sample_results.append({
                        'child': child,
                        'head': head,
                        'true_rel': true_rel,
                        'pred_rel': pred_rel
                    })

    uas = correct_arc / total
    las = correct_rel / total

    print(f"UAS: {uas:.4f} | LAS: {las:.4f}")
    print("\n前5个样本预测结果：")
    for res in sample_results:
        print(
            f"子词: {res['child']:<6} 中心词: {res['head']:<6} | 真实关系: {res['true_rel']:<8} 预测关系: {res['pred_rel']}")


if __name__ == '__main__':
    checkpoint = torch.load('model.pt', weights_only=False)
    model = DependencyParser(output_dim=len(checkpoint['rel2idx']) + 1)
    model.load_state_dict(checkpoint['model_state'])
    testdata = DependencyDataset('text.test.conll', 'word_emb300.txt')
    evaluate(model, testdata)

