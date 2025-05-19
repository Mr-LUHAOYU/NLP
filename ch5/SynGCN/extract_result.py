import numpy as np
import pandas as pd


result = {
    'loss': [],
    'acc': [],
    'test_acc': [],
    'f1': [],
}
cur = []
with open('log/Restaurants.log', 'r', encoding='utf') as f:
    for s in f.readlines() + ['epoch']:
        if s[:5] == 'epoch':
            if cur:
                cur = np.matrix(cur)
                cur = np.mean(cur, axis=0)
                # print(cur)
                result['loss'].append(cur[0, 0])
                result['acc'].append(cur[0, 1])
                result['test_acc'].append(cur[0, 2])
                result['f1'].append(cur[0, 3])
            cur = []

        if s[:4] == 'loss':
            parts = s.split(", ")
            numbers = [float(p.split(": ")[1]) for p in parts]
            cur.append(numbers)

pd.DataFrame(result).to_csv('result.csv')