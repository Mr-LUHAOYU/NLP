from sentence_transformers import SentenceTransformer
import sqlite3
import numpy as np

# 连接到数据库
conn = sqlite3.connect(f"knowledge_base.db")
cursor = conn.cursor()
print('数据库连接成功')

# 创建表（如果不存在）
cursor.execute(
    """
CREATE TABLE IF NOT EXISTS embeddings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sentence TEXT NOT NULL,
    embedding BLOB NOT NULL
)
""")

# 清空表
cursor.execute("""
DELETE FROM embeddings;
""")

print('表加载成功')

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
print('模型已经成功导入')

f = open('content.txt', 'r', encoding='utf-8')
sentences = f.readlines()
f.close()
print('测试语句成功导入')

for sentence in sentences:
    embeddings = model.encode(sentence.strip())
    # 将 embeddings 转换为 bytes 以存储在 SQLite 中
    embedding_bytes = np.array(embeddings).tobytes()
    print(sentence, 'embedding shape:', embeddings.shape)

    # 插入数据
    cursor.execute(
        "INSERT INTO embeddings (sentence, embedding) VALUES (?, ?)",
        (sentence, embedding_bytes),
    )
    print(f"Stored {sentence} in the database.")

# 提交更改并关闭连接
conn.commit()
conn.close()

print("Embeddings have been successfully stored in the database.")
