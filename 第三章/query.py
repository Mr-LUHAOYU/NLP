import sqlite3
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


def cosine_similarity(v1, v2):
    """计算两个向量之间的余弦相似度"""
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    return dot_product / (norm_v1 * norm_v2)


def search(query, top_k=1):
    # 连接到数据库
    conn = sqlite3.connect(f'knowledge_base.db')
    cursor = conn.cursor()

    # 将查询转换为向量
    query_vector = model.encode(query)

    # 获取所有存储的embeddings
    cursor.execute('SELECT sentence, embedding FROM embeddings')
    results = cursor.fetchall()

    # 计算相似度并排序
    similarities = []
    for sentence, embedding_bytes in results:
        embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
        similarity = cosine_similarity(query_vector, embedding)  # 使用余弦相似度
        similarities.append((sentence, similarity))

    # 按相似度降序排序
    similarities.sort(key=lambda x: x[1], reverse=True)

    # 关闭连接
    conn.close()

    # 返回前 top_k 个结果
    return similarities[:top_k]


if __name__ == "__main__":
    # 测试搜索功能
    while True:
        query = input('请输入查询的关键词：').strip()
        if not query: break
        results = search(query)
        print(f"Top {len(results)} results for query: '{query}'")
        for sentence, similarity in results:
            print(f"sentence: {sentence.strip()}, Similarity: {similarity:.4f}")

"""
湿度
量子纠缠
怎样泡绿茶最好？
无竞争市场
政府权力来源
"""