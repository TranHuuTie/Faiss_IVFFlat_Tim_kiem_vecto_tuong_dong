import numpy as np
import faiss
import time

# --- Load index ---
index = faiss.read_index('ivfflat_index_cosine_similarity.index')

# --- Load vectors ---
vectors = np.load('../random_vectors_1M.npy')

# --- Chuẩn hóa vectors ---
def normalize(vectors):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    np.clip(norms, 1e-10, None, out=norms)  # Tránh chia cho 0
    return vectors / norms

normalized_vectors = normalize(vectors)

# --- Tìm kiếm ---
total_queries = 10000  # Tăng số lượng vector truy vấn lên 10,000
k = 100
lookalike_threshold = 0.835
query_vectors = normalized_vectors[:total_queries]

# Số cụm kiểm tra khi tìm kiếm
nprobe = 1000
index.nprobe = nprobe

start_time = time.time()
distances, indices = index.search(query_vectors, k)
end_time = time.time()

# --- Lọc và sắp xếp kết quả ---
all_distances = distances.flatten()
all_indices = indices.flatten()

# Lọc theo ngưỡng look alike
valid_mask = all_distances >= lookalike_threshold
valid_distances = all_distances[valid_mask]
valid_indices = all_indices[valid_mask]

# Sắp xếp theo độ tương đồng (cosine similarity giảm dần)
sorted_indices = np.argsort(-valid_distances)[:100000]
top_100k_indices = valid_indices[sorted_indices]
top_100k_distances = valid_distances[sorted_indices]

all_matched_vectors = []

print(f"Thời gian tìm kiếm: {end_time - start_time:.4f} giây")
print(f"Số lượng vector có look alike >= {lookalike_threshold}: {len(valid_indices)}")

# Hiển thị các vector tương đồng
print("Kết quả tìm kiếm:")
for i, query_idx in enumerate(range(total_queries)):
    matched_vectors = [(query_idx, indices[i, j], distances[i, j]) for j in range(k) if distances[i, j] >= lookalike_threshold]
    all_matched_vectors.extend(matched_vectors)
    print(f"\nQuery Vector {query_idx} (tìm thấy {len(matched_vectors)} vector tương đồng):")
    for q_idx, idx, dist in matched_vectors:
        print(f"  -> Query Vector {q_idx} tương đồng với Vector {idx} - Độ tương đồng: {dist:.4f}")

# Sắp xếp và lấy ra 100,000 vector có độ tương đồng cao nhất
top_100k_vectors = sorted(all_matched_vectors, key=lambda x: -x[2])[:100000]

# Ghi kết quả ra file
with open('top_100k_vectors_0.84_nprobe1000_gpu.txt', 'w') as f:
    f.write("Query Vector, Matched Vector, Similarity\n")
    for q_idx, idx, dist in top_100k_vectors:
        f.write(f"{q_idx}, {idx}, {dist:.4f}\n")

print("\nTop 100,000 vector có độ tương đồng cao nhất:")
for q_idx, idx, dist in top_100k_vectors:
    print(f"Query Vector {q_idx} tương đồng với Vector {idx} - Độ tương đồng: {dist:.4f}")

print(f"Thời gian tìm kiếm: {end_time - start_time:.4f} giây")
print(f"Số lượng vector có look alike >= {lookalike_threshold}: {len(valid_indices)}")
