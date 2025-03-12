import numpy as np
import faiss
import time

# --- Load và chuẩn hóa vector ---
vectors = np.load('../random_vectors_1M.npy')
print("Loaded vectors shape:", vectors.shape)

def normalize(vectors):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    np.clip(norms, 1e-10, None, out=norms)  # Tránh chia cho 0
    return vectors / norms

normalized_vectors = normalize(vectors)

# --- Cấu hình IVF index ---
dimension = vectors.shape[1]
nlist = 1000  # Số lượng cụm để chia
quantizer = faiss.IndexFlatIP(dimension) # faiss.IndexFlatL2(dimension)  # Sử dụng cho bước gán cụm
index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)

# Huấn luyện index
index.train(normalized_vectors)
print("Index trained")

# Thêm vector vào index
index.add(normalized_vectors)
print(f"Number of vectors in index: {index.ntotal}")

# Lưu index
faiss.write_index(index, 'ivf_index_cosine_similarity.index')
print("Index saved to 'ivf_index_cosine_similarity.index'")

# import numpy as np
# import faiss
# import time
# import psutil

# # --- Load và chuẩn hóa vector ---
# vectors = np.load('../random_vectors_1M.npy')
# print("Loaded vectors shape:", vectors.shape)

# def normalize(vectors):
#     norms = np.linalg.norm(vectors, axis=1, keepdims=True)
#     np.clip(norms, 1e-10, None, out=norms)  # Tránh chia cho 0
#     return vectors / norms

# normalized_vectors = normalize(vectors)

# # --- Cấu hình IVF index ---
# dimension = vectors.shape[1]
# nlist = 100  # Số cụm (cluster)
# quantizer = faiss.IndexFlatIP(dimension)
# index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)

# # Train trước khi add dữ liệu
# index.train(normalized_vectors)
# index.add(normalized_vectors)
# print(f"Number of vectors in index: {index.ntotal}")

# # --- Lưu index ---
# faiss.write_index(index, 'ivf_index_cosine_similarity_v1.index')

# # --- Load lại index để tìm kiếm ---
# index = faiss.read_index('ivf_index_cosine_similarity_v1.index')

# # --- Tối ưu tốc độ tìm kiếm ---
# available_ram = psutil.virtual_memory().available / (1024 ** 3)  # GB
# num_threads = min(4, int(available_ram // 0.5))
# faiss.omp_set_num_threads(max(1, num_threads))

# index.nprobe = 10  # Số cụm sẽ được kiểm tra trong quá trình tìm kiếm

# # --- Tìm kiếm vector tương đồng ---
# total_queries = 1000
# k = 10000  # Tìm kiếm top 10000 vector giống nhất với 1000 query
# lookalike_threshold = 0.8

# query_vectors = normalized_vectors[:total_queries]

# # Chia batch để tối ưu tốc độ
# batch_size = 200
# results = []

# start_time = time.time()
# for i in range(0, total_queries, batch_size):
#     batch_queries = query_vectors[i : i + batch_size]
#     distances, indices = index.search(batch_queries, k // total_queries)
#     results.append((distances, indices))

# # Gộp kết quả lại
# distances = np.concatenate([r[0] for r in results])
# indices = np.concatenate([r[1] for r in results])
# end_time = time.time()

# # --- Lọc kết quả theo ngưỡng look alike ---
# mask = distances >= lookalike_threshold
# valid_indices = indices[mask]
# valid_distances = distances[mask]

# # Sắp xếp theo độ tương đồng (giảm dần)
# sorted_order = np.argsort(-valid_distances)
# top_10k_indices = valid_indices[sorted_order][:10000]
# top_10k_distances = valid_distances[sorted_order][:10000]

# # --- Hiển thị kết quả ---
# print(f"Thời gian tìm kiếm: {end_time - start_time:.4f} giây")
# print(f"Số lượng vector có look alike >= {lookalike_threshold}: {len(valid_indices)}")
# print("Top 10,000 vector tương đồng nhất:", top_10k_indices)
# print("Độ tương đồng tương ứng:", top_10k_distances)
