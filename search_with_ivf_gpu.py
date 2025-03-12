import numpy as np 
import faiss
import time

# --- Chuyển FAISS sang chế độ GPU ---
print("Đang khởi tạo GPU resources và giới hạn bộ nhớ...")
res = faiss.StandardGpuResources()
res.setTempMemory(512 * 1024 * 1024)  # Giới hạn tạm thời bộ nhớ GPU còn 512MB
res.setPinnedMemory(512 * 1024 * 1024)  # Dùng pinned memory để tối ưu chuyển dữ liệu GPU

# --- Load index và chuyển sang GPU ---
print("Đang load index và chuyển sang GPU...")
index = faiss.read_index('ivfflat_index_cosine_similarity.index')
index = faiss.index_cpu_to_gpu(res, 0, index)

# --- Load vectors ---
print("Đang load vectors...")
vectors = np.load('../random_vectors_1M.npy')

# --- Chuẩn hóa vectors ---
def normalize(vectors):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    np.clip(norms, 1e-10, None, out=norms)
    return vectors / norms

normalized_vectors = normalize(vectors)

# --- Cấu hình tìm kiếm ---
total_queries = 10000
k = 200
lookalike_threshold = 0.836
query_vectors = normalized_vectors[:total_queries]

# quét toàn bộ 1000 cụm
nprobe = 1000
index.nprobe = nprobe

# --- Tối ưu batch size ---
print("Chia nhỏ batch size để giảm tải bộ nhớ...")
batch_size = 2000  # Chia thành 5 batch (2000x5 = 10000)
all_distances = []
all_indices = []

# --- Tìm kiếm trên GPU ---
print("Bắt đầu tìm kiếm trên GPU...")
start_time = time.time()

for i in range(0, total_queries, batch_size):
    batch_queries = query_vectors[i:i + batch_size]
    distances, indices = index.search(batch_queries, k)
    all_distances.append(distances)
    all_indices.append(indices)

# Gộp kết quả
distances = np.vstack(all_distances)
indices = np.vstack(all_indices)

end_time = time.time()

# --- Lọc và sắp xếp kết quả ---
print("Đang lọc và sắp xếp kết quả...")
all_distances = distances.flatten()
all_indices = indices.flatten()

valid_mask = all_distances >= lookalike_threshold
valid_distances = all_distances[valid_mask]
valid_indices = all_indices[valid_mask]

sorted_indices = np.argsort(-valid_distances)[:100000]
top_100k_indices = valid_indices[sorted_indices]
top_100k_distances = valid_distances[sorted_indices]

# Ghi kết quả
print(f"Thời gian tìm kiếm trên GPU: {end_time - start_time:.4f} giây")
print(f"Số lượng vector có look alike >= {lookalike_threshold}: {len(valid_indices)}")

with open('top_100k_vectors_0.835_nprobe500_GPU.txt', 'w') as f:
    f.write("Query Vector, Matched Vector, Similarity\n")
    for i in range(total_queries):
        matched_vectors = [(i, indices[i, j], distances[i, j]) for j in range(k) if distances[i, j] >= lookalike_threshold]
        for q_idx, idx, dist in matched_vectors:
            f.write(f"{q_idx}, {idx}, {dist:.4f}\n")

print(f"\nĐã ghi kết quả vào 'top_100k_vectors_0.835_nprobe1000_GPU.txt'")
print(f"Tổng số vector tìm thấy: {len(valid_distances)}")


# mục tiêu là dùng GPU để tăng tốc và giữ nguyên nprobe = 1000 và total_queries = 10,000. -> giải quyết lỗi out of memory mà vẫn tận dụng tối đa GPU. 🚀

# tối ưu theo hướng:

# Giảm bộ nhớ tạm của FAISS nhưng vẫn đảm bảo GPU acceleration.
# Dùng "pinned memory" để tối ưu luồng dữ liệu giữa RAM và VRAM.
# Chia batch thông minh: vẫn giữ 10,000 truy vấn nhưng chia nhỏ xử lý từng phần trong GPU.

# res.setTempMemory():

# Giới hạn bộ nhớ GPU cho FAISS.
# Ban đầu set 512MB, 
# res.setPinnedMemory():

# Cho phép dùng "pinned memory" để tối ưu dữ liệu giữa RAM và VRAM, giảm gánh nặng bộ nhớ GPU.
# Batch size:

# Thay vì xử lý toàn bộ 10,000 truy vấn một lần, chia thành 5 batch (2000 mỗi batch).
# Duy trì tốc độ tìm kiếm nhanh nhưng hạn chế "out of memory".


# code cũ chưa tối ưu nên lỗi OOM
# import numpy as np
# import faiss
# import time

# # # --- Chuyển FAISS sang chế độ GPU --- 
# # print("Đang khởi tạo GPU resources...")
# # res = faiss.StandardGpuResources()

# # --- Chuyển FAISS sang chế độ GPU với giới hạn bộ nhớ ---
# print("Đang khởi tạo GPU resources và giới hạn bộ nhớ...")
# res = faiss.StandardGpuResources()
# res.setTempMemory(2 * 1024 * 1024 * 1024)  # Giới hạn bộ nhớ GPU còn 2GB


# # --- Load index và chuyển sang GPU ---
# print("Đang load index và chuyển sang GPU...")
# index = faiss.read_index('ivfflat_index_cosine_similarity.index')
# index = faiss.index_cpu_to_gpu(res, 0, index)

# # --- Load vectors ---
# print("Đang load vectors...")
# vectors = np.load('../random_vectors_1M.npy')

# # --- Chuẩn hóa vectors ---
# def normalize(vectors):
#     norms = np.linalg.norm(vectors, axis=1, keepdims=True)
#     np.clip(norms, 1e-10, None, out=norms)  # Tránh chia cho 0
#     return vectors / norms

# normalized_vectors = normalize(vectors)

# # --- Cấu hình tìm kiếm ---
# total_queries = 10000  # Tăng số lượng vector truy vấn lên 10,000
# k = 100
# lookalike_threshold = 0.84
# query_vectors = normalized_vectors[:total_queries]

# # Số cụm kiểm tra khi tìm kiếm
# nprobe = 1000
# index.nprobe = nprobe

# # --- Tìm kiếm trên GPU ---
# print("Bắt đầu tìm kiếm trên GPU...")
# start_time = time.time()
# distances, indices = index.search(query_vectors, k)
# end_time = time.time()

# # --- Lọc và sắp xếp kết quả ---
# print("Đang lọc và sắp xếp kết quả...")
# all_distances = distances.flatten()
# all_indices = indices.flatten()

# # Lọc theo ngưỡng look alike
# valid_mask = all_distances >= lookalike_threshold
# valid_distances = all_distances[valid_mask]
# valid_indices = all_indices[valid_mask]

# # Sắp xếp theo độ tương đồng (cosine similarity giảm dần)
# sorted_indices = np.argsort(-valid_distances)[:100000]
# top_100k_indices = valid_indices[sorted_indices]
# top_100k_distances = valid_distances[sorted_indices]

# all_matched_vectors = []

# # Hiển thị thời gian và kết quả
# print(f"Thời gian tìm kiếm trên GPU: {end_time - start_time:.4f} giây")
# print(f"Số lượng vector có look alike >= {lookalike_threshold}: {len(valid_indices)}")

# # Ghi kết quả ra file
# with open('top_100k_vectors_0.84_nprobe1000_GPU.txt', 'w') as f:
#     f.write("Query Vector, Matched Vector, Similarity\n")
#     for i in range(total_queries):
#         matched_vectors = [(i, indices[i, j], distances[i, j]) for j in range(k) if distances[i, j] >= lookalike_threshold]
#         all_matched_vectors.extend(matched_vectors)
#         for q_idx, idx, dist in matched_vectors:
#             f.write(f"{q_idx}, {idx}, {dist:.4f}\n")

# print(f"\nĐã ghi kết quả vào 'top_100k_vectors_0.84_nprobe1000_GPU.txt'")

# print(f"Tổng số vector tìm thấy: {len(all_matched_vectors)}")
