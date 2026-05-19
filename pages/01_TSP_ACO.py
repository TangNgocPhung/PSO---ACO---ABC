"""Trang demo bài toán TSP với ACO."""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from algorithms.common import page_header, back_button, plot_convergence, metric_row
from algorithms.tsp_aco import aco_tsp, SAMPLE_DATASETS, random_cities

st.set_page_config(page_title="01 · TSP ACO", page_icon="🗺️", layout="wide")
back_button()
page_header("🗺️", "Bài toán 01 – TSP",
            "Traveling Salesman Problem giải bằng Ant Colony Optimization", "ACO")

with st.expander("📖 Mô tả bài toán", expanded=False):
    st.markdown("""
    **Bài toán Người du lịch (TSP)** – tìm chu trình ngắn nhất đi qua tất cả thành phố.
    Đây là bài toán NP-Hard kinh điển.

    **Thuật toán ACO:**
    - Mỗi con kiến xây dựng 1 chu trình theo xác suất:
      $P_{ij} = \\frac{\\tau_{ij}^\\alpha \\cdot \\eta_{ij}^\\beta}{\\sum_k \\tau_{ik}^\\alpha \\cdot \\eta_{ik}^\\beta}$
    - $\\tau$ = pheromone, $\\eta = 1/d$ (heuristic). Sau mỗi vòng, pheromone bay hơi
    rồi được tăng cường theo độ tốt của lời giải.
    """)

# -------- Sidebar params --------
st.sidebar.markdown("## ⚙️ Tham số")
dataset_name = st.sidebar.selectbox("📚 Dataset", list(SAMPLE_DATASETS.keys()))
if dataset_name == "Ngẫu nhiên (20)":
    coords = random_cities(20)
elif dataset_name == "Ngẫu nhiên (50)":
    coords = random_cities(50)
else:
    coords = SAMPLE_DATASETS[dataset_name]

n_ants = st.sidebar.slider("Số kiến (n_ants)", 10, 100, 30)
n_iter = st.sidebar.slider("Số vòng lặp", 10, 300, 80)
alpha = st.sidebar.slider("Alpha (pheromone weight)", 0.1, 5.0, 1.0)
beta = st.sidebar.slider("Beta (heuristic weight)", 0.1, 5.0, 2.0)
rho = st.sidebar.slider("Rho (bay hơi)", 0.05, 0.95, 0.5)
seed = st.sidebar.number_input("Random seed", 0, 9999, 42)

st.write(f"**Số thành phố:** {len(coords)}")
fig0, ax0 = plt.subplots(figsize=(7, 5))
ax0.scatter(coords[:, 0], coords[:, 1], c="red", s=60, zorder=3)
for i, (x, y) in enumerate(coords):
    ax0.text(x, y, f" {i}", fontsize=8)
ax0.set_title("Bản đồ thành phố")
ax0.grid(True, alpha=.3)
st.pyplot(fig0)

if st.button("🚀 Chạy ACO-TSP", type="primary"):
    progress = st.progress(0)
    status = st.empty()
    def cb(it, total, best):
        progress.progress(it / total)
        status.write(f"Vòng {it}/{total} – Best distance: **{best:.2f}**")
    with st.spinner("Đang chạy..."):
        path, dist, hist, rt = aco_tsp(coords, n_ants=n_ants, n_iter=n_iter,
                                       alpha=alpha, beta=beta, rho=rho,
                                       seed=seed, progress_cb=cb)
    status.empty()
    st.success(f"✅ Hoàn tất sau {rt:.2f}s")

    metric_row([
        ("📏 Tổng quãng đường", f"{dist:.2f}", "Khoảng cách của chu trình tốt nhất"),
        ("⏱️ Thời gian", f"{rt:.2f}s", "Thời gian chạy"),
        ("🐜 Số kiến", f"{n_ants}", None),
        ("🔁 Vòng lặp", f"{n_iter}", None),
    ])

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🛣️ Chu trình tối ưu")
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(coords[:, 0], coords[:, 1], c="red", s=60, zorder=3)
        order = path + [path[0]]
        ax.plot(coords[order, 0], coords[order, 1], "g--", lw=1.5, alpha=.8)
        for i, (x, y) in enumerate(coords):
            ax.text(x, y, f" {i}", fontsize=8)
        ax.set_title(f"Chu trình – {dist:.2f}")
        ax.grid(True, alpha=.3)
        st.pyplot(fig)
    with col2:
        st.subheader("📉 Hội tụ")
        st.pyplot(plot_convergence(hist, "Hội tụ TSP-ACO", "Best Distance"))

    with st.expander("📋 Chi tiết lời giải"):
        st.code(" → ".join(map(str, path + [path[0]])))
