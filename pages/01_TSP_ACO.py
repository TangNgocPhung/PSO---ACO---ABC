"""Trang demo bài toán TSP với ACO."""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from algorithms.common import (page_header, back_button, plot_convergence, metric_row,
                                generate_pdf_report, download_pdf_button)
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
fig0, ax0 = plt.subplots(figsize=(8, 6))
ax0.scatter(coords[:, 0], coords[:, 1], c="#ef4444", s=90, zorder=3,
            edgecolor="white", linewidth=1.5)
bbox_style = dict(boxstyle="round,pad=0.2", fc="white",
                  ec="#94a3b8", lw=0.5, alpha=0.9)
# Chỉ hiển thị nhãn cho ≤ 30 thành phố để không bị rối
if len(coords) <= 30:
    for i, (x, y) in enumerate(coords):
        ax0.annotate(f"{i}", (x, y), xytext=(7, 5),
                     textcoords="offset points",
                     fontsize=8, fontweight="700", color="#1e1b4b",
                     bbox=bbox_style, zorder=5)
ax0.set_title("Bản đồ thành phố", color="#1e1b4b", fontweight="bold")
ax0.grid(True, alpha=.3)
ax0.margins(0.1)
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

    with st.expander("📖 Giải thích kết quả", expanded=False):
        st.markdown(f"""
        ### 🎯 Các chỉ số đầu ra

        - **📏 Tổng quãng đường = `{dist:.2f}`**
          Đây là **tổng độ dài Euclid** của chu trình tối ưu mà đàn kiến tìm được —
          đi qua **tất cả {len(coords)} thành phố đúng 1 lần** rồi quay về điểm xuất phát.
          Càng nhỏ càng tốt (vì TSP là bài toán tối thiểu hoá).

        - **⏱️ Thời gian = `{rt:.2f} giây`** — Tổng thời gian CPU để chạy `{n_iter}` vòng lặp
          × `{n_ants}` con kiến. Phụ thuộc vào kích thước bài toán.

        - **🐜 Số kiến / 🔁 Vòng lặp** — Tham số đầu vào của ACO. Nhiều kiến/vòng lặp thường
          cho lời giải tốt hơn nhưng tốn thời gian hơn.

        ### 🖼️ Các hình minh hoạ

        - **🛣️ Chu trình tối ưu** — đường nét xanh lá kết nối các thành phố theo thứ tự
          mà con kiến tốt nhất đi. Các nhãn số trong khung trắng = chỉ số thành phố
          (đã ẩn nếu > 30 để tránh rối).

        - **📉 Đồ thị hội tụ** — đường cong **best distance theo từng vòng lặp**.
          Đường giảm nhanh ở các vòng đầu, sau đó **plateau** (đi ngang)
          → dấu hiệu thuật toán đã hội tụ. Nếu còn dao động nhiều → tăng `n_iter`
          hoặc giảm `rho` để pheromone không bay hơi quá nhanh.

        ### 💡 Diễn giải hành vi thuật toán

        Ở vòng 1, kiến đi gần như ngẫu nhiên (pheromone đồng đều) →
        khoảng đường ~`{hist[0]:.0f}`. Theo thời gian, các cạnh tốt được kiến đi nhiều
        → pheromone trên đó tăng → kiến sau theo dấu pheromone → hội tụ về lời giải
        cuối ~`{hist[-1]:.0f}` (giảm **{(hist[0]-hist[-1])/hist[0]*100:.1f}%**).
        """)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🛣️ Chu trình tối ưu")
        fig1, ax = plt.subplots(figsize=(8, 6.5))
        order = path + [path[0]]
        ax.plot(coords[order, 0], coords[order, 1], "-", color="#10b981",
                lw=2, alpha=.85, zorder=2)
        ax.scatter(coords[:, 0], coords[:, 1], c="#ef4444", s=90, zorder=3,
                   edgecolor="white", linewidth=1.5)
        bbox2 = dict(boxstyle="round,pad=0.2", fc="white",
                     ec="#94a3b8", lw=0.5, alpha=0.9)
        if len(coords) <= 30:
            for i, (x, y) in enumerate(coords):
                ax.annotate(f"{i}", (x, y), xytext=(7, 5),
                            textcoords="offset points",
                            fontsize=8, fontweight="700", color="#1e1b4b",
                            bbox=bbox2, zorder=5)
        ax.set_title(f"Chu trình – {dist:.2f}",
                     color="#1e1b4b", fontweight="bold")
        ax.grid(True, alpha=.3)
        ax.margins(0.1)
        st.pyplot(fig1)
    with col2:
        st.subheader("📉 Hội tụ")
        fig2 = plot_convergence(hist, "Hội tụ TSP-ACO", "Best Distance")
        st.pyplot(fig2)

    with st.expander("📋 Chi tiết lời giải"):
        st.code(" → ".join(map(str, path + [path[0]])))

    # ---------- 📄 NÚT TẢI PDF ----------
    st.markdown("---")
    st.subheader("📥 Xuất báo cáo PDF")
    pdf_bytes = generate_pdf_report(
        problem_title="Bài toán 01 – TSP",
        problem_subtitle="Traveling Salesman Problem giải bằng Ant Colony Optimization",
        algorithm="ACO",
        inputs={
            "Dataset": dataset_name,
            "Số thành phố": len(coords),
            "Số kiến (n_ants)": n_ants,
            "Số vòng lặp": n_iter,
            "Alpha (pheromone)": alpha,
            "Beta (heuristic)": beta,
            "Rho (bay hơi)": rho,
            "Random seed": seed,
        },
        outputs={
            "Tổng quãng đường": f"{dist:.2f}",
            "Thời gian chạy": f"{rt:.2f} giây",
            "Chu trình tốt nhất": " → ".join(map(str, path[:10])) + (" → ..." if len(path) > 10 else ""),
        },
        figures=[
            ("Hình 0: Bản đồ thành phố (input)", fig0),
            ("Hình 1: Chu trình tối ưu", fig1),
            ("Hình 2: Đồ thị hội tụ", fig2),
        ],
    )
    download_pdf_button(pdf_bytes,
                       file_name=f"BaoCao_TSP_ACO_n{len(coords)}.pdf",
                       key="pdf_tsp")
