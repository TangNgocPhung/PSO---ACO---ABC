"""Trang demo bài toán VRP với ACO."""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from algorithms.common import page_header, back_button, plot_convergence, metric_row
from algorithms.vrp_aco import aco_cvrp, make_random_cvrp

st.set_page_config(page_title="02 · VRP ACO", page_icon="🚚", layout="wide")
back_button()
page_header("🚚", "Bài toán 02 – CVRP",
            "Capacitated Vehicle Routing Problem giải bằng ACO", "ACO")

with st.expander("📖 Mô tả bài toán"):
    st.markdown("""
    **CVRP** – một depot, nhiều khách hàng có nhu cầu giao hàng,
    đội xe có sức chứa cố định. Mục tiêu tối thiểu tổng quãng đường mà không xe
    nào vượt sức chứa.
    """)

st.sidebar.markdown("## ⚙️ Tham số")
n_customers = st.sidebar.slider("Số khách hàng", 5, 30, 15)
capacity = st.sidebar.slider("Capacity / xe", 30, 200, 80)
n_ants = st.sidebar.slider("Số kiến", 10, 60, 20)
n_iter = st.sidebar.slider("Vòng lặp", 20, 200, 80)
alpha = st.sidebar.slider("Alpha", 0.1, 5.0, 1.0)
beta = st.sidebar.slider("Beta", 0.1, 5.0, 2.0)
rho = st.sidebar.slider("Rho", 0.05, 0.95, 0.4)
seed = st.sidebar.number_input("Seed", 0, 9999, 42)

coords, demands, cap = make_random_cvrp(n_customers, capacity, seed)
st.write(f"**Depot:** node 0 tại ({coords[0,0]:.0f},{coords[0,1]:.0f}) – **Capacity:** {capacity}")

fig0, ax0 = plt.subplots(figsize=(7, 5))
ax0.scatter(coords[1:, 0], coords[1:, 1], c="blue", s=80, label="Khách hàng")
ax0.scatter(coords[0, 0], coords[0, 1], c="red", s=200, marker="s", label="Depot")
for i in range(1, len(coords)):
    ax0.text(coords[i, 0], coords[i, 1], f" {i}(d={demands[i]})", fontsize=7)
ax0.legend()
ax0.grid(True, alpha=.3)
ax0.set_title("Bản đồ depot & khách hàng")
st.pyplot(fig0)

if st.button("🚀 Chạy ACO-CVRP", type="primary"):
    progress = st.progress(0)
    status = st.empty()
    def cb(it, total, best):
        progress.progress(it / total)
        status.write(f"Vòng {it}/{total} – Best total: **{best:.2f}**")
    with st.spinner("Đang chạy..."):
        routes, total, hist, rt = aco_cvrp(coords, demands, capacity,
                                            n_ants=n_ants, n_iter=n_iter,
                                            alpha=alpha, beta=beta, rho=rho,
                                            seed=seed, progress_cb=cb)
    status.empty()
    st.success(f"✅ Hoàn tất sau {rt:.2f}s")

    metric_row([
        ("📏 Tổng quãng đường", f"{total:.2f}", None),
        ("🚚 Số xe", f"{len(routes)}", None),
        ("⏱️ Thời gian", f"{rt:.2f}s", None),
        ("👥 Khách hàng", f"{n_customers}", None),
    ])

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🛣️ Các tuyến đường")
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(coords[1:, 0], coords[1:, 1], c="blue", s=80)
        ax.scatter(coords[0, 0], coords[0, 1], c="red", s=200, marker="s")
        colors = plt.cm.tab10(np.linspace(0, 1, max(len(routes), 1)))
        for k, r in enumerate(routes):
            xs = [coords[0, 0]] + [coords[c, 0] for c in r] + [coords[0, 0]]
            ys = [coords[0, 1]] + [coords[c, 1] for c in r] + [coords[0, 1]]
            ax.plot(xs, ys, "-o", color=colors[k], lw=1.8,
                    label=f"Xe {k+1} (load={sum(demands[c] for c in r)})")
        for i in range(1, len(coords)):
            ax.text(coords[i, 0], coords[i, 1], f" {i}", fontsize=7)
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=.3)
        st.pyplot(fig)
    with col2:
        st.subheader("📉 Hội tụ")
        st.pyplot(plot_convergence(hist, "Hội tụ ACO-CVRP", "Tổng quãng đường"))

    with st.expander("📋 Chi tiết tuyến"):
        for k, r in enumerate(routes):
            st.write(f"**Xe {k+1}** (load={sum(demands[c] for c in r)}/{capacity}): "
                     f"0 → {' → '.join(map(str, r))} → 0")
