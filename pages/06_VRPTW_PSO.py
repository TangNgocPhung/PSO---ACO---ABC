"""Trang demo VRPTW với PSO."""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from algorithms.common import page_header, back_button, plot_convergence, metric_row
from algorithms.vrptw_pso import pso_vrptw, make_solomon_like

st.set_page_config(page_title="06 · VRPTW PSO", page_icon="⏰", layout="wide")
back_button()
page_header("⏰", "Bài toán 06 – VRPTW",
            "Vehicle Routing Problem with Time Windows – giải bằng PSO", "PSO")

with st.expander("📖 Mô tả bài toán"):
    st.markdown("""
    **VRPTW** – mở rộng của CVRP: mỗi khách hàng có khung thời gian phục vụ
    [ready, due]. Xe phải đến trong khung này (chờ nếu sớm, phạt nếu trễ).
    Mục tiêu: tối thiểu tổng quãng đường + số xe + phạt thời gian.
    """)

st.sidebar.markdown("## ⚙️ Tham số")
n_customers = st.sidebar.slider("Số khách hàng", 10, 50, 20)
capacity = st.sidebar.slider("Capacity", 50, 400, 200)
n_particles = st.sidebar.slider("Số particles", 10, 100, 30)
n_iter = st.sidebar.slider("Vòng lặp", 30, 300, 80)
seed = st.sidebar.number_input("Seed", 0, 9999, 42)

customers, cap = make_solomon_like(n_customers, capacity, seed)
depot = customers[0]
st.write(f"**Depot:** ({depot['x']:.0f},{depot['y']:.0f}) | **Capacity:** {capacity}")

fig0, ax0 = plt.subplots(figsize=(7, 5))
xs = [c["x"] for c in customers[1:]]
ys = [c["y"] for c in customers[1:]]
ax0.scatter(xs, ys, c="blue", s=80, label="Khách hàng")
ax0.scatter(depot["x"], depot["y"], c="red", s=200, marker="s", label="Depot")
for c in customers[1:]:
    ax0.text(c["x"], c["y"], f" {c['id']}", fontsize=7)
ax0.legend(); ax0.grid(True, alpha=.3)
ax0.set_title("Bản đồ depot & khách hàng (kèm khung thời gian)")
st.pyplot(fig0)

if st.button("🚀 Chạy PSO-VRPTW", type="primary"):
    progress = st.progress(0); status = st.empty()
    def cb(it, total, best):
        progress.progress(it / total)
        status.write(f"Vòng {it}/{total} – Best fit: **{best:.2f}**")
    with st.spinner("Đang chạy..."):
        routes, total_dist, n_vehicles, hist, rt = pso_vrptw(
            customers, capacity, n_particles=n_particles, n_iter=n_iter,
            seed=seed, progress_cb=cb)
    status.empty()
    st.success(f"✅ {n_vehicles} xe, tổng đường {total_dist:.2f}")

    metric_row([
        ("🚚 Số xe", f"{n_vehicles}", None),
        ("📏 Tổng đường", f"{total_dist:.2f}", None),
        ("⏱️ Thời gian chạy", f"{rt:.2f}s", None),
        ("👥 Khách hàng", f"{n_customers}", None),
    ])

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🛣️ Tuyến đường")
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(xs, ys, c="blue", s=80)
        ax.scatter(depot["x"], depot["y"], c="red", s=200, marker="s")
        colors = plt.cm.tab10(np.linspace(0, 1, max(len(routes), 1)))
        for k, r in enumerate(routes):
            rx = [depot["x"]] + [c["x"] for c in r] + [depot["x"]]
            ry = [depot["y"]] + [c["y"] for c in r] + [depot["y"]]
            ax.plot(rx, ry, "-o", color=colors[k], lw=1.8,
                    label=f"Xe {k+1} (load={sum(c['demand'] for c in r)})")
        ax.legend(loc="best", fontsize=8); ax.grid(True, alpha=.3)
        st.pyplot(fig)
    with col2:
        st.subheader("📉 Hội tụ")
        st.pyplot(plot_convergence(hist, "Hội tụ PSO-VRPTW", "Fitness"))

    with st.expander("📋 Chi tiết tuyến + thời gian"):
        for k, r in enumerate(routes):
            seq = [str(c["id"]) for c in r]
            st.write(f"**Xe {k+1}:** depot → {' → '.join(seq)} → depot "
                     f"(load={sum(c['demand'] for c in r)})")
