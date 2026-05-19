"""Trang demo Tối ưu hàm 2D với ACO – các hàm benchmark Rastrigin/Ackley/..."""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from algorithms.common import page_header, back_button, plot_convergence, metric_row
from algorithms.func2d_aco import aco_function_opt, FUNCTIONS

st.set_page_config(page_title="11 · Function 2D ACO", page_icon="📈", layout="wide")
back_button()
page_header("📈", "Bài toán 11 – Function 2D Optimization",
            "Tìm điểm thấp nhất trên một địa hình mấp mô (Rastrigin/Ackley/...)", "ACO")

with st.expander("📖 Mô tả bài toán"):
    st.markdown("""
    **Tối ưu hàm liên tục 2D**: tìm $(x^*, y^*) = \\arg\\min f(x,y)$ trên miền $[lo, hi]^2$.

    Các hàm benchmark có nhiều cực trị địa phương:
    - **Rastrigin**: $f(x,y) = 20 + (x^2 - 10\\cos 2\\pi x) + (y^2 - 10\\cos 2\\pi y)$, min tại (0,0)
    - **Ackley, Sphere, Himmelblau**: mỗi hàm có đặc tính landscape khác nhau.

    ACO chia miền thành lưới $M \\times M$ ô, sample theo pheromone × heuristic.
    """)

st.sidebar.markdown("## ⚙️ Tham số")
func_name = st.sidebar.selectbox("Hàm mục tiêu", list(FUNCTIONS.keys()))
n_ants = st.sidebar.slider("Số kiến", 10, 100, 40)
n_iter = st.sidebar.slider("Vòng lặp", 30, 300, 100)
grid_m = st.sidebar.slider("Kích thước lưới M", 10, 60, 30)
alpha = st.sidebar.slider("Alpha", 0.1, 5.0, 1.0)
beta = st.sidebar.slider("Beta", 0.1, 5.0, 2.0)
rho = st.sidebar.slider("Rho", 0.05, 0.5, 0.1)
local_steps = st.sidebar.slider("Local search steps", 0, 50, 15)
seed = st.sidebar.number_input("Seed", 0, 9999, 42)

func, (lo, hi), (tx, ty, tf) = FUNCTIONS[func_name]
st.write(f"**Hàm:** {func_name} | **Miền:** [{lo}, {hi}]² | "
         f"**Optimum đã biết:** ({tx}, {ty}) = {tf}")

# Plot landscape
X, Y = np.meshgrid(np.linspace(lo, hi, 80), np.linspace(lo, hi, 80))
Z = func(X, Y)

fig0, ax0 = plt.subplots(figsize=(7, 5))
cs = ax0.contourf(X, Y, Z, levels=30, cmap="viridis")
plt.colorbar(cs, ax=ax0)
ax0.scatter(tx, ty, c="red", s=200, marker="*", label=f"Optimum ({tx},{ty})")
ax0.legend(); ax0.set_title(f"Landscape – {func_name}")
st.pyplot(fig0)

if st.button("🚀 Chạy ACO", type="primary"):
    progress = st.progress(0); status = st.empty()
    def cb(it, total, best):
        progress.progress(it / total)
        status.write(f"Vòng {it}/{total} – Best f = **{best:.4f}**")
    with st.spinner("Đang chạy..."):
        bx, by, bf, hist, pher, rt = aco_function_opt(
            func, lo, hi, n_ants=n_ants, n_iter=n_iter, grid_m=grid_m,
            alpha=alpha, beta=beta, rho=rho, local_steps=local_steps,
            seed=seed, progress_cb=cb)
    status.empty()
    err = abs(bf - tf)
    st.success(f"✅ Best = ({bx:.4f}, {by:.4f}) → f = {bf:.6f} (lỗi {err:.6f})")

    metric_row([
        ("x*", f"{bx:.4f}", f"true {tx}"),
        ("y*", f"{by:.4f}", f"true {ty}"),
        ("f(x*,y*)", f"{bf:.6f}", f"true {tf}"),
        ("⏱️ Runtime", f"{rt:.2f}s", None),
    ])

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🗺️ Vị trí tìm thấy")
        fig, ax = plt.subplots(figsize=(7, 5))
        cs = ax.contourf(X, Y, Z, levels=30, cmap="viridis", alpha=.85)
        plt.colorbar(cs, ax=ax)
        ax.scatter(tx, ty, c="red", s=200, marker="*", label="Optimum")
        ax.scatter(bx, by, c="cyan", s=200, marker="X", edgecolor="black",
                   label="ACO best")
        ax.legend()
        st.pyplot(fig)
    with col2:
        st.subheader("📉 Hội tụ")
        st.pyplot(plot_convergence(hist, f"Hội tụ ACO – {func_name}", "f(x,y)"))

    st.subheader("🔥 Pheromone heatmap")
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    im = ax3.imshow(pher.T, origin="lower", cmap="hot",
                    extent=[lo, hi, lo, hi], aspect="auto")
    plt.colorbar(im, ax=ax3, label="Pheromone")
    ax3.set_title("Phân bố pheromone cuối")
    st.pyplot(fig3)
