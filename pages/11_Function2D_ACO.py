"""Trang demo Tối ưu hàm 2D với ACO – các hàm benchmark Rastrigin/Ackley/..."""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from algorithms.common import (page_header, back_button, plot_convergence, metric_row,
                                generate_pdf_report, download_pdf_button)
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
        fig1, ax = plt.subplots(figsize=(7, 5))
        cs = ax.contourf(X, Y, Z, levels=30, cmap="viridis", alpha=.85)
        plt.colorbar(cs, ax=ax)
        ax.scatter(tx, ty, c="red", s=200, marker="*", label="Optimum")
        ax.scatter(bx, by, c="cyan", s=200, marker="X", edgecolor="black",
                   label="ACO best")
        ax.legend()
        st.pyplot(fig1)
    with col2:
        st.subheader("📉 Hội tụ")
        fig2 = plot_convergence(hist, f"Hội tụ ACO – {func_name}", "f(x,y)")
        st.pyplot(fig2)

    st.subheader("🔥 Pheromone heatmap")
    from matplotlib.colors import LogNorm, PowerNorm
    scale_choice = st.radio(
        "Thang màu",
        ["Log (khuyên dùng)", "Power γ=0.3", "Tuyến tính"],
        horizontal=True,
        help="ACO hội tụ mạnh về 1 ô → thang tuyến tính làm các ô khác "
             "trông toàn đen. Dùng Log/Power để thấy rõ phân bố."
    )

    pher_disp = pher.T.copy() + 1e-3  # offset tránh log(0)
    fig3, ax3 = plt.subplots(figsize=(9, 5))
    if scale_choice.startswith("Log"):
        norm = LogNorm(vmin=max(pher_disp.min(), 1e-3), vmax=pher_disp.max())
    elif scale_choice.startswith("Power"):
        norm = PowerNorm(gamma=0.3, vmin=pher_disp.min(), vmax=pher_disp.max())
    else:
        norm = None

    im = ax3.imshow(pher_disp, origin="lower", cmap="inferno",
                    extent=[lo, hi, lo, hi], aspect="auto", norm=norm,
                    interpolation="bilinear")
    cbar = plt.colorbar(im, ax=ax3, label="Pheromone (log/power scale)")
    cbar.ax.tick_params(labelsize=8)
    # đánh dấu vị trí best & optimum
    ax3.scatter(tx, ty, c="cyan", s=180, marker="*",
                edgecolor="white", linewidth=1.2,
                label=f"Optimum ({tx},{ty})", zorder=5)
    ax3.scatter(bx, by, c="lime", s=160, marker="X",
                edgecolor="black", linewidth=1.2,
                label="ACO best", zorder=5)
    ax3.legend(loc="upper right", fontsize=9,
               facecolor="white", framealpha=.9)
    ax3.set_xlabel("x"); ax3.set_ylabel("y")
    ax3.set_title(f"Phân bố pheromone cuối ({scale_choice})",
                  color="#1e1b4b", fontweight="bold")
    st.pyplot(fig3)

    # ---------- 📄 NÚT TẢI PDF ----------
    st.markdown("---")
    st.subheader("📥 Xuất báo cáo PDF")
    pdf_bytes = generate_pdf_report(
        problem_title="Bài toán 11 – Function 2D Optimization",
        problem_subtitle=f"Tối ưu hàm {func_name} trong miền [{lo}, {hi}]²",
        algorithm="ACO",
        inputs={
            "Hàm mục tiêu": func_name,
            "Miền": f"[{lo}, {hi}]²",
            "Optimum lý thuyết": f"({tx}, {ty}) = {tf}",
            "Số kiến": n_ants,
            "Số vòng lặp": n_iter,
            "Kích thước lưới M": grid_m,
            "Alpha": alpha,
            "Beta": beta,
            "Rho": rho,
            "Local search steps": local_steps,
            "Seed": seed,
        },
        outputs={
            "x*": f"{bx:.6f}",
            "y*": f"{by:.6f}",
            "f(x*, y*)": f"{bf:.8f}",
            "Sai số": f"{abs(bf - tf):.8f}",
            "Thời gian chạy": f"{rt:.2f} giây",
            "Pheromone max": f"{pher.max():.2f}",
            "Pheromone median": f"{np.median(pher):.4f}",
        },
        figures=[
            ("Hình 0: Landscape hàm số", fig0),
            ("Hình 1: Vị trí tìm thấy", fig1),
            ("Hình 2: Đồ thị hội tụ", fig2),
            ("Hình 3: Phân bố pheromone", fig3),
        ],
    )
    download_pdf_button(pdf_bytes,
                       file_name=f"BaoCao_Function2D_ACO_{func_name}.pdf",
                       key="pdf_func2d")
