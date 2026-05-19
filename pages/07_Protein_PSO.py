"""Trang demo Protein Folding (HP Model 2D) với PSO."""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from algorithms.common import page_header, back_button, plot_convergence, metric_row
from algorithms.protein_pso import pso_protein, DATASETS, decode, fitness

st.set_page_config(page_title="07 · Protein PSO", page_icon="🧬", layout="wide")
back_button()
page_header("🧬", "Bài toán 07 – Protein Folding",
            "HP Model 2D – dự đoán cấu trúc gấp cuộn protein bằng PSO", "PSO")

with st.expander("📖 Mô tả bài toán"):
    st.markdown("""
    **HP Model** – chuỗi amino acid được mã hoá nhị phân:
    - **H** = hydrophobic (kỵ nước)
    - **P** = polar (phân cực)

    Mục tiêu: tìm cách gấp chuỗi trên lưới 2D sao cho **số cặp H-H không liền kề chuỗi
    nhưng liền kề trên lưới** là nhiều nhất (năng lượng càng âm càng tốt).
    """)

st.sidebar.markdown("## ⚙️ Tham số")
seq_name = st.sidebar.selectbox("Chuỗi protein", list(DATASETS.keys()))
n_particles = st.sidebar.slider("Số particles", 20, 200, 80)
n_iter = st.sidebar.slider("Vòng lặp", 50, 500, 150)
penalty = st.sidebar.slider("Penalty (chồng chéo)", 10, 100, 40)
seed = st.sidebar.number_input("Seed", 0, 9999, 42)

seq, opt_energy = DATASETS[seq_name]
st.write(f"**Chuỗi:** `{seq}` | **Độ dài:** {len(seq)} | **Năng lượng tối ưu đã biết:** {opt_energy}")

if st.button("🚀 Chạy PSO-Protein", type="primary"):
    progress = st.progress(0); status = st.empty()
    def cb(it, total, best):
        progress.progress(it / total)
        status.write(f"Vòng {it}/{total} – fitness: **{best:.2f}**")
    with st.spinner("Đang chạy..."):
        gbest, fit, hist_f, hist_e, rt = pso_protein(
            seq, n_particles=n_particles, n_iter=n_iter,
            penalty_weight=penalty, seed=seed, progress_cb=cb)
    status.empty()
    h_indices = {i for i, aa in enumerate(seq) if aa == "H"}
    _, energy, pen = fitness(seq, gbest, h_indices, penalty)
    quality = abs(energy) / abs(opt_energy) * 100 if opt_energy != 0 else 0
    st.success(f"✅ Năng lượng tìm được = {energy} (mục tiêu {opt_energy}, chất lượng {quality:.1f}%)")

    metric_row([
        ("⚡ Energy", f"{energy}", f"Optimum {opt_energy}"),
        ("📊 Chất lượng", f"{quality:.1f}%", None),
        ("⚠️ Penalty", f"{pen}", None),
        ("⏱️ Thời gian", f"{rt:.2f}s", None),
    ])

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🧬 Cấu trúc gấp cuộn 2D")
        coords = decode(gbest)
        fig, ax = plt.subplots(figsize=(7, 6))
        xs = [c[0] for c in coords]; ys = [c[1] for c in coords]
        ax.plot(xs, ys, "-", color="gray", lw=1)
        for i, (x, y) in enumerate(coords):
            color = "red" if seq[i] == "H" else "blue"
            ax.scatter(x, y, c=color, s=180, zorder=3, edgecolor="black")
            ax.text(x, y, seq[i], ha="center", va="center", color="white",
                    fontsize=8, fontweight="bold", zorder=4)
        ax.set_aspect("equal"); ax.grid(True, alpha=.3)
        ax.set_title(f"Cấu trúc tối ưu – E={energy}")
        # legend
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(color="red", label="H (kỵ nước)"),
                           Patch(color="blue", label="P (phân cực)")])
        st.pyplot(fig)
    with col2:
        st.subheader("📉 Hội tụ Energy")
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        ax2.plot(hist_e, color="green", lw=2)
        ax2.axhline(opt_energy, color="red", ls="--", label=f"Optimum {opt_energy}")
        ax2.set_xlabel("Vòng lặp"); ax2.set_ylabel("Energy")
        ax2.set_title("Năng lượng theo vòng lặp")
        ax2.legend(); ax2.grid(True, alpha=.3)
        st.pyplot(fig2)
