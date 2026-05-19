"""Trang demo Protein Folding HP Model 2D & 3D với PSO."""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (cần để bật projection 3d)
from matplotlib.patches import Patch
from algorithms.common import page_header, back_button, plot_convergence, metric_row
from algorithms.protein_pso import (pso_protein, DATASETS_2D, DATASETS_3D,
                                     decode, get_fitness_details)

st.set_page_config(page_title="07 · Protein PSO", page_icon="🧬", layout="wide")
back_button()
page_header("🧬", "Bài toán 07 – Protein Folding",
            "HP Model 2D & 3D – dự đoán cấu trúc gấp cuộn protein bằng PSO", "PSO")

with st.expander("📖 Mô tả bài toán", expanded=False):
    st.markdown("""
    **HP Model** – chuỗi amino acid được mã hoá nhị phân:
    - **H** = hydrophobic (kỵ nước)
    - **P** = polar (phân cực)

    Mục tiêu: tìm cách gấp chuỗi trên **lưới (2D hoặc 3D)** sao cho số **cặp H-H
    không liền kề trên chuỗi nhưng liền kề trên lưới** là nhiều nhất
    (năng lượng càng âm càng tốt).

    | Chiều | Số hướng | Số láng giềng tối đa | Không gian tìm kiếm |
    |-------|----------|----------------------|----------------------|
    | **2D** | 4 (±x, ±y) | 4 | $4^{L-1}$ |
    | **3D** | 6 (±x, ±y, ±z) | 6 | $6^{L-1}$ |

    3D có không gian lớn hơn nhưng cho phép cấu trúc gấp cuộn phong phú hơn,
    số cặp H-H tối ưu thường cao hơn 2D.
    """)

# -------- Sidebar --------
st.sidebar.markdown("## ⚙️ Cấu hình")
mode = st.sidebar.radio("🧊 Chiều không gian", ["2D", "3D"], horizontal=True,
                        help="2D = lưới phẳng (4 hướng) | 3D = lưới khối (6 hướng)")
datasets = DATASETS_2D if mode == "2D" else DATASETS_3D
seq_name = st.sidebar.selectbox("📋 Chuỗi protein", list(datasets.keys()))
n_particles = st.sidebar.slider("Số particles", 20, 200, 80)
n_iter = st.sidebar.slider("Vòng lặp tối đa", 50, 500, 200)
penalty = st.sidebar.slider("Penalty (chồng chéo)", 10, 100, 40)
early_stop = st.sidebar.slider("Early stop (vòng không cải thiện)", 10, 100, 40)
seed = st.sidebar.number_input("Seed", 0, 9999, 42)

seq, opt_contacts = datasets[seq_name]
h_count = seq.count("H")
p_count = seq.count("P")

# Thông tin chuỗi
st.markdown(f"""
**Chuỗi:** `{seq}`

**Độ dài:** {len(seq)} amino acid · **H:** {h_count} · **P:** {p_count} ·
**Tối ưu đã biết:** {opt_contacts} cặp H-H · **Chế độ:** {mode}
""")

# Hiển thị chuỗi với màu sắc
seq_html = ""
for aa in seq:
    if aa == "H":
        seq_html += (f"<span style='background:#ef4444;color:white;padding:4px 8px;"
                     f"margin:2px;border-radius:8px;font-family:JetBrains Mono;"
                     f"font-weight:700;display:inline-block;'>H</span>")
    else:
        seq_html += (f"<span style='background:#3b82f6;color:white;padding:4px 8px;"
                     f"margin:2px;border-radius:8px;font-family:JetBrains Mono;"
                     f"font-weight:700;display:inline-block;'>P</span>")
st.markdown(seq_html, unsafe_allow_html=True)
st.markdown("&nbsp;", unsafe_allow_html=True)

# -------- Chạy thuật toán --------
if st.button(f"🚀 Chạy PSO-Protein ({mode})", type="primary"):
    progress = st.progress(0); status = st.empty()
    def cb(it, total, best):
        progress.progress(it / total)
        status.write(f"Vòng {it}/{total} – Fitness (= -contacts): **{best:.2f}**")

    with st.spinner(f"Đang gấp cuộn protein trong không gian {mode}..."):
        gbest, fit, hist_f, hist_e, rt = pso_protein(
            seq, mode=mode, n_particles=n_particles, n_iter=n_iter,
            penalty_weight=penalty, early_stop=early_stop,
            seed=seed, progress_cb=cb)
    status.empty()

    h_indices = {i for i, aa in enumerate(seq) if aa == "H"}
    _, contacts, pen = get_fitness_details(seq, gbest, h_indices, mode, penalty)
    quality = contacts / opt_contacts * 100 if opt_contacts > 0 else 0

    if contacts >= opt_contacts:
        st.success(f"🏆 Đạt/Vượt optimum! {contacts} cặp H-H (mục tiêu {opt_contacts}).")
    elif contacts >= 0.8 * opt_contacts:
        st.success(f"✅ Kết quả tốt: {contacts}/{opt_contacts} cặp H-H ({quality:.1f}%).")
    else:
        st.warning(f"⚠️ Còn xa optimum: {contacts}/{opt_contacts} cặp H-H ({quality:.1f}%).")

    # ----- Metrics -----
    metric_row([
        ("🔗 H-H contacts", f"{contacts}", f"Optimum {opt_contacts}"),
        ("📊 Chất lượng", f"{quality:.1f}%", None),
        ("⚠️ Penalty", f"{pen}", "Chồng chéo trên lưới"),
        ("⏱️ Runtime", f"{rt:.2f}s", f"{len(hist_f)} vòng đã chạy"),
    ])

    # ----- Visualisation -----
    coords = decode(gbest, mode)
    col1, col2 = st.columns([3, 2])

    with col1:
        st.subheader(f"🧬 Cấu trúc gấp cuộn {mode}")
        if mode == "2D":
            fig, ax = plt.subplots(figsize=(7, 6))
            xs = [c[0] for c in coords]
            ys = [c[1] for c in coords]
            # Backbone
            ax.plot(xs, ys, "-", color="gray", lw=1.5, alpha=.6, zorder=1)
            # H-H contacts (đường đứt xanh)
            pos_map = {c: i for i, c in enumerate(coords)}
            for i in range(len(seq)):
                if seq[i] != "H":
                    continue
                for j in range(i + 2, len(seq)):
                    if seq[j] != "H":
                        continue
                    d = abs(coords[i][0] - coords[j][0]) + abs(coords[i][1] - coords[j][1])
                    if d == 1:
                        ax.plot([coords[i][0], coords[j][0]],
                                [coords[i][1], coords[j][1]],
                                "--", color="#10b981", lw=1.5, alpha=.7, zorder=2)
            # Residues
            for i, (x, y) in enumerate(coords):
                color = "#ef4444" if seq[i] == "H" else "#3b82f6"
                ax.scatter(x, y, c=color, s=320, zorder=3,
                           edgecolor="black", linewidth=1.2)
                ax.text(x, y, seq[i], ha="center", va="center",
                        color="white", fontsize=9, fontweight="bold", zorder=4)
            ax.set_aspect("equal")
            ax.grid(True, alpha=.25)
            ax.set_title(f"Cấu trúc 2D – {contacts} cặp H-H",
                         color="#1e1b4b", fontweight="bold")
            ax.legend(handles=[
                Patch(color="#ef4444", label="H (kỵ nước)"),
                Patch(color="#3b82f6", label="P (phân cực)"),
                Patch(color="#10b981", label="H-H contact"),
            ], loc="best")
            st.pyplot(fig)
        else:  # 3D
            fig = plt.figure(figsize=(8, 7))
            ax = fig.add_subplot(111, projection="3d")
            xs = [c[0] for c in coords]
            ys = [c[1] for c in coords]
            zs = [c[2] for c in coords]
            # Backbone
            ax.plot(xs, ys, zs, "-", color="gray", lw=2.5, alpha=.7, zorder=1)
            # H-H contacts
            for i in range(len(seq)):
                if seq[i] != "H":
                    continue
                for j in range(i + 2, len(seq)):
                    if seq[j] != "H":
                        continue
                    d = (abs(coords[i][0] - coords[j][0])
                         + abs(coords[i][1] - coords[j][1])
                         + abs(coords[i][2] - coords[j][2]))
                    if d == 1:
                        ax.plot([coords[i][0], coords[j][0]],
                                [coords[i][1], coords[j][1]],
                                [coords[i][2], coords[j][2]],
                                "--", color="#10b981", lw=1.6, alpha=.85)
            # Residues
            for i, (x, y, z) in enumerate(coords):
                if seq[i] == "H":
                    ax.scatter(x, y, z, c="#ef4444", s=260, edgecolor="black",
                               linewidth=0.9, depthshade=True, zorder=3)
                else:
                    ax.scatter(x, y, z, c="#60a5fa", s=140, edgecolor="black",
                               linewidth=0.9, depthshade=True, zorder=3)
            ax.set_xlabel("X", fontweight="bold")
            ax.set_ylabel("Y", fontweight="bold")
            ax.set_zlabel("Z", fontweight="bold")
            ax.set_title(f"Cấu trúc 3D – {contacts} cặp H-H",
                         color="#1e1b4b", fontweight="bold")
            ax.legend(handles=[
                Patch(color="#ef4444", label="H (kỵ nước)"),
                Patch(color="#60a5fa", label="P (phân cực)"),
                Patch(color="#10b981", label="H-H contact"),
            ], loc="upper left", fontsize=8)
            fig.tight_layout()
            st.pyplot(fig)

            # góc nhìn khác (nhìn từ trên xuống)
            with st.expander("🔄 Xem nhiều góc khác"):
                fig2 = plt.figure(figsize=(11, 4))
                for k, (elev, azim, title) in enumerate(
                    [(30, 45, "Mặc định"), (90, 0, "Trên xuống (Top)"),
                     (0, 0, "Mặt trước")]):
                    ax2 = fig2.add_subplot(1, 3, k + 1, projection="3d")
                    ax2.plot(xs, ys, zs, "-", color="gray", lw=2, alpha=.6)
                    for i, (x, y, z) in enumerate(coords):
                        if seq[i] == "H":
                            ax2.scatter(x, y, z, c="#ef4444", s=120,
                                        edgecolor="black", lw=.7, depthshade=True)
                        else:
                            ax2.scatter(x, y, z, c="#60a5fa", s=70,
                                        edgecolor="black", lw=.7, depthshade=True)
                    ax2.view_init(elev=elev, azim=azim)
                    ax2.set_title(title, fontsize=10)
                fig2.tight_layout()
                st.pyplot(fig2)

    with col2:
        st.subheader("📈 Hội tụ năng lượng")
        fig3, ax3 = plt.subplots(figsize=(7, 4.5))
        ax3.plot(hist_e, color="#10b981", lw=2.5, marker="o", markersize=3,
                 markerfacecolor="white", markeredgecolor="#10b981")
        ax3.axhline(opt_contacts, color="#ef4444", ls="--", lw=2,
                    label=f"Optimum = {opt_contacts}")
        ax3.fill_between(range(len(hist_e)), hist_e, alpha=.15, color="#10b981")
        ax3.set_xlabel("Vòng lặp", fontweight="bold")
        ax3.set_ylabel("Số cặp H-H", fontweight="bold")
        ax3.set_title("Số H-H contacts theo vòng lặp",
                      color="#1e1b4b", fontweight="bold")
        ax3.legend(loc="lower right")
        ax3.grid(True, alpha=.3)
        st.pyplot(fig3)

        st.subheader("📉 Hội tụ fitness")
        st.pyplot(plot_convergence(hist_f, "Fitness (-contacts + penalty)",
                                    "Fitness (thấp = tốt)"))

    # ----- Chi tiết hướng -----
    with st.expander("📋 Chi tiết chuỗi hướng tối ưu"):
        dir_label = (["+x", "+y", "-x", "-y"] if mode == "2D"
                     else ["+x", "-x", "+y", "-y", "+z", "-z"])
        st.code(" → ".join(dir_label[int(d)] for d in gbest), language=None)
        st.markdown(f"**Toạ độ các residue:**")
        coord_str = "\n".join(
            f"  {i:3d}. {seq[i]} → {tuple(coords[i])}"
            for i in range(len(seq))
        )
        st.code(coord_str, language=None)
