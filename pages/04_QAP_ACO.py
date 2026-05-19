"""Trang demo QAP với ACO – phân công 8 phòng ban × 8 vị trí."""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from algorithms.common import (page_header, back_button, plot_convergence, metric_row,
                                generate_pdf_report, download_pdf_button)
from algorithms.qap_aco import (aco_qap, FLOW, DISTANCE, DEPARTMENT_NAMES,
                                 SHORT, LOCATION_NAMES, COLORS, qap_cost)

st.set_page_config(page_title="04 · QAP ACO", page_icon="🏢", layout="wide")
back_button()
page_header("🏢", "Bài toán 04 – QAP",
            "Phân công 8 phòng ban vào 8 vị trí trong toà nhà", "ACO")

with st.expander("📖 Mô tả bài toán"):
    st.markdown("""
    **QAP** – gán $n$ thực thể vào $n$ vị trí sao cho tổng chi phí
    $\\sum_{i,j} f_{ij} \\cdot d_{\\pi(i),\\pi(j)}$ là nhỏ nhất.
    Trong demo: 8 phòng ban × 8 vị trí (toà nhà 4 tầng, 2 phòng/tầng).
    """)

st.sidebar.markdown("## ⚙️ Tham số")
n_ants = st.sidebar.slider("Số kiến", 10, 80, 30)
n_iter = st.sidebar.slider("Vòng lặp", 30, 500, 150)
alpha = st.sidebar.slider("Alpha", 0.1, 5.0, 1.0)
beta = st.sidebar.slider("Beta", 0.1, 5.0, 2.0)
rho = st.sidebar.slider("Rho", 0.05, 0.95, 0.15)
seed = st.sidebar.number_input("Seed", 0, 9999, 42)

col_a, col_b = st.columns(2)
with col_a:
    st.subheader("🔁 Flow matrix")
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(FLOW, cmap="YlOrRd")
    ax.set_xticks(range(8)); ax.set_yticks(range(8))
    ax.set_xticklabels(SHORT, rotation=45); ax.set_yticklabels(SHORT)
    for i in range(8):
        for j in range(8):
            ax.text(j, i, int(FLOW[i, j]), ha="center", va="center", fontsize=7)
    plt.colorbar(im, ax=ax); ax.set_title("Flow (lần/tuần)")
    st.pyplot(fig)
with col_b:
    st.subheader("📏 Distance matrix")
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(DISTANCE, cmap="Blues")
    ax.set_xticks(range(8)); ax.set_yticks(range(8))
    ax.set_xticklabels([f"P{i+1}" for i in range(8)], rotation=45)
    ax.set_yticklabels([f"P{i+1}" for i in range(8)])
    for i in range(8):
        for j in range(8):
            ax.text(j, i, int(DISTANCE[i, j]), ha="center", va="center", fontsize=7)
    plt.colorbar(im, ax=ax); ax.set_title("Distance (m)")
    st.pyplot(fig)

if st.button("🚀 Chạy ACO-QAP", type="primary"):
    progress = st.progress(0); status = st.empty()
    def cb(it, total, best):
        progress.progress(it / total)
        status.write(f"Vòng {it}/{total} – Cost: **{best:.0f}**")
    with st.spinner("Đang chạy..."):
        assign, cost, hist, rt = aco_qap(FLOW, DISTANCE, num_ants=n_ants,
                                          num_iter=n_iter, alpha=alpha, beta=beta,
                                          rho=rho, seed=seed, progress_cb=cb)
    status.empty()
    st.success(f"✅ Cost tối ưu = **{cost:.0f}**")

    metric_row([
        ("💰 Tổng chi phí", f"{cost:.0f}", None),
        ("⏱️ Thời gian", f"{rt:.2f}s", None),
        ("🐜 Số kiến", f"{n_ants}", None),
        ("🔁 Vòng lặp", f"{n_iter}", None),
    ])

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🏢 Sơ đồ toà nhà")
        fig1, ax = plt.subplots(figsize=(6, 7))
        for floor in range(4):
            for slot in range(2):
                loc = floor * 2 + slot
                dept = assign.index(loc) if loc in assign else None
                x, y = slot * 3, (3 - floor) * 2
                color = COLORS[dept] if dept is not None else "lightgray"
                ax.add_patch(Rectangle((x, y), 2.8, 1.8, facecolor=color, edgecolor="black"))
                if dept is not None:
                    ax.text(x + 1.4, y + 0.9, f"{SHORT[dept]}\n(P{loc+1})",
                            ha="center", va="center", fontsize=11, fontweight="bold")
        ax.set_xlim(-0.5, 6.3); ax.set_ylim(-0.5, 8.5)
        ax.set_aspect("equal"); ax.axis("off")
        ax.set_title("Phân bố phòng ban (4 tầng × 2 phòng)")
        st.pyplot(fig1)
    with col2:
        st.subheader("📉 Hội tụ")
        fig2 = plot_convergence(hist, "Hội tụ QAP-ACO", "Cost")
        st.pyplot(fig2)

    with st.expander("📋 Chi tiết phân công"):
        for dept_idx, loc_idx in enumerate(assign):
            st.write(f"- **{DEPARTMENT_NAMES[dept_idx]}** → {LOCATION_NAMES[loc_idx]}")

    # ---------- 📄 NÚT TẢI PDF ----------
    st.markdown("---")
    st.subheader("📥 Xuất báo cáo PDF")
    pdf_bytes = generate_pdf_report(
        problem_title="Bài toán 04 – QAP",
        problem_subtitle="Quadratic Assignment Problem – 8 phòng ban × 8 vị trí",
        algorithm="ACO",
        inputs={
            "Số phòng ban": 8,
            "Số vị trí": 8,
            "Số kiến": n_ants,
            "Số vòng lặp": n_iter,
            "Alpha": alpha,
            "Beta": beta,
            "Rho": rho,
            "Seed": seed,
        },
        outputs={
            "Tổng chi phí": f"{cost:.0f}",
            "Thời gian chạy": f"{rt:.2f} giây",
            **{f"{DEPARTMENT_NAMES[d]}": LOCATION_NAMES[l]
               for d, l in enumerate(assign)},
        },
        figures=[
            ("Hình 1: Sơ đồ toà nhà (4 tầng × 2 phòng)", fig1),
            ("Hình 2: Đồ thị hội tụ", fig2),
        ],
    )
    download_pdf_button(pdf_bytes,
                       file_name=f"BaoCao_QAP_ACO.pdf",
                       key="pdf_qap")
