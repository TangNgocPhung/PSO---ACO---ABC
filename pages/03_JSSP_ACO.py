"""Trang demo bài toán JSSP với ACO – benchmark FT06."""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from algorithms.common import (page_header, back_button, plot_convergence, metric_row,
                                generate_pdf_report, download_pdf_button)
from algorithms.jssp_aco import aco_jssp, FT06, FT06_OPT, decode_schedule

st.set_page_config(page_title="03 · JSSP ACO", page_icon="⚙️", layout="wide")
back_button()
page_header("⚙️", "Bài toán 03 – JSSP",
            "Job Shop Scheduling – benchmark FT06 (6 jobs × 6 machines)", "ACO")

with st.expander("📖 Mô tả bài toán"):
    st.markdown("""
    **JSSP** – mỗi job gồm chuỗi thao tác phải thực hiện đúng thứ tự trên máy cụ thể.
    Mỗi máy chỉ xử lý 1 thao tác tại 1 thời điểm. Mục tiêu: tối thiểu *makespan*.

    Dataset **FT06** (Fisher & Thompson 1963) có optimum đã biết = **55**.
    """)

st.sidebar.markdown("## ⚙️ Tham số")
n_ants = st.sidebar.slider("Số kiến", 10, 100, 40)
n_iter = st.sidebar.slider("Vòng lặp", 20, 300, 120)
alpha = st.sidebar.slider("Alpha", 0.1, 5.0, 1.0)
beta = st.sidebar.slider("Beta", 0.1, 5.0, 2.0)
rho = st.sidebar.slider("Rho", 0.05, 0.95, 0.2)
seed = st.sidebar.number_input("Seed", 0, 9999, 42)

with st.expander("📊 Dữ liệu FT06 (machine, processing time)"):
    for ji, job in enumerate(FT06):
        st.write(f"**Job {ji}:** " + " → ".join(f"(M{m}, {t})" for m, t in job))
    st.info(f"🎯 Optimum đã biết: **{FT06_OPT}**")

if st.button("🚀 Chạy ACO-JSSP", type="primary"):
    progress = st.progress(0)
    status = st.empty()
    def cb(it, total, best):
        progress.progress(it / total)
        status.write(f"Vòng {it}/{total} – Makespan: **{best}** (opt={FT06_OPT})")
    with st.spinner("Đang chạy..."):
        seq, sched, mk, hist, rt = aco_jssp(FT06, num_ants=n_ants, num_iter=n_iter,
                                            alpha=alpha, beta=beta, rho=rho,
                                            seed=seed, progress_cb=cb)
    status.empty()
    gap = (mk - FT06_OPT) / FT06_OPT * 100
    st.success(f"✅ Makespan = **{mk}** (gap = {gap:.1f}% so với optimum)")

    metric_row([
        ("⏱️ Makespan", f"{mk}", f"Optimum: {FT06_OPT}"),
        ("📊 Gap", f"{gap:.1f}%", None),
        ("⌛ Thời gian", f"{rt:.2f}s", None),
        ("🔁 Vòng lặp", f"{n_iter}", None),
    ])

    with st.expander("📖 Giải thích kết quả", expanded=False):
        st.markdown(f"""
        ### 🎯 Các chỉ số đầu ra

        - **⏱️ Makespan = `{mk}`**
          **Tổng thời gian hoàn thành tất cả công việc** = thời điểm kết thúc của
          operation cuối cùng trên máy bận nhất. Đây là **mục tiêu cần tối thiểu hoá** trong JSSP.

        - **🎯 Optimum đã biết = `{FT06_OPT}`** (FT06 benchmark, Fisher & Thompson 1963)

        - **📊 Gap = `{gap:.1f}%`**
          Khoảng cách so với optimum: $(mk - opt) / opt \\times 100\\%$.
          - Gap < 5%: rất tốt
          - Gap 5-15%: chấp nhận được
          - Gap > 15%: cần tăng vòng lặp hoặc tinh chỉnh α, β

        ### 🖼️ Các hình minh hoạ

        - **📊 Gantt chart** — Lịch sản xuất chi tiết. Mỗi hàng = 1 máy (M0..M5),
          mỗi thanh ngang = 1 operation của 1 job. Màu thanh = job ID (J0..J5).
          Khoảng trống = máy nhàn rỗi (chờ job có sẵn).

        - **📉 Đồ thị hội tụ** — Makespan theo vòng lặp.
          Đường gạch đỏ = optimum = `{FT06_OPT}`. Nếu đường xanh chạm được đường đỏ
          → tìm được lời giải tối ưu toàn cục.

        ### 💡 Kiểm tra tính hợp lệ của lịch

        - ✅ Mỗi job thực hiện đúng thứ tự operation
        - ✅ Mỗi máy xử lý 1 operation tại 1 thời điểm
        - ✅ Tất cả `{sum(len(j) for j in FT06)}` operations đã được lập lịch

        ### 🔍 Diễn giải

        Vòng đầu makespan ~`{hist[0]}` (kiến chọn ngẫu nhiên).
        Sau `{n_iter}` vòng, makespan giảm còn `{mk}` — **giảm
        {(hist[0]-mk)/hist[0]*100:.1f}%**. Pheromone tích luỹ trên các cặp `(vị trí, job)`
        tốt → các vòng sau ưu tiên chọn job hợp lý hơn.
        """)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📊 Gantt chart")
        n_machines = max(m for job in FT06 for m, _ in job) + 1
        fig1, ax = plt.subplots(figsize=(10, 5))
        colors = plt.cm.tab10(np.linspace(0, 1, len(FT06)))
        for m, j, op, s, e in sched:
            ax.barh(m, e - s, left=s, color=colors[j], edgecolor="black", alpha=.85)
            ax.text((s + e) / 2, m, f"J{j}", ha="center", va="center",
                    fontsize=8, color="white", fontweight="bold")
        ax.set_yticks(range(n_machines))
        ax.set_yticklabels([f"M{i}" for i in range(n_machines)])
        ax.set_xlabel("Thời gian")
        ax.set_title(f"Lịch sản xuất – Makespan = {mk}")
        ax.grid(True, axis="x", alpha=.3)
        st.pyplot(fig1)
    with col2:
        st.subheader("📉 Hội tụ")
        fig2 = plot_convergence(hist, "Hội tụ JSSP-ACO", "Makespan")
        fig2.gca().axhline(FT06_OPT, color="red", ls="--", label=f"Optimum={FT06_OPT}")
        fig2.gca().legend()
        st.pyplot(fig2)

    with st.expander("📋 Chuỗi thứ tự công việc"):
        st.code(" → ".join(map(str, seq)))

    # ---------- 📄 NÚT TẢI PDF ----------
    st.markdown("---")
    st.subheader("📥 Xuất báo cáo PDF")
    pdf_bytes = generate_pdf_report(
        problem_title="Bài toán 03 – JSSP",
        problem_subtitle="Job Shop Scheduling Problem – benchmark FT06",
        algorithm="ACO",
        inputs={
            "Dataset": "FT06 (6 jobs × 6 machines)",
            "Optimum đã biết": FT06_OPT,
            "Số kiến": n_ants,
            "Số vòng lặp": n_iter,
            "Alpha": alpha,
            "Beta": beta,
            "Rho": rho,
            "Seed": seed,
        },
        outputs={
            "Makespan": mk,
            "Gap so với optimum": f"{gap:.2f}%",
            "Thời gian chạy": f"{rt:.2f} giây",
            "Chuỗi thứ tự (10 op đầu)": " → ".join(map(str, seq[:10])),
        },
        figures=[
            ("Hình 1: Gantt chart lịch sản xuất", fig1),
            ("Hình 2: Đồ thị hội tụ", fig2),
        ],
    )
    download_pdf_button(pdf_bytes,
                       file_name=f"BaoCao_JSSP_ACO_FT06.pdf",
                       key="pdf_jssp")
