"""Trang demo PID Tuning với PSO."""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from algorithms.common import (page_header, back_button, plot_convergence, metric_row,
                                generate_pdf_report, download_pdf_button)
from algorithms.pid_pso import pso_pid, simulate_pid, performance_metrics

st.set_page_config(page_title="08 · PID PSO", page_icon="🎛️", layout="wide")
back_button()
page_header("🎛️", "Bài toán 08 – PID Controller Tuning",
            "Tinh chỉnh tham số (Kp, Ki, Kd) cho bộ điều khiển PID bằng PSO", "PSO")

with st.expander("📖 Mô tả bài toán"):
    st.markdown("""
    Bộ điều khiển PID: $u(t) = K_p e + K_i \\int e\\,dt + K_d \\frac{de}{dt}$

    **Plant:** $G(s) = \\frac{1}{s^3 + 6s^2 + 11s + 6}$

    PSO sẽ tối ưu (Kp, Ki, Kd) để tối thiểu sai số tích phân (ITAE/ISE/IAE).
    """)

st.sidebar.markdown("## ⚙️ Tham số PSO")
n_particles = st.sidebar.slider("Số particles", 10, 80, 30)
n_iter = st.sidebar.slider("Vòng lặp", 20, 200, 50)
w = st.sidebar.slider("Inertia w", 0.1, 1.0, 0.7)
c1 = st.sidebar.slider("c1 (cognitive)", 0.1, 3.0, 1.5)
c2 = st.sidebar.slider("c2 (social)", 0.1, 3.0, 1.5)
metric = st.sidebar.selectbox("Hàm mục tiêu", ["ITAE", "ISE", "IAE"])
seed = st.sidebar.number_input("Seed", 0, 9999, 42)

st.markdown("**Bounds mặc định:** Kp ∈ [0.1, 50], Ki ∈ [0, 20], Kd ∈ [0, 20]")

if st.button("🚀 Chạy PSO-PID", type="primary"):
    progress = st.progress(0); status = st.empty()
    def cb(it, total, best):
        progress.progress(it / total)
        status.write(f"Vòng {it}/{total} – {metric} = **{best:.4f}**")
    with st.spinner("Đang chạy..."):
        gbest, gbest_fit, hist, rt = pso_pid(
            n_particles=n_particles, n_iter=n_iter, w=w, c1=c1, c2=c2,
            metric=metric, seed=seed, progress_cb=cb)
    status.empty()
    Kp, Ki, Kd = gbest
    st.success(f"✅ Tối ưu (Kp,Ki,Kd) = ({Kp:.3f}, {Ki:.3f}, {Kd:.3f})")

    t, y = simulate_pid(Kp, Ki, Kd)
    m = performance_metrics(t, y)

    metric_row([
        ("Kp", f"{Kp:.3f}", None),
        ("Ki", f"{Ki:.3f}", None),
        ("Kd", f"{Kd:.3f}", None),
        (metric, f"{m[metric]:.4f}", None),
    ])
    metric_row([
        ("Overshoot %", f"{m['Overshoot%']:.2f}", None),
        ("Rise time", f"{m['Rise']:.3f}s", None),
        ("Settle time", f"{m['Settle']:.3f}s", None),
        ("⏱️ Runtime", f"{rt:.2f}s", None),
    ])

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📈 Step response")
        fig1, ax = plt.subplots(figsize=(7, 5))
        ax.plot(t, y, "b-", lw=2, label="PSO-tuned PID")
        ax.axhline(1.0, color="red", ls="--", label="Setpoint")
        # PID Ziegler-Nichols baseline (rough)
        try:
            t2, y2 = simulate_pid(10.0, 5.0, 1.0)
            ax.plot(t2, y2, "g--", lw=1.5, alpha=.7, label="Baseline (10,5,1)")
        except Exception:
            pass
        ax.set_xlabel("Thời gian (s)"); ax.set_ylabel("Output")
        ax.legend(); ax.grid(True, alpha=.3)
        st.pyplot(fig1)
    with col2:
        st.subheader("📉 Hội tụ")
        fig2 = plot_convergence(hist, "Hội tụ PSO-PID", metric)
        st.pyplot(fig2)

    # ---------- 📄 NÚT TẢI PDF ----------
    st.markdown("---")
    st.subheader("📥 Xuất báo cáo PDF")
    pdf_bytes = generate_pdf_report(
        problem_title="Bài toán 08 – PID Tuning",
        problem_subtitle="Tinh chỉnh tham số bộ điều khiển PID bằng PSO",
        algorithm="PSO",
        inputs={
            "Số particles": n_particles,
            "Số vòng lặp": n_iter,
            "w (inertia)": w,
            "c1 (cognitive)": c1,
            "c2 (social)": c2,
            "Hàm mục tiêu": metric,
            "Seed": seed,
        },
        outputs={
            "Kp": f"{Kp:.4f}",
            "Ki": f"{Ki:.4f}",
            "Kd": f"{Kd:.4f}",
            f"{metric}": f"{m[metric]:.4f}",
            "Overshoot %": f"{m['Overshoot%']:.2f}",
            "Rise time": f"{m['Rise']:.3f} s",
            "Settle time": f"{m['Settle']:.3f} s",
            "Thời gian chạy": f"{rt:.2f} giây",
        },
        figures=[
            ("Hình 1: Step response", fig1),
            ("Hình 2: Đồ thị hội tụ", fig2),
        ],
    )
    download_pdf_button(pdf_bytes,
                       file_name=f"BaoCao_PID_PSO_{metric}.pdf",
                       key="pdf_pid")
