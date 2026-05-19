"""Trang demo Feature Selection – Binary PSO trên Breast Cancer dataset."""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from algorithms.common import (page_header, back_button, plot_convergence, metric_row,
                                generate_pdf_report, download_pdf_button)
from algorithms.feature_selection_bpso import binary_pso_feature_selection

st.set_page_config(page_title="10 · Feature Selection BPSO", page_icon="🔬", layout="wide")
back_button()
page_header("🔬", "Bài toán 10 – Feature Selection",
            "Binary PSO chọn đặc trưng quan trọng – Breast Cancer (569×30)", "PSO")

with st.expander("📖 Mô tả bài toán"):
    st.markdown("""
    **Feature Selection**: chọn $k < d$ đặc trưng hữu ích nhất để giữ accuracy mô hình
    cao đồng thời giảm chiều dữ liệu.

    **Binary PSO**: mỗi particle là vector nhị phân (0/1) đại diện cho việc chọn
    feature hay không. Cập nhật vận tốc theo PSO, vị trí được sample bằng:
    $x_i = 1$ nếu $rand() < sigmoid(v_i)$.

    **Fitness:** $\\alpha \\cdot err + (1-\\alpha) \\cdot \\frac{|S|}{d}$
    """)

st.sidebar.markdown("## ⚙️ Tham số")
n_particles = st.sidebar.slider("Số particles", 10, 60, 20)
n_iter = st.sidebar.slider("Vòng lặp", 10, 100, 25)
w = st.sidebar.slider("Inertia", 0.1, 1.0, 0.7)
c1 = st.sidebar.slider("c1", 0.1, 3.0, 1.5)
c2 = st.sidebar.slider("c2", 0.1, 3.0, 1.5)
alpha = st.sidebar.slider("Alpha (trade-off acc vs #features)", 0.5, 1.0, 0.9)
seed = st.sidebar.number_input("Seed", 0, 9999, 42)

st.info("📊 Dataset: sklearn `load_breast_cancer` (569 mẫu × 30 đặc trưng, "
        "phân loại nhị phân ác/lành tính)")

if st.button("🚀 Chạy Binary PSO", type="primary"):
    progress = st.progress(0); status = st.empty()
    def cb(it, total, best):
        progress.progress(it / total)
        status.write(f"Vòng {it}/{total} – fitness: **{best:.4f}**")
    with st.spinner("Đang chạy (có thể mất chút thời gian do train LogisticRegression)..."):
        res = binary_pso_feature_selection(n_particles=n_particles, n_iter=n_iter,
                                            w=w, c1=c1, c2=c2, alpha=alpha,
                                            seed=seed, progress_cb=cb)
    status.empty()
    st.success(f"✅ Đã chọn {res['n_features_selected']}/{res['n_features_total']} đặc trưng")

    metric_row([
        ("🎯 Accuracy gốc", f"{res['baseline_acc']*100:.2f}%",
         f"Dùng {res['n_features_total']} features"),
        ("🚀 Accuracy chọn", f"{res['selected_acc']*100:.2f}%",
         f"Dùng {res['n_features_selected']} features"),
        ("📉 Giảm features", f"{(1-res['n_features_selected']/res['n_features_total'])*100:.0f}%",
         None),
        ("⏱️ Runtime", f"{res['runtime']:.2f}s", None),
    ])

    with st.expander("📖 Giải thích kết quả", expanded=False):
        delta = (res['selected_acc'] - res['baseline_acc']) * 100
        st.markdown(f"""
        ### 🎯 Các chỉ số đầu ra

        - **🎯 Accuracy gốc = `{res['baseline_acc']*100:.2f}%`**
          Độ chính xác của LogisticRegression khi dùng **toàn bộ {res['n_features_total']}
          đặc trưng** (mean radius, mean texture, ..., worst fractal dimension).

        - **🚀 Accuracy với features chọn = `{res['selected_acc']*100:.2f}%`**
          Độ chính xác **sau khi chọn lọc** chỉ giữ {res['n_features_selected']} đặc trưng "quan trọng".
          Δ = **{delta:+.2f}%** so với baseline.

        - **📉 Giảm features = `{(1-res['n_features_selected']/res['n_features_total'])*100:.0f}%`**
          Giảm số features từ {res['n_features_total']} → {res['n_features_selected']}
          → **mô hình nhẹ hơn, training nhanh hơn, ít overfitting hơn**.

        - **⏱️ Runtime = `{res['runtime']:.2f}s`** — Bao gồm fit/eval LogisticRegression
          tại mỗi particle × mỗi vòng lặp.

        ### 🖼️ Các hình minh hoạ

        - **✅ Đặc trưng được chọn** — Liệt kê {res['n_features_selected']} features mà Binary PSO
          đánh giá là quan trọng nhất.

        - **📉 Hội tụ BPSO** — Fitness theo vòng lặp:
          $$fitness = \\alpha \\cdot err + (1-\\alpha) \\cdot \\frac{{|S|}}{{d}}$$
          với $\\alpha={alpha}$. Càng nhỏ càng tốt (vừa muốn err thấp, vừa muốn |S| nhỏ).

        - **📊 So sánh trước & sau** (2 bar chart):
          - Trái: Accuracy gốc vs sau chọn lọc
          - Phải: Số features gốc vs sau chọn lọc

        ### 💡 Phân tích

        {"🏆 **Feature selection THÀNH CÔNG!** Giảm "
         f"{(1-res['n_features_selected']/res['n_features_total'])*100:.0f}% features "
         f"mà accuracy {'tăng' if delta > 0 else 'giảm chỉ'} {abs(delta):.2f}%."
         if abs(delta) < 3 and res['n_features_selected'] < res['n_features_total']
         else "📊 Kết quả cần xem xét — accuracy có thay đổi đáng kể, cần kiểm tra trade-off."}

        ### 🔬 Binary PSO khác PSO chuẩn ra sao?

        | | PSO chuẩn | **Binary PSO** |
        |---|---|---|
        | Vị trí | Số thực $x \\in \\mathbb{{R}}^d$ | Vector nhị phân $x \\in \\{{0,1\\}}^d$ |
        | Cập nhật vị trí | $x = x + v$ | $x_i = 1$ nếu $rand() < sigmoid(v_i)$ |
        | Ứng dụng | Tối ưu liên tục | Feature selection, subset selection |

        ### 📚 Dataset Breast Cancer

        - **569 mẫu**, **30 đặc trưng** thuộc các nhóm: mean / standard error / worst
          (radius, texture, perimeter, area, smoothness, ...).
        - **Phân loại nhị phân**: ác tính (malignant) vs lành tính (benign).
        """)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("✅ Đặc trưng được chọn")
        names = res["feature_names"]
        mask = res["gbest"]
        chosen = [names[i] for i in range(len(mask)) if mask[i] == 1]
        for i, name in enumerate(chosen):
            st.markdown(f"<span style='background:#dcfce7;padding:3px 8px;border-radius:6px;"
                        f"margin:2px;display:inline-block;'>{i+1}. {name}</span>",
                        unsafe_allow_html=True)
    with col2:
        st.subheader("📉 Hội tụ")
        fig1 = plot_convergence(res["history"], "Hội tụ BPSO", "Fitness")
        st.pyplot(fig1)

    st.subheader("📊 So sánh trước & sau")
    fig2, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(["Gốc (30)", f"Chọn ({res['n_features_selected']})"],
                [res["baseline_acc"], res["selected_acc"]],
                color=["#dbeafe", "#16a34a"], edgecolor="black")
    axes[0].set_ylabel("Accuracy"); axes[0].set_ylim(0, 1.05)
    axes[0].set_title("Accuracy"); axes[0].grid(True, axis="y", alpha=.3)
    for i, v in enumerate([res["baseline_acc"], res["selected_acc"]]):
        axes[0].text(i, v + 0.01, f"{v*100:.2f}%", ha="center", fontweight="bold")
    axes[1].bar(["Gốc", "Chọn"], [res["n_features_total"], res["n_features_selected"]],
                color=["#fef3c7", "#f97316"], edgecolor="black")
    axes[1].set_ylabel("Số đặc trưng"); axes[1].set_title("Số features")
    axes[1].grid(True, axis="y", alpha=.3)
    st.pyplot(fig2)

    # ---------- 📄 NÚT TẢI PDF ----------
    st.markdown("---")
    st.subheader("📥 Xuất báo cáo PDF")
    chosen = [res["feature_names"][i] for i in range(len(res["gbest"])) if res["gbest"][i] == 1]
    pdf_bytes = generate_pdf_report(
        problem_title="Bài toán 10 – Feature Selection",
        problem_subtitle="Binary PSO chọn đặc trưng – Breast Cancer dataset (569×30)",
        algorithm="Binary PSO",
        inputs={
            "Số particles": n_particles,
            "Số vòng lặp": n_iter,
            "w (inertia)": w,
            "c1": c1,
            "c2": c2,
            "Alpha (trade-off)": alpha,
            "Seed": seed,
            "Dataset": "Breast Cancer (sklearn)",
            "Total features": res["n_features_total"],
        },
        outputs={
            "Số features được chọn": res["n_features_selected"],
            "Giảm features": f"{(1-res['n_features_selected']/res['n_features_total'])*100:.1f}%",
            "Accuracy gốc": f"{res['baseline_acc']*100:.2f}%",
            "Accuracy với features chọn": f"{res['selected_acc']*100:.2f}%",
            "Thời gian chạy": f"{res['runtime']:.2f} giây",
            "Features chọn": ", ".join(chosen[:5]) + (f" ... (+{len(chosen)-5} nữa)" if len(chosen) > 5 else ""),
        },
        figures=[
            ("Hình 1: Đồ thị hội tụ BPSO", fig1),
            ("Hình 2: So sánh accuracy & số features", fig2),
        ],
    )
    download_pdf_button(pdf_bytes,
                       file_name=f"BaoCao_FeatureSelection_BPSO.pdf",
                       key="pdf_fs")
