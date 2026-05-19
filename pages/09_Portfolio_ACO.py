"""Trang demo Portfolio Optimization (Markowitz) với ACO."""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from algorithms.common import (page_header, back_button, plot_convergence, metric_row,
                                generate_pdf_report, download_pdf_button)
from algorithms.portfolio_aco import (aco_portfolio, STOCK_NAMES, SECTORS,
                                       EXPECTED_RETURNS, STD_DEVS, COV_MATRIX,
                                       CORRELATION, RF, portfolio_return,
                                       portfolio_risk, sharpe_ratio)

st.set_page_config(page_title="09 · Portfolio ACO", page_icon="💹", layout="wide")
back_button()
page_header("💹", "Bài toán 09 – Portfolio Optimization",
            "Tối ưu danh mục đầu tư Markowitz với 10 cổ phiếu Việt Nam – ACO", "ACO")

with st.expander("📖 Mô tả bài toán"):
    st.markdown("""
    **Markowitz Mean-Variance**: phân bổ trọng số $w_i$ cho $n$ tài sản để
    tối đa hoá **Sharpe ratio** = $(R_p - R_f) / \\sigma_p$.

    ACO rời rạc hoá weight thành các mức (0%, 2.5%, ..., 50%) và lan truyền pheromone
    trên ma trận (cổ phiếu × mức).
    """)

st.sidebar.markdown("## ⚙️ Tham số")
n_ants = st.sidebar.slider("Số kiến", 10, 100, 30)
n_iter = st.sidebar.slider("Vòng lặp", 50, 500, 200)
alpha = st.sidebar.slider("Alpha", 0.1, 5.0, 1.0)
beta = st.sidebar.slider("Beta", 0.1, 5.0, 2.0)
rho = st.sidebar.slider("Rho", 0.05, 0.5, 0.1)
seed = st.sidebar.number_input("Seed", 0, 9999, 42)

# Bảng cổ phiếu
st.subheader("📊 Dữ liệu cổ phiếu")
import pandas as pd
df = pd.DataFrame({
    "Mã": STOCK_NAMES,
    "Ngành": SECTORS,
    "Expected Return (%)": (EXPECTED_RETURNS * 100).round(2),
    "Std Dev (%)": (STD_DEVS * 100).round(2),
})
st.dataframe(df, use_container_width=True, hide_index=True)

if st.button("🚀 Chạy ACO-Portfolio", type="primary"):
    progress = st.progress(0); status = st.empty()
    def cb(it, total, best):
        progress.progress(it / total)
        status.write(f"Vòng {it}/{total} – Sharpe = **{best:.4f}**")
    with st.spinner("Đang chạy..."):
        w, sr, hist, rt = aco_portfolio(n_ants=n_ants, n_iter=n_iter,
                                         alpha=alpha, beta=beta, rho=rho,
                                         seed=seed, progress_cb=cb)
    status.empty()
    r = portfolio_return(w); risk = portfolio_risk(w)
    st.success(f"✅ Sharpe = {sr:.4f} | Return = {r*100:.2f}% | Risk = {risk*100:.2f}%")

    metric_row([
        ("📈 Return", f"{r*100:.2f}%", None),
        ("📉 Risk", f"{risk*100:.2f}%", None),
        ("⚡ Sharpe", f"{sr:.4f}", f"Rf = {RF*100:.0f}%"),
        ("⏱️ Runtime", f"{rt:.2f}s", None),
    ])

    # Tính idx (thứ tự giảm dần theo weight) – dùng cả trong giải thích và biểu đồ
    idx = np.argsort(-w)

    with st.expander("📖 Giải thích kết quả", expanded=False):
        st.markdown(f"""
        ### 🎯 Các chỉ số đầu ra

        - **📈 Expected Return = `{r*100:.2f}%`**
          $R_p = \\sum_i w_i \\cdot R_i$ — Lợi nhuận kỳ vọng hàng năm của danh mục,
          tính theo bình quân gia quyền các lợi nhuận từng cổ phiếu.

        - **📉 Risk (σ) = `{risk*100:.2f}%`**
          $\\sigma_p = \\sqrt{{w^T \\Sigma w}}$ — Độ lệch chuẩn danh mục,
          đo lường mức biến động giá. Càng nhỏ → danh mục càng ổn định.

        - **⚡ Sharpe Ratio = `{sr:.4f}`** *(càng lớn càng tốt)*
          $$Sharpe = \\frac{{R_p - R_f}}{{\\sigma_p}}$$
          với $R_f = {RF*100:.0f}\\%$ (lãi suất phi rủi ro).
          - **Sharpe > 1**: tốt
          - **Sharpe > 2**: rất tốt
          - **Sharpe > 3**: xuất sắc

        - **⏱️ Runtime = `{rt:.2f}s`**

        ### 🖼️ Các hình minh hoạ

        - **💼 Phân bổ danh mục** (bar chart):
          Trọng số $w_i$ của mỗi cổ phiếu trong danh mục.
          Sắp xếp giảm dần. Tổng các $w_i = 100\\%$.

        - **📉 Hội tụ Sharpe** — Sharpe Ratio tăng dần theo vòng lặp ACO.

        - **🔥 Ma trận tương quan** (heatmap) — Tương quan giữa từng cặp cổ phiếu.
          - **Đỏ (gần 1)**: tương quan dương mạnh (di chuyển cùng chiều)
          - **Xanh (gần 0)**: ít tương quan (đa dạng hoá tốt)

        ### 💡 Diễn giải đầu tư

        Theo lý thuyết **Markowitz (1952)**:
        - Tăng trọng số cổ phiếu có **Sharpe cao cá nhân** → tăng return chung
        - Kết hợp các cổ phiếu **ít tương quan** → giảm rủi ro tổng thể (diversification)

        ACO khám phá nhiều combinations weights → tìm portfolio **maximize Sharpe**.

        ### 📊 Top 3 cổ phiếu trong danh mục

        {chr(10).join(f"{i+1}. **{STOCK_NAMES[ix]}** ({SECTORS[ix]}) — weight = **{w[ix]*100:.2f}%**, "
                       f"expected return = {EXPECTED_RETURNS[ix]*100:.1f}%, σ = {STD_DEVS[ix]*100:.1f}%"
                       for i, ix in enumerate(idx[:3]))}

        ### ⚠️ Lưu ý

        - Đây là **Markowitz mean-variance** thuần — giả định Returns phân bố chuẩn.
        - Không tính các yếu tố như: thuế, phí giao dịch, ràng buộc thị trường.
        - Quá khứ không đảm bảo cho tương lai — dùng tham khảo, không thay tư vấn tài chính.
        """)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("💼 Phân bổ danh mục")
        fig1, ax = plt.subplots(figsize=(7, 5))
        ax.barh([STOCK_NAMES[i] for i in idx], [w[i] * 100 for i in idx],
                color="#2a5298", edgecolor="black")
        ax.set_xlabel("Weight (%)"); ax.invert_yaxis()
        ax.set_title("Trọng số tối ưu")
        for i, v in enumerate([w[j] * 100 for j in idx]):
            ax.text(v + 0.3, i, f"{v:.1f}%", va="center", fontsize=8)
        st.pyplot(fig1)
    with col2:
        st.subheader("📉 Hội tụ Sharpe")
        fig2 = plot_convergence(hist, "Hội tụ ACO-Portfolio", "Sharpe Ratio")
        st.pyplot(fig2)

    # Correlation heatmap
    with st.expander("🔥 Ma trận tương quan"):
        fig3, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(CORRELATION, cmap="RdYlBu_r", vmin=0, vmax=1)
        ax.set_xticks(range(10)); ax.set_yticks(range(10))
        ax.set_xticklabels(STOCK_NAMES, rotation=45)
        ax.set_yticklabels(STOCK_NAMES)
        for i in range(10):
            for j in range(10):
                ax.text(j, i, f"{CORRELATION[i,j]:.2f}", ha="center", va="center",
                        fontsize=7)
        plt.colorbar(im, ax=ax)
        st.pyplot(fig3)

    # ---------- 📄 NÚT TẢI PDF ----------
    st.markdown("---")
    st.subheader("📥 Xuất báo cáo PDF")
    pdf_bytes = generate_pdf_report(
        problem_title="Bài toán 09 – Portfolio Optimization",
        problem_subtitle="Tối ưu danh mục đầu tư Markowitz – 10 cổ phiếu Việt Nam",
        algorithm="ACO",
        inputs={
            "Số kiến": n_ants,
            "Số vòng lặp": n_iter,
            "Alpha": alpha,
            "Beta": beta,
            "Rho": rho,
            "Risk-free rate": f"{RF*100:.1f}%",
            "Seed": seed,
        },
        outputs={
            "Sharpe Ratio": f"{sr:.4f}",
            "Expected Return": f"{r*100:.2f}%",
            "Risk (σ)": f"{risk*100:.2f}%",
            "Thời gian chạy": f"{rt:.2f} giây",
            **{f"Weight {STOCK_NAMES[i]}": f"{w[i]*100:.2f}%" for i in idx if w[i] > 0.001},
        },
        figures=[
            ("Hình 1: Phân bổ danh mục tối ưu", fig1),
            ("Hình 2: Đồ thị hội tụ Sharpe", fig2),
            ("Hình 3: Ma trận tương quan", fig3),
        ],
    )
    download_pdf_button(pdf_bytes,
                       file_name=f"BaoCao_Portfolio_ACO.pdf",
                       key="pdf_portfolio")
