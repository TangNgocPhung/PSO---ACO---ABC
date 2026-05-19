"""Trang demo bài toán VRP với ACO."""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from algorithms.common import (page_header, back_button, plot_convergence, metric_row,
                                generate_pdf_report, download_pdf_button)
from algorithms.vrp_aco import aco_cvrp, make_random_cvrp

st.set_page_config(page_title="02 · VRP ACO", page_icon="🚚", layout="wide")
back_button()
page_header("🚚", "Bài toán 02 – CVRP",
            "Capacitated Vehicle Routing Problem giải bằng ACO", "ACO")

with st.expander("📖 Mô tả bài toán"):
    st.markdown("""
    **CVRP** – một depot, nhiều khách hàng có nhu cầu giao hàng,
    đội xe có sức chứa cố định. Mục tiêu tối thiểu tổng quãng đường mà không xe
    nào vượt sức chứa.
    """)

st.sidebar.markdown("## ⚙️ Tham số")
n_customers = st.sidebar.slider("Số khách hàng", 5, 30, 15)
capacity = st.sidebar.slider("Capacity / xe", 30, 200, 80)
n_ants = st.sidebar.slider("Số kiến", 10, 60, 20)
n_iter = st.sidebar.slider("Vòng lặp", 20, 200, 80)
alpha = st.sidebar.slider("Alpha", 0.1, 5.0, 1.0)
beta = st.sidebar.slider("Beta", 0.1, 5.0, 2.0)
rho = st.sidebar.slider("Rho", 0.05, 0.95, 0.4)
seed = st.sidebar.number_input("Seed", 0, 9999, 42)

coords, demands, cap = make_random_cvrp(n_customers, capacity, seed)
st.write(f"**Depot:** node 0 tại ({coords[0,0]:.0f},{coords[0,1]:.0f}) – **Capacity:** {capacity}")

fig0, ax0 = plt.subplots(figsize=(8, 6))
ax0.scatter(coords[1:, 0], coords[1:, 1], c="#3b82f6", s=110,
            edgecolor="white", linewidth=1.5, label="Khách hàng", zorder=3)
ax0.scatter(coords[0, 0], coords[0, 1], c="#ef4444", s=240, marker="s",
            edgecolor="white", linewidth=2, label="Depot", zorder=4)
# Annotate có offset + bbox trắng để không bị đè lên chấm
bbox_style = dict(boxstyle="round,pad=0.25", fc="white",
                  ec="#3b82f6", lw=0.6, alpha=0.9)
for i in range(1, len(coords)):
    ax0.annotate(f"{i} · d={demands[i]}",
                 (coords[i, 0], coords[i, 1]),
                 xytext=(9, 7), textcoords="offset points",
                 fontsize=8, fontweight="600", color="#1e1b4b",
                 bbox=bbox_style, zorder=5)
ax0.legend(loc="upper right", fontsize=9)
ax0.grid(True, alpha=.3)
ax0.set_title("Bản đồ depot & khách hàng",
              color="#1e1b4b", fontweight="bold")
ax0.margins(0.1)  # thêm padding mép để label không bị cắt
st.pyplot(fig0)

if st.button("🚀 Chạy ACO-CVRP", type="primary"):
    progress = st.progress(0)
    status = st.empty()
    def cb(it, total, best):
        progress.progress(it / total)
        status.write(f"Vòng {it}/{total} – Best total: **{best:.2f}**")
    with st.spinner("Đang chạy..."):
        routes, total, hist, rt = aco_cvrp(coords, demands, capacity,
                                            n_ants=n_ants, n_iter=n_iter,
                                            alpha=alpha, beta=beta, rho=rho,
                                            seed=seed, progress_cb=cb)
    status.empty()
    st.success(f"✅ Hoàn tất sau {rt:.2f}s")

    metric_row([
        ("📏 Tổng quãng đường", f"{total:.2f}", None),
        ("🚚 Số xe", f"{len(routes)}", None),
        ("⏱️ Thời gian", f"{rt:.2f}s", None),
        ("👥 Khách hàng", f"{n_customers}", None),
    ])

    with st.expander("📖 Giải thích kết quả", expanded=False):
        st.markdown(f"""
        ### 🎯 Các chỉ số đầu ra

        - **📏 Tổng quãng đường = `{total:.2f}`**
          Tổng chiều dài Euclid của tất cả `{len(routes)}` tuyến (đi + về depot).
          Càng nhỏ càng tốt — đây là mục tiêu chính cần tối thiểu hoá.

        - **🚚 Số xe = `{len(routes)}`**
          Số xe được sử dụng để giao hàng. Mỗi xe có **sức chứa = {capacity}** đơn vị.
          ACO tự quyết định số xe tối thiểu (đóng tuyến khi xe đã đầy → mở tuyến mới).

        - **⏱️ Thời gian** — Thời gian chạy `{n_iter}` vòng × `{n_ants}` kiến.

        - **👥 Khách hàng** — Số điểm cần phục vụ (không tính depot).

        ### 🖼️ Các hình minh hoạ

        - **🛣️ Các tuyến đường** — Mỗi xe có màu riêng (xanh dương, nâu, cyan…),
          xuất phát từ **depot (vuông đỏ)** đi qua các khách hàng rồi về depot.
          Trong legend: `Xe k (load=X/Y)` với X = tổng demand đã phục vụ, Y = capacity.

        - **📉 Đồ thị hội tụ** — Best total distance theo vòng lặp.
          Đường giảm → ACO ngày càng tìm được lời giải tốt hơn.

        ### 💡 Kiểm tra ràng buộc

        - ✅ **Capacity**: Không xe nào vượt {capacity} đơn vị (mỗi tuyến `load ≤ {capacity}`)
        - ✅ **Coverage**: {n_customers} khách hàng phục vụ đầy đủ
        - ✅ **Closed tour**: Mỗi xe quay lại depot

        ### 📊 Các tuyến cụ thể

        {chr(10).join(f"- **Xe {k+1}**: load = {sum(demands[c] for c in r)}/{capacity}, qua {len(r)} khách"
                       for k, r in enumerate(routes))}
        """)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🛣️ Các tuyến đường")
        fig1, ax = plt.subplots(figsize=(8, 6.5))
        colors = plt.cm.tab10(np.linspace(0, 1, max(len(routes), 1)))
        # Vẽ các tuyến trước (lwer zorder để chấm đè lên)
        for k, r in enumerate(routes):
            xs = [coords[0, 0]] + [coords[c, 0] for c in r] + [coords[0, 0]]
            ys = [coords[0, 1]] + [coords[c, 1] for c in r] + [coords[0, 1]]
            ax.plot(xs, ys, "-", color=colors[k], lw=2, alpha=.85,
                    label=f"Xe {k+1} (load={sum(demands[c] for c in r)})",
                    zorder=2)
        # Chấm khách hàng phía trên
        ax.scatter(coords[1:, 0], coords[1:, 1], c="#3b82f6", s=110,
                   edgecolor="white", linewidth=1.5, zorder=3)
        ax.scatter(coords[0, 0], coords[0, 1], c="#ef4444", s=240, marker="s",
                   edgecolor="white", linewidth=2, zorder=4)
        # Label offset + bbox
        bbox2 = dict(boxstyle="round,pad=0.2", fc="white",
                     ec="#94a3b8", lw=0.5, alpha=0.9)
        for i in range(1, len(coords)):
            ax.annotate(f"{i}", (coords[i, 0], coords[i, 1]),
                        xytext=(8, 6), textcoords="offset points",
                        fontsize=8, fontweight="700", color="#1e1b4b",
                        bbox=bbox2, zorder=5)
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=.3)
        ax.margins(0.1)
        st.pyplot(fig1)
    with col2:
        st.subheader("📉 Hội tụ")
        fig2 = plot_convergence(hist, "Hội tụ ACO-CVRP", "Tổng quãng đường")
        st.pyplot(fig2)

    with st.expander("📋 Chi tiết tuyến"):
        for k, r in enumerate(routes):
            st.write(f"**Xe {k+1}** (load={sum(demands[c] for c in r)}/{capacity}): "
                     f"0 → {' → '.join(map(str, r))} → 0")

    # ---------- 📄 NÚT TẢI PDF ----------
    st.markdown("---")
    st.subheader("📥 Xuất báo cáo PDF")
    pdf_bytes = generate_pdf_report(
        problem_title="Bài toán 02 – CVRP",
        problem_subtitle="Capacitated Vehicle Routing Problem giải bằng ACO",
        algorithm="ACO",
        inputs={
            "Số khách hàng": n_customers,
            "Capacity / xe": capacity,
            "Số kiến": n_ants,
            "Số vòng lặp": n_iter,
            "Alpha": alpha,
            "Beta": beta,
            "Rho": rho,
            "Seed": seed,
        },
        outputs={
            "Tổng quãng đường": f"{total:.2f}",
            "Số xe sử dụng": len(routes),
            "Thời gian chạy": f"{rt:.2f} giây",
            **{f"Tuyến xe {k+1} (load)": f"{sum(demands[c] for c in r)}/{capacity}"
               for k, r in enumerate(routes)},
        },
        figures=[
            ("Hình 0: Bản đồ depot + khách hàng (input)", fig0),
            ("Hình 1: Các tuyến đường tối ưu", fig1),
            ("Hình 2: Đồ thị hội tụ", fig2),
        ],
    )
    download_pdf_button(pdf_bytes,
                       file_name=f"BaoCao_CVRP_ACO_n{n_customers}.pdf",
                       key="pdf_cvrp")
