"""Trang demo VRPTW với PSO (chuẩn) – hỗ trợ upload Solomon CSV."""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from algorithms.common import (page_header, back_button, plot_convergence, metric_row,
                                generate_pdf_report, download_pdf_button)
from algorithms.vrptw_pso import (pso_vrptw, make_solomon_like, load_solomon_csv)

st.set_page_config(page_title="06 · VRPTW PSO", page_icon="⏰", layout="wide")
back_button()
page_header("⏰", "Bài toán 06 – VRPTW",
            "Vehicle Routing Problem with Time Windows – PSO chuẩn (có gbest-guided move)", "PSO")

with st.expander("📖 Mô tả bài toán & thuật toán", expanded=False):
    st.markdown("""
    **VRPTW** mở rộng của CVRP: mỗi khách hàng có khung thời gian phục vụ
    $[ready_i, due_i]$. Xe phải đến trong khung này – nếu sớm thì chờ,
    nếu trễ thì bị phạt nặng.

    **Hàm fitness:**

    $$f = \\text{total\\_dist} + \\lambda \\cdot \\text{time\\_penalty} + 50 \\cdot \\text{n\\_vehicles}$$

    với $\\lambda$ = `penalty_coef` (mặc định 1000 — theo notebook gốc).

    **PSO chuẩn cho VRPTW** kết hợp 2 toán tử:
    1. **Swap operator** (cognitive) — tráo đổi 2 khách ngẫu nhiên trong particle
    2. **gbest-guided gene copy** (social) — sao chép 1 "gene" từ gbest sang particle
       với xác suất `gbest_prob` → giúp particle học từ giải pháp tốt nhất toàn cục

    Đây chính là phần *social learning* đã được bổ sung so với phiên bản trước
    (chỉ có swap đơn thuần thì giống local search hơn là PSO thực thụ).
    """)

# -------- Sidebar --------
st.sidebar.markdown("## 📂 Nguồn dữ liệu")
data_src = st.sidebar.radio(
    "Chọn nguồn",
    ["Tự sinh (Solomon-like)", "Upload Solomon CSV"],
    help="Tự sinh: random ngẫu nhiên kiểu Solomon. Upload: dùng dataset thật C101/R101/RC101..."
)

if data_src == "Tự sinh (Solomon-like)":
    n_customers = st.sidebar.slider("Số khách hàng", 10, 50, 20)
    capacity = st.sidebar.slider("Capacity", 50, 400, 200)
    seed = st.sidebar.number_input("Seed (tạo dữ liệu)", 0, 9999, 42)
    customers, cap = make_solomon_like(n_customers, capacity, seed)
else:
    uploaded = st.sidebar.file_uploader(
        "Solomon CSV (cột: CUST_NO, X, Y, DEMAND, READY_TIME, DUE_DATE, SERVICE_TIME)",
        type=["csv", "txt"],
    )
    capacity = st.sidebar.slider("Capacity", 50, 1000, 200)
    normalize = st.sidebar.checkbox("Chuẩn hoá toạ độ (MinMax 0-100)", value=False)
    seed = st.sidebar.number_input("Seed (PSO)", 0, 9999, 42)
    if uploaded is None:
        st.warning("📤 Vui lòng upload file Solomon CSV để chạy tiếp.")
        st.info("Bạn có thể tải các dataset Solomon chuẩn tại "
                "[neo.lcc.uma.es/vrp/vrp-instances/](https://neo.lcc.uma.es/vrp/vrp-instances/)")
        st.stop()
    try:
        df = pd.read_csv(uploaded)
        customers, cap = load_solomon_csv(df, capacity, normalize=normalize)
        st.success(f"✅ Đã load {len(customers)} điểm (1 depot + {len(customers)-1} khách)")
    except Exception as e:
        st.error(f"❌ Lỗi load CSV: {e}")
        st.stop()

st.sidebar.markdown("## ⚙️ Tham số PSO")
n_particles = st.sidebar.slider("Số particles", 10, 100, 30)
n_iter = st.sidebar.slider("Vòng lặp", 30, 300, 80)
penalty_coef = st.sidebar.slider(
    "Penalty coef λ", 10, 5000, 1000, step=10,
    help="Hệ số phạt khi đến trễ. Càng lớn → buộc tuyệt đối tuân thủ time window."
)
gbest_prob = st.sidebar.slider(
    "Gbest-guided prob", 0.0, 1.0, 0.5,
    help="Xác suất sao chép gene từ gbest (thành phần social)."
)
swap_prob = st.sidebar.slider(
    "Swap prob", 0.0, 1.0, 0.7,
    help="Xác suất swap 2 khách (thành phần cognitive)."
)
seed_pso = st.sidebar.number_input("Seed (PSO)", 0, 9999, 42, key="seed_pso_unique")

# -------- Thông tin dữ liệu --------
depot = customers[0]
st.markdown(f"""
**Depot:** ({depot['x']:.1f}, {depot['y']:.1f}) · **Capacity:** {cap} ·
**Số khách:** {len(customers)-1}
""")

col_map, col_tbl = st.columns([3, 2])
with col_map:
    fig0, ax0 = plt.subplots(figsize=(7, 5))
    xs = [c["x"] for c in customers[1:]]
    ys = [c["y"] for c in customers[1:]]
    ax0.scatter(xs, ys, c="#3b82f6", s=70, label="Khách hàng", zorder=3,
                edgecolor="white", linewidth=1.5)
    ax0.scatter(depot["x"], depot["y"], c="#ef4444", s=250, marker="s",
                label="Depot", zorder=4, edgecolor="white", linewidth=1.5)
    for c in customers[1:]:
        ax0.annotate(f"{c['id']}", (c["x"], c["y"]),
                     xytext=(7, 5), textcoords="offset points",
                     fontsize=8, fontweight="700", color="#1e1b4b",
                     bbox=dict(boxstyle="round,pad=0.2", fc="white",
                               ec="#94a3b8", lw=0.5, alpha=0.9),
                     zorder=5)
    ax0.legend(loc="best", fontsize=9)
    ax0.set_title("Bản đồ depot & khách hàng", color="#1e1b4b", fontweight="bold")
    st.pyplot(fig0)

with col_tbl:
    st.markdown("**Bảng dữ liệu khách hàng (5 dòng đầu):**")
    preview_df = pd.DataFrame(customers[:6])
    st.dataframe(preview_df, hide_index=True, use_container_width=True, height=240)

# -------- Chạy thuật toán --------
if st.button("🚀 Chạy PSO-VRPTW (PSO chuẩn)", type="primary"):
    progress = st.progress(0); status = st.empty()
    def cb(it, total, best):
        progress.progress(it / total)
        status.write(f"Vòng {it}/{total} – Best fitness: **{best:.2f}**")

    with st.spinner("Đang chạy PSO với gbest-guided move..."):
        routes, total_dist, n_vehicles, hist, rt, n_late = pso_vrptw(
            customers, cap,
            n_particles=n_particles, n_iter=n_iter,
            penalty_coef=penalty_coef, gbest_prob=gbest_prob, swap_prob=swap_prob,
            seed=seed_pso, progress_cb=cb)
    status.empty()

    if n_late == 0:
        st.success(f"✅ Hoàn tất! {n_vehicles} xe, tổng đường {total_dist:.2f}, "
                   f"**không có khách nào bị trễ**.")
    else:
        st.warning(f"⚠️ {n_vehicles} xe, tổng đường {total_dist:.2f}, "
                   f"còn **{n_late} khách bị trễ** (tăng penalty_coef hoặc vòng lặp).")

    # ----- Metrics -----
    metric_row([
        ("🚚 Số xe", f"{n_vehicles}", None),
        ("📏 Tổng đường", f"{total_dist:.2f}", None),
        ("⏰ Khách trễ", f"{n_late}", "Vi phạm time window"),
        ("⏱️ Runtime", f"{rt:.2f}s", None),
    ])

    with st.expander("📖 Giải thích kết quả", expanded=False):
        st.markdown(f"""
        ### 🎯 Các chỉ số đầu ra

        - **🚚 Số xe = `{n_vehicles}`** — Số xe cần huy động để phục vụ {len(customers)-1}
          khách thoả mãn cả capacity và time window.

        - **📏 Tổng đường = `{total_dist:.2f}`** — Tổng quãng đường (Euclid)
          của tất cả xe (depot → khách → khách → ... → depot).

        - **⏰ Số khách trễ = `{n_late}`**
          Số khách hàng mà xe đến **sau due_date** (vi phạm time window cứng).
          {"✅ Lời giải HỢP LỆ — không khách nào bị trễ." if n_late == 0
           else "⚠️ Tăng `penalty_coef` lên 1500-2000 hoặc tăng số particles/vòng lặp."}

        - **⏱️ Runtime = `{rt:.2f}s`** — Thời gian PSO chạy `{n_iter}` vòng.

        ### 🖼️ Các hình minh hoạ

        - **🛣️ Các tuyến đường tối ưu** — Mỗi xe có màu riêng, xuất phát/kết thúc
          tại **depot (vuông đỏ)**. Legend hiển thị `load = X/{cap}` cho mỗi xe.

        - **📉 Hội tụ PSO** — Fitness theo vòng lặp.
          $$fitness = total\\_dist + \\lambda \\cdot tardiness + 50 \\cdot n\\_vehicles$$
          Đường giảm → PSO ngày càng tốt. Có thể dao động hơn ACO do swap-based.

        ### ⏰ Kiểm tra time window

        Mỗi khách có khung $[ready_i, due_i]$. Xe đến tại thời điểm $t$:
        - $t < ready_i$ → **chờ** đến $ready_i$ (không phạt nhưng tốn thời gian)
        - $ready_i \\le t \\le due_i$ → **đúng giờ** ✓
        - $t > due_i$ → **trễ** ✗ → phạt $\\lambda_2 \\cdot (t - due_i)$

        Xem chi tiết time window cho từng khách trong expander **"📋 Chi tiết tuyến"** ở dưới.

        ### 💡 Mẹo

        - Tăng `Gbest-guided prob` → particles học gbest nhiều hơn → hội tụ nhanh hơn nhưng dễ kẹt
        - Tăng `Swap prob` → nhiều exploration cá nhân hơn → đa dạng hơn nhưng chậm
        """)

    # ----- Trực quan hoá -----
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🛣️ Các tuyến đường tối ưu")
        fig1, ax = plt.subplots(figsize=(8, 6.5))
        ax.scatter(xs, ys, c="#3b82f6", s=70, zorder=3,
                   edgecolor="white", linewidth=1.5)
        ax.scatter(depot["x"], depot["y"], c="#ef4444", s=250, marker="s",
                   zorder=4, edgecolor="white", linewidth=1.5)
        colors = plt.cm.tab10(np.linspace(0, 1, max(len(routes), 1)))
        for k, r in enumerate(routes):
            rx = [depot["x"]] + [c["x"] for c in r] + [depot["x"]]
            ry = [depot["y"]] + [c["y"] for c in r] + [depot["y"]]
            load = sum(c["demand"] for c in r)
            ax.plot(rx, ry, "-o", color=colors[k], lw=1.8, markersize=5,
                    label=f"Xe {k+1} ({load}/{cap})")
        for c in customers[1:]:
            ax.annotate(f"{c['id']}", (c["x"], c["y"]),
                        xytext=(7, 5), textcoords="offset points",
                        fontsize=8, fontweight="700", color="#1e1b4b",
                        bbox=dict(boxstyle="round,pad=0.2", fc="white",
                                  ec="#94a3b8", lw=0.5, alpha=0.9),
                        zorder=5)
        ax.legend(loc="best", fontsize=8)
        ax.set_title(f"Lời giải tối ưu ({n_vehicles} xe)",
                     color="#1e1b4b", fontweight="bold")
        st.pyplot(fig1)

    with col2:
        st.subheader("📉 Hội tụ PSO")
        fig2 = plot_convergence(hist, "Hội tụ PSO-VRPTW", "Fitness (lower = better)")
        st.pyplot(fig2)

    # ----- Chi tiết tuyến + kiểm tra time window -----
    with st.expander("📋 Chi tiết tuyến + kiểm tra time window"):
        for k, r in enumerate(routes):
            load = sum(c["demand"] for c in r)
            st.markdown(f"**🚚 Xe {k+1}** – Load: {load}/{cap}")
            time_now = 0.0
            prev_id = depot["id"]
            from algorithms.vrptw_pso import build_distance_matrix
            dist = build_distance_matrix(customers)
            log_rows = []
            for c in r:
                d = dist[prev_id, c["id"]]
                arrival = time_now + d
                wait = max(0, c["ready_time"] - arrival)
                start_service = max(arrival, c["ready_time"])
                late = max(0, start_service - c["due_date"])
                end_service = start_service + c["service_time"]
                log_rows.append({
                    "khách": c["id"],
                    "demand": c["demand"],
                    "đến": f"{arrival:.1f}",
                    "ready/due": f"[{c['ready_time']:.0f}, {c['due_date']:.0f}]",
                    "chờ": f"{wait:.1f}",
                    "trễ": f"{late:.1f}" if late > 0 else "—",
                    "service": f"{start_service:.1f}→{end_service:.1f}",
                })
                time_now = end_service
                prev_id = c["id"]
            st.dataframe(pd.DataFrame(log_rows), hide_index=True,
                         use_container_width=True)

    # ---------- 📄 NÚT TẢI PDF ----------
    st.markdown("---")
    st.subheader("📥 Xuất báo cáo PDF")
    pdf_bytes = generate_pdf_report(
        problem_title="Bài toán 06 – VRPTW (PSO)",
        problem_subtitle="Vehicle Routing Problem with Time Windows giải bằng PSO chuẩn",
        algorithm="PSO",
        inputs={
            "Nguồn dữ liệu": data_src,
            "Số khách hàng": len(customers) - 1,
            "Capacity": cap,
            "Số particles": n_particles,
            "Số vòng lặp": n_iter,
            "Penalty coef λ": penalty_coef,
            "Gbest-guided prob": gbest_prob,
            "Swap prob": swap_prob,
            "Seed": seed_pso,
        },
        outputs={
            "Số xe sử dụng": n_vehicles,
            "Tổng quãng đường": f"{total_dist:.2f}",
            "Khách bị trễ": n_late,
            "Thời gian chạy": f"{rt:.2f} giây",
        },
        figures=[
            ("Hình 0: Bản đồ depot + khách hàng (input)", fig0),
            ("Hình 1: Các tuyến đường tối ưu", fig1),
            ("Hình 2: Đồ thị hội tụ PSO", fig2),
        ],
    )
    download_pdf_button(pdf_bytes,
                       file_name=f"BaoCao_VRPTW_PSO_n{len(customers)-1}.pdf",
                       key="pdf_vrptw_pso")
