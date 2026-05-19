"""Trang demo VRPTW với PSO (chuẩn) – hỗ trợ upload Solomon CSV."""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from algorithms.common import page_header, back_button, plot_convergence, metric_row
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
        ax0.text(c["x"], c["y"], f" {c['id']}", fontsize=7, color="#1e1b4b")
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

    # ----- Trực quan hoá -----
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🛣️ Các tuyến đường tối ưu")
        fig, ax = plt.subplots(figsize=(8, 6.5))
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
            ax.text(c["x"], c["y"], f" {c['id']}", fontsize=7, color="#1e1b4b")
        ax.legend(loc="best", fontsize=8)
        ax.set_title(f"Lời giải tối ưu ({n_vehicles} xe)",
                     color="#1e1b4b", fontweight="bold")
        st.pyplot(fig)

    with col2:
        st.subheader("📉 Hội tụ PSO")
        st.pyplot(plot_convergence(hist, "Hội tụ PSO-VRPTW", "Fitness (lower = better)"))

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
