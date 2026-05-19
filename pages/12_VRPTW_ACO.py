"""Trang demo VRPTW với ACO (port từ notebook gốc)."""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from algorithms.common import (page_header, back_button, plot_convergence, metric_row,
                                generate_pdf_report, download_pdf_button)
from algorithms.vrptw_aco import (aco_vrptw, generate_solomon_like,
                                    distance_matrix, evaluate_routes)

st.set_page_config(page_title="12 · VRPTW ACO", page_icon="🐜", layout="wide")
back_button()
page_header("🐜⏰", "Bài toán 12 – VRPTW",
            "Vehicle Routing Problem with Time Windows giải bằng Ant Colony Optimization", "ACO")

with st.expander("📖 Mô tả bài toán & thuật toán", expanded=False):
    st.markdown("""
    **VRPTW** – tương tự bài toán 06, nhưng ở đây giải bằng **ACO** (page 06 dùng PSO).

    **Tại sao ACO cho VRPTW?**
    - VRPTW có cấu trúc đồ thị rõ ràng (cạnh giữa các khách hàng) → pheromone tự nhiên
    - Constraint check tại từng bước → kiến chỉ chọn khách KHẢ THI tại thời điểm đó
    - Pheromone tích luỹ → các edge "tốt" được kế thừa qua nhiều thế hệ

    **Mã giả ACO-VRPTW:**

    ```
    1. Khởi tạo τ[i,j] = τ₀, tính d[i,j], η[i,j] = 1/d[i,j]
    2. FOR iter = 1..num_iter:
    3.   FOR each ant k:
    4.     unvisited = {1, ..., N}
    5.     WHILE unvisited:
    6.       Bắt đầu tuyến mới với xe rỗng tại depot
    7.       WHILE còn khách khả thi (capacity OK & time window OK):
    8.         next ← chọn theo τ^α · η^β
    9.         Cập nhật time, load, route
    10.    Tính cost = quãng + λ₁·#xe + λ₂·tardiness
    11.    Cập nhật best toàn cục
    12.  τ ← (1-ρ)·τ + bổ sung pheromone cho top-3 ant + elitist best
    ```

    **So sánh ACO (page 12) vs PSO (page 06) cho VRPTW:**

    |  | ACO (page 12) | PSO (page 06) |
    |---|---|---|
    | Đại diện | Pheromone trên cạnh | Hoán vị khách |
    | Cập nhật | Bay hơi + bổ sung | Vận tốc + swap |
    | Constraint | Check tại từng bước (kiến chỉ chọn khách feasible) | Check sau khi build (penalty) |
    | Hội tụ | Mượt, dần dần | Có thể nhanh hơn nhưng dễ dao động |
    """)

# -------- Sidebar --------
st.sidebar.markdown("## 📊 Dataset")
n_customers = st.sidebar.slider("Số khách hàng", 10, 50, 20)
capacity = st.sidebar.slider("Capacity / xe", 50, 400, 200)
data_seed = st.sidebar.number_input("Seed dữ liệu", 0, 9999, 42)

st.sidebar.markdown("## 🐜 Tham số ACO")
num_ants = st.sidebar.slider("Số kiến (num_ants)", 5, 80, 25)
num_iter = st.sidebar.slider("Vòng lặp (num_iter)", 30, 300, 100)
alpha = st.sidebar.slider("α (pheromone weight)", 0.1, 5.0, 1.0)
beta = st.sidebar.slider("β (heuristic weight)", 0.1, 5.0, 2.5)
rho = st.sidebar.slider("ρ (evaporation rate)", 0.05, 0.5, 0.15)
Q = st.sidebar.slider("Q (deposit factor)", 10, 500, 100)
elitist = st.sidebar.checkbox("Elitist (bổ sung mạnh cho best)", value=True)
aco_seed = st.sidebar.number_input("Seed ACO", 0, 9999, 42, key="aco_seed")

st.sidebar.markdown("## ⚖️ Tham số fitness")
lambda_veh = st.sidebar.slider("λ₁ (vehicle penalty)", 0, 200, 50)
lambda_late = st.sidebar.slider("λ₂ (lateness penalty)", 100, 5000, 1000, step=100)

# -------- Sinh dữ liệu --------
customers, cap = generate_solomon_like(n_customers, capacity, data_seed)
depot = customers[0]

st.markdown(f"""
**Depot:** ({depot['x']:.0f}, {depot['y']:.0f}) · **Capacity:** {cap} ·
**Số khách:** {len(customers)-1}
""")

# Bản đồ input
fig0, ax0 = plt.subplots(figsize=(8, 5.5))
xs = [c["x"] for c in customers[1:]]
ys = [c["y"] for c in customers[1:]]
window_widths = np.array([c["due_date"] - c["ready_time"] for c in customers[1:]])
sc = ax0.scatter(xs, ys, c=window_widths, cmap="RdYlGn", s=120,
                 edgecolor="black", linewidth=1, zorder=3)
ax0.scatter(depot["x"], depot["y"], c="black", s=300, marker="s",
            label="Depot", zorder=4)
for c in customers[1:]:
    ax0.annotate(f"{c['id']}", (c["x"], c["y"]),
                 xytext=(7, 5), textcoords="offset points",
                 fontsize=8, fontweight="700", color="#1e1b4b",
                 bbox=dict(boxstyle="round,pad=0.2", fc="white",
                           ec="#94a3b8", lw=0.5, alpha=0.9),
                 zorder=5)
plt.colorbar(sc, ax=ax0, label="Độ rộng time window (lớn = dễ phục vụ)")
ax0.legend(); ax0.set_xlabel("X"); ax0.set_ylabel("Y")
ax0.set_title("Bản đồ depot + khách hàng (màu theo độ rộng time window)",
              color="#1e1b4b", fontweight="bold")
st.pyplot(fig0)

# -------- Chạy ACO --------
if st.button("🚀 Chạy ACO-VRPTW", type="primary"):
    progress = st.progress(0); status = st.empty()
    def cb(it, total, best):
        progress.progress(it / total)
        status.write(f"Vòng {it}/{total} – Best cost: **{best:.2f}**")

    with st.spinner("Đàn kiến đang xây dựng tuyến..."):
        res = aco_vrptw(
            customers, cap,
            num_ants=num_ants, num_iter=num_iter,
            alpha=alpha, beta=beta, rho=rho, Q=Q,
            lambda_veh=lambda_veh, lambda_late=lambda_late,
            elitist=elitist, seed=aco_seed, progress_cb=cb)
    status.empty()

    if res["tardiness"] == 0 and res["cap_violations"] == 0:
        st.success(f"✅ Hoàn tất! {res['n_vehicles']} xe, tổng đường {res['total_dist']:.2f}, "
                   "không vi phạm ràng buộc nào.")
    else:
        st.warning(f"⚠️ {res['n_vehicles']} xe, đường {res['total_dist']:.2f}, "
                   f"tardiness {res['tardiness']:.1f}, "
                   f"capacity violations {res['cap_violations']}")

    # ---------- Metrics ----------
    metric_row([
        ("🚚 Số xe", f"{res['n_vehicles']}", None),
        ("📏 Tổng đường", f"{res['total_dist']:.2f}", None),
        ("⏰ Tardiness", f"{res['tardiness']:.1f}", "Tổng thời gian trễ"),
        ("⏱️ Runtime", f"{res['runtime']:.2f}s", f"{num_iter} vòng × {num_ants} kiến"),
    ])

    # ---------- Hình 1: Routes ----------
    st.subheader("🛣️ Hình 1: Các tuyến đường tối ưu")
    fig1, ax = plt.subplots(figsize=(10, 6.5))
    ax.scatter(xs, ys, c="#3b82f6", s=120, edgecolor="white",
               linewidth=1.5, zorder=3)
    ax.scatter(depot["x"], depot["y"], c="#ef4444", s=300, marker="s",
               edgecolor="white", linewidth=2, zorder=4, label="Depot")
    for c in customers[1:]:
        ax.annotate(f"{c['id']}", (c["x"], c["y"]),
                    xytext=(7, 5), textcoords="offset points",
                    fontsize=8, fontweight="700", color="#1e1b4b",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white",
                              ec="#94a3b8", lw=0.5, alpha=0.9),
                    zorder=5)
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(res["best_routes"]), 1)))
    for k, r in enumerate(res["best_routes"]):
        load = sum(customers[cid]["demand"] for cid in r)
        rx = [depot["x"]] + [customers[cid]["x"] for cid in r] + [depot["x"]]
        ry = [depot["y"]] + [customers[cid]["y"] for cid in r] + [depot["y"]]
        ax.plot(rx, ry, "-o", color=colors[k], lw=2, markersize=6,
                label=f"Xe {k+1} ({load} đv)", alpha=0.85)
    ax.legend(loc="best", fontsize=9)
    ax.set_xlabel("X"); ax.set_ylabel("Y")
    ax.set_title(f"ACO-VRPTW: {res['n_vehicles']} xe, "
                 f"tổng đường = {res['total_dist']:.1f}",
                 color="#1e1b4b", fontweight="bold")
    st.pyplot(fig1)

    # ---------- Hình 2: Hội tụ ----------
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📉 Hình 2: Hội tụ Best/Avg")
        fig2, ax = plt.subplots(figsize=(8, 5))
        its = range(1, len(res["history_best"]) + 1)
        ax.plot(its, res["history_best"], color="#10b981", lw=2.5,
                marker="o", markersize=4, label="Best")
        ax.plot(its, res["history_avg"], color="#f59e0b", lw=1.5,
                alpha=0.7, ls="--", label="Avg")
        ax.fill_between(its, res["history_best"], res["history_avg"],
                        alpha=0.15, color="#10b981")
        ax.set_xlabel("Vòng lặp"); ax.set_ylabel("Cost (fitness)")
        ax.set_title("Hội tụ của ACO-VRPTW",
                     color="#1e1b4b", fontweight="bold")
        ax.legend()
        st.pyplot(fig2)

    # ---------- Hình 3: Pheromone heatmap ----------
    with col2:
        st.subheader("🔥 Hình 3: Pheromone heatmap")
        fig3, ax = plt.subplots(figsize=(8, 6))
        pher_disp = res["pheromone"] + 1e-3
        im = ax.imshow(pher_disp, cmap="inferno",
                       norm=LogNorm(vmin=pher_disp.min(), vmax=pher_disp.max()),
                       aspect="auto", interpolation="nearest")
        plt.colorbar(im, ax=ax, label="Pheromone (log scale)")
        ax.set_xlabel("Khách đến (j)"); ax.set_ylabel("Khách đi (i)")
        ax.set_title("Ma trận pheromone cuối cùng",
                     color="#1e1b4b", fontweight="bold")
        st.pyplot(fig3)

    # ---------- Hình 4: Gantt chart ----------
    st.subheader("📅 Hình 4: Gantt chart lịch trình các xe")
    D = res["distance_matrix"]
    fig4, ax = plt.subplots(figsize=(14, 1.5 + 0.8 * len(res["best_routes"])))
    for k, r in enumerate(res["best_routes"]):
        prev = 0
        t = 0.0
        for cid in r:
            c = customers[cid]
            arrival = t + D[prev, cid]
            # khung thời gian (background)
            ax.barh(k, c["due_date"] - c["ready_time"], left=c["ready_time"],
                    color="lightgray", alpha=0.3, edgecolor="gray")
            # travel
            ax.barh(k, D[prev, cid], left=t, color="#6366f1",
                    alpha=0.6, height=0.5)
            start_service = max(arrival, c["ready_time"])
            # wait
            if start_service > arrival:
                ax.barh(k, start_service - arrival, left=arrival,
                        color="#fbbf24", alpha=0.7, height=0.5)
            # service
            ax.barh(k, c["service_time"], left=start_service,
                    color=colors[k], alpha=0.9, height=0.7,
                    edgecolor="black", lw=0.5)
            ax.text(start_service + c["service_time"]/2, k, f"{cid}",
                    ha="center", va="center", fontsize=8,
                    fontweight="bold", color="white")
            t = start_service + c["service_time"]
            prev = cid
    ax.set_yticks(range(len(res["best_routes"])))
    ax.set_yticklabels([f"Xe {i+1}" for i in range(len(res["best_routes"]))])
    ax.set_xlabel("Thời gian")
    ax.set_title("Gantt chart (xám = window | tím = travel | "
                 "vàng = wait | màu = service)",
                 color="#1e1b4b", fontweight="bold")
    st.pyplot(fig4)

    # ---------- 📄 NÚT TẢI PDF BÁO CÁO ----------
    st.markdown("---")
    st.subheader("📥 Xuất báo cáo PDF")
    st.markdown(
        "Tải file PDF gồm **đầy đủ tham số đầu vào, kết quả và 5 hình minh hoạ** "
        "để đưa vào báo cáo / lưu trữ."
    )
    inputs_dict = {
        "Số khách hàng": n_customers,
        "Capacity / xe": cap,
        "Seed dữ liệu": data_seed,
        "Số kiến (num_ants)": num_ants,
        "Số vòng lặp (num_iter)": num_iter,
        "α (pheromone)": alpha,
        "β (heuristic)": beta,
        "ρ (evaporation)": rho,
        "Q (deposit)": Q,
        "Elitist": "Bật" if elitist else "Tắt",
        "λ₁ vehicle penalty": lambda_veh,
        "λ₂ lateness penalty": lambda_late,
        "Seed ACO": aco_seed,
    }
    outputs_dict = {
        "Số xe sử dụng": res["n_vehicles"],
        "Tổng quãng đường": f"{res['total_dist']:.2f}",
        "Tổng tardiness": f"{res['tardiness']:.2f}",
        "Vi phạm capacity": res["cap_violations"],
        "Best fitness (cost)": f"{res['best_cost']:.2f}",
        "Thời gian chạy": f"{res['runtime']:.2f} giây",
    }
    figures_list = [
        ("Hình 0: Bản đồ depot + khách hàng (input)", fig0),
        ("Hình 1: Các tuyến đường tối ưu", fig1),
        ("Hình 2: Đồ thị hội tụ Best/Avg", fig2),
        ("Hình 3: Heatmap pheromone (log scale)", fig3),
        ("Hình 4: Gantt chart lịch trình các xe", fig4),
    ]
    pdf_bytes = generate_pdf_report(
        problem_title="Bài toán 12 – VRPTW (ACO)",
        problem_subtitle="Vehicle Routing Problem with Time Windows giải bằng Ant Colony Optimization",
        algorithm="ACO",
        inputs=inputs_dict,
        outputs=outputs_dict,
        figures=figures_list,
        note=("Mẫu báo cáo tự sinh từ giao diện CSTT Demo. "
              "Xám = time window, tím = travel, vàng = wait, màu = service."),
    )
    download_pdf_button(pdf_bytes,
                       file_name=f"BaoCao_VRPTW_ACO_n{n_customers}_iter{num_iter}.pdf",
                       key="pdf_vrptw_aco")

    # ---------- Chi tiết tuyến ----------
    with st.expander("📋 Chi tiết từng tuyến + kiểm tra time window"):
        import pandas as pd
        for k, r in enumerate(res["best_routes"]):
            load = sum(customers[cid]["demand"] for cid in r)
            st.markdown(f"**🚚 Xe {k+1}** – Load: {load}/{cap}")
            rows = []
            prev = 0
            t = 0.0
            for cid in r:
                c = customers[cid]
                arrival = t + D[prev, cid]
                wait = max(0, c["ready_time"] - arrival)
                start_service = max(arrival, c["ready_time"])
                late = max(0, start_service - c["due_date"])
                end_service = start_service + c["service_time"]
                rows.append({
                    "khách": cid,
                    "demand": c["demand"],
                    "đến": f"{arrival:.1f}",
                    "ready/due": f"[{c['ready_time']:.0f}, {c['due_date']:.0f}]",
                    "chờ": f"{wait:.1f}",
                    "trễ": f"{late:.1f}" if late > 0 else "—",
                    "service": f"{start_service:.1f}→{end_service:.1f}",
                })
                t = end_service
                prev = cid
            st.dataframe(pd.DataFrame(rows), hide_index=True,
                         use_container_width=True)
