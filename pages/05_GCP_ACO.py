"""Trang demo GCP với ACO – Karate Club graph."""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from algorithms.common import (page_header, back_button, plot_convergence, metric_row,
                                generate_pdf_report, download_pdf_button)
from algorithms.gcp_aco import aco_gcp

st.set_page_config(page_title="05 · GCP ACO", page_icon="🎨", layout="wide")
back_button()
page_header("🎨", "Bài toán 05 – GCP",
            "Graph Coloring Problem – Karate Club Graph", "ACO")

with st.expander("📖 Mô tả bài toán"):
    st.markdown("""
    **Graph Coloring** – tô màu các đỉnh sao cho 2 đỉnh kề nhau khác màu, dùng ít màu nhất.
    Bài toán NP-Hard. Sử dụng Karate Club (Zachary 1977) – 34 đỉnh, 78 cạnh.
    """)

st.sidebar.markdown("## ⚙️ Tham số")
graph_choice = st.sidebar.selectbox("Đồ thị", ["Karate Club (34)", "Petersen (10)", "Ngẫu nhiên (20)"])
max_colors = st.sidebar.slider("Số màu tối đa", 2, 10, 5)
n_ants = st.sidebar.slider("Số kiến", 10, 80, 30)
n_iter = st.sidebar.slider("Vòng lặp", 20, 200, 80)
alpha = st.sidebar.slider("Alpha", 0.1, 5.0, 1.0)
beta = st.sidebar.slider("Beta", 0.1, 5.0, 2.0)
rho = st.sidebar.slider("Rho", 0.05, 0.95, 0.3)
seed = st.sidebar.number_input("Seed", 0, 9999, 42)

if graph_choice == "Karate Club (34)":
    G = nx.karate_club_graph()
elif graph_choice == "Petersen (10)":
    G = nx.petersen_graph()
else:
    G = nx.erdos_renyi_graph(20, 0.25, seed=seed)

st.write(f"**|V|** = {G.number_of_nodes()} | **|E|** = {G.number_of_edges()}")

pos = nx.spring_layout(G, seed=seed)
fig0, ax0 = plt.subplots(figsize=(7, 5))
nx.draw(G, pos, with_labels=True, node_color="lightblue", node_size=350,
        font_size=9, ax=ax0, edge_color="gray")
ax0.set_title("Đồ thị ban đầu")
st.pyplot(fig0)

if st.button("🚀 Chạy ACO-GCP", type="primary"):
    progress = st.progress(0); status = st.empty()
    def cb(it, total, best):
        progress.progress(it / total)
        status.write(f"Vòng {it}/{total} – Best fitness: **{best:.0f}**")
    with st.spinner("Đang chạy..."):
        coloring, fit, conflicts, hist, rt = aco_gcp(G, max_colors=max_colors,
                                                      num_ants=n_ants, num_iter=n_iter,
                                                      alpha=alpha, beta=beta, rho=rho,
                                                      seed=seed, progress_cb=cb)
    status.empty()
    n_used = len(set(coloring.values()))
    if conflicts == 0:
        st.success(f"✅ Tô màu hợp lệ – dùng **{n_used}** màu")
    else:
        st.warning(f"⚠️ Còn {conflicts} xung đột – cần tăng max_colors hoặc số vòng lặp")

    metric_row([
        ("🎨 Số màu dùng", f"{n_used}", None),
        ("⚠️ Xung đột", f"{conflicts}", None),
        ("🎯 Fitness", f"{fit:.0f}", None),
        ("⏱️ Thời gian", f"{rt:.2f}s", None),
    ])

    with st.expander("📖 Giải thích kết quả", expanded=False):
        st.markdown(f"""
        ### 🎯 Các chỉ số đầu ra

        - **🎨 Số màu dùng = `{n_used}`**
          Số màu phân biệt được dùng để tô tất cả {G.number_of_nodes()} đỉnh.
          Càng nhỏ càng tốt. **Chromatic number** $\\chi(G)$ là số màu tối thiểu cần thiết.

        - **⚠️ Số xung đột = `{conflicts}`**
          Số cạnh $(u,v) \\in E$ có **2 đỉnh cùng màu** — vi phạm ràng buộc.
          {"**Lời giải HỢP LỆ** — không vi phạm cạnh nào!" if conflicts == 0
           else f"**Lời giải VI PHẠM** — cần tăng max_colors hoặc số vòng lặp."}

        - **🎯 Fitness = `{fit:.0f}`**
          $$f = penalty \\cdot conflicts + n_{{used}}$$
          (penalty = 100). Khi conflicts = 0, fitness chính là số màu dùng.

        - **⏱️ Thời gian = `{rt:.2f}s`**

        ### 🖼️ Các hình minh hoạ

        - **🎨 Đồ thị đã tô màu** — Các đỉnh có cùng màu = cùng nhóm.
          Quan sát: 2 đỉnh kề nhau (có cạnh nối) **phải khác màu**.
          Nếu thấy 2 đỉnh kề cùng màu → đó là **conflict**.

        - **📉 Đồ thị hội tụ** — Fitness theo vòng lặp.

        ### 💡 Đánh giá

        - **Đồ thị input**: |V| = {G.number_of_nodes()} đỉnh, |E| = {G.number_of_edges()} cạnh
        - **Mật độ cạnh**: {G.number_of_edges()/(G.number_of_nodes()*(G.number_of_nodes()-1)/2)*100:.1f}%
        - **Bậc trung bình**: {2*G.number_of_edges()/G.number_of_nodes():.1f}
          → cần ít nhất {2*G.number_of_edges()//G.number_of_nodes() + 1} màu trên lý thuyết

        ### 📚 Ứng dụng thực tế

        - **Phân bổ tần số** trong mạng di động (tránh nhiễu)
        - **Lập lịch thi cử** (môn cùng SV không xếp cùng giờ)
        - **Phân bổ thanh ghi** trong compiler
        """)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🎨 Đồ thị đã tô màu")
        palette = plt.cm.tab10(np.linspace(0, 1, max(max_colors, 1)))
        node_colors = [palette[coloring[v]] for v in G.nodes()]
        fig1, ax = plt.subplots(figsize=(7, 6))
        nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=380,
                font_size=9, ax=ax, edge_color="gray")
        ax.set_title(f"Coloring – {n_used} màu, {conflicts} conflict")
        st.pyplot(fig1)
    with col2:
        st.subheader("📉 Hội tụ")
        fig2 = plot_convergence(hist, "Hội tụ GCP-ACO", "Fitness")
        st.pyplot(fig2)

    # ---------- 📄 NÚT TẢI PDF ----------
    st.markdown("---")
    st.subheader("📥 Xuất báo cáo PDF")
    pdf_bytes = generate_pdf_report(
        problem_title="Bài toán 05 – GCP",
        problem_subtitle=f"Graph Coloring Problem – {graph_choice}",
        algorithm="ACO",
        inputs={
            "Đồ thị": graph_choice,
            "|V| (số đỉnh)": G.number_of_nodes(),
            "|E| (số cạnh)": G.number_of_edges(),
            "Số màu tối đa": max_colors,
            "Số kiến": n_ants,
            "Số vòng lặp": n_iter,
            "Alpha": alpha,
            "Beta": beta,
            "Rho": rho,
            "Seed": seed,
        },
        outputs={
            "Số màu sử dụng": n_used,
            "Số xung đột": conflicts,
            "Fitness": f"{fit:.0f}",
            "Thời gian chạy": f"{rt:.2f} giây",
            "Trạng thái": "Hợp lệ" if conflicts == 0 else f"Còn {conflicts} xung đột",
        },
        figures=[
            ("Hình 0: Đồ thị ban đầu (input)", fig0),
            ("Hình 1: Đồ thị đã tô màu", fig1),
            ("Hình 2: Đồ thị hội tụ", fig2),
        ],
    )
    download_pdf_button(pdf_bytes,
                       file_name=f"BaoCao_GCP_ACO.pdf",
                       key="pdf_gcp")
