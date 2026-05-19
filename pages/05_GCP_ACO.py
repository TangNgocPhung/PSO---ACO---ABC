"""Trang demo GCP với ACO – Karate Club graph."""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from algorithms.common import page_header, back_button, plot_convergence, metric_row
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

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🎨 Đồ thị đã tô màu")
        palette = plt.cm.tab10(np.linspace(0, 1, max(max_colors, 1)))
        node_colors = [palette[coloring[v]] for v in G.nodes()]
        fig, ax = plt.subplots(figsize=(7, 6))
        nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=380,
                font_size=9, ax=ax, edge_color="gray")
        ax.set_title(f"Coloring – {n_used} màu, {conflicts} conflict")
        st.pyplot(fig)
    with col2:
        st.subheader("📉 Hội tụ")
        st.pyplot(plot_convergence(hist, "Hội tụ GCP-ACO", "Fitness"))
