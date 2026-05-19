"""Tiện ích chung + CSS theme cho toàn bộ app."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np


# ============================================================================
#  THEME / CSS GLOBAL
# ============================================================================
GLOBAL_CSS = """
<style>
/* ------- IMPORT FONT ------- */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"]  {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* ------- TỔNG THỂ NỀN APP ------- */
.stApp {
    background:
        radial-gradient(at 0% 0%, rgba(99,102,241,0.10) 0px, transparent 50%),
        radial-gradient(at 100% 0%, rgba(236,72,153,0.08) 0px, transparent 50%),
        radial-gradient(at 50% 100%, rgba(59,130,246,0.08) 0px, transparent 50%),
        linear-gradient(180deg, #f8fafc 0%, #eef2ff 100%);
}

/* ------- BLOCK CHÍNH ------- */
.block-container {
    padding-top: 1.2rem !important;
    padding-bottom: 2rem !important;
    max-width: 1280px !important;
}

/* ------- HEADER TRANG ------- */
.page-hero {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
    color: white;
    padding: 1.8rem 2rem;
    border-radius: 20px;
    margin-bottom: 1.4rem;
    box-shadow: 0 20px 50px -12px rgba(118,75,162,.35);
    position: relative;
    overflow: hidden;
}
.page-hero::before {
    content: "";
    position: absolute;
    top: -50%; right: -10%;
    width: 400px; height: 400px;
    background: radial-gradient(circle, rgba(255,255,255,.15), transparent 70%);
    border-radius: 50%;
}
.page-hero h1, .page-hero h2 {
    margin: 0 0 .25rem 0 !important;
    color: white !important;
    font-weight: 800 !important;
    letter-spacing: -.5px;
}
.page-hero p {
    margin: 0 !important;
    color: rgba(255,255,255,.92) !important;
    font-size: 1rem;
}

/* ------- BADGE (tag thuật toán) ------- */
.badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 999px;
    font-weight: 600;
    font-size: .75rem;
    letter-spacing: .3px;
    margin-left: 8px;
    backdrop-filter: blur(6px);
}
.badge-pso { background: rgba(59,130,246,.95); color: white; border:1px solid rgba(255,255,255,.3); }
.badge-aco { background: rgba(245,158,11,.95); color: white; border:1px solid rgba(255,255,255,.3); }
.badge-abc { background: rgba(34,197,94,.95);  color: white; border:1px solid rgba(255,255,255,.3); }

/* ------- CARD CHO TRANG CHỦ ------- */
.problem-card {
    background: rgba(255,255,255,.85);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,.6);
    border-radius: 18px;
    padding: 1.25rem 1.4rem;
    margin-bottom: 0.9rem;
    box-shadow: 0 10px 25px -10px rgba(99,102,241,.15);
    transition: all 0.35s cubic-bezier(.4,0,.2,1);
    cursor: pointer;
    position: relative;
    overflow: hidden;
}
.problem-card::after {
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 4px;
    background: linear-gradient(90deg, #667eea, #f093fb);
    border-radius: 18px 18px 0 0;
}
.problem-card:hover {
    transform: translateY(-6px) scale(1.02);
    box-shadow: 0 25px 50px -10px rgba(99,102,241,.35);
    border-color: rgba(102,126,234,.5);
}
.problem-card h4 {
    color: #1e1b4b;
    font-weight: 700;
    font-size: 1.05rem;
    margin: 0 0 .4rem 0 !important;
    display: flex;
    align-items: center;
}
.problem-card .icon-circle {
    width: 44px; height: 44px;
    border-radius: 12px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 1.4rem;
    margin-right: 10px;
    background: linear-gradient(135deg, #e0e7ff, #fce7f3);
    box-shadow: 0 4px 10px rgba(99,102,241,.18);
}
.problem-card p {
    color: #475569;
    font-size: .88rem;
    margin: .5rem 0 0 0 !important;
    line-height: 1.45;
}

/* ------- THẺ MÔ TẢ THUẬT TOÁN (3 cột PSO/ACO/ABC) ------- */
.algo-card {
    background: rgba(255,255,255,.8);
    backdrop-filter: blur(10px);
    border-radius: 18px;
    padding: 1.4rem 1.5rem;
    border-left: 5px solid #6366f1;
    box-shadow: 0 10px 30px -12px rgba(99,102,241,.25);
    height: 100%;
    transition: transform .3s;
}
.algo-card:hover { transform: translateY(-4px); }
.algo-card h4 {
    color: #1e1b4b;
    margin: 0 0 .5rem 0 !important;
    font-weight: 700;
}
.algo-card p {
    color: #475569;
    margin: 0 !important;
    font-size: .92rem;
    line-height: 1.55;
}
.algo-card.pso { border-left-color: #3b82f6; }
.algo-card.aco { border-left-color: #f59e0b; }
.algo-card.abc { border-left-color: #10b981; }

/* ------- METRIC ------- */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, rgba(255,255,255,.95), rgba(241,245,249,.85));
    border: 1px solid rgba(99,102,241,.18);
    border-radius: 16px;
    padding: 1rem 1.2rem;
    box-shadow: 0 6px 16px -6px rgba(99,102,241,.18);
    transition: all .25s;
}
[data-testid="stMetric"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 24px -8px rgba(99,102,241,.28);
    border-color: rgba(99,102,241,.35);
}
[data-testid="stMetricLabel"] {
    color: #64748b !important;
    font-weight: 600 !important;
    font-size: .8rem !important;
    letter-spacing: .3px;
    text-transform: uppercase;
}
[data-testid="stMetricValue"] {
    color: #1e1b4b !important;
    font-weight: 800 !important;
    font-size: 1.6rem !important;
    background: linear-gradient(135deg, #4f46e5, #ec4899);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

/* ------- SIDEBAR ------- */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1e1b4b 0%, #312e81 100%);
}
[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stMarkdown li,
[data-testid="stSidebar"] .stMarkdown span:not(.badge):not(.pill) {
    color: #e0e7ff !important;
}
/* Chữ in đậm / nghiêng trong sidebar - tô vàng nhạt để nổi bật trên nền tối */
[data-testid="stSidebar"] .stMarkdown strong,
[data-testid="stSidebar"] .stMarkdown b {
    color: #fde68a !important;
    font-weight: 700 !important;
}
[data-testid="stSidebar"] .stMarkdown em,
[data-testid="stSidebar"] .stMarkdown i {
    color: #fbcfe8 !important;
}
/* Link trong sidebar (đoạn markdown) */
[data-testid="stSidebar"] .stMarkdown a {
    color: #93c5fd !important;
    text-decoration: underline;
}
/* Inline code trong sidebar */
[data-testid="stSidebar"] .stMarkdown code {
    background: rgba(255,255,255,.15) !important;
    color: #fde68a !important;
    border: 1px solid rgba(253,230,138,.3) !important;
}
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] label * {
    color: #c7d2fe !important;
    font-weight: 600 !important;
    font-size: .88rem !important;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] h4 {
    color: white !important;
    border-bottom: 2px solid rgba(255,255,255,.15);
    padding-bottom: .5rem;
}

/* ----- Slider trong sidebar ----- */
[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] > div > div {
    background: linear-gradient(90deg, #f093fb, #f5576c) !important;
}
/* Giá trị hiển thị bên cạnh thanh trượt */
[data-testid="stSidebar"] [data-testid="stTickBar"],
[data-testid="stSidebar"] [data-testid="stTickBarMin"],
[data-testid="stSidebar"] [data-testid="stTickBarMax"] {
    color: #a5b4fc !important;
}
[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] [role="slider"] {
    background: #f093fb !important;
    box-shadow: 0 0 0 3px rgba(240,147,251,.3) !important;
}

/* ----- Select box / Multi-select trong sidebar ----- */
[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background: rgba(255,255,255,.95) !important;
    border: 1px solid rgba(255,255,255,.3) !important;
    border-radius: 10px !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] > div *,
[data-testid="stSidebar"] [data-baseweb="select"] input {
    color: #1e1b4b !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] svg {
    fill: #6366f1 !important;
}

/* ----- Number input / Text input trong sidebar (chuyển nền sang trắng + chữ tím đậm) ----- */
[data-testid="stSidebar"] [data-testid="stNumberInput"] input,
[data-testid="stSidebar"] [data-testid="stTextInput"] input,
[data-testid="stSidebar"] [data-testid="stTextArea"] textarea,
[data-testid="stSidebar"] input[type="number"],
[data-testid="stSidebar"] input[type="text"] {
    background: #ffffff !important;
    color: #1e1b4b !important;
    border: 1px solid rgba(99,102,241,.4) !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    caret-color: #6366f1 !important;
}
[data-testid="stSidebar"] [data-testid="stNumberInput"] input:focus,
[data-testid="stSidebar"] [data-testid="stTextInput"] input:focus {
    border-color: #f093fb !important;
    box-shadow: 0 0 0 3px rgba(240,147,251,.25) !important;
}
/* Nút +/- của number input */
[data-testid="stSidebar"] [data-testid="stNumberInput"] button {
    background: rgba(255,255,255,.18) !important;
    color: #ffffff !important;
    border: 1px solid rgba(255,255,255,.25) !important;
    border-radius: 8px !important;
}
[data-testid="stSidebar"] [data-testid="stNumberInput"] button:hover {
    background: rgba(240,147,251,.45) !important;
    border-color: #f093fb !important;
}
[data-testid="stSidebar"] [data-testid="stNumberInput"] button svg {
    fill: #ffffff !important;
}

/* ----- Radio / Checkbox trong sidebar ----- */
[data-testid="stSidebar"] [data-testid="stRadio"] label,
[data-testid="stSidebar"] [data-testid="stCheckbox"] label {
    color: #e0e7ff !important;
}
[data-testid="stSidebar"] [data-baseweb="radio"] div[role="radio"] {
    border-color: rgba(255,255,255,.5) !important;
}

/* ----- Dropdown menu (khi click vào select) ----- */
[data-baseweb="popover"] [role="listbox"] {
    background: #ffffff !important;
    border: 1px solid rgba(99,102,241,.3) !important;
    border-radius: 10px !important;
    box-shadow: 0 12px 32px -8px rgba(99,102,241,.35) !important;
}
[data-baseweb="popover"] [role="option"] {
    color: #1e1b4b !important;
}
[data-baseweb="popover"] [role="option"]:hover {
    background: linear-gradient(90deg, rgba(99,102,241,.12), rgba(240,147,251,.12)) !important;
}

/* ============================================================
   SIDEBAR NAV (danh sách 11 trang Streamlit tự sinh)
   Trước: chữ đen trên nền tím đậm → không thấy.
   Sau:  chữ trắng + hover gradient hồng-tím.
   ============================================================ */
[data-testid="stSidebarNav"] {
    background: rgba(255,255,255,.05) !important;
    border-radius: 12px !important;
    padding: 8px 6px !important;
    margin-bottom: 1rem !important;
}
[data-testid="stSidebarNav"] ul {
    padding-left: 0 !important;
}
[data-testid="stSidebarNav"] li {
    list-style: none !important;
    margin: 2px 0 !important;
}
/* Link trong nav (mọi phiên bản Streamlit) */
[data-testid="stSidebarNav"] a,
[data-testid="stSidebarNav"] a span,
[data-testid="stSidebarNav"] a p,
[data-testid="stSidebarNavLink"],
[data-testid="stSidebarNavLink"] span,
[data-testid="stSidebarNavLink"] p,
[data-testid="stSidebarNavItems"] a,
[data-testid="stSidebarNavItems"] a span,
[data-testid="stSidebarNavItems"] a p {
    color: #e0e7ff !important;
    font-weight: 500 !important;
    font-size: .92rem !important;
    text-decoration: none !important;
}
/* Link container */
[data-testid="stSidebarNav"] a,
[data-testid="stSidebarNavLink"],
[data-testid="stSidebarNavItems"] a {
    display: block !important;
    padding: 8px 14px !important;
    border-radius: 10px !important;
    transition: all .2s ease !important;
    background: transparent !important;
}
/* Hover */
[data-testid="stSidebarNav"] a:hover,
[data-testid="stSidebarNavLink"]:hover,
[data-testid="stSidebarNavItems"] a:hover {
    background: linear-gradient(90deg, rgba(240,147,251,.25), rgba(99,102,241,.25)) !important;
    transform: translateX(3px);
    box-shadow: inset 3px 0 0 #f093fb;
}
[data-testid="stSidebarNav"] a:hover *,
[data-testid="stSidebarNavLink"]:hover *,
[data-testid="stSidebarNavItems"] a:hover * {
    color: #ffffff !important;
    font-weight: 600 !important;
}
/* Trang đang active */
[data-testid="stSidebarNav"] a[aria-current="page"],
[data-testid="stSidebarNavLink"][aria-current="page"],
[data-testid="stSidebarNavItems"] a[aria-current="page"] {
    background: linear-gradient(90deg, #f093fb, #f5576c) !important;
    box-shadow: 0 6px 16px -6px rgba(245,87,108,.55) !important;
}
[data-testid="stSidebarNav"] a[aria-current="page"] *,
[data-testid="stSidebarNavLink"][aria-current="page"] *,
[data-testid="stSidebarNavItems"] a[aria-current="page"] * {
    color: #ffffff !important;
    font-weight: 700 !important;
}

/* Mũi tên collapse sidebar (<<) */
[data-testid="stSidebarCollapseButton"] svg,
[data-testid="stBaseButton-headerNoPadding"] svg {
    fill: #c7d2fe !important;
}
[data-testid="stSidebarCollapseButton"]:hover svg,
[data-testid="stBaseButton-headerNoPadding"]:hover svg {
    fill: #ffffff !important;
}

/* ------- BUTTON ------- */
.stButton > button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    border-radius: 12px;
    padding: .6rem 1.4rem;
    font-weight: 600;
    letter-spacing: .3px;
    box-shadow: 0 8px 20px -6px rgba(102,126,234,.55);
    transition: all .25s cubic-bezier(.4,0,.2,1);
}
.stButton > button:hover {
    transform: translateY(-2px) scale(1.02);
    box-shadow: 0 12px 28px -6px rgba(102,126,234,.7);
    background: linear-gradient(135deg, #5a67d8 0%, #6b46c1 100%);
}
.stButton > button:active { transform: translateY(0); }
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    box-shadow: 0 8px 20px -6px rgba(245,87,108,.55);
}
.stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #ec80f7 0%, #f04060 100%);
}

/* ------- EXPANDER ------- */
[data-testid="stExpander"] {
    background: rgba(255,255,255,.7);
    backdrop-filter: blur(8px);
    border: 1px solid rgba(99,102,241,.15);
    border-radius: 14px;
    margin-bottom: .8rem;
    box-shadow: 0 4px 14px -6px rgba(99,102,241,.12);
}
[data-testid="stExpander"] summary {
    font-weight: 600 !important;
    color: #4338ca !important;
    padding: .5rem 0;
}

/* ------- ALERT (success / info / warn / error) – BODY chính ------- */
.stApp [data-testid="stAlert"] {
    border-radius: 14px;
    border: 1px solid rgba(99,102,241,.2);
    box-shadow: 0 4px 14px -6px rgba(99,102,241,.18);
    backdrop-filter: blur(6px);
}
/* Body alerts giữ màu mặc định Streamlit (info xanh, success xanh lá, ...) */

/* ------- ALERT TRONG SIDEBAR – luôn nền TRẮNG + chữ TÍM ĐẬM ------- */
[data-testid="stSidebar"] [data-testid="stAlert"] {
    background: rgba(255,255,255,.95) !important;
    border: 1px solid rgba(99,102,241,.35) !important;
    border-radius: 14px !important;
    box-shadow: 0 6px 18px -8px rgba(0,0,0,.4) !important;
    color: #1e1b4b !important;
}
[data-testid="stSidebar"] [data-testid="stAlert"] *,
[data-testid="stSidebar"] [data-testid="stAlert"] p,
[data-testid="stSidebar"] [data-testid="stAlert"] span,
[data-testid="stSidebar"] [data-testid="stAlert"] div {
    color: #1e1b4b !important;
}
[data-testid="stSidebar"] [data-testid="stAlert"] strong,
[data-testid="stSidebar"] [data-testid="stAlert"] b {
    color: #4c1d95 !important;
    font-weight: 700 !important;
}
[data-testid="stSidebar"] [data-testid="stAlert"] svg {
    fill: #6366f1 !important;
}

/* ------- PROGRESS BAR ------- */
.stProgress > div > div {
    background: linear-gradient(90deg, #667eea, #f093fb, #f5576c) !important;
    border-radius: 999px !important;
}
.stProgress > div > div > div {
    background: transparent !important;
}

/* ------- SUBHEADER ------- */
.stApp h2, .stApp h3 {
    color: #1e1b4b !important;
    font-weight: 700 !important;
    letter-spacing: -.3px;
}
.stApp h3 {
    font-size: 1.15rem !important;
    border-left: 4px solid #667eea;
    padding-left: 12px;
    margin-top: 1.2rem !important;
}

/* ------- DATAFRAME (body chính – nền sáng chữ tối) ------- */
.stApp [data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 4px 14px -6px rgba(99,102,241,.18);
}
.stApp [data-testid="stDataFrame"] * {
    color: #1e1b4b !important;
}
/* Body markdown - đảm bảo bold trong main area là tím đậm */
.stApp .main .stMarkdown strong,
.stApp .main .stMarkdown b {
    color: #4338ca !important;
    font-weight: 700 !important;
}

/* ------- RADIO / CHECKBOX trong body chính (không sidebar) ------- */
.stApp .main [data-testid="stRadio"] label,
.stApp .main [data-testid="stCheckbox"] label {
    color: #1e1b4b !important;
}

/* ------- CODE BLOCK (body chính) ------- */
.stApp code {
    background: linear-gradient(135deg, #fef3c7, #fde68a) !important;
    color: #92400e !important;
    padding: 2px 6px !important;
    border-radius: 5px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: .85rem !important;
}
.stApp pre {
    background: #1e1b4b !important;
    border-radius: 12px !important;
    padding: 1rem !important;
    box-shadow: 0 8px 20px -8px rgba(30,27,75,.4);
}
.stApp pre code,
.stApp pre code * {
    background: transparent !important;
    color: #c7d2fe !important;
}
/* st.code block (st.code()) - dark with light text */
.stApp [data-testid="stCodeBlock"] {
    background: #1e1b4b !important;
    border-radius: 12px !important;
}
.stApp [data-testid="stCodeBlock"] code,
.stApp [data-testid="stCodeBlock"] pre,
.stApp [data-testid="stCodeBlock"] span {
    color: #e0e7ff !important;
    background: transparent !important;
}

/* ------- DIVIDER ------- */
hr {
    border: none !important;
    height: 1px !important;
    background: linear-gradient(90deg, transparent, rgba(99,102,241,.3), transparent) !important;
    margin: 1.5rem 0 !important;
}

/* ------- FOOTER (chân trang) ------- */
.footer-box {
    background: rgba(255,255,255,.6);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(99,102,241,.15);
    border-radius: 16px;
    padding: 1.2rem 1.5rem;
    margin-top: 2rem;
    text-align: center;
    color: #475569;
    box-shadow: 0 4px 14px -6px rgba(99,102,241,.15);
}
.footer-box p { margin: .25rem 0 !important; font-size: .9rem; }
.footer-box .authors {
    font-weight: 600;
    color: #1e1b4b;
}

/* ------- ANIMATION KEYFRAMES ------- */
@keyframes float {
    0%,100% { transform: translateY(0); }
    50%     { transform: translateY(-6px); }
}
.floating { animation: float 4s ease-in-out infinite; }

@keyframes shimmer {
    0%   { background-position: -200% 0; }
    100% { background-position: 200% 0; }
}
</style>
"""


def inject_css():
    """Inject CSS toàn cục — gọi ở đầu mỗi trang."""
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)


def page_header(icon: str, title: str, subtitle: str, algo: str):
    """Header gradient cho mỗi trang bài toán."""
    inject_css()
    badge_class = {"PSO": "badge-pso", "ACO": "badge-aco", "ABC": "badge-abc"}.get(algo, "badge-pso")
    st.markdown(
        f"""
        <div class="page-hero">
            <h2><span class="floating" style="display:inline-block;">{icon}</span>
                {title}
                <span class="badge {badge_class}">{algo}</span>
            </h2>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def back_button():
    """Nút quay về trang chủ ở sidebar."""
    if st.sidebar.button("🏠 Về trang chủ", use_container_width=True, key="back_home"):
        st.switch_page("app.py")
    st.sidebar.markdown("<hr style='margin:.5rem 0;'/>", unsafe_allow_html=True)


# ============================================================================
#  MATPLOTLIB THEME (đồng bộ với palette UI)
# ============================================================================
def _setup_mpl_theme():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelcolor": "#475569",
        "axes.edgecolor": "#cbd5e1",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#e2e8f0",
        "grid.alpha": 0.6,
        "xtick.color": "#64748b",
        "ytick.color": "#64748b",
        "figure.facecolor": "#ffffff",
        "axes.facecolor": "#fbfaff",
    })

_setup_mpl_theme()


def plot_convergence(history, title="Đồ thị hội tụ", ylabel="Best Fitness"):
    """Vẽ đường hội tụ với gradient fill đẹp mắt."""
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(history))
    ax.plot(x, history, color="#6366f1", lw=2.5, marker="o",
            markersize=4, markerfacecolor="#f093fb",
            markeredgecolor="white", markeredgewidth=1.2,
            label="Best fitness", zorder=3)
    ax.fill_between(x, history, max(history) if history else 0,
                    color="#6366f1", alpha=0.10, zorder=1)
    ax.set_xlabel("Vòng lặp", fontweight="600")
    ax.set_ylabel(ylabel, fontweight="600")
    ax.set_title(title, color="#1e1b4b", pad=15)
    ax.legend(loc="best", frameon=True, fancybox=True, shadow=False,
              facecolor="white", edgecolor="#cbd5e1")
    fig.tight_layout()
    return fig


def metric_row(items):
    """items: list of (label, value, help) tuples."""
    cols = st.columns(len(items))
    for c, (label, value, help_) in zip(cols, items):
        c.metric(label, value, help=help_)


def section_title(text: str, icon: str = "📊"):
    """Tiêu đề section có icon + gradient bar."""
    st.markdown(
        f"""
        <h3 style="margin-top:1.4rem;">
            <span style="display:inline-block;margin-right:8px;">{icon}</span>{text}
        </h3>
        """,
        unsafe_allow_html=True,
    )
