"""
╔═══════════════════════════════════════════════════════════════════════╗
║   HỆ THỐNG MINH HỌA CÁC HỆ CƠ SỞ TRI THỨC                            ║
║   Thuật toán Tối ưu hóa Bầy đàn: PSO, ACO, ABC                      ║
║   Giảng viên: PGS.TS Lê Hoàng Thái                                   ║
║   Học phần: Các hệ cơ sở tri thức – Khóa 36 (2025–2027)              ║
╚═══════════════════════════════════════════════════════════════════════╝
"""
import streamlit as st
from algorithms.common import inject_css

st.set_page_config(
    page_title="CSTT – Swarm Intelligence Demo",
    page_icon="🐝",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_css()

# ---------------------------------------------------------------------------
#  CSS RIÊNG CHO TRANG CHỦ (hero banner lớn)
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .home-hero {
        background:
            radial-gradient(circle at 20% 30%, rgba(245,87,108,.4), transparent 40%),
            radial-gradient(circle at 80% 70%, rgba(102,126,234,.4), transparent 40%),
            linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        color: white;
        padding: 3rem 2.5rem;
        border-radius: 28px;
        text-align: center;
        margin-bottom: 1.8rem;
        box-shadow: 0 30px 70px -15px rgba(118,75,162,.45);
        position: relative;
        overflow: hidden;
    }
    .home-hero::before, .home-hero::after {
        content: "";
        position: absolute;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(255,255,255,.2), transparent 60%);
    }
    .home-hero::before { width: 300px; height: 300px; top: -100px; left: -50px; }
    .home-hero::after  { width: 250px; height: 250px; bottom: -80px; right: -40px; }
    .home-hero h1 {
        font-size: 2.8rem !important;
        font-weight: 800 !important;
        margin: 0 0 .5rem 0 !important;
        color: white !important;
        letter-spacing: -1px;
        text-shadow: 0 4px 20px rgba(0,0,0,.2);
    }
    .home-hero h3 {
        font-size: 1.25rem !important;
        font-weight: 500 !important;
        color: rgba(255,255,255,.95) !important;
        margin: 0 0 .8rem 0 !important;
    }
    .home-hero .subtitle {
        font-size: .95rem;
        color: rgba(255,255,255,.85);
        margin-top: 1rem;
    }
    .home-hero .pill {
        display: inline-block;
        padding: 5px 14px;
        background: rgba(255,255,255,.18);
        border: 1px solid rgba(255,255,255,.3);
        border-radius: 999px;
        margin: 4px;
        font-size: .85rem;
        font-weight: 500;
        backdrop-filter: blur(8px);
    }

    .section-heading {
        text-align: center;
        margin: 2rem 0 1.5rem 0;
    }
    .section-heading h2 {
        font-size: 1.8rem !important;
        font-weight: 800 !important;
        background: linear-gradient(135deg, #4f46e5, #ec4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0 !important;
    }
    .section-heading p {
        color: #64748b;
        font-size: .95rem;
        margin-top: .3rem !important;
    }

    /* Sidebar trang chủ */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e1b4b 0%, #4c1d95 50%, #831843 100%);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
#  HERO (banner đầu trang)
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="home-hero">
        <h1>🐝  CÁC HỆ CƠ SỞ TRI THỨC</h1>
        <h3>Demo PSO – ACO – ABC trên 12 bài toán tối ưu kinh điển</h3>
        <div>
            <span class="pill">📚 PGS.TS Lê Hoàng Thái</span>
            <span class="pill">🎓 Khoá 36 (2025–2027)</span>
            <span class="pill">💻 Khoa học Máy tính</span>
        </div>
        <p class="subtitle">Trường Đại học Sư phạm TP. Hồ Chí Minh</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
#  3 CARD GIỚI THIỆU THUẬT TOÁN
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="section-heading">
        <h2>✨ Ba thuật toán cốt lõi</h2>
        <p>Tối ưu hoá bầy đàn – tri thức nổi lên từ hành vi tập thể</p>
    </div>
    """,
    unsafe_allow_html=True,
)

c1, c2, c3 = st.columns(3, gap="medium")
with c1:
    st.markdown(
        """
        <div class="algo-card pso">
            <h4>🐦 PSO – Particle Swarm Optimization</h4>
            <p>Mô phỏng hành vi bay đàn của chim và cá. Mỗi cá thể cập nhật vị trí
            dựa trên <b>vị trí tốt nhất cá nhân</b> (pbest) và
            <b>vị trí tốt nhất toàn cục</b> (gbest). Phù hợp với tối ưu liên tục.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        """
        <div class="algo-card aco">
            <h4>🐜 ACO – Ant Colony Optimization</h4>
            <p>Mô phỏng đàn kiến tìm đường bằng <b>pheromone</b>. Bay hơi & tích luỹ
            pheromone tạo ra "trí tuệ tập thể". Phù hợp các bài toán tổ hợp:
            <b>TSP, VRP, JSSP, QAP, GCP</b>.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with c3:
    st.markdown(
        """
        <div class="algo-card abc">
            <h4>🐝 ABC – Artificial Bee Colony</h4>
            <p>Mô phỏng đàn ong mật với 3 vai trò: <b>ong thợ</b>,
            <b>ong quan sát</b>, <b>ong trinh sát</b>. Cân bằng khai phá (exploration)
            và khai thác (exploitation) rất tốt cho tối ưu liên tục.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
#  11 BÀI TOÁN
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="section-heading">
        <h2>🎯 12 Bài toán minh hoạ</h2>
        <p>Click vào nút <b>"Mở demo"</b> trên mỗi card để chạy thực nghiệm</p>
    </div>
    """,
    unsafe_allow_html=True,
)

problems = [
    ("01 · TSP",               "🗺️", "ACO", "Người du lịch (Traveling Salesman Problem)",      "pages/01_TSP_ACO.py"),
    ("02 · CVRP",              "🚚", "ACO", "Định tuyến phương tiện có sức chứa (Capacitated VRP)", "pages/02_CVRP_ACO.py"),
    ("03 · JSSP",              "⚙️", "ACO", "Lập lịch công việc phân xưởng (FT06 benchmark)",  "pages/03_JSSP_ACO.py"),
    ("04 · QAP",               "🏢", "ACO", "Phân công Bậc hai – bố trí 8 phòng × 8 vị trí",    "pages/04_QAP_ACO.py"),
    ("05 · GCP",               "🎨", "ACO", "Tô màu đồ thị – Karate Club / Petersen",           "pages/05_GCP_ACO.py"),
    ("06 · VRPTW",             "⏰", "PSO", "Định tuyến phương tiện có Khung thời gian",        "pages/06_VRPTW_PSO.py"),
    ("07 · Protein Folding",   "🧬", "PSO", "Gấp cuộn Protein (HP Model 2D/3D)",                "pages/07_Protein_PSO.py"),
    ("08 · PID Tuning",        "🎛️", "PSO", "Tinh chỉnh tham số bộ điều khiển PID",            "pages/08_PID_PSO.py"),
    ("09 · Portfolio",         "💹", "ACO", "Tối ưu danh mục đầu tư (Markowitz)",              "pages/09_Portfolio_ACO.py"),
    ("10 · Feature Selection", "🔬", "PSO", "Binary PSO chọn đặc trưng (Breast Cancer)",        "pages/10_FeatureSelection_BPSO.py"),
    ("11 · Function 2D",       "📈", "ACO", "Tối ưu hàm 2D (Rastrigin / Ackley / …)",          "pages/11_Function2D_ACO.py"),
    ("12 · VRPTW (ACO)",       "🐜", "ACO", "VRPTW giải bằng ACO – so sánh với page 06 (PSO)", "pages/12_VRPTW_ACO.py"),
]

# Render thành lưới 3 cột
cols = st.columns(3, gap="medium")
for i, (title, icon, algo, desc, path) in enumerate(problems):
    badge_cls = {"PSO": "badge-pso", "ACO": "badge-aco", "ABC": "badge-abc"}[algo]
    with cols[i % 3]:
        st.markdown(
            f"""
            <div class="problem-card">
                <h4><span class="icon-circle">{icon}</span> {title}
                    <span class="badge {badge_cls}" style="margin-left:auto;">{algo}</span>
                </h4>
                <p>{desc}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button(f"▶️  Mở demo", key=f"btn_{i}", use_container_width=True, type="primary"):
            st.switch_page(path)

# ---------------------------------------------------------------------------
#  SIDEBAR
# ---------------------------------------------------------------------------
st.sidebar.markdown(
    """
    <div style="text-align:center;padding:1rem 0;">
        <div style="font-size:3rem;">🐝</div>
        <h3 style="color:white;margin:.5rem 0;">CSTT Demo</h3>
        <p style="color:#c7d2fe;font-size:.85rem;margin:0;">Swarm Intelligence Lab</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    ### 📖 Về dự án
    Giao diện minh hoạ **12 bài toán tối ưu** thuộc Chương 4 báo cáo:
    - Tối ưu **tổ hợp** (TSP, VRP, JSSP, QAP, GCP, VRPTW × 2)
    - Tối ưu **liên tục** (PID, Function 2D)
    - Tối ưu **ứng dụng** (Portfolio, Protein, Feature Selection)
    """
)
st.sidebar.markdown("---")
# Custom div với inline !important (bulletproof: thắng cả CSS sidebar !important)
st.sidebar.markdown(
    """
    <div class="cstt-tip-card" style="
        background: #ffffff !important;
        padding: 0.9rem 1rem;
        border-radius: 14px;
        margin-top: 0.8rem;
        box-shadow: 0 6px 18px -8px rgba(0,0,0,0.5);
        font-size: 0.9rem;
        line-height: 1.6;
        border-left: 4px solid #f093fb;
    ">
        <span style="color:#1e1b4b !important; font-size:1rem;">💡</span>
        <b style="color:#4c1d95 !important; font-weight:700;"> Mẹo:</b>
        <span style="color:#1e1b4b !important;">
            Click vào một bài toán, sau đó điều chỉnh tham số ở sidebar và nhấn
        </span>
        <b style="color:#4c1d95 !important; font-weight:700;">🚀 Chạy</b><span style="color:#1e1b4b !important;">.</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
#  FOOTER
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="footer-box">
        <p class="authors">👨‍🎓 Nhóm tác giả</p>
        <p>Chế Chí Công · Lê Thị Mai Len · Huỳnh Phát Lợi · Trần Võ Khải Nguyên ·
        Tăng Ngọc Phụng · Hoàng Châu Ngọc Phương · Võ Phú Vinh</p>
        <p style="font-size:.8rem;color:#94a3b8;margin-top:.6rem !important;">
            📍 TP. Hồ Chí Minh – 20/05/2026 ·
            🎓 Trường Đại học Sư phạm TP.HCM ·
            👨‍🏫 GVHD: PGS.TS Lê Hoàng Thái
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)
