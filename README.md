# 🐝 CSTT – Hệ thống demo Swarm Intelligence

Giao diện web (Streamlit) minh hoạ **11 bài toán** trong báo cáo
*"Thuật ngữ và các toán tử của GA – Tổng quan về thuật toán Tối ưu Bầy đàn (PSO)
và thuật toán Tối ưu Đàn kiến (ACO)"* – Học phần **Các hệ cơ sở tri thức**,
GVHD **PGS.TS Lê Hoàng Thái**, Khoá 36 (2025–2027).

## 📋 Danh sách bài toán

| # | Bài toán | Thuật toán | File trang |
|---|----------|-----------|-----------|
| 01 | TSP – Traveling Salesman | ACO | `pages/01_TSP_ACO.py` |
| 02 | CVRP – Vehicle Routing | ACO | `pages/02_VRP_ACO.py` |
| 03 | JSSP – Job Shop Scheduling (FT06) | ACO | `pages/03_JSSP_ACO.py` |
| 04 | QAP – Quadratic Assignment | ACO | `pages/04_QAP_ACO.py` |
| 05 | GCP – Graph Coloring | ACO | `pages/05_GCP_ACO.py` |
| 06 | VRPTW – VRP with Time Windows | PSO | `pages/06_VRPTW_PSO.py` |
| 07 | Protein Folding (HP 2D **& 3D**) | PSO | `pages/07_Protein_PSO.py` |
| 08 | PID Controller Tuning | PSO | `pages/08_PID_PSO.py` |
| 09 | Portfolio Optimization (Markowitz) | ACO | `pages/09_Portfolio_ACO.py` |
| 10 | Feature Selection (Breast Cancer) | Binary PSO | `pages/10_FeatureSelection_BPSO.py` |
| 11 | Function 2D (Rastrigin/Ackley/...) | ACO | `pages/11_Function2D_ACO.py` |

## 🚀 Cách chạy

### Cách 1 – chạy nhanh trên Windows
```cmd
double-click vào file run.bat
```

### Cách 2 – chạy thủ công
```bash
pip install -r requirements.txt
streamlit run app.py
```
Mở trình duyệt tại địa chỉ: <http://localhost:8501>

## 🗂️ Cấu trúc thư mục

```
app/
├── app.py                              # Trang chủ (landing page)
├── requirements.txt                    # Thư viện cần cài
├── run.bat                             # Script chạy nhanh
├── README.md                           # File này
├── algorithms/                         # 11 module thuật toán
│   ├── common.py                       # Tiện ích UI dùng chung
│   ├── tsp_aco.py                      # ACO cho TSP
│   ├── vrp_aco.py                      # ACO cho CVRP
│   ├── jssp_aco.py                     # ACO cho JSSP
│   ├── qap_aco.py                      # ACO cho QAP
│   ├── gcp_aco.py                      # ACO cho GCP
│   ├── vrptw_pso.py                    # PSO cho VRPTW
│   ├── protein_pso.py                  # PSO cho HP-Model
│   ├── pid_pso.py                      # PSO cho PID
│   ├── portfolio_aco.py                # ACO cho Portfolio
│   ├── feature_selection_bpso.py       # Binary PSO cho Feature Selection
│   └── func2d_aco.py                   # ACO cho hàm 2D
└── pages/                              # 11 trang Streamlit
    ├── 01_TSP_ACO.py
    ├── 02_VRP_ACO.py
    ├── ...
    └── 11_Function2D_ACO.py
```

## 🎛️ Cách sử dụng

1. **Trang chủ**: Hiển thị tóm tắt 3 thuật toán (PSO/ACO/ABC) và 11 bài toán.
   Click vào nút **"▶️ Mở demo"** trên từng card để vào trang bài toán.
2. **Trong từng trang bài toán**:
   - Đọc phần *"Mô tả bài toán"* (expander).
   - Điều chỉnh tham số trong sidebar (số kiến/particle, alpha/beta/rho, vòng lặp, seed, …).
   - Nhấn nút **🚀 Chạy** để chạy thuật toán.
   - Xem trực tiếp: progress bar → kết quả → đồ thị hội tụ → trực quan hoá lời giải.
3. **Quay về trang chủ**: nút **🏠 Về trang chủ** ở đầu sidebar.

## 📚 Tham khảo nguồn

Các thuật toán được viết lại độc lập dựa trên 11 file Jupyter notebook gốc
trong thư mục `../CSTT_ThayThai/` (do nhóm thực hiện):
- `01_JSSP_ACO_FT06.ipynb`
- `02_GCP_ACO_KarateClub_Chuong6.ipynb`
- `03_Feature_Selection_Binary_PSO_BreastCancer_Chuong6.ipynb`
- `Func2d_aco.ipynb`
- `PSO_PID_Tuning (1).ipynb`
- `PSO_Protein_Folding_Problem_(HP_Model_2D_3D).ipynb`
- `Portfolio_aco.ipynb`
- `QAP_Solver.ipynb`
- `TSP_CacHeCoSoTriThuc_ThayThai.ipynb`
- `VRP_CacHeCoSoTriThuc_ThayThai_NhomNhungChuOngChamChi.ipynb`
- `Vehicle_Routing_Problem_with_Time_Windows.ipynb`

## 👨‍🎓 Nhóm tác giả

- Chế Chí Công – KHMT836005
- Lê Thị Mai Len – KHMT836015
- Huỳnh Phát Lợi – KHMT836016
- Trần Võ Khôi Nguyên – KHMT836022
- Tăng Ngọc Phụng – KHMT836027
- Hoàng Châu Ngọc Phương – KHMT836028
- Võ Phú Vinh – KHMT836036

📍 TP. Hồ Chí Minh – 20/05/2026
