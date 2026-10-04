"""
Quy hoạch động (DP) tìm biên dạng vận tốc tối ưu đa mục tiêu cho AMR / Thang máy di chuyển thẳng đứng.
Mã nguồn được viết lại theo phong cách C++ trong sáng, tường minh, chuẩn mực toán học và dễ đọc.

Mục tiêu: min sum_k [ w_time * dt - w_energy * E + w_jerk * (jerk^2 * dt) ]
"""

from dataclasses import dataclass, replace
import argparse
import sys
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Đảm bảo mã hóa UTF-8 chuẩn cho Windows Console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')


# ==============================================================================
# 1. CẤU TRÚC THAM SỐ (C++ Style Struct)
# ==============================================================================
@dataclass
class Params:
    # Thông số Cơ - Điện
    m: float = 10.0          # Khối lượng tải + khung [kg]
    r: float = 0.04          # Bán kính puly / bánh răng kéo [m]
    N: float = 5.0           # Tỉ số truyền động cơ
    Kt: float = 0.08         # Hằng số mô-men [Nm/A]
    Ke: float = 0.08         # Hằng số sức điện động [V.s/rad]
    R: float = 0.25          # Điện trở cuộn dây + mạch [Ohm]
    g: float = 9.81          # Gia tốc trọng trường [m/s^2]
    
    # Ràng buộc Động lực học & Phần cứng
    V_max: float = 1.5       # Vận tốc tối đa [m/s]
    A_min: float = -9.81     # Gia tốc đứng nhỏ nhất [m/s^2]
    A_max: float = 9.81      # Gia tốc đứng lớn nhất [m/s^2]
    I_high: float = 40.0     # Dòng điện cực đại cho phép [A]
    
    # Hành trình
    y_start: float = 10.0     # Vị trí đầu [m]
    y_end: float = 0.0       # Vị trí cuối [m]
    
    # Bước lưới rời rạc hóa
    dy: float = 0.05         # Bước vị trí [m]
    dv: float = 0.025        # Bước vận tốc [m/s]
    
    # Trọng số Hàm chi phí đa mục tiêu
    w_time: float = 1.0    # Trọng số thời gian [J/s] 
    w_energy: float = 1.0    # Trọng số thu hồi năng lượng
    w_jerk: float = 1.0      # Trọng số làm mượt gia tốc (Jerk)
    
    count_consumption: bool = True  # True: Tính cả điện tiêu thụ (E<0); False: Chỉ tính phần thu hồi (E>=0)


# ==============================================================================
# 2. CÁC HÀM TRỢ GIÚP ĐỘNG HỌC & ĐIỆN NĂNG (C++ Style Functions)
# ==============================================================================
def calc_accel_y(vc: float, vn: float, dy: float, direction: int) -> float:
    """Gia tốc phương thẳng đứng: a_y = d * (v_next^2 - v_curr^2) / (2 * dy)"""
    a_trav = (vn**2 - vc**2) / (2.0 * dy)
    return direction * a_trav


def calc_segment_time(vc: float, vn: float, dy: float) -> float:
    """Thời gian di chuyển đoạn dy: dt = 2 * dy / (v_curr + v_next)"""
    v_sum = vc + vn
    if v_sum <= 0:
        return 1e6
    return (2.0 * dy) / v_sum


def calc_motor_current(a_y: float, p: Params) -> float:
    """Dòng điện động cơ cần thiết: I = m * r * (g + a_y) / (Kt * N)"""
    return (p.m * p.r * (p.g + a_y)) / (p.Kt * p.N)


def calc_segment_energy(v_avg: float, a_y: float, dt: float, direction: int, p: Params) -> float:
    """
    Năng lượng tái tạo (Joule) trong khoảng dt: E = -(Ke * w_s * I + I^2 * R) * dt
    >0: Thu hồi năng lượng nạp pin; <0: Tiêu thụ điện năng
    """
    I = calc_motor_current(a_y, p)
    w_s = direction * v_avg * p.N / p.r  # Tốc độ góc động cơ có dấu [rad/s]
    
    P_in = p.Ke * w_s * I + (I**2) * p.R  # Công suất điện tiêu thụ từ nguồn
    E = -P_in * dt                         # Năng lượng tái tạo nạp pin
    
    if not p.count_consumption:
        E = max(E, 0.0)
    return E


# ==============================================================================
# 3. THUẬT TOÁN QUY HOẠCH ĐỘNG (DP Solver - C++ Style Explicit Loop)
# ==============================================================================
def solve(p: Params = Params()) -> dict:
    # 1. Khởi tạo Lưới vị trí và Lưới vận tốc
    dist = abs(p.y_end - p.y_start)
    n_steps = int(round(dist / p.dy))
    direction = 1 if p.y_end > p.y_start else -1
    y_nodes = p.y_start + direction * p.dy * np.arange(n_steps + 1)
    
    n_v = int(round(p.V_max / p.dv)) + 1
    v_grid = np.arange(n_v) * p.dv

    # 2. Pre-calculate Ma trận Chuyển trạng thái 2D (curr_v -> next_v)
    vc_grid = v_grid[:, None]
    vn_grid = v_grid[None, :]
    vsum_grid = vc_grid + vn_grid
    vavg_grid = vsum_grid / 2.0
    
    a_matrix = direction * (vn_grid**2 - vc_grid**2) / (2.0 * p.dy)
    
    vsum_safe = np.where(vsum_grid > 0, vsum_grid, 1.0)
    dt_matrix = np.where(vsum_grid > 0, 2.0 * p.dy / vsum_safe, np.inf)
    
    I_matrix = (p.m * p.r * (p.g + a_matrix)) / (p.Kt * p.N)
    w_s_matrix = direction * vavg_grid * p.N / p.r
    
    dt_safe = np.where(np.isfinite(dt_matrix), dt_matrix, 1.0)
    E_matrix = -(p.Ke * w_s_matrix * I_matrix + (I_matrix**2) * p.R) * dt_safe
    if not p.count_consumption:
        E_matrix = np.maximum(E_matrix, 0.0)

    # Kiểm tra tính khả thi của từng chuyển tiếp (c -> n)
    feas_matrix = (
        (vsum_grid > 0) &
        (a_matrix >= p.A_min - 1e-6) & (a_matrix <= p.A_max + 1e-6) &
        (np.abs(I_matrix) <= p.I_high)
    )

    # 3. Khởi tạo Bảng DP & Mảng Lưu đường đi truy vết (Parent Policy)
    # dp_table[k][p][c]: Chi phí nhỏ nhất tới bước k với v_{k-1}=v_grid[p] và v_k=v_grid[c]
    dp_table = np.full((n_steps + 1, n_v, n_v), np.inf)
    parent = np.full((n_steps + 1, n_v, n_v), -1, dtype=int)
    
    # Điều kiện biên k=0: Xuất phát từ nghỉ (v=0 m/s, a=0 m/s^2)
    dp_table[0][0][0] = 0.0

    # Danh sách node vận tốc hợp lệ cho từng bước k (Pruning)
    allowed_v = []
    for k in range(n_steps + 1):
        if k == 0 or k == n_steps:
            allowed_v.append([0])                   # Bắt đầu và kết thúc bắt buộc v = 0 m/s
        else:
            allowed_v.append(list(range(1, n_v)))   # Các điểm trung gian v > 0 m/s

    # 4. Vòng lặp DP Quy hoạch động tiến (Forward DP Recursion - Fast & Clean C++ Style)
    for k in range(n_steps):
        curr_allowed = allowed_v[k]
        next_allowed = allowed_v[k + 1]
        prev_arr = np.array([0] if k == 0 else allowed_v[k - 1])

        for c in curr_allowed:
            costs_prev_c = dp_table[k][prev_arr, c]
            valid_p_mask = np.isfinite(costs_prev_c)
            if not np.any(valid_p_mask):
                continue

            p_valid = prev_arr[valid_p_mask]
            costs_p_valid = costs_prev_c[valid_p_mask]

            a_prev = np.zeros_like(p_valid, dtype=float) if k == 0 else a_matrix[p_valid, c]
            dt_prev = np.full_like(p_valid, 0.01, dtype=float) if k == 0 else dt_matrix[p_valid, c]

            for n in next_allowed:
                if not feas_matrix[c, n]:
                    continue

                dt_curr = dt_matrix[c, n]
                a_curr = a_matrix[c, n]
                E_curr = E_matrix[c, n]

                tau = 0.5 * (dt_prev + dt_curr)
                jerk = (a_curr - a_prev) / dt_curr
                q_jerk = (jerk**2) * dt_curr

                # CÔNG THỨC CHI PHÍ ĐOẠN (Stage Cost) = w_time*dt - w_energy*E + w_jerk*(jerk^2*dt)
                stage_cost = (p.w_time * dt_curr) - (p.w_energy * E_curr) + (p.w_jerk * q_jerk)
                tot_costs = costs_p_valid + stage_cost

                # Cập nhật Bellman
                best_idx = np.argmin(tot_costs)
                if tot_costs[best_idx] < dp_table[k + 1][c][n]:
                    dp_table[k + 1][c][n] = tot_costs[best_idx]
                    parent[k + 1][c][n] = p_valid[best_idx]

    # 5. Chi phí dừng hẳn tại nút cuối N (Terminal Cost for v_N = 0)
    a_last = a_matrix[:, 0]
    dt_last = dt_safe[:, 0]
    terminal_term = p.w_jerk * (a_last**2) / dt_last
    
    final_costs = dp_table[n_steps, :, 0] + terminal_term
    best_last_c = int(np.argmin(final_costs))
    best_total_cost = float(final_costs[best_last_c])

    if not np.isfinite(best_total_cost):
        raise RuntimeError("Không tìm thấy đường đi DP hợp lệ. Hãy kiểm tra lại A_max, A_min hoặc bước lưới dy/dv.")

    # 6. Truy vết ngược (Backtracking Path Reconstruction)
    idx = [0, best_last_c]
    for k in range(n_steps, 1, -1):
        prev_p = parent[k][idx[-1]][idx[-2]]
        idx.append(prev_p)
    idx = idx[::-1]

    # 7. Tổng hợp Kết quả Động học & Năng lượng
    v = v_grid[idx]
    jc = idx[:-1]
    jn = idx[1:]
    
    dt_k = dt_matrix[jc, jn]
    t = np.concatenate([[0.0], np.cumsum(dt_k)])
    a_k = a_matrix[jc, jn]
    I_k = I_matrix[jc, jn]
    E_k = E_matrix[jc, jn]
    
    a_ext = np.concatenate([[0.0], a_k, [0.0]])
    jerk_k = np.diff(a_ext)[:-1] / dt_k

    res = {
        "idx": idx, "y": y_nodes, "v": v, "t": t, "dt": dt_k,
        "a": a_k, "I": I_k, "E": E_k, "jerk": jerk_k,
        "cost": best_total_cost, "T": float(t[-1]),
        "E_total": float(np.sum(E_k)),
        "E_ideal": float(p.m * p.g * abs(p.y_start - p.y_end))
    }
    return res


# ==============================================================================
# 4. HÀM VẼ ĐỒ THỊ & SO SÁNH TRỌNG SỐ (Visualization)
# ==============================================================================
def plot_result(res: dict, p: Params, path: str = "dp_velocity_profile.png") -> str:
    t, y, v = res["t"], res["y"], res["v"]
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    ax = ax.ravel()

    ax[0].plot(t, v, "b.-")
    ax[0].set(title="Vận tốc theo thời gian", xlabel="t [s]", ylabel="v [m/s]")

    ax[1].plot(y, v, "g.-")
    ax[1].set(title="Vận tốc theo vị trí", xlabel="y [m]", ylabel="v [m/s]")

    ax[2].plot(t, y, "k.-")
    ax[2].set(title="Quãng đường y theo thời gian", xlabel="t [s]", ylabel="y [m]")

    ax[3].step(t, np.append(res["a"], res["a"][-1]), where="post", color="r")
    ax[3].axhline(p.A_max, ls="--", c="gray")
    ax[3].axhline(p.A_min, ls="--", c="gray")
    ax[3].set(title="Gia tốc thẳng đứng a_y", xlabel="t [s]", ylabel="a [m/s²]")

    ax[4].step(t, np.append(res["jerk"], res["jerk"][-1]), where="post", color="m")
    ax[4].set(title="Jerk = Δa/Δt", xlabel="t [s]", ylabel="jerk [m/s³]")

    E_cum = np.concatenate([[0.0], np.cumsum(res["E"])])
    P = res["E"] / res["dt"]
    ax[5].plot(t, E_cum, "orange", label="E tích luỹ [J]")
    ax[5].set(title="Năng lượng thu hồi theo thời gian", xlabel="t [s]", ylabel="E [J]")
    ax5b = ax[5].twinx()
    ax5b.step(t, np.append(P, P[-1]), where="post", color="c", alpha=0.7, label="P [W]")
    ax5b.set_ylabel("P [W]")
    ax[5].legend(loc="upper left")
    ax5b.legend(loc="lower right")

    for a_ in ax:
        a_.grid(alpha=0.3)
        
    fig.suptitle(f"DP Velocity Profile | T = {res['T']:.2f} s | "
                 f"E_net = {res['E_total']:.1f} J (Thế năng m*g*H = {res['E_ideal']:.1f} J) | "
                 f"Weights: (Time {p.w_time}, Energy {p.w_energy}, Jerk {p.w_jerk})")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def compare_weights(base: Params, configs: dict, path: str = "dp_weight_comparison.png") -> str:
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for name, kw in configs.items():
        r = solve(replace(base, **kw))
        ax[0].plot(r["t"], r["v"], label=f"{name}: T={r['T']:.2f}s, E={r['E_total']:.1f}J")
        ax[1].step(r["t"], np.append(r["a"], r["a"][-1]), where="post", label=name)
        
    ax[0].set(title="Vận tốc v(t)", xlabel="t [s]", ylabel="v [m/s]")
    ax[1].set(title="Gia tốc a(t)", xlabel="t [s]", ylabel="a [m/s²]")
    for a_ in ax:
        a_.grid(alpha=0.3)
        a_.legend(fontsize=8)
        
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_pareto_frontier(base: Params = Params(), path: str = "dp_pareto_frontier.png") -> str:
    """
    Phân tích Pareto Frontier đa mục tiêu: Thời gian di chuyển (T) vs Năng lượng thu hồi (E).
    Quét qua tỷ lệ trọng số w_energy / w_time để chiếu không gian nghiệm Pareto.
    Biểu đồ được thiết kế theo phong cách TỐI GIẢN (Minimalist), tinh tế và rõ ràng.
    """
    # Quét độc lập cả 2 biến w_time và w_energy theo lưới 2D (10x10 = 100 điểm)
    w_time_vals = np.linspace(0.5, 3.0, 10)    # Quét w_time từ ~3.16 đến 1000
    w_energy_vals = np.linspace(0.0, 3.0, 10)  # Quét w_energy từ 1.0 đến 1000
    
    evaluated_points = []
    
    for wt in w_time_vals:
        for we in w_energy_vals:
            p_test = replace(base, w_time=float(wt), w_energy=float(we))
            try:
                res = solve(p_test)
                evaluated_points.append({
                    "T": res["T"],
                    "E": res["E_total"],
                    "w_time": wt,
                    "w_energy": we,
                    "ratio": we / wt if wt > 0 else 0.0
                })
            except Exception:
                continue

    if not evaluated_points:
        print("Không có nghiệm DP hợp lệ nào trong quá trình quét Pareto.")
        return ""

    # Trích xuất Pareto Frontier (Điểm không bị áp đảo)
    pareto_pts = []
    for pt in evaluated_points:
        is_dominated = False
        for other in evaluated_points:
            if (other["T"] <= pt["T"] and other["E"] >= pt["E"]) and (other["T"] < pt["T"] or other["E"] > pt["E"]):
                is_dominated = True
                break
        if not is_dominated:
            pareto_pts.append(pt)

    # Sắp xếp các điểm Pareto theo thời gian T tăng dần
    pareto_pts = sorted(pareto_pts, key=lambda x: x["T"])
    
    T_all = [p["T"] for p in evaluated_points]
    E_all = [p["E"] for p in evaluated_points]
    
    T_par = [p["T"] for p in pareto_pts]
    E_par = [p["E"] for p in pareto_pts]

    E_ideal = base.m * base.g * abs(base.y_start - base.y_end)

    # Vẽ biểu đồ Tối giản (Minimalist Style)
    plt.rcParams.update({
        'font.size': 10,
        'axes.edgecolor': '#cccccc',
        'axes.linewidth': 0.8
    })
    
    fig, ax1 = plt.subplots(figsize=(8.5, 5.0))
    fig.patch.set_facecolor('#ffffff')
    ax1.set_facecolor('#fafafa')

    # 1. Vẽ tất cả các nghiệm thử nghiệm (Scatter plot mờ)
    ax1.scatter(T_all, E_all, color='#94a3b8', alpha=0.4, s=35, zorder=2, label='Không gian nghiệm DP')

    # 2. Vẽ Đường cong Pareto Frontier
    ax1.plot(T_par, E_par, color='#059669', lw=2.2, linestyle='-', zorder=3, label='Biên Pareto (Pareto Frontier)')
    ax1.scatter(T_par, E_par, color='#10b981', edgecolor='#047857', s=55, linewidth=1.2, zorder=4)

    # 3. Đánh dấu các điểm đặc trưng trên biên Pareto
    if len(pareto_pts) > 0:
        # Điểm Thời gian ngắn nhất (T min)
        pt_min_t = pareto_pts[0]
        ax1.annotate(
            f"Ưu tiên Thời gian\nT={pt_min_t['T']:.2f}s, E={pt_min_t['E']:.1f}J",
            xy=(pt_min_t['T'], pt_min_t['E']),
            xytext=(pt_min_t['T'] + 0.1, pt_min_t['E'] - 18),
            arrowprops=dict(arrowstyle="->", color="#059669", lw=1.2),
            fontsize=8.5, fontweight='bold', color="#047857"
        )
        
        # Điểm Năng lượng cao nhất (E max)
        pt_max_e = pareto_pts[-1]
        ax1.annotate(
            f"Ưu tiên Năng lượng\nT={pt_max_e['T']:.2f}s, E={pt_max_e['E']:.1f}J",
            xy=(pt_max_e['T'], pt_max_e['E']),
            xytext=(pt_max_e['T'] - 0.45, pt_max_e['E'] + 8),
            arrowprops=dict(arrowstyle="->", color="#059669", lw=1.2),
            fontsize=8.5, fontweight='bold', color="#047857"
        )

        # Điểm Cân bằng (Knee Point - giữa danh sách)
        if len(pareto_pts) >= 3:
            mid_idx = len(pareto_pts) // 2
            pt_mid = pareto_pts[mid_idx]
            ax1.annotate(
                f"Điểm cân bằng (Knee)\nT={pt_mid['T']:.2f}s, E={pt_mid['E']:.1f}J",
                xy=(pt_mid['T'], pt_mid['E']),
                xytext=(pt_mid['T'] + 0.12, pt_mid['E'] + 10),
                arrowprops=dict(arrowstyle="->", color="#0284c7", lw=1.2),
                fontsize=8.5, fontweight='bold', color="#0369a1"
            )

    # Nhãn trục X và Trục Y (Trái)
    ax1.set_xlabel("Thời gian hoàn thành hành trình T [s]", fontsize=10.5, fontweight='bold', labelpad=6)
    ax1.set_ylabel("Năng lượng hãm tái sinh E [J]", fontsize=10.5, fontweight='bold', color='#059669', labelpad=6)
    ax1.tick_params(axis='y', labelcolor='#059669')

    # Trục Y phụ (Phải) - Hiệu suất % theo Thế năng m*g*H
    ax2 = ax1.twinx()
    y1_min, y1_max = ax1.get_ylim()
    ax2.set_ylim((y1_min / E_ideal) * 100.0, (y1_max / E_ideal) * 100.0)
    ax2.set_ylabel("Hiệu suất thu hồi so với Thế năng m·g·H (%)", fontsize=9.5, color='#475569', labelpad=6)
    ax2.tick_params(axis='y', labelcolor='#475569')

    ax1.grid(True, linestyle=':', alpha=0.35, color='#94a3b8')
    ax1.legend(loc='lower right', frameon=True, framealpha=0.95, facecolor='#ffffff', edgecolor='#e2e8f0', fontsize=9.0)

    plt.title(f"PARETO FRONTIER: NĂNG LƯỢNG THU HỒI vs THỜI GIAN DI CHUYỂN\n"
              f"Thế năng m·g·H = {E_ideal:.1f} J | Tải m = {base.m} kg, Hành trình H = {abs(base.y_start - base.y_end)} m",
              fontsize=11.0, fontweight='bold', pad=10)

    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return path


# ==============================================================================
# 5. EXECUTION ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    p = Params()
    res = solve(p)
    print("="*65)
    print("KẾT QUẢ TỐI ƯU DP PROFILE VẬN TỐC THANG MÁY (REPROMPTED DP)")
    print("="*65)
    print(f"Thời gian di chuyển : {res['T']:.3f} s")
    print(f"Vận tốc đỉnh        : {res['v'].max():.3f} m/s")
    print(f"Gia tốc min/max     : {res['a'].min():.2f} / {res['a'].max():.2f} m/s²")
    print(f"|I| lớn nhất        : {np.abs(res['I']).max():.1f} A (Giới hạn {p.I_high} A)")
    print(f"Max |jerk|          : {np.abs(res['jerk']).max():.1f} m/s³")
    print(f"Năng lượng thu hồi  : {res['E_total']:.1f} J (Thế năng m*g*H = {res['E_ideal']:.1f} J)")
    print("="*65)
    print("Đã lưu biểu đồ chính vào:", plot_result(res, p))

    cfgs = {
        "Chỉ thời gian":      dict(w_energy=0.0, w_jerk=0.0),
        "Thời gian + Jerk":   dict(w_energy=0.0, w_jerk=2.0),
        "Cân bằng":           dict(),
        "Ưu tiên năng lượng": dict(w_energy=5.0),
    }
    print("Đã lưu biểu đồ so sánh trọng số vào:", compare_weights(p, cfgs))

    print("Đang tính toán Pareto Frontier giữa Năng lượng thu hồi và Thời gian...")
    pareto_img = plot_pareto_frontier(p, "dp_pareto_frontier.png")
    print("Đã lưu biểu đồ Pareto Frontier vào:", pareto_img)

