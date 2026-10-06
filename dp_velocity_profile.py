"""
Quy hoạch động (DP) tìm biên dạng vận tốc tối ưu NĂNG LƯỢNG THU HỒI TỐI ĐA (Pure Energy Recovery).
Ràng buộc: Thời gian di chuyển T <= T_max và Gia tốc giật |Jerk| <= Jerk_max.

Bài toán: Cho khoảng thời gian T_max (và giới hạn Jerk), tìm biên dạng vận tốc v(t) 
sao cho thu hồi năng lượng nạp pin E_total là LỚN NHẤT.

Hàm chi phí: Min Cost = -E_total  (tương đương Max E_total)
"""

from dataclasses import dataclass, replace
import argparse
import sys
import time
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
    m: float = 2       # Khối lượng tải + khung [kg]
    r: float = 0.03          # Bán kính puly / bánh răng kéo [m]
    N: float = 20           # Tỉ số truyền động cơ
    Kt: float = 0.018         # Hằng số mô-men [Nm/A]
    Ke: float = 0.018         # Hằng số sức điện động [V.s/rad]
    R: float = 2.4        # Điện trở cuộn dây + mạch [Ohm]
    g: float = 9.81          # Gia tốc trọng trường [m/s^2]
    
    # Ràng buộc Động lực học & Phần cứng
    V_max: float = 0.9     # Vận tốc tối đa [m/s]
    A_min: float = -9.81     # Gia tốc đứng nhỏ nhất [m/s^2]
    A_max: float = 2       # Gia tốc đứng lớn nhất [m/s^2]
    I_high: float = 5     # Dòng điện cực đại cho phép [A]
    jerk_max: float = 10.0   # Ràng buộc Jerk tối đa cho phép [m/s^3]
    T_min: float = 0.0       # Ràng buộc thời gian di chuyển nhỏ nhất [s]
    T_max: float = 10      # Ràng buộc thời gian di chuyển lớn nhất [s]
    
    # Hành trình
    y_start: float = 1.5     # Vị trí đầu [m]
    y_end: float = 0.0       # Vị trí cuối [m]
    
    # Bước lưới rời rạc hóa
    dy: float = 0.05          # Bước vị trí [m]
    dv: float = 0.02        # Bước vận tốc [m/s]
    dt_t: float = 0.02       # Bước lưới thời gian cho DP [s]
    
    count_consumption: bool = False  # True: Tính cả điện tiêu thụ (E<0); False: Chỉ tính phần thu hồi (E>=0)


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
# 3. THUẬT TOÁN QUY HOẠCH ĐỘNG (DP Solver - Pure Energy Maximization with T & Jerk Constraints)
# ==============================================================================
def solve(p: Params = Params()) -> dict:
    t0 = time.time()
    # 1. Khởi tạo Lưới vị trí và Lưới vận tốc
    dist = abs(p.y_end - p.y_start)
    n_steps = int(round(dist / p.dy))
    direction = 1 if p.y_end > p.y_start else -1
    y_nodes = p.y_start + direction * p.dy * np.arange(n_steps + 1)
    
    n_v = int(round(p.V_max / p.dv)) + 1
    v_grid = np.arange(n_v) * p.dv

    n_t = int(round(p.T_max / p.dt_t)) + 1

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

    # Kiểm tra tính khả thi của từng chuyển tiếp (Vận tốc sum > 0, Gia tốc trong giới hạn, Dòng điện cho phép)
    feas_matrix = (
        (vsum_grid > 0) &
        (a_matrix >= p.A_min - 1e-6) & (a_matrix <= p.A_max + 1e-6) &
        (np.abs(I_matrix) <= p.I_high)
    )

    # Quy đổi bước dt sang chỉ số bước lưới thời gian (int)
    dt_indices = np.where(np.isfinite(dt_matrix), np.round(dt_matrix / p.dt_t), 0).astype(int)

    # 3. Khởi tạo Bảng DP 4D: dp_table[k][v_prev_idx][v_curr_idx][t_accumulated_idx]
    # Lưu Năng lượng thu hồi tích lũy TỐI ĐA (Max E)
    dp_table = np.full((n_steps + 1, n_v, n_v, n_t), -np.inf)
    parent_p = np.full((n_steps + 1, n_v, n_v, n_t), -1, dtype=int)
    parent_t = np.full((n_steps + 1, n_v, n_v, n_t), -1, dtype=int)
    
    # Điều kiện biên k=0: Xuất phát từ nghỉ (v=0, a=0, t=0 s, E=0 J)
    dp_table[0][0][0][0] = 0.0

    # Pruning: Vị trí xuất phát k=0 và kết thúc k=n_steps bắt buộc v=0
    allowed_v = []
    for k in range(n_steps + 1):
        if k == 0 or k == n_steps:
            allowed_v.append([0])
        else:
            allowed_v.append(list(range(1, n_v)))

    # 4. Vòng lặp Quy hoạch động tiến (Forward DP Recursion)
    for k in range(n_steps):
        curr_allowed = allowed_v[k]
        next_allowed = allowed_v[k + 1]
        prev_arr = np.array([0] if k == 0 else allowed_v[k - 1])

        for c in curr_allowed:
            # Lấy không gian trạng thái khả thi tại bước k đối với v_curr = v_grid[c]
            sub_dp = dp_table[k, prev_arr, c, :]
            p_indices, t_indices = np.where(np.isfinite(sub_dp))
            if len(p_indices) == 0:
                continue
            
            p_valids = prev_arr[p_indices]
            energies_valid = sub_dp[p_indices, t_indices]

            for n in next_allowed:
                if not feas_matrix[c, n]:
                    continue

                dt_curr = dt_matrix[c, n]
                dt_idx_step = dt_indices[c, n]
                a_curr = a_matrix[c, n]
                E_curr = E_matrix[c, n]

                # RÀNG BUỘC JERK: |jerk| = |(a_curr - a_prev) / dt_curr| <= jerk_max
                if k == 0:
                    a_prev = 0.0
                    jerk = a_curr / dt_curr
                    jerk_mask = np.abs(jerk) <= (p.jerk_max + 1e-5)
                else:
                    a_prev = a_matrix[p_valids, c]
                    jerk = (a_curr - a_prev) / dt_curr
                    jerk_mask = np.abs(jerk) <= (p.jerk_max + 1e-5)

                if not np.any(jerk_mask):
                    continue

                p_sel = p_valids[jerk_mask]
                t_sel = t_indices[jerk_mask]
                e_sel = energies_valid[jerk_mask]

                # RÀNG BUỘC THỜI GIAN: t_next <= T_max
                t_next_sel = t_sel + dt_idx_step
                valid_t_mask = t_next_sel < n_t
                if not np.any(valid_t_mask):
                    continue

                p_final = p_sel[valid_t_mask]
                t_final = t_sel[valid_t_mask]
                tn_final = t_next_sel[valid_t_mask]
                tot_energies = e_sel[valid_t_mask] + E_curr  # Thuần Tối ưu Năng lượng (Max E)

                # Vectorized Bellman Update (Cập nhật đường đi có năng lượng LỚN HƠN)
                sort_idx = np.argsort(tot_energies)
                p_final = p_final[sort_idx]
                t_final = t_final[sort_idx]
                tn_final = tn_final[sort_idx]
                tot_energies = tot_energies[sort_idx]

                mask_better = tot_energies > dp_table[k + 1, c, n, tn_final]
                tn_better = tn_final[mask_better]
                dp_table[k + 1, c, n, tn_better] = tot_energies[mask_better]
                parent_p[k + 1, c, n, tn_better] = p_final[mask_better]
                parent_t[k + 1, c, n, tn_better] = t_final[mask_better]

    # 5. Tìm trạng thái kết thúc (v_N = 0) có năng lượng thu hồi TỐI ĐA (Max E) trong khoảng T_min <= T <= T_max
    best_energy = -np.inf
    best_c, best_t = -1, -1

    min_t_idx = int(round(p.T_min / p.dt_t))
    max_t_idx = int(round(p.T_max / p.dt_t))

    for c in range(n_v):
        sub_final = dp_table[n_steps, c, 0, :]
        valid_range = sub_final[min_t_idx:max_t_idx + 1]
        if len(valid_range) > 0 and np.any(np.isfinite(valid_range)):
            local_idx = np.argmax(valid_range)
            if valid_range[local_idx] > best_energy:
                best_energy = valid_range[local_idx]
                best_c = c
                best_t = min_t_idx + local_idx

    if not np.isfinite(best_energy):
        raise RuntimeError(f"Không tìm thấy đường đi DP hợp lệ thỏa mãn {p.T_min:.2f}s <= T <= {p.T_max:.2f}s và jerk_max = {p.jerk_max}m/s^3.")

    # 6. Truy vết ngược (Backtracking Path Reconstruction)
    idx = [0, best_c]
    t_idx_path = [best_t]
    curr_c, curr_n = best_c, 0
    curr_t = best_t

    for k in range(n_steps, 0, -1):
        prev_p = parent_p[k, curr_c, curr_n, curr_t]
        prev_t = parent_t[k, curr_c, curr_n, curr_t]
        if k > 1:
            idx.append(prev_p)
            t_idx_path.append(prev_t)
        curr_n = curr_c
        curr_c = prev_p
        curr_t = prev_t

    idx = idx[::-1]
    t_idx_path = t_idx_path[::-1]

    # 7. Tổng hợp Kết quả Động học & Năng lượng
    v = v_grid[idx]
    jc = idx[:-1]
    jn = idx[1:]

    dt_k = dt_matrix[jc, jn]
    t_path = np.concatenate([[0.0], np.cumsum(dt_k)])
    a_k = a_matrix[jc, jn]
    I_k = I_matrix[jc, jn]
    E_k = E_matrix[jc, jn]
    
    a_ext = np.concatenate([[0.0], a_k, [0.0]])
    jerk_k = np.diff(a_ext)[:-1] / dt_k

    res = {
        "idx": idx, "y": y_nodes, "v": v, "t": t_path, "dt": dt_k,
        "a": a_k, "I": I_k, "E": E_k, "jerk": jerk_k,
        "cost": float(best_energy), "T": float(t_path[-1]),
        "E_total": float(np.sum(E_k)),
        "E_ideal": float(p.m * p.g * abs(p.y_start - p.y_end)),
        "solve_time": time.time() - t0
    }
    return res


# ==============================================================================
# 4. HÀM VẼ ĐỒ THỊ & PHÂN TÍCH (Visualization)
# ==============================================================================
def plot_result(res: dict, p: Params, path: str = "dp_velocity_profile.png") -> str:
    t, y, v = res["t"], res["y"], res["v"]
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    ax = ax.ravel()

    ax[0].plot(t, v, "b.-")
    ax[0].set(title="Vận tốc theo thời gian v(t)", xlabel="t [s]", ylabel="v [m/s]")

    ax[1].plot(y, v, "g.-")
    ax[1].set(title="Vận tốc theo vị trí v(y)", xlabel="y [m]", ylabel="v [m/s]")

    ax[2].plot(t, y, "k.-")
    ax[2].set(title="Quãng đường y theo thời gian y(t)", xlabel="t [s]", ylabel="y [m]")

    ax[3].step(t, np.append(res["a"], res["a"][-1]), where="post", color="r")
    ax[3].axhline(p.A_max, ls="--", c="gray", label="A_max")
    ax[3].axhline(p.A_min, ls="--", c="gray", label="A_min")
    ax[3].set(title="Gia tốc thẳng đứng a_y", xlabel="t [s]", ylabel="a [m/s²]")
    ax[3].legend(loc="upper right", fontsize=8)

    ax[4].step(t, np.append(res["jerk"], res["jerk"][-1]), where="post", color="m")
    ax[4].axhline(p.jerk_max, ls="--", c="purple", label=f"Jerk max ({p.jerk_max})")
    ax[4].axhline(-p.jerk_max, ls="--", c="purple")
    ax[4].set(title="Jerk = Δa/Δt", xlabel="t [s]", ylabel="jerk [m/s³]")
    ax[4].legend(loc="upper right", fontsize=8)

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
        
    fig.suptitle(f"DP Velocity Profile (Pure Energy Maximization)\n"
                 f"T = {res['T']:.2f} s (Ràng buộc T_max = {p.T_max:.1f} s) | "
                 f"Max |Jerk| = {np.abs(res['jerk']).max():.2f} m/s³ (Ràng buộc {p.jerk_max:.1f} m/s³) | "
                 f"E_net = {res['E_total']:.1f} J / Thế năng = {res['E_ideal']:.1f} J",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_energy_vs_time_sweep(base: Params, t_targets: list = None, path: str = "dp_energy_vs_time.png") -> str:
    """
    Quét qua các khoảng thời gian t (t_target) khác nhau,
    vẽ đồ thị Năng lượng Thu hồi Tối đa E_max(t) và so sánh các biên dạng vận tốc.
    """
    if t_targets is None:
        t_targets = np.linspace(1.2, 5.0, 15)

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))

    E_results = []
    T_actuals = []
    valid_t_targets = []

    for t_target in t_targets:
        t_min_val = float(t_target) - 0.5
        t_max_val = float(t_target) + 0.5
        p_test = replace(base, T_min=t_min_val, T_max=t_max_val)
        try:
            r = solve(p_test)
            valid_t_targets.append(t_target)
            E_results.append(r["E_total"])
            T_actuals.append(r["T"])
            print(f"E = {r['E_total']:.2f} J  |  T_actual = {r['T']:.3f} s [{t_min_val}s <= T <= {t_max_val}s]")

            ax[0].plot(r["t"], r["v"], label=f"t_target={t_target:.2f}s (T={r['T']:.2f}s, E={r['E_total']:.1f}J)")
        except Exception as e:
            print(f"Bỏ qua t_target = {t_target:.2f}s (Không khả thi: {e})")

    if not valid_t_targets:
        print("Không có điểm thời gian nào hợp lệ để vẽ sweep.")
        return ""

    ax[0].set(title="Biên dạng vận tốc v(t) theo khoảng thời gian t_target", xlabel="t [s]", ylabel="v [m/s]")
    ax[0].grid(alpha=0.3)
    ax[0].legend(fontsize=8)

    # Đồ thị Năng lượng thu hồi tối đa theo khoảng thời gian t_target
    ax[1].plot(valid_t_targets, E_results, "o-", color="#059669", lw=2, markersize=7, label="E_total [J]")
    ax[1].axhline(base.m * base.g * abs(base.y_start - base.y_end), ls="--", color="gray", label=f"Thế năng m*g*H ({base.m * base.g * abs(base.y_start - base.y_end):.1f}J)")
    ax[1].set(title="Năng lượng thu hồi tối đa theo Khoảng thời gian t_target", xlabel="Giới hạn khoảng thời gian t_target [s]", ylabel="E_total [J]")
    ax[1].grid(alpha=0.3)
    ax[1].legend(fontsize=9)

    fig.suptitle("PHÂN TÍCH TỐI ƯU NĂNG LƯỢNG THU HỒI THEO KHOẢNG THỜI GIAN t", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# ==============================================================================
# 5. EXECUTION ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    p = Params()
    print("=" * 65)
    print("QUY HOẠCH ĐỘNG (DP) - TỐI ƯU NĂNG LƯỢNG THU HỒI TỐI ĐA")
    print("Ràng buộc: Thời gian di chuyển T <= T_max & Jerk <= Jerk_max")
    print("=" * 65)
    
    res = solve(p)
    
    print(f"Ràng buộc T_max     : {p.T_max:.2f} s")
    print(f"Ràng buộc Jerk max  : {p.jerk_max:.2f} m/s³")
    print(f"Thời gian thực tế T : {res['T']:.3f} s")
    print(f"Vận tốc đỉnh        : {res['v'].max():.3f} m/s")
    print(f"Gia tốc min/max     : {res['a'].min():.2f} / {res['a'].max():.2f} m/s²")
    print(f"|I| lớn nhất        : {np.abs(res['I']).max():.1f} A (Giới hạn {p.I_high} A)")
    print(f"Max |jerk|          : {np.abs(res['jerk']).max():.2f} m/s³ (Giới hạn {p.jerk_max} m/s³)")
    print(f"Năng lượng thu hồi  : {res['E_total']:.1f} J (Thế năng m*g*H = {res['E_ideal']:.1f} J)")
    print(f"Thời gian tính DP   : {res['solve_time']:.3f} s")
    print("=" * 65)
    
    img_result = plot_result(res, p)
    print("Đã lưu biểu đồ chính vào:", img_result)

    print("Đang quét phân tích Năng lượng thu hồi theo các khoảng thời gian T_max...")
    img_sweep = plot_energy_vs_time_sweep(p)
    print("Đã lưu biểu đồ quét thời gian vào:", img_sweep)
