"""
BENCHMARK: biên dạng vận tốc DP  vs  Trapezoidal  vs  S-curve (cùng bộ thông số Params).

Đặt file này cùng thư mục với dp_velocity_profile.py rồi chạy:
    python benchmark_profiles.py
    python benchmark_profiles.py --sweep 2.2 2.5 3 4 5 --detail-T 3.0 --jerk-max 10

Nguyên tắc so sánh công bằng
----------------------------
1. Cùng Params: cùng hành trình H, V_max, A_min/A_max, I_high, jerk_max, mô hình điện (m, r, N, Kt, Ke, R)
   và cùng cờ count_consumption.
2. Cùng THỜI GIAN di chuyển T: profile Trapezoid / S-curve luôn được dựng với đúng T mà DP đạt được.
3. Cùng một hàm tính năng lượng (công thức giống hệt calc_segment_energy trong DP),
   tích phân trên lưới thời gian mịn dt = 1 ms. DP cũng được đánh giá lại bằng hàm này
   (so khớp với E_total do DP tự báo cáo -> cột kiểm tra).
4. Mỗi họ profile có 2 biến thể:
     - "chuẩn"  : gia tốc đối xứng a = min(|A_min|, A_max)  (cách dựng sách giáo khoa)
     - "tối ưu" : quét a_acc (tăng tốc) và a_dec (hãm) không đối xứng để E lớn nhất
                  => đối thủ mạnh nhất mà họ profile đó có thể làm được.
5. Trapezoid có gia tốc nhảy bậc nên jerk = vô cùng: nó KHÔNG thể thoả jerk_max.
   Biến thể tối ưu của nó vẫn được tối ưu với các ràng buộc còn lại và bị đánh dấu vi phạm jerk.
   S-curve dùng đúng jerk = jerk_max.
"""

from dataclasses import dataclass, field, replace
import argparse
import csv
import os
import sys
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dp_velocity_profile import Params, solve

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


# ==============================================================================
# 1. DỰNG PROFILE TỪ CÁC ĐOẠN (duration, a_start, jerk)  -- toạ độ theo chiều chuyển động
#    a > 0: đang tăng tốc theo chiều chuyển động.  a_y = direction * a  (giống DP)
# ==============================================================================
def build_trapezoid(T, H, a_acc, a_dec, v_max):
    """Hình thang: tăng tốc a_acc -> chạy đều -> hãm a_dec, tổng thời gian đúng T."""
    c = 0.5 / a_acc + 0.5 / a_dec
    disc = T * T - 4.0 * c * H
    if disc < 0:
        return None
    v = (T - np.sqrt(disc)) / (2.0 * c)  # nghiệm nhỏ -> cruise dài nhất
    if v > v_max + 1e-9:
        return None
    t1, t3 = v / a_acc, v / a_dec
    t2 = T - t1 - t3
    if t2 < -1e-9:
        return None
    return [(t1, a_acc, 0.0), (max(t2, 0.0), 0.0, 0.0), (t3, -a_dec, 0.0)]


def _s_phase_duration(v, a_pk, J):
    v = np.asarray(v, dtype=float)
    return np.where(v >= a_pk * a_pk / J, v / a_pk + a_pk / J, 2.0 * np.sqrt(np.maximum(v, 0.0) / J))


def _s_phase(v, a_pk, J, sign):
    """Pha tăng (sign=+1) hoặc hãm (sign=-1) từ 0 -> v (hoặc v -> 0) bằng 3 đoạn jerk +J/0/-J."""
    if v >= a_pk * a_pk / J:
        tj = a_pk / J
        ta = v / a_pk - a_pk / J
        return [(tj, 0.0, sign * J), (ta, sign * a_pk, 0.0), (tj, sign * a_pk, -sign * J)]
    tj = np.sqrt(v / J)
    return [(tj, 0.0, sign * J), (tj, sign * J * tj, -sign * J)]


def build_scurve(T, H, a_acc, a_dec, J, v_max):
    """S-curve 7 đoạn (jerk bị chặn = J), tổng thời gian đúng T, quãng đường đúng H."""
    vs = np.linspace(0.0, v_max, 4001)
    g = vs * (T - 0.5 * (_s_phase_duration(vs, a_acc, J) + _s_phase_duration(vs, a_dec, J))) - H
    cand = np.where(g >= 0)[0]
    if len(cand) == 0:
        return None
    i = cand[0]
    lo, hi = vs[max(i - 1, 0)], vs[i]
    for _ in range(60):  # bisection tìm v_peak
        mid = 0.5 * (lo + hi)
        gm = mid * (T - 0.5 * (_s_phase_duration(mid, a_acc, J) + _s_phase_duration(mid, a_dec, J))) - H
        lo, hi = (mid, hi) if gm < 0 else (lo, mid)
    v = hi
    Ta, Td = float(_s_phase_duration(v, a_acc, J)), float(_s_phase_duration(v, a_dec, J))
    tc = T - Ta - Td
    if tc < -1e-9:
        return None
    return _s_phase(v, a_acc, J, +1) + [(max(tc, 0.0), 0.0, 0.0)] + _s_phase(v, a_dec, J, -1)


# ==============================================================================
# 2. ĐÁNH GIÁ PROFILE (cùng mô hình năng lượng với DP)
# ==============================================================================
@dataclass
class Result:
    name: str
    T: float = 0.0
    E: float = 0.0
    eta: float = 0.0
    v_peak: float = 0.0
    ay_min: float = 0.0
    ay_max: float = 0.0
    I_max: float = 0.0
    jerk_an: float = 0.0     # jerk giải tích (inf nếu gia tốc nhảy bậc)
    jerk_grid: float = 0.0   # jerk đo trên lưới dy giống cách DP định nghĩa: Δa/Δt_đoạn
    s_err: float = 0.0
    viol: list = field(default_factory=list)
    info: str = ""
    # dữ liệu vẽ
    t: np.ndarray = None
    v: np.ndarray = None
    a_y: np.ndarray = None
    I: np.ndarray = None
    E_cum: np.ndarray = None
    tj: np.ndarray = None
    jerk_t: np.ndarray = None

    @property
    def ok(self):
        return not self.viol


def _breakpoints(segs):
    tb, vb, sb = [0.0], [0.0], [0.0]
    for d, a0, j in segs:
        v0, s0 = vb[-1], sb[-1]
        vb.append(v0 + a0 * d + 0.5 * j * d * d)
        sb.append(s0 + v0 * d + 0.5 * a0 * d * d + j * d ** 3 / 6.0)
        tb.append(tb[-1] + d)
    return np.array(tb), np.array(vb), np.array(sb)


def _energy(v, a_trav, dt, p: Params, direction):
    """Giống hệt calc_segment_energy: E = -(Ke*w*I + I^2*R)*dt (clip nếu count_consumption=False)."""
    a_y = direction * a_trav
    I = p.m * p.r * (p.g + a_y) / (p.Kt * p.N)
    w = direction * v * p.N / p.r
    E = -(p.Ke * w * I + I * I * p.R) * dt
    return E if p.count_consumption else np.maximum(E, 0.0)


def _grid_jerk(tn, sn, vn, p: Params, direction):
    n_steps = int(round(abs(p.y_end - p.y_start) / p.dy))
    s_nodes = np.arange(n_steps + 1) * p.dy
    v_nodes = np.interp(s_nodes, sn, vn)
    v_nodes[0] = v_nodes[-1] = 0.0
    vc, vnx = v_nodes[:-1], v_nodes[1:]
    a_k = direction * (vnx ** 2 - vc ** 2) / (2.0 * p.dy)
    dt_k = 2.0 * p.dy / np.maximum(vc + vnx, 1e-9)
    jerk = np.diff(np.concatenate([[0.0], a_k, [0.0]]))[:-1] / dt_k
    t_nodes = np.interp(s_nodes, sn, tn)
    return t_nodes, jerk


def evaluate(name, segs, p: Params, dt=1e-3, jerk_mode="analytic", check_jerk=True, full=True) -> Result:
    segs = [s for s in segs if s[0] > 1e-12]
    H = abs(p.y_end - p.y_start)
    direction = 1 if p.y_end > p.y_start else -1
    tb, vb, sb = _breakpoints(segs)
    T = tb[-1]
    d = np.array([s[0] for s in segs])
    a0 = np.array([s[1] for s in segs])
    jj = np.array([s[2] for s in segs])

    def state(tt):
        idx = np.clip(np.searchsorted(tb, tt, side="right") - 1, 0, len(segs) - 1)
        tau = tt - tb[idx]
        a = a0[idx] + jj[idx] * tau
        v = vb[idx] + a0[idx] * tau + 0.5 * jj[idx] * tau ** 2
        s = sb[idx] + vb[idx] * tau + 0.5 * a0[idx] * tau ** 2 + jj[idx] * tau ** 3 / 6.0
        return a, v, s

    n = max(int(np.ceil(T / dt)), 2)
    h = T / n
    tn = np.linspace(0.0, T, n + 1)
    tm = 0.5 * (tn[:-1] + tn[1:])
    an, vn, sn = state(tn)
    am, vm, _ = state(tm)
    Em = _energy(vm, am, h, p, direction)

    r = Result(name=name, T=T, E=float(Em.sum()))
    r.eta = r.E / (p.m * p.g * H)
    r.v_peak = float(vn.max())
    r.s_err = float(sn[-1] - H)

    a_s = a0
    a_e = a0 + jj * d
    a_y_ends = direction * np.concatenate([a_s, a_e])
    I_ends = p.m * p.r * (p.g + a_y_ends) / (p.Kt * p.N)
    r.ay_min, r.ay_max = float(a_y_ends.min()), float(a_y_ends.max())
    r.I_max = float(np.abs(I_ends).max())

    jumps = np.concatenate([np.abs(a_s[1:] - a_e[:-1]), [abs(a_s[0]), abs(a_e[-1])]])
    r.jerk_an = np.inf if jumps.max() > 1e-9 else float(np.abs(jj).max())

    if full or jerk_mode == "grid":
        tj, jerk_grid = _grid_jerk(tn, sn, vn, p, direction)
        r.jerk_grid = float(np.abs(jerk_grid).max())
        r.tj, r.jerk_t = tj, jerk_grid

    # kiểm tra ràng buộc
    tol = 1e-6
    if r.v_peak > p.V_max + tol:
        r.viol.append("V_max")
    if r.ay_min < p.A_min - tol or r.ay_max > p.A_max + tol:
        r.viol.append("gia tốc")
    if r.I_max > p.I_high + tol:
        r.viol.append("I_high")
    if abs(r.s_err) > 1e-4:
        r.viol.append("hành trình")
    if T > p.T_max + tol or T < p.T_min - tol:
        r.viol.append("T")
    if check_jerk:
        jk = r.jerk_an if jerk_mode == "analytic" else r.jerk_grid
        if jk > p.jerk_max + 1e-3:
            r.viol.append("jerk")

    if full:
        r.t, r.v, r.a_y = tn, vn, direction * an
        r.I = p.m * p.r * (p.g + r.a_y) / (p.Kt * p.N)
        r.E_cum = np.concatenate([[0.0], np.cumsum(Em)])
    return r


# ==============================================================================
# 3. TỐI ƯU THAM SỐ GIA TỐC CHO TRAPEZOID / S-CURVE (đối thủ mạnh nhất của từng họ)
# ==============================================================================
def optimise_accels(build, p: Params, a_lim, d_lim, check_jerk, n0=17, n_zoom=9, rounds=3):
    lo_a, hi_a = a_lim
    lo_d, hi_d = d_lim
    best = (-np.inf, None, None, None)
    for rd in range(rounds):
        n = n0 if rd == 0 else n_zoom
        As, Ds = np.linspace(lo_a, hi_a, n), np.linspace(lo_d, hi_d, n)
        for a in As:
            for dd in Ds:
                segs = build(a, dd)
                if segs is None:
                    continue
                r = evaluate("", segs, p, dt=4e-3, check_jerk=check_jerk, full=False)
                if r.ok and r.E > best[0]:
                    best = (r.E, a, dd, segs)
        if best[1] is None:
            return None
        sa, sd = As[1] - As[0], Ds[1] - Ds[0]
        lo_a, hi_a = max(a_lim[0], best[1] - sa), min(a_lim[1], best[1] + sa)
        lo_d, hi_d = max(d_lim[0], best[2] - sd), min(d_lim[1], best[2] + sd)
    return best[1], best[2], best[3]


def make_baselines(T, p: Params) -> dict:
    """Dựng 4 profile đối chứng ở đúng thời gian T. Trả về {tên: Result | None}."""
    H = abs(p.y_end - p.y_start)
    p = replace(p, T_min=0.0, T_max=T + 1e-6)  # T đã cố định theo DP, không kiểm tra lại cửa sổ T
    a_std = min(abs(p.A_min), p.A_max)
    a_lim, d_lim = (0.3, abs(p.A_min)), (0.3, p.A_max)  # a_y tăng tốc >= A_min ; hãm <= A_max
    out = {}

    def finish(name, segs, info, check_jerk=True):
        if segs is None:
            out[name] = None
            return
        r = evaluate(name, segs, p, dt=1e-3, jerk_mode="analytic")
        r.info = info
        out[name] = r

    # --- chuẩn (đối xứng) ---
    finish("Trapezoid chuẩn", build_trapezoid(T, H, a_std, a_std, p.V_max), f"a={a_std:.2f}")
    finish("S-curve chuẩn", build_scurve(T, H, a_std, a_std, p.jerk_max, p.V_max), f"a={a_std:.2f}, J={p.jerk_max:g}")

    # --- tối ưu (bất đối xứng); trapezoid được tối ưu bỏ qua jerk (vì không thể thoả) ---
    opt = optimise_accels(lambda a, d: build_trapezoid(T, H, a, d, p.V_max), p, a_lim, d_lim, check_jerk=False)
    finish("Trapezoid tối ưu", opt[2] if opt else None, f"a↑={opt[0]:.2f}, a↓={opt[1]:.2f}" if opt else "")
    opt = optimise_accels(lambda a, d: build_scurve(T, H, a, d, p.jerk_max, p.V_max), p, a_lim, d_lim, check_jerk=True)
    finish("S-curve tối ưu", opt[2] if opt else None, f"a↑={opt[0]:.2f}, a↓={opt[1]:.2f}, J={p.jerk_max:g}" if opt else "")
    return out


def dp_as_profile(res: dict, p: Params) -> Result:
    """Đánh giá lại nghiệm DP bằng cùng hàm evaluate (gia tốc không đổi trên mỗi đoạn dy)."""
    direction = 1 if p.y_end > p.y_start else -1
    segs = [(float(dt), direction * float(a), 0.0) for dt, a in zip(res["dt"], res["a"])]
    p = replace(p, T_min=0.0, T_max=res["T"] + 1e-6)
    r = evaluate("DP", segs, p, dt=1e-3, jerk_mode="grid")
    r.info = f"dy={p.dy:g}, dv={p.dv:g}, dt={p.dt_t:g}"
    r.jerk_an = np.inf
    return r


# ==============================================================================
# 4. CHẠY SO SÁNH TẠI MỘT THỜI GIAN & IN BẢNG
# ==============================================================================
ORDER = ["DP", "S-curve tối ưu", "S-curve chuẩn", "Trapezoid tối ưu", "Trapezoid chuẩn"]
STYLE = {
    "DP": dict(color="k", lw=2.6, ls="-"),
    "S-curve tối ưu": dict(color="tab:blue", lw=1.8, ls="-"),
    "S-curve chuẩn": dict(color="tab:blue", lw=1.5, ls="--"),
    "Trapezoid tối ưu": dict(color="tab:red", lw=1.8, ls="-"),
    "Trapezoid chuẩn": dict(color="tab:red", lw=1.5, ls="--"),
}


def compare_at(p_case: Params, verbose=True):
    res = solve(p_case)
    dp = dp_as_profile(res, p_case)
    chk = dp.E - res["E_total"]
    base = make_baselines(res["T"], p_case)
    all_r = {"DP": dp, **base}
    return res, all_r, chk


def _fmt_jerk(x):
    return "∞" if not np.isfinite(x) else f"{x:.1f}"


def print_table(all_r: dict, p: Params, T: float, chk: float = None):
    E_dp = all_r["DP"].E
    print(f"\n--- So sánh tại T = {T:.3f} s  (thế năng m·g·H = {p.m * p.g * abs(p.y_start - p.y_end):.2f} J) ---")
    hdr = f"{'Profile':<18}{'E [J]':>8}{'η':>7}{'ΔE vs DP':>10}{'v_peak':>8}{'a_y min/max':>14}{'|I|max':>8}{'jerk gt':>9}{'jerk lưới':>10}  Hợp lệ"
    print(hdr)
    print("-" * len(hdr))
    for k in ORDER:
        r = all_r.get(k)
        if r is None:
            print(f"{k:<18}{'—':>8}  không khả thi ở T này (không dựng được profile thoả V_max/A/T)")
            continue
        d = (r.E - E_dp) / E_dp * 100.0
        flag = "OK" if r.ok else "✗ " + ",".join(r.viol)
        print(f"{k:<18}{r.E:>8.2f}{r.eta * 100:>6.1f}%{d:>+9.1f}%{r.v_peak:>8.3f}"
              f"{r.ay_min:>8.2f}/{r.ay_max:<5.2f}{r.I_max:>8.2f}{_fmt_jerk(r.jerk_an):>9}{r.jerk_grid:>10.1f}  {flag}")
    if chk is not None:
        print(f"(kiểm tra: E_DP đánh giá lại − E_total do DP báo = {chk:+.4f} J)")
    print(f"Giới hạn: V_max={p.V_max}, a_y∈[{p.A_min}, {p.A_max}], I_high={p.I_high}, jerk_max={p.jerk_max}")


# ==============================================================================
# 5. ĐỒ THỊ
# ==============================================================================
def plot_detail(all_r: dict, p: Params, T: float, path: str):
    fig, ax = plt.subplots(2, 3, figsize=(16, 8.5))
    ax = ax.ravel()
    for k in ORDER:
        r = all_r.get(k)
        if r is None:
            continue
        st = STYLE[k]
        lab = f"{k} (E={r.E:.1f}J)"
        ax[0].plot(r.t, r.v, label=lab, **st)
        ax[1].plot(r.t, r.a_y, **st)
        ax[2].step(r.tj, np.append(r.jerk_t, r.jerk_t[-1]), where="post", **st)
        ax[3].plot(r.t, r.I, **st)
        ax[4].plot(r.t, r.E_cum, **st)
    ax[0].set(title="Vận tốc v(t)", xlabel="t [s]", ylabel="v [m/s]")
    ax[1].axhline(p.A_max, c="gray", ls=":")
    ax[1].axhline(p.A_min, c="gray", ls=":")
    ax[1].set(title="Gia tốc thẳng đứng a_y(t)", xlabel="t [s]", ylabel="a_y [m/s²]")
    ax[2].axhline(p.jerk_max, c="purple", ls=":")
    ax[2].axhline(-p.jerk_max, c="purple", ls=":")
    ax[2].set(title=f"Jerk đo trên lưới dy (định nghĩa của DP), giới hạn ±{p.jerk_max:g}",
              xlabel="t [s]", ylabel="jerk [m/s³]")
    ax[3].axhline(p.I_high, c="gray", ls=":")
    ax[3].set(title="Dòng điện động cơ I(t)", xlabel="t [s]", ylabel="I [A]")
    ax[4].set(title="Năng lượng thu hồi tích luỹ", xlabel="t [s]", ylabel="E [J]")

    names = [k for k in ORDER if all_r.get(k) is not None]
    Es = [all_r[k].E for k in names]
    cols = [STYLE[k]["color"] for k in names]
    bars = ax[5].barh(range(len(names)), Es, color=cols, alpha=0.8)
    for i, k in enumerate(names):
        mark = "" if all_r[k].ok else "  ✗ " + ",".join(all_r[k].viol)
        ax[5].text(Es[i], i, f" {Es[i]:.2f} J ({all_r[k].eta * 100:.0f}%){mark}", va="center", fontsize=8)
    ax[5].set_yticks(range(len(names)))
    ax[5].set_yticklabels(names, fontsize=9)
    ax[5].invert_yaxis()
    ax[5].set(title="E thu hồi tại cùng T  (η = E / mgH)", xlabel="E [J]")
    ax[5].set_xlim(0, max(Es) * 1.45)
    for a_ in ax:
        a_.grid(alpha=0.3)
    ax[0].legend(fontsize=8, loc="lower center")
    fig.suptitle(f"BENCHMARK DP vs S-curve vs Trapezoid  |  T = {T:.2f} s, jerk_max = {p.jerk_max:g}, "
                 f"V_max = {p.V_max}, A ∈ [{p.A_min}, {p.A_max}]", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_sweep(rows: list, p: Params, path: str):
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.2))
    Ts = [r["T"] for r in rows]
    E_ideal = p.m * p.g * abs(p.y_start - p.y_end)
    for k in ORDER:
        E = [row["E"].get(k, np.nan) for row in rows]
        bad = [row["valid"].get(k, True) is False for row in rows]
        ax[0].plot(Ts, E, marker="o", ms=5, label=k + ("  (vi phạm jerk)" if "Trapezoid" in k else ""), **STYLE[k])
        if k != "DP":
            E_dp = np.array([row["E"]["DP"] for row in rows])
            ax[1].plot(Ts, (np.array(E) - E_dp) / E_dp * 100.0, marker="o", ms=5, **STYLE[k])
    ax[0].axhline(E_ideal, c="gray", ls=":", label=f"m·g·H = {E_ideal:.1f} J")
    ax[0].set(title="Năng lượng thu hồi theo thời gian di chuyển T", xlabel="T [s]", ylabel="E [J]")
    ax[0].legend(fontsize=8)
    ax[1].axhline(0, c="k", lw=1)
    ax[1].set(title="Chênh lệch so với DP:  (E − E_DP) / E_DP", xlabel="T [s]", ylabel="[%]")
    for a_ in ax:
        a_.grid(alpha=0.3)
    fig.suptitle("BENCHMARK THEO T  (khoảng trống = profile đó không khả thi ở T này)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# ==============================================================================
# 6. MAIN
# ==============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", type=float, nargs="*", default=[2.5, 3.0, 3.5, 4.0, 5.0],
                    help="Các T mục tiêu để quét (rỗng = bỏ qua quét)")
    ap.add_argument("--window", type=float, default=0.1, help="DP chạy với T trong [T_target ± window]")
    ap.add_argument("--detail-T", type=float, default=3.0, help="T để vẽ đồ thị chi tiết thứ hai (<=0: bỏ qua)")
    ap.add_argument("--jerk-max", type=float, default=None)
    ap.add_argument("--count-consumption", action="store_true",
                    help="Tính cả điện tiêu thụ (E<0) cho mọi profile — nên bật để so sánh thực tế hơn")
    ap.add_argument("--outdir", default=".")
    args = ap.parse_args()

    p = Params()
    if args.jerk_max is not None:
        p = replace(p, jerk_max=args.jerk_max)
    if args.count_consumption:
        p = replace(p, count_consumption=True)
    tag = "_consume" if p.count_consumption else ""
    os.makedirs(args.outdir, exist_ok=True)
    out = lambda f: os.path.join(args.outdir, f)

    print("=" * 78)
    print("BENCHMARK: DP  vs  S-curve  vs  Trapezoid  (cùng Params, cùng T, cùng mô hình năng lượng)")
    print("=" * 78)

    # (1) DP tự chọn T tối ưu trong [T_min, T_max] rồi so sánh tại đúng T đó
    t0 = time.time()
    res, all_r, chk = compare_at(p)
    print_table(all_r, p, res["T"], chk)
    print(f"[DP tự chọn T* trong [{p.T_min}, {p.T_max}] s — {time.time() - t0:.1f}s]")
    print("Đã lưu:", plot_detail(all_r, p, res["T"], out("benchmark_detail_Tstar" + tag + ".png")))

    # (2) Quét theo T
    rows = []
    for T_target in args.sweep:
        pc = replace(p, T_min=max(T_target - args.window, 0.0), T_max=T_target + args.window)
        try:
            res_i, all_i, chk_i = compare_at(pc)
        except RuntimeError as e:
            print(f"\nBỏ qua T={T_target}: DP không khả thi ({e})")
            continue
        print_table(all_i, pc, res_i["T"], chk_i)
        print(f"(T mục tiêu {T_target:g} s, cửa sổ DP [{pc.T_min:.2f}, {pc.T_max:.2f}] s -> T thật của nghiệm DP = {res_i['T']:.3f} s;"
              f" DP làm tròn từng đoạn lên lưới dt_t={pc.dt_t:g} s nên T thật có thể lệch cửa sổ)")
        rows.append({
            "T": res_i["T"],
            "E": {k: all_i[k].E for k in ORDER if all_i.get(k) is not None},
            "valid": {k: all_i[k].ok for k in ORDER if all_i.get(k) is not None},
            "jerk": {k: all_i[k].jerk_grid for k in ORDER if all_i.get(k) is not None},
        })
        if args.detail_T > 0 and abs(T_target - args.detail_T) < 1e-9:
            print("Đã lưu:", plot_detail(all_i, pc, res_i["T"], out(f"benchmark_detail_T{T_target:g}" + tag + ".png")))

    if rows:
        print("Đã lưu:", plot_sweep(rows, p, out("benchmark_sweep" + tag + ".png")))
        with open(out("benchmark_sweep" + tag + ".csv"), "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["T [s]"] + [f"E {k} [J]" for k in ORDER] + [f"hợp lệ {k}" for k in ORDER])
            for row in rows:
                w.writerow([f"{row['T']:.3f}"]
                           + [f"{row['E'][k]:.3f}" if k in row["E"] else "" for k in ORDER]
                           + [row["valid"].get(k, "") for k in ORDER])
        print("Đã lưu:", out("benchmark_sweep" + tag + ".csv"))


if __name__ == "__main__":
    main()
