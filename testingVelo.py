import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

# ==============================================================================
# 1. KINEMATIC ASYMMETRIC S-CURVE GENERATOR
# Reference: 'A Complete Solution to Asymmetric S-curve Motion Profile: Theory & Experiments'
# ==============================================================================
def generate_profile(beta, gamma, distance, v_max, a_max, dt=0.001):
    beta = max(0.001, min(beta, 0.5))
    gamma = max(1.0, gamma)

    dt_j_star = beta * (v_max / a_max)
    dt_a_star = (1.0 - beta) * (v_max / a_max)
    j_max = a_max / dt_j_star

    delta_s_star = (1.0 + gamma) * (beta**2) * (v_max**2 / a_max)
    delta_l_star = (1.0 + gamma) * ((1.0 + beta) / 2.0) * (v_max**2 / a_max)

    if distance < delta_s_star:
        mode = "Short (S2)"
        dt_a, dt_v = 0.0, 0.0
        dt_j = ((dt_j_star / ((1.0 + gamma) * a_max)) * distance) ** (1.0 / 3.0)
    elif distance <= delta_l_star:
        mode = "Medium (S3)"
        dt_j, dt_v = dt_j_star, 0.0
        term1 = (3.0 * beta * v_max) / (2.0 * a_max)
        term2 = ((beta * v_max) / (2.0 * a_max)) ** 2
        term3 = (2.0 * distance) / (a_max * (1.0 + gamma))
        dt_a = -term1 + np.sqrt(term2 + term3)
    else:
        mode = "Long (S4)"
        dt_j, dt_a = dt_j_star, dt_a_star
        dt_v = (distance - delta_l_star) / v_max

    durations = [
        dt_j,           # Phase 1: Jerk Up
        dt_a,           # Phase 2: Const Accel
        dt_j,           # Phase 3: Jerk Down
        dt_v,           # Phase 4: Cruise
        gamma * dt_j,   # Phase 5: Decel Jerk Down
        gamma * dt_a,   # Phase 6: Const Decel
        gamma * dt_j    # Phase 7: Decel Jerk Up to 0
    ]
    
    t_breaks = np.cumsum([0.0] + durations)
    t_total = t_breaks[-1]
    n_points = max(2, int(np.ceil(t_total / dt)))
    time_arr = np.linspace(0, t_total, n_points)

    jerk_arr = np.zeros_like(time_arr)
    accel_arr = np.zeros_like(time_arr)
    vel_arr = np.zeros_like(time_arr)
    pos_arr = np.zeros_like(time_arr)

    a1 = j_max * dt_j
    v1 = 0.5 * j_max * (dt_j**2)
    p1 = (1.0 / 6.0) * j_max * (dt_j**3)

    a2 = a1
    v2 = v1 + a1 * dt_a
    p2 = p1 + v1 * dt_a + 0.5 * a1 * (dt_a**2)

    a3 = 0.0
    v3 = v2 + a1 * dt_j - 0.5 * j_max * (dt_j**2)
    p3 = p2 + v2 * dt_j + 0.5 * a1 * (dt_j**2) - (1.0 / 6.0) * j_max * (dt_j**3)

    a4 = 0.0
    v4 = v3
    p4 = p3 + v3 * dt_v

    j_brake = j_max / (gamma**2)
    a_brake = a1 / gamma

    for idx, t in enumerate(time_arr):
        if t <= t_breaks[1]:  # Phase 1
            tau = t
            jerk_arr[idx] = j_max
            accel_arr[idx] = j_max * tau
            vel_arr[idx] = 0.5 * j_max * (tau**2)
            pos_arr[idx] = (1.0 / 6.0) * j_max * (tau**3)

        elif t <= t_breaks[2]:  # Phase 2
            tau = t - t_breaks[1]
            jerk_arr[idx] = 0.0
            accel_arr[idx] = a1
            vel_arr[idx] = v1 + a1 * tau
            pos_arr[idx] = p1 + v1 * tau + 0.5 * a1 * (tau**2)

        elif t <= t_breaks[3]:  # Phase 3
            tau = t - t_breaks[2]
            jerk_arr[idx] = -j_max
            accel_arr[idx] = a1 - j_max * tau
            vel_arr[idx] = v2 + a1 * tau - 0.5 * j_max * (tau**2)
            pos_arr[idx] = p2 + v2 * tau + 0.5 * a1 * (tau**2) - (1.0 / 6.0) * j_max * (tau**3)

        elif t <= t_breaks[4]:  # Phase 4 (Cruise)
            tau = t - t_breaks[3]
            jerk_arr[idx] = 0.0
            accel_arr[idx] = 0.0
            vel_arr[idx] = v4
            pos_arr[idx] = p3 + v4 * tau

        elif t <= t_breaks[5]:  # Phase 5 (Decel Jerk)
            tau = t - t_breaks[4]
            jerk_arr[idx] = -j_brake
            accel_arr[idx] = -j_brake * tau
            vel_arr[idx] = v4 - 0.5 * j_brake * (tau**2)
            pos_arr[idx] = p4 + v4 * tau - (1.0 / 6.0) * j_brake * (tau**3)

        elif t <= t_breaks[6]:  # Phase 6 (Const Decel)
            tau = t - t_breaks[5]
            tau5 = gamma * dt_j
            v5 = v4 - 0.5 * j_brake * (tau5**2)
            p5 = p4 + v4 * tau5 - (1.0 / 6.0) * j_brake * (tau5**3)
            jerk_arr[idx] = 0.0
            accel_arr[idx] = -a_brake
            vel_arr[idx] = v5 - a_brake * tau
            pos_arr[idx] = p5 + v5 * tau - 0.5 * a_brake * (tau**2)

        else:  # Phase 7 (Decel to Stop)
            tau = t - t_breaks[6]
            tau_rem = max(0.0, (gamma * dt_j) - tau)
            jerk_arr[idx] = j_brake
            accel_arr[idx] = -j_brake * tau_rem
            vel_arr[idx] = 0.5 * j_brake * (tau_rem**2)
            
            tau5 = gamma * dt_j
            v5 = v4 - 0.5 * j_brake * (tau5**2)
            p5 = p4 + v4 * tau5 - (1.0 / 6.0) * j_brake * (tau5**3)
            tau6 = gamma * dt_a
            v6 = v5 - a_brake * tau6
            p6 = p5 + v5 * tau6 - 0.5 * a_brake * (tau6**2)
            pos_arr[idx] = p6 + v6 * tau - 0.5 * a_brake * (tau**2) + (1.0 / 6.0) * j_brake * (tau**3)

    metadata = {
        "mode": mode,
        "delta_s_star": delta_s_star,
        "delta_l_star": delta_l_star,
        "v_peak": v4,
        "a_peak": a1,
        "t_total": t_total,
        "t_breaks": t_breaks
    }

    return time_arr, jerk_arr, accel_arr, vel_arr, pos_arr, metadata

# ==============================================================================
# 2. ELECTROMECHANICAL & BOOST REGENERATIVE BRAKING MODEL WITH GRAVITY
# ==============================================================================
MOTOR_PARAMS = {
    'mass': 10.0,         # Descending payload mass (kg)
    'gear_ratio': 5.0,   # Gearbox / Transmission reduction ratio N
    'eta_gear': 0.90,    # Gearbox mechanical efficiency (90%)
    'g': 9.81,           # Gravity (m/s^2)
    'r_spool': 0.04,     # Physical spool / pinion radius (m)
    'J_rotor': 0.001,    # Motor rotor inertia (kg*m^2)
    'B': 0.0005,         # Viscous friction coefficient (N*m*s/rad)
    'Ke': 0.08,          # Back-EMF constant (V*s/rad)
    'Kt': 0.08,          # Torque constant (N*m/A)
    'R_motor': 0.25,     # Motor armature resistance (Ohms)
    'L_motor': 0.002,    # Motor armature inductance (H) [2 mH]
    'V_battery': 12.0,   # Battery storage voltage (V)
    'delta_I': 0.5,      # Hysteresis current ripple (A)
    'V_diode': 0.7,      # Switch / Diode forward drop (V)
    'I_max': 30.0,       # Max allowable current limit (A)
    'eta_battery': 0.95  # Battery charge acceptance efficiency (95%)
}

def calculate_regen_energy(time_arr, vel_arr, accel_arr, params=MOTOR_PARAMS):
    mass = params['mass']
    g = params['g']
    N = params.get('gear_ratio', 1.0)
    eta_gear = params.get('eta_gear', 0.90)
    r_spool = params.get('r_spool', 0.02)
    r_eff = r_spool / N  # Effective radius reflected to motor shaft
    
    J_rotor = params['J_rotor']
    B = params['B']
    Ke = params['Ke']
    Kt = params['Kt']
    R = params['R_motor']
    L = params['L_motor']
    V_bat = params['V_battery']
    delta_I = params['delta_I']
    V_diode = params['V_diode']
    I_max = params['I_max']
    eta_battery = params.get('eta_battery', 0.95)

    # Total equivalent inertia reflected to motor shaft
    J_total = J_rotor + mass * (r_eff ** 2)

    # Angular Kinematics (descending direction is positive v > 0)
    omega = vel_arr / r_eff
    alpha = accel_arr / r_eff

    # Gravitational overhauling torque reflected to motor shaft (incorporating gear efficiency)
    T_g = mass * g * r_eff * eta_gear

    # Dynamic Torque Balance for Downward Motion:
    # J_total * alpha(t) = T_g - T_e(t) - B * omega(t)
    # => T_e(t) = T_g - J_total * alpha(t) - B * omega(t)
    T_e_raw = T_g - J_total * alpha - B * omega
    
    # Generator mode when T_e > 0 and omega > 0
    generator_mask = (T_e_raw > 0) & (omega > 1e-4)
    Te = np.where(generator_mask, T_e_raw, 0.0)
    I_target = np.clip(Te / Kt, 0.0, I_max)

    # Back-EMF: Eb(t) = Ke * omega(t)
    Eb = Ke * omega

    # Boost Converter Switching Times:
    Eb_clamped = np.clip(Eb, 0.05, V_bat - 0.05)
    T_charge = (L * delta_I) / Eb_clamped
    T_discharge = (L * delta_I) / (V_bat - Eb_clamped)
    T_switch = T_charge + T_discharge
    duty_cycle = np.where(generator_mask, T_charge / T_switch, 0.0)

    # Actual Current Injected into Battery (Battery Current = Motor Current * (1 - D))
    I_bat = np.where(generator_mask, I_target * (1.0 - duty_cycle), 0.0)

    # Power calculations:
    # Ideal electrical power generated at motor terminals
    P_ideal = Eb * I_target
    # Electrical losses (Copper loss RMS considering ripple + Diode forward loss)
    I_rms_sq = (I_target ** 2) + ((delta_I ** 2) / 12.0)
    P_cu = I_rms_sq * R
    P_diode = V_diode * I_bat
    P_loss = P_cu + P_diode
    
    # Net regenerated power delivered into the battery (incorporating battery charging efficiency)
    P_net_elec = np.maximum(0.0, P_ideal - P_loss)
    P_net = np.where(generator_mask, P_net_elec * eta_battery, 0.0)

    # Cumulative Regenerated Energy: E_regen(t) = integral(P_net * dt)
    if len(time_arr) > 1:
        dt = time_arr[1] - time_arr[0]
        E_cum = np.cumsum(P_net) * dt
    else:
        E_cum = np.zeros_like(time_arr)

    E_total = E_cum[-1] if len(E_cum) > 0 else 0.0

    # Theoretical Energy Available
    v_max_reached = np.max(vel_arr) if len(vel_arr) > 0 else 0.0
    dist_total = (time_arr[-1] * 0) # from profile
    E_potential = mass * g * (vel_arr[-1] if len(vel_arr)==0 else 0.0)

    return {
        'omega': omega,
        'alpha': alpha,
        'Te': Te,
        'Eb': Eb,
        'I_target': I_target,
        'I_bat': I_bat,
        'duty_cycle': duty_cycle,
        'P_ideal': P_ideal,
        'P_loss': P_loss,
        'P_net': P_net,
        'E_cum': E_cum,
        'E_total': E_total,
        'r_eff': r_eff
    }

# ==============================================================================
# 3. OPTIMIZER: FIND (BETA*, GAMMA*) TO MAXIMIZE REGENERATED ENERGY
# ==============================================================================
def optimize_parameters(distance, v_max, a_max, mass=5.0, gear_ratio=5.0, params=MOTOR_PARAMS):
    p_copy = params.copy()
    p_copy['mass'] = mass
    p_copy['gear_ratio'] = gear_ratio
    
    beta_vals = np.linspace(0.05, 0.50, 15)
    gamma_vals = np.linspace(1.0, 4.0, 20)
    
    best_E = -1.0
    best_beta = 0.2
    best_gamma = 2.0

    for b in beta_vals:
        for g in gamma_vals:
            t_arr, j_arr, a_arr, v_arr, p_arr, _ = generate_profile(b, g, distance, v_max, a_max, dt=0.002)
            res = calculate_regen_energy(t_arr, v_arr, a_arr, p_copy)
            if res['E_total'] > best_E:
                best_E = res['E_total']
                best_beta = b
                best_gamma = g

    return best_beta, best_gamma, best_E

# ==============================================================================
# 4. INTERACTIVE VISUALIZATION GUI (2-Column Dashboard)
# ==============================================================================
TARGET_DIST = 3.0   # meters
V_MAX = 1.5         # m/s
A_MAX = 9.81        # m/s^2
MASS_DEFAULT = 10.0  # kg
GEAR_DEFAULT = 5.0  # 5:1 reduction

fig = plt.figure(figsize=(15, 9.0))
plt.subplots_adjust(bottom=0.25, top=0.92, left=0.07, right=0.96, hspace=0.42, wspace=0.25)

# Left Column: Kinematics (p, v, a, j)
ax_p = plt.subplot(4, 2, 1)
ax_v = plt.subplot(4, 2, 3, sharex=ax_p)
ax_a = plt.subplot(4, 2, 5, sharex=ax_p)
ax_j = plt.subplot(4, 2, 7, sharex=ax_p)

# Right Column: Electrical & Regen Power (Eb/I, Duty Cycle, Power, Energy)
ax_elec = plt.subplot(4, 2, 2, sharex=ax_p)
ax_duty = plt.subplot(4, 2, 4, sharex=ax_p)
ax_pow  = plt.subplot(4, 2, 6, sharex=ax_p)
ax_en   = plt.subplot(4, 2, 8, sharex=ax_p)

# Initial calculations
t, j, a, v, p, meta = generate_profile(beta=0.2, gamma=2.0, distance=TARGET_DIST, v_max=V_MAX, a_max=A_MAX)
curr_params = MOTOR_PARAMS.copy()
curr_params['mass'] = MASS_DEFAULT
curr_params['gear_ratio'] = GEAR_DEFAULT
reg = calculate_regen_energy(t, v, a, curr_params)

# Kinematic Curves
line_p, = ax_p.plot(t, p, color='#6b21a8', lw=2)
line_v, = ax_v.plot(t, v, color='#1d4ed8', lw=2)
line_a, = ax_a.plot(t, a, color='#b91c1c', lw=2)
line_j, = ax_j.plot(t, j, color='#047857', lw=2)

ax_p.set_ylabel('Position (m)')
ax_v.set_ylabel('Velocity (m/s)')
ax_a.set_ylabel('Accel (m/s²)')
ax_j.set_ylabel('Jerk (m/s³)')
ax_j.set_xlabel('Time (s)')

# Electrical / Regen Curves
line_Eb, = ax_elec.plot(t, reg['Eb'], color='#0284c7', lw=2, label='Back-EMF $E_b$ (V)')
ax_elec_twin = ax_elec.twinx()
line_I, = ax_elec_twin.plot(t, reg['I_target'], color='#ea580c', lw=2, linestyle='--', label='Current $I_{tgt}$ (A)')
ax_elec.set_ylabel('$E_b$ (V)', color='#0284c7')
ax_elec_twin.set_ylabel('$I_{tgt}$ (A)', color='#ea580c')

line_duty, = ax_duty.plot(t, reg['duty_cycle'], color='#0891b2', lw=2)
ax_duty.set_ylabel('Duty Cycle $D$')

line_pideal, = ax_pow.plot(t, reg['P_ideal'], color='#94a3b8', lw=1.5, linestyle=':', label='Ideal Power ($E_b \cdot I$)')
line_ploss, = ax_pow.plot(t, reg['P_loss'], color='#ef4444', lw=1.5, linestyle='--', label='Losses ($I^2R + V_d I$)')
line_pnet, = ax_pow.plot(t, reg['P_net'], color='#16a34a', lw=2, label='Net Battery Power')
ax_pow.set_ylabel('Power (W)')
ax_pow.legend(loc='upper right', fontsize=8)

line_en, = ax_en.plot(t, reg['E_cum'], color='#15803d', lw=2.5)
ax_en.set_ylabel('Regen Energy (J)')
ax_en.set_xlabel('Time (s)')

for ax in [ax_p, ax_v, ax_a, ax_j, ax_elec, ax_duty, ax_pow, ax_en]:
    ax.grid(True, linestyle='--', alpha=0.5)

# Theoretical Potential Energy: E_pot = m * g * distance
E_pot_init = MASS_DEFAULT * 9.81 * TARGET_DIST
eff_init = (reg['E_total'] / E_pot_init * 100.0) if E_pot_init > 0 else 0.0

# Header Title
title_text = fig.suptitle(
    f"Descent Regen | {meta['mode']} | E_regen: {reg['E_total']:.2f} J ({reg['E_total']/3.6:.1f} mWh) | E_pot: {E_pot_init:.1f} J | Eff: {eff_init:.1f}%",
    fontsize=12, fontweight='bold'
)

# Sliders & Buttons
ax_beta  = plt.axes([0.08, 0.17, 0.52, 0.02])
ax_gamma = plt.axes([0.08, 0.13, 0.52, 0.02])
ax_dist  = plt.axes([0.08, 0.09, 0.52, 0.02])
ax_mass  = plt.axes([0.08, 0.05, 0.52, 0.02])
ax_gear  = plt.axes([0.08, 0.01, 0.52, 0.02])
ax_opt_btn    = plt.axes([0.66, 0.11, 0.28, 0.065])
ax_pareto_btn = plt.axes([0.66, 0.025, 0.28, 0.065])

s_beta  = Slider(ax_beta, r'Jerk $\beta$', 0.01, 0.50, valinit=0.2, valstep=0.01)
s_gamma = Slider(ax_gamma, r'Decel $\gamma$', 1.0, 5.0, valinit=2.0, valstep=0.1)
s_dist  = Slider(ax_dist, r'Dist $\delta$ (m)', 0.01, 10.0, valinit=TARGET_DIST, valstep=0.05)
s_mass  = Slider(ax_mass, r'Mass $m$ (kg)', 0.5, 20.0, valinit=MASS_DEFAULT, valstep=0.5)
s_gear  = Slider(ax_gear, r'Gearbox $N$', 1.0, 10.0, valinit=GEAR_DEFAULT, valstep=1.0)

btn_opt = Button(ax_opt_btn, '⚡ Auto-Optimize (β*, γ*)', color='#dcfce7', hovercolor='#86efac')
btn_pareto = Button(ax_pareto_btn, '📊 Plot Pareto Curve (β, γ, N)', color='#e0e7ff', hovercolor='#c7d2fe')

def update_plots(beta_val, gamma_val, dist_val, mass_val, gear_val):
    t_new, j_new, a_new, v_new, p_new, meta_new = generate_profile(
        beta=beta_val, gamma=gamma_val, distance=dist_val, v_max=V_MAX, a_max=A_MAX
    )
    p_eval = MOTOR_PARAMS.copy()
    p_eval['mass'] = mass_val
    p_eval['gear_ratio'] = gear_val
    reg_new = calculate_regen_energy(t_new, v_new, a_new, p_eval)

    line_p.set_data(t_new, p_new)
    line_v.set_data(t_new, v_new)
    line_a.set_data(t_new, a_new)
    line_j.set_data(t_new, j_new)

    line_Eb.set_data(t_new, reg_new['Eb'])
    line_I.set_data(t_new, reg_new['I_target'])
    line_duty.set_data(t_new, reg_new['duty_cycle'])
    
    line_pideal.set_data(t_new, reg_new['P_ideal'])
    line_ploss.set_data(t_new, reg_new['P_loss'])
    line_pnet.set_data(t_new, reg_new['P_net'])
    line_en.set_data(t_new, reg_new['E_cum'])

    for ax in [ax_p, ax_v, ax_a, ax_j, ax_elec, ax_duty, ax_pow, ax_en]:
        ax.relim()
        ax.autoscale_view()
        ax.set_xlim(0, t_new[-1] * 1.05)
        
    ax_elec_twin.relim()
    ax_elec_twin.autoscale_view()

    E_pot = mass_val * 9.81 * dist_val
    eff = (reg_new['E_total'] / E_pot * 100.0) if E_pot > 0 else 0.0

    title_text.set_text(
        f"Descent Regen | {meta_new['mode']} | E_regen: {reg_new['E_total']:.2f} J ({reg_new['E_total']/3.6:.1f} mWh) | E_pot: {E_pot:.1f} J | Eff: {eff:.1f}%"
    )
    fig.canvas.draw_idle()

def on_slider_change(val):
    update_plots(s_beta.val, s_gamma.val, s_dist.val, s_mass.val, s_gear.val)

s_beta.on_changed(on_slider_change)
s_gamma.on_changed(on_slider_change)
s_dist.on_changed(on_slider_change)
s_mass.on_changed(on_slider_change)
s_gear.on_changed(on_slider_change)

def on_opt_click(event):
    opt_beta, opt_gamma, opt_E = optimize_parameters(
        s_dist.val, V_MAX, A_MAX, mass=s_mass.val, gear_ratio=s_gear.val
    )
    s_beta.set_val(opt_beta)
    s_gamma.set_val(opt_gamma)

btn_opt.on_clicked(on_opt_click)

def on_pareto_click(event):
    from pareto_analysis import run_parameter_sweep, plot_pareto_analysis
    print(f"Calculating Pareto frontier for Distance={s_dist.val:.2f}m, Mass={s_mass.val:.2f}kg...")
    t_vals, e_vals, p_list = run_parameter_sweep(
        distance=s_dist.val,
        v_max=V_MAX,
        a_max=A_MAX,
        mass=s_mass.val,
        beta_range=(0.02, 0.50, 15),
        gamma_range=(1.0, 5.0, 20),
        gear_range=(1.0, 10.0, 10),
        dt=0.002
    )
    plot_pareto_analysis(t_vals, e_vals, p_list, distance=s_dist.val, mass=s_mass.val, show_plot=False)
    plt.show()

btn_pareto.on_clicked(on_pareto_click)

if __name__ == "__main__":
    plt.show()