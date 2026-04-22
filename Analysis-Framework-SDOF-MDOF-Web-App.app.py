import math
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# ---------------------------
# Page config + CSS
# ---------------------------
st.set_page_config(page_title="Preliminary Tall Building Designer", layout="wide")

st.markdown(
    """
    <style>
    .main .block-container {
        padding-top: 0.8rem;
        padding-bottom: 1.0rem;
        max-width: 100%;
    }
    h1, h2, h3 { letter-spacing: 0.2px; }
    div[data-testid="stExpander"] {
        border: 1px solid #e6e9ef;
        border-radius: 14px;
        margin-bottom: 0.7rem;
        background-color: #fafbfc;
    }
    div[data-testid="stExpander"] details summary {
        font-weight: 700; font-size: 1rem;
    }
    div[data-testid="stNumberInput"] > div,
    div[data-testid="stSelectbox"] > div,
    div[data-testid="stTextInput"] > div {
        border-radius: 12px;
    }
    .stButton button {
        width: 100%; border-radius: 12px; font-weight: 700; height: 3rem;
    }
    div[data-testid="metric-container"] {
        background-color: #f8f9fb;
        border: 1px solid #e8ecf2;
        padding: 12px 16px;
        border-radius: 14px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04);
    }
    textarea { border-radius: 12px !important; }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Preliminary Tall Building Designer (Iterative Target)")

# ---------------------------
# Data classes
# ---------------------------
@dataclass
class BuildingInput:
    # Geometry
    plan_shape: str
    n_story: int
    n_basement: int
    story_height: float
    basement_height: float
    plan_x: float
    plan_y: float
    n_bays_x: int
    n_bays_y: int
    bay_x: float
    bay_y: float

    # Core / transport
    stair_count: int
    elevator_count: int
    elevator_area_each: float
    stair_area_each: float
    service_area: float

    # Loads / materials
    fck: float
    Ec: float
    fy: float
    DL: float
    LL: float
    slab_finish_allowance: float
    facade_line_load: float

    # Controls
    Ct: float
    x_period: float
    upper_period_factor: float
    target_position_factor: float
    drift_limit_ratio: float
    effective_modal_mass_ratio: float

    # Dimension limits
    min_wall_thickness: float
    max_wall_thickness: float
    min_column_dim: float
    max_column_dim: float
    min_beam_width: float
    min_beam_depth: float
    min_slab_thickness: float
    max_slab_thickness: float

    # Advanced (kept)
    wall_cracked_factor: float
    column_cracked_factor: float
    wall_rebar_ratio: float
    column_rebar_ratio: float
    beam_rebar_ratio: float
    slab_rebar_ratio: float
    seismic_mass_factor: float
    max_story_wall_slenderness: float
    perimeter_column_factor: float
    corner_column_factor: float
    lower_zone_wall_count: int
    middle_zone_wall_count: int
    upper_zone_wall_count: int
    basement_retaining_wall_thickness: float
    perimeter_shear_wall_ratio: float


@dataclass
class ModalResult:
    periods_s: List[float]
    frequencies_hz: List[float]
    effective_mass_ratios: List[float]
    cumulative_effective_mass_ratios: List[float]


@dataclass
class Quantities:
    wall_conc_m3: float
    col_conc_m3: float
    beam_conc_m3: float
    slab_conc_m3: float
    wall_steel_kg: float
    col_steel_kg: float
    beam_steel_kg: float
    slab_steel_kg: float
    total_steel_kg: float


@dataclass
class DesignResult:
    T_ref: float
    T_target: float
    T_upper: float
    T_est: float
    period_error: float
    K_est: float
    drift: float
    drift_ratio: float
    period_ok: bool
    drift_ok: bool
    core_scale: float
    col_scale: float
    beam_b: float
    beam_h: float
    slab_t: float
    core_outer: Tuple[float, float]
    core_opening: Tuple[float, float]
    wall_t: float
    col_dims: Tuple[float, float, float]  # (corner, perim, interior)
    modal: ModalResult
    qty: Quantities
    beta: float


# ---------------------------
# Input panel (organized)
# ---------------------------
def input_panel() -> BuildingInput:
    with st.expander("1. Plan Shape and Geometry", expanded=True):
        plan_shape = st.radio("Plan shape", ["square", "triangle"], horizontal=True)
        c1, c2 = st.columns(2)
        with c1:
            n_story = st.number_input("Above-grade stories", 1, 150, 50)
            basement_height = st.number_input("Basement height (m)", 2.5, 6.0, 3.0)
            plan_x = st.number_input("Plan X (m)", 10.0, 300.0, 80.0)
            n_bays_x = st.number_input("Bays in X", 1, 30, 8)
            bay_x = st.number_input("Bay X (m)", 2.0, 20.0, 10.0)
        with c2:
            n_basement = st.number_input("Basement stories", 0, 20, 10)
            story_height = st.number_input("Story height (m)", 2.5, 6.0, 3.2)
            plan_y = st.number_input("Plan Y (m)", 10.0, 300.0, 80.0)
            n_bays_y = st.number_input("Bays in Y", 1, 30, 8)
            bay_y = st.number_input("Bay Y (m)", 2.0, 20.0, 10.0)

    with st.expander("2. Core and Vertical Transportation"):
        c3, c4 = st.columns(2)
        with c3:
            stair_count = st.number_input("Stairs", 0, 20, 2)
            elevator_area_each = st.number_input("Elevator area each (m²)", 0.0, 20.0, 3.5)
            service_area = st.number_input("Service area (m²)", 0.0, 200.0, 35.0)
        with c4:
            elevator_count = st.number_input("Elevators", 0, 30, 4)
            stair_area_each = st.number_input("Stair area each (m²)", 0.0, 50.0, 20.0)

    with st.expander("3. Loads and Materials"):
        c5, c6 = st.columns(2)
        with c5:
            fck = st.number_input("fck (MPa)", 20.0, 120.0, 70.0)
            fy = st.number_input("fy (MPa)", 200.0, 700.0, 420.0)
            DL = st.number_input("DL (kN/m²)", 0.0, 20.0, 3.0)
            slab_finish_allowance = st.number_input("Slab / fit-out allowance (kN/m²)", 0.0, 10.0, 1.5)
        with c6:
            Ec = st.number_input("Ec (MPa)", 20000.0, 60000.0, 36000.0)
            LL = st.number_input("LL (kN/m²)", 0.0, 20.0, 2.5)
            facade_line_load = st.number_input("Facade line load (kN/m)", 0.0, 50.0, 1.0)

    with st.expander("4. Design Controls"):
        c7, c8 = st.columns(2)
        with c7:
            Ct = st.number_input("Ct", 0.001, 0.200, 0.0488, format="%.4f")
            upper_period_factor = st.number_input("Upper period factor", 1.0, 3.0, 2.0, step=0.05)
            target_position_factor = st.number_input("Target position factor (β)", 0.10, 0.95, 0.85, step=0.05)
        with c8:
            x_period = st.number_input("x exponent", 0.10, 1.50, 0.75, step=0.01)
            drift_denominator = st.number_input("Drift denominator", 100.0, 2000.0, 500.0)
            effective_modal_mass_ratio = st.number_input("Effective modal mass ratio", 0.10, 1.0, 0.80)

    with st.expander("5. Dimension Limits"):
        c9, c10 = st.columns(2)
        with c9:
            min_wall_thickness = st.number_input("Min wall thickness (m)", 0.10, 2.0, 0.30)
            min_column_dim = st.number_input("Min column dimension (m)", 0.10, 3.0, 0.70)
            min_beam_width = st.number_input("Min beam width (m)", 0.10, 3.0, 0.40)
            min_beam_depth = st.number_input("Min beam depth (m)", 0.10, 3.0, 0.75)
            min_slab_thickness = st.number_input("Min slab thickness (m)", 0.05, 1.0, 0.22)
        with c10:
            max_wall_thickness = st.number_input("Max wall thickness (m)", 0.10, 3.0, 1.20)
            max_column_dim = st.number_input("Max column dimension (m)", 0.10, 5.0, 1.80)
            max_slab_thickness = st.number_input("Max slab thickness (m)", 0.05, 1.0, 0.40)

    with st.expander("6. Advanced Settings"):
        st.caption("Affects cracked stiffness, reinforcement quantities, seismic mass, and zone stiffness.")
        c11, c12 = st.columns(2)
        with c11:
            wall_cracked_factor = st.number_input("Wall cracked factor", 0.10, 1.0, 0.70)
            column_cracked_factor = st.number_input("Column cracked factor", 0.10, 1.0, 0.70)
            wall_rebar_ratio = st.number_input("Wall rebar ratio", 0.0, 0.10, 0.0030, format="%.4f")
            column_rebar_ratio = st.number_input("Column rebar ratio", 0.0, 0.10, 0.0100, format="%.4f")
            beam_rebar_ratio = st.number_input("Beam rebar ratio", 0.0, 0.10, 0.0150, format="%.4f")
            slab_rebar_ratio = st.number_input("Slab rebar ratio", 0.0, 0.10, 0.0035, format="%.4f")
        with c12:
            seismic_mass_factor = st.number_input("Seismic mass factor", 0.10, 2.0, 1.00)
            max_story_wall_slenderness = st.number_input("Max wall slenderness", 1.0, 50.0, 12.0)
            perimeter_column_factor = st.number_input("Perimeter column factor", 1.0, 3.0, 1.10)
            corner_column_factor = st.number_input("Corner column factor", 1.0, 3.0, 1.30)
            lower_zone_wall_count = st.number_input("Lower zone wall count", 4, 12, 8)
            middle_zone_wall_count = st.number_input("Middle zone wall count", 4, 12, 6)
            upper_zone_wall_count = st.number_input("Upper zone wall count", 4, 12, 4)
            basement_retaining_wall_thickness = st.number_input("Basement retaining wall thickness (m)", 0.10, 2.0, 0.50)
            perimeter_shear_wall_ratio = st.number_input("Perimeter shear wall ratio", 0.0, 1.0, 0.20, format="%.3f")

    return BuildingInput(
        plan_shape=plan_shape,
        n_story=int(n_story),
        n_basement=int(n_basement),
        story_height=float(story_height),
        basement_height=float(basement_height),
        plan_x=float(plan_x),
        plan_y=float(plan_y),
        n_bays_x=int(n_bays_x),
        n_bays_y=int(n_bays_y),
        bay_x=float(bay_x),
        bay_y=float(bay_y),
        stair_count=int(stair_count),
        elevator_count=int(elevator_count),
        elevator_area_each=float(elevator_area_each),
        stair_area_each=float(stair_area_each),
        service_area=float(service_area),
        fck=float(fck),
        Ec=float(Ec),
        fy=float(fy),
        DL=float(DL),
        LL=float(LL),
        slab_finish_allowance=float(slab_finish_allowance),
        facade_line_load=float(facade_line_load),
        Ct=float(Ct),
        x_period=float(x_period),
        upper_period_factor=float(upper_period_factor),
        target_position_factor=float(target_position_factor),
        drift_limit_ratio=1.0 / float(drift_denominator),
        effective_modal_mass_ratio=float(effective_modal_mass_ratio),
        min_wall_thickness=float(min_wall_thickness),
        max_wall_thickness=float(max_wall_thickness),
        min_column_dim=float(min_column_dim),
        max_column_dim=float(max_column_dim),
        min_beam_width=float(min_beam_width),
        min_beam_depth=float(min_beam_depth),
        min_slab_thickness=float(min_slab_thickness),
        max_slab_thickness=float(max_slab_thickness),
        wall_cracked_factor=float(wall_cracked_factor),
        column_cracked_factor=float(column_cracked_factor),
        wall_rebar_ratio=float(wall_rebar_ratio),
        column_rebar_ratio=float(column_rebar_ratio),
        beam_rebar_ratio=float(beam_rebar_ratio),
        slab_rebar_ratio=float(slab_rebar_ratio),
        seismic_mass_factor=float(seismic_mass_factor),
        max_story_wall_slenderness=float(max_story_wall_slenderness),
        perimeter_column_factor=float(perimeter_column_factor),
        corner_column_factor=float(corner_column_factor),
        lower_zone_wall_count=int(lower_zone_wall_count),
        middle_zone_wall_count=int(middle_zone_wall_count),
        upper_zone_wall_count=int(upper_zone_wall_count),
        basement_retaining_wall_thickness=float(basement_retaining_wall_thickness),
        perimeter_shear_wall_ratio=float(perimeter_shear_wall_ratio),
    )


# ---------------------------
# Core calculations
# ---------------------------
def code_period(H: float, Ct: float, x: float) -> float:
    return Ct * (H ** x)


def initial_sections(inp: BuildingInput):
    # start from mid-range
    wall_t = 0.5 * (inp.min_wall_thickness + inp.max_wall_thickness)
    col = 0.5 * (inp.min_column_dim + inp.max_column_dim)
    beam_b = max(inp.min_beam_width, 0.45)
    beam_h = max(inp.min_beam_depth, 0.8)
    slab_t = 0.5 * (inp.min_slab_thickness + inp.max_slab_thickness)
    return wall_t, col, beam_b, beam_h, slab_t


def estimate_core_geometry(inp: BuildingInput):
    # simple rectangular core around shafts
    area_core = (
        inp.elevator_count * inp.elevator_area_each
        + inp.stair_count * inp.stair_area_each
        + inp.service_area
    )
    side = max(8.0, math.sqrt(area_core) * 2.0)
    opening = side * 0.6
    return (side, side), (opening, opening)


def estimate_stiffness(inp: BuildingInput, wall_t: float, col_dim: float):
    H = inp.n_story * inp.story_height
    # crude lateral stiffness proxy: walls + columns
    E = inp.Ec * 1e6  # MPa -> N/m^2
    I_wall = (wall_t * (H ** 3)) / 12.0
    I_col = (col_dim ** 4) / 12.0

    k_walls = E * I_wall / (H ** 3 + 1e-9)
    k_cols = E * I_col * (inp.n_bays_x * inp.n_bays_y) / (H ** 3 + 1e-9)
    return k_walls * inp.wall_cracked_factor + k_cols * inp.column_cracked_factor


def estimate_mass(inp: BuildingInput, slab_t: float):
    area = inp.plan_x * inp.plan_y
    DL = inp.DL + inp.slab_finish_allowance
    LL = inp.LL
    w = (DL + 0.25 * LL) * area * inp.n_story  # kN
    m = (w * 1e3) / 9.81 * inp.seismic_mass_factor
    return m, w


def compute_period(mass: float, stiffness: float):
    if stiffness <= 0:
        return 1e9
    return 2.0 * math.pi * math.sqrt(mass / stiffness)


def drift_estimate(inp: BuildingInput, K: float, weight_kN: float):
    H = inp.n_story * inp.story_height
    # crude: delta ~ F/K ; F ~ 0.1 W
    F = 0.1 * weight_kN * 1e3
    delta = F / (K + 1e-9)
    return delta, delta / (H + 1e-9)


def modal_stub(T1: float) -> ModalResult:
    # simple 5-mode stub consistent with T1
    periods = [T1, T1 / 3.0, T1 / 5.0, T1 / 7.0, T1 / 9.0]
    freqs = [1.0 / (p + 1e-9) for p in periods]
    ratios = [0.75, 0.15, 0.05, 0.03, 0.02]
    cum = np.cumsum(ratios).tolist()
    return ModalResult(periods, freqs, ratios, cum)


def quantities(inp: BuildingInput, wall_t, col_dim, beam_b, beam_h, slab_t) -> Quantities:
    area = inp.plan_x * inp.plan_y
    H = inp.n_story * inp.story_height

    wall_conc = 4 * inp.plan_x * wall_t * H * 0.25  # crude
    col_conc = (col_dim ** 2) * inp.n_bays_x * inp.n_bays_y * H
    beam_len = inp.n_bays_x * inp.plan_y + inp.n_bays_y * inp.plan_x
    beam_conc = beam_len * beam_b * beam_h * inp.n_story
    slab_conc = area * slab_t * inp.n_story

    # steel by ratios (kg ~ ratio * conc * density factor)
    def steel(vol, ratio):
        return vol * ratio * 7850.0

    w_s = steel(wall_conc, inp.wall_rebar_ratio)
    c_s = steel(col_conc, inp.column_rebar_ratio)
    b_s = steel(beam_conc, inp.beam_rebar_ratio)
    s_s = steel(slab_conc, inp.slab_rebar_ratio)

    return Quantities(
        wall_conc, col_conc, beam_conc, slab_conc,
        w_s, c_s, b_s, s_s, w_s + c_s + b_s + s_s
    )


# ---------------------------
# Iterative design
# ---------------------------
def run_design(inp: BuildingInput) -> DesignResult:
    H = inp.n_story * inp.story_height

    T_ref = code_period(H, inp.Ct, inp.x_period)
    T_upper = inp.upper_period_factor * T_ref
    beta = inp.target_position_factor
    T_target = T_ref + beta * (T_upper - T_ref)

    wall_t, col_dim, beam_b, beam_h, slab_t = initial_sections(inp)
    core_scale = 1.0
    col_scale = 1.0

    for _ in range(15):
        (core_outer, core_open) = estimate_core_geometry(inp)
        K = estimate_stiffness(inp, wall_t * core_scale, col_dim * col_scale)
        m, w = estimate_mass(inp, slab_t)
        T_est = compute_period(m, K)

        err = abs(T_est - T_target) / max(T_target, 1e-9)
        if err < 0.05:
            break

        # update scales (physics-consistent)
        scale = (T_est / max(T_target, 1e-9)) ** 2
        core_scale *= scale
        col_scale *= scale

        # clamp to limits
        wall_t = np.clip(wall_t * core_scale, inp.min_wall_thickness, inp.max_wall_thickness)
        col_dim = np.clip(col_dim * col_scale, inp.min_column_dim, inp.max_column_dim)

    # final evaluation
    (core_outer, core_open) = estimate_core_geometry(inp)
    K = estimate_stiffness(inp, wall_t, col_dim)
    m, w = estimate_mass(inp, slab_t)
    T_est = compute_period(m, K)
    delta, drift_ratio = drift_estimate(inp, K, w)

    period_error = abs(T_est - T_target) / max(T_target, 1e-9)
    period_ok = T_est <= T_upper
    drift_ok = drift_ratio <= inp.drift_limit_ratio

    modal = modal_stub(T_est)
    qty = quantities(inp, wall_t, col_dim, beam_b, beam_h, slab_t)

    return DesignResult(
        T_ref, T_target, T_upper, T_est, period_error,
        K, delta, drift_ratio, period_ok, drift_ok,
        core_scale, col_scale, beam_b, beam_h, slab_t,
        core_outer, core_open, wall_t,
        (col_dim * inp.corner_column_factor,
         col_dim * inp.perimeter_column_factor,
         col_dim),
        modal, qty, beta
    )


# ---------------------------
# Plot
# ---------------------------
def plot_plan(inp: BuildingInput, res: DesignResult):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot([0, inp.plan_x, inp.plan_x, 0, 0],
            [0, 0, inp.plan_y, inp.plan_y, 0], 'k-')

    # grid
    for i in range(inp.n_bays_x + 1):
        x = i * inp.bay_x
        ax.plot([x, x], [0, inp.plan_y], color="#e5e7eb", lw=0.8)
    for j in range(inp.n_bays_y + 1):
        y = j * inp.bay_y
        ax.plot([0, inp.plan_x], [y, y], color="#e5e7eb", lw=0.8)

    # columns (simple)
    cc, pc, ic = res.col_dims
    for i in range(inp.n_bays_x + 1):
        for j in range(inp.n_bays_y + 1):
            x = i * inp.bay_x
            y = j * inp.bay_y
            if (i in [0, inp.n_bays_x]) and (j in [0, inp.n_bays_y]):
                d = cc; col = "#1f77b4"
            elif (i in [0, inp.n_bays_x]) or (j in [0, inp.n_bays_y]):
                d = pc; col = "#2ca02c"
            else:
                d = ic; col = "#ff7f0e"
            ax.add_patch(plt.Rectangle((x - d/2, y - d/2), d, d, color=col, alpha=0.9))

    # core + opening
    cx0 = (inp.plan_x - res.core_outer[0]) / 2
    cy0 = (inp.plan_y - res.core_outer[1]) / 2
    ax.add_patch(plt.Rectangle((cx0, cy0), res.core_outer[0], res.core_outer[1],
                               fill=False, lw=2.2, color="#111827"))
    ix0 = (inp.plan_x - res.core_opening[0]) / 2
    iy0 = (inp.plan_y - res.core_opening[1]) / 2
    ax.add_patch(plt.Rectangle((ix0, iy0), res.core_opening[0], res.core_opening[1],
                               fill=False, lw=1.2, ls="--", color="#6b7280"))

    # labels
    ax.text(cx0 + res.core_outer[0]/2, cy0 - 1.5, "CORE", ha="center", fontsize=9, fontweight="bold")
    ax.text(ix0 + res.core_opening[0]/2, iy0 + res.core_opening[1]/2,
            "OPENING", ha="center", va="center", fontsize=8)

    # info block
    info = [
        f"Core outer = {res.core_outer[0]:.2f} x {res.core_outer[1]:.2f} m",
        f"Core opening = {res.core_opening[0]:.2f} x {res.core_opening[1]:.2f} m",
        f"Wall t = {res.wall_t:.2f} m",
        f"Corner col = {res.col_dims[0]:.2f} m",
        f"Perim col = {res.col_dims[1]:.2f} m",
        f"Interior col = {res.col_dims[2]:.2f} m",
        f"Beam = {res.beam_b:.2f} x {res.beam_h:.2f} m",
        f"Slab t = {res.slab_t:.2f} m",
    ]
    for k, txt in enumerate(info):
        ax.text(inp.plan_x + 3, inp.plan_y - 2 - 3*k, txt, fontsize=8)

    ax.set_aspect("equal")
    ax.set_xlim(-2, inp.plan_x + 35)
    ax.set_ylim(inp.plan_y + 6, -6)
    ax.axis("off")
    return fig


# ---------------------------
# Run
# ---------------------------
inp = input_panel()

if st.button("Analyze"):
    res = run_design(inp)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Reference period (s)", f"{res.T_ref:.3f}")
    c2.metric("Design target (s)", f"{res.T_target:.3f}")
    c3.metric("Estimated (s)", f"{res.T_est:.3f}")
    c4.metric("Upper limit (s)", f"{res.T_upper:.3f}")

    c5, c6, c7 = st.columns(3)
    c5.metric("Period error (%)", f"{100*res.period_error:.2f}")
    c6.metric("Top drift (m)", f"{res.drift:.3f}")
    c7.metric("Drift ratio", f"{res.drift_ratio:.5f}")

    st.caption(
        f"Target formula: T_target = T_ref + β (T_upper - T_ref), β = {res.beta:.2f}"
    )

    st.pyplot(plot_plan(inp, res))

    st.subheader("Quantities")
    q = res.qty
    st.write({
        "Wall concrete (m3)": round(q.wall_conc_m3, 1),
        "Column concrete (m3)": round(q.col_conc_m3, 1),
        "Beam concrete (m3)": round(q.beam_conc_m3, 1),
        "Slab concrete (m3)": round(q.slab_conc_m3, 1),
        "Total steel (kg)": round(q.total_steel_kg, 0),
    })

    st.subheader("Modal (stub)")
    st.write({
        "Periods (s)": [round(x, 3) for x in res.modal.periods_s],
        "Mass ratios": [round(100*x, 1) for x in res.modal.effective_mass_ratios],
    })
