
import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Preliminary Tall Building Designer", layout="wide")

st.markdown(
    """
    <style>
    .main .block-container {padding-top: 0.8rem; padding-bottom: 1rem; max-width: 100%;}
    div[data-testid="stExpander"] {border: 1px solid #e6e9ef; border-radius: 14px; margin-bottom: 0.7rem; background-color: #fafbfc;}
    div[data-testid="stExpander"] details summary {font-weight: 700; font-size: 1rem;}
    .stButton button {width: 100%; border-radius: 12px; font-weight: 700; height: 3rem;}
    div[data-testid="metric-container"] {background-color: #f8f9fb; border: 1px solid #e8ecf2; padding: 12px 16px; border-radius: 14px; box-shadow: 0 1px 2px rgba(0,0,0,0.04);}
    textarea {border-radius: 12px !important;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Preliminary Tall Building Designer")
st.caption("Iterative target with real MDOF shear-building modal analysis")


@dataclass
class BuildingInput:
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
    stair_count: int
    elevator_count: int
    elevator_area_each: float
    stair_area_each: float
    service_area: float
    fck: float
    Ec: float
    fy: float
    DL: float
    LL: float
    slab_finish_allowance: float
    facade_line_load: float
    Ct: float
    x_period: float
    upper_period_factor: float
    target_position_factor: float
    drift_limit_ratio: float
    effective_modal_mass_ratio: float
    min_wall_thickness: float
    max_wall_thickness: float
    min_column_dim: float
    max_column_dim: float
    min_beam_width: float
    min_beam_depth: float
    min_slab_thickness: float
    max_slab_thickness: float
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
    mode_shapes: List[List[float]]
    story_masses_kg: List[float]
    story_stiffnesses_npm: List[float]


@dataclass
class Quantities:
    wall_conc_m3: float
    column_conc_m3: float
    beam_conc_m3: float
    slab_conc_m3: float
    wall_steel_kg: float
    column_steel_kg: float
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
    drift_m: float
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
    col_dims: Tuple[float, float, float]
    modal: ModalResult
    qty: Quantities
    beta: float
    governing_issue: str
    redesign_suggestions: List[str]
    iteration_log: List[Tuple[int, float, float, float, float, float]]
    active_core_walls: int
    core_wall_segments: List[Tuple[Tuple[float,float], Tuple[float,float]]]


def streamlit_input_panel() -> BuildingInput:
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

    with st.expander("2. Core and Vertical Transportation", expanded=False):
        c3, c4 = st.columns(2)
        with c3:
            stair_count = st.number_input("Stairs", 0, 20, 2)
            elevator_area_each = st.number_input("Elevator area each (m²)", 0.0, 20.0, 3.5)
            service_area = st.number_input("Service area (m²)", 0.0, 200.0, 35.0)
        with c4:
            elevator_count = st.number_input("Elevators", 0, 30, 4)
            stair_area_each = st.number_input("Stair area each (m²)", 0.0, 50.0, 20.0)

    with st.expander("3. Loads and Materials", expanded=False):
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

    with st.expander("4. Design Controls", expanded=False):
        c7, c8 = st.columns(2)
        with c7:
            Ct = st.number_input("Ct", 0.001, 0.200, 0.0488, format="%.4f")
            upper_period_factor = st.number_input("Upper period factor", 1.0, 3.0, 2.0, step=0.05)
            target_position_factor = st.number_input("Target position factor β", 0.10, 0.95, 0.85, step=0.05)
        with c8:
            x_period = st.number_input("x exponent", 0.10, 1.50, 0.75, step=0.01)
            drift_denominator = st.number_input("Drift denominator", 100.0, 2000.0, 500.0)
            effective_modal_mass_ratio = st.number_input("Effective modal mass ratio", 0.10, 1.0, 0.80)

    with st.expander("5. Dimension Limits", expanded=False):
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

    with st.expander("6. Advanced Settings", expanded=False):
        st.caption("These parameters affect cracked stiffness, reinforcement quantities, seismic mass, zone-based stiffness distribution, and core/perimeter behavior.")
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


def code_period(H: float, Ct: float, x: float) -> float:
    return Ct * (H ** x)


def slab_prelim(inp: BuildingInput) -> float:
    return float(np.clip(max(inp.bay_x, inp.bay_y) / 28.0, inp.min_slab_thickness, inp.max_slab_thickness))


def beam_prelim(inp: BuildingInput) -> Tuple[float, float]:
    span = max(inp.bay_x, inp.bay_y)
    h = max(inp.min_beam_depth, span / 12.0)
    b = max(inp.min_beam_width, 0.45 * h)
    return float(b), float(h)


def floor_area(inp: BuildingInput) -> float:
    return 0.5 * inp.plan_x * inp.plan_y if inp.plan_shape == "triangle" else inp.plan_x * inp.plan_y


def total_weight_and_mass(inp: BuildingInput, slab_t: float) -> Tuple[float, float]:
    area = floor_area(inp)
    slab_self = slab_t * 25.0
    q = inp.DL + inp.LL + inp.slab_finish_allowance + slab_self
    W = q * area * inp.n_story
    W += 1.10 * q * area * inp.n_basement
    W += inp.facade_line_load * 2.0 * (inp.plan_x + inp.plan_y) * inp.n_story
    W *= inp.seismic_mass_factor
    M = W * 1000.0 / 9.81
    return W, M


def core_geometry(inp: BuildingInput, wall_t: float):
    req_open = inp.elevator_count * inp.elevator_area_each + inp.stair_count * inp.stair_area_each + inp.service_area
    opening_side = max(6.0, math.sqrt(req_open))
    open_x = 1.25 * opening_side
    open_y = 0.80 * opening_side
    outer_x = min(inp.plan_x * 0.40, open_x + 2.0 * wall_t + 5.0)
    outer_y = min(inp.plan_y * 0.40, open_y + 2.0 * wall_t + 5.0)
    cx0 = (inp.plan_x - outer_x) / 2.0
    cy0 = (inp.plan_y - outer_y) / 2.0
    cx1 = cx0 + outer_x
    cy1 = cy0 + outer_y
    segs = []
    if inp.plan_shape == "square":
        mx = (cx0 + cx1) / 2
        my = (cy0 + cy1) / 2
        segs += [((cx0, cy0), (mx - wall_t, cy0)), ((mx + wall_t, cy0), (cx1, cy0))]
        segs += [((cx0, cy1), (mx - wall_t, cy1)), ((mx + wall_t, cy1), (cx1, cy1))]
        segs += [((cx0, cy0), (cx0, my - wall_t)), ((cx0, my + wall_t), (cx0, cy1))]
        segs += [((cx1, cy0), (cx1, my - wall_t)), ((cx1, my + wall_t), (cx1, cy1))]
    return (outer_x, outer_y), (open_x, open_y), segs


def stiffness_total(inp: BuildingInput, wall_t: float, col_dim: float, core_scale: float, col_scale: float) -> float:
    H = inp.n_story * inp.story_height
    E = inp.Ec * 1e6
    core_outer, _, _ = core_geometry(inp, wall_t * core_scale)
    outer_x, outer_y = core_outer
    t = wall_t * core_scale
    I_core_x = 2.0 * (outer_x * t**3 / 12.0) + 2.0 * (t * outer_y**3 / 12.0)
    I_core_y = 2.0 * (outer_y * t**3 / 12.0) + 2.0 * (t * outer_x**3 / 12.0)
    I_core = min(I_core_x, I_core_y) * inp.wall_cracked_factor
    d = col_dim * col_scale
    I_col = d**4 / 12.0
    ncol = (inp.n_bays_x + 1) * (inp.n_bays_y + 1)
    I_cols = I_col * ncol * inp.column_cracked_factor
    K_core = 3.0 * E * I_core / (H**3 + 1e-9)
    K_cols = 3.0 * E * I_cols / (H**3 + 1e-9)
    return max(1e5, K_core + K_cols)


def build_story_masses(inp: BuildingInput, total_mass: float) -> List[float]:
    return [total_mass / inp.n_story for _ in range(inp.n_story)]


def build_story_stiffnesses(inp: BuildingInput, K_total: float) -> List[float]:
    n = inp.n_story
    raw = [1.35 - 0.55 * (i / max(n - 1, 1)) for i in range(n)]
    inv_sum = sum(1.0 / a for a in raw)
    c = K_total * inv_sum
    return [c * a for a in raw]


def assemble_m_k(masses: List[float], ks: List[float]):
    n = len(masses)
    M = np.diag(masses)
    K = np.zeros((n, n), dtype=float)
    for i in range(n):
        ki = ks[i]
        if i == 0:
            K[i, i] += ki
        else:
            K[i, i] += ki
            K[i, i - 1] -= ki
            K[i - 1, i] -= ki
            K[i - 1, i - 1] += ki
    return M, K


def solve_mdof(inp: BuildingInput, total_mass: float, K_total: float, n_modes: int = 5) -> ModalResult:
    masses = build_story_masses(inp, total_mass)
    ks = build_story_stiffnesses(inp, K_total)
    M, K = assemble_m_k(masses, ks)
    A = np.linalg.inv(M) @ K
    eigvals, eigvecs = np.linalg.eig(A)
    eigvals = np.real(eigvals)
    eigvecs = np.real(eigvecs)
    mask = eigvals > 1e-12
    eigvals = eigvals[mask]
    eigvecs = eigvecs[:, mask]
    order = np.argsort(eigvals)
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    omegas = np.sqrt(eigvals)
    periods = [2.0 * math.pi / w for w in omegas[:n_modes]]
    freqs = [w / (2.0 * math.pi) for w in omegas[:n_modes]]

    ones = np.ones((len(masses), 1))
    total_mass_scalar = np.sum(np.diag(M)).item()
    ratios, cumulative, mode_shapes = [], [], []
    cum = 0.0
    for i in range(min(n_modes, eigvecs.shape[1])):
        phi = eigvecs[:, i].reshape(-1, 1)
        denom = (phi.T @ M @ phi).item()
        gamma = ((phi.T @ M @ ones) / denom).item()
        meff = gamma**2 * denom
        ratio = meff / total_mass_scalar
        cum += ratio
        ph = phi.flatten().copy()
        if abs(ph[-1]) > 1e-12:
            ph = ph / ph[-1]
        if ph[-1] < 0:
            ph = -ph
        ratios.append(ratio)
        cumulative.append(cum)
        mode_shapes.append(ph.tolist())

    return ModalResult(periods, freqs, ratios, cumulative, mode_shapes, masses, ks)


def drift_estimate(inp: BuildingInput, weight_kN: float, K_est: float):
    H = inp.n_story * inp.story_height
    F = 0.10 * weight_kN * 1000.0
    d = F / (K_est + 1e-9)
    return d, d / (H + 1e-9)


def quantities(inp: BuildingInput, wall_t: float, col_dim: float, beam_b: float, beam_h: float, slab_t: float,
               core_scale: float, col_scale: float) -> Quantities:
    area = floor_area(inp)
    H = inp.n_story * inp.story_height
    outer, opening, _ = core_geometry(inp, wall_t * core_scale)
    ring_area = max(0.0, outer[0] * outer[1] - opening[0] * opening[1])
    wall_conc = ring_area * H
    ncol = (inp.n_bays_x + 1) * (inp.n_bays_y + 1)
    d = col_dim * col_scale
    col_conc = (d * d) * H * ncol
    total_beam_len = ((inp.n_bays_x * inp.plan_y) + (inp.n_bays_y * inp.plan_x)) * inp.n_story
    beam_conc = total_beam_len * beam_b * beam_h
    slab_conc = area * slab_t * inp.n_story

    def steel(vol, ratio):
        return vol * ratio * 7850.0

    ws = steel(wall_conc, inp.wall_rebar_ratio)
    cs = steel(col_conc, inp.column_rebar_ratio)
    bs = steel(beam_conc, inp.beam_rebar_ratio)
    ss = steel(slab_conc, inp.slab_rebar_ratio)
    return Quantities(wall_conc, col_conc, beam_conc, slab_conc, ws, cs, bs, ss, ws + cs + bs + ss)


def redesign_suggestions(period_ok: bool, drift_ok: bool, T_est: float, T_target: float, T_upper: float, drift_ratio: float, drift_limit: float):
    out = []
    issue = "OK"
    if not period_ok and T_est > T_upper:
        issue = "Estimated dynamic period exceeds upper limit"
        out.append("Increase lateral stiffness: enlarge core wall thickness and/or perimeter columns.")
        out.append("Increase active wall count in middle and upper zones.")
        out.append("Consider a larger core footprint or additional outrigger/belt system.")
    elif T_est > T_target * 1.05:
        issue = "Estimated period is larger than design target"
        out.append("Increase core and column stiffness moderately.")
    elif T_est < T_target * 0.95:
        issue = "System is stiffer than needed"
        out.append("Sections can be reduced for a more economical design.")
        out.append("Reduce core wall thickness and/or column dimensions while rechecking drift.")
    if not drift_ok:
        issue = "Drift exceeds allowable limit"
        out.append("Increase global stiffness; lower-story wall/column strengthening is recommended.")
        out.append("Check story stiffness distribution and consider additional perimeter walls.")
    if not out:
        out.append("System appears preliminarily adequate.")
    return issue, out


def run_design(inp: BuildingInput) -> DesignResult:
    H = inp.n_story * inp.story_height
    T_ref = code_period(H, inp.Ct, inp.x_period)
    T_upper = inp.upper_period_factor * T_ref
    beta = inp.target_position_factor
    T_target = T_ref + beta * (T_upper - T_ref)

    slab_t = slab_prelim(inp)
    beam_b, beam_h = beam_prelim(inp)
    wall_t = 0.5 * (inp.min_wall_thickness + inp.max_wall_thickness)
    col_dim = 0.5 * (inp.min_column_dim + inp.max_column_dim)
    core_scale = 1.0
    col_scale = 1.0

    log = []
    for i in range(1, 21):
        W, M_total = total_weight_and_mass(inp, slab_t)
        K_est = stiffness_total(inp, wall_t, col_dim, core_scale, col_scale)
        modal = solve_mdof(inp, M_total * inp.effective_modal_mass_ratio, K_est, 5)
        T_est = modal.periods_s[0]
        drift_m, drift_ratio = drift_estimate(inp, W, K_est)
        err = abs(T_est - T_target) / max(T_target, 1e-9)
        log.append((i, core_scale, col_scale, T_est, T_target, drift_ratio))
        if err < 0.05 and drift_ratio <= inp.drift_limit_ratio and T_est <= T_upper:
            break
        ratio = float(np.clip((T_est / max(T_target, 1e-9)) ** 2, 0.75, 1.25))
        core_scale *= ratio
        col_scale *= ratio
        if drift_ratio > inp.drift_limit_ratio:
            core_scale *= 1.05
            col_scale *= 1.03
        core_scale = float(np.clip(core_scale, inp.min_wall_thickness / max(wall_t, 1e-9), inp.max_wall_thickness / max(wall_t, 1e-9)))
        col_scale = float(np.clip(col_scale, inp.min_column_dim / max(col_dim, 1e-9), inp.max_column_dim / max(col_dim, 1e-9)))

    W, M_total = total_weight_and_mass(inp, slab_t)
    K_est = stiffness_total(inp, wall_t, col_dim, core_scale, col_scale)
    modal = solve_mdof(inp, M_total * inp.effective_modal_mass_ratio, K_est, 5)
    T_est = modal.periods_s[0]
    drift_m, drift_ratio = drift_estimate(inp, W, K_est)
    period_error = abs(T_est - T_target) / max(T_target, 1e-9)
    period_ok = T_est <= T_upper
    drift_ok = drift_ratio <= inp.drift_limit_ratio
    core_outer, core_opening, segs = core_geometry(inp, wall_t * core_scale)
    qty = quantities(inp, wall_t, col_dim, beam_b, beam_h, slab_t, core_scale, col_scale)
    cc = col_dim * col_scale * inp.corner_column_factor
    pc = col_dim * col_scale * inp.perimeter_column_factor
    ic = col_dim * col_scale
    issue, sugg = redesign_suggestions(period_ok, drift_ok, T_est, T_target, T_upper, drift_ratio, inp.drift_limit_ratio)

    return DesignResult(T_ref, T_target, T_upper, T_est, period_error, K_est, drift_m, drift_ratio, period_ok, drift_ok,
                        core_scale, col_scale, beam_b, beam_h, slab_t, core_outer, core_opening, wall_t * core_scale,
                        (cc, pc, ic), modal, qty, beta, issue, sugg, log, inp.lower_zone_wall_count, segs)


def plot_modes(res: DesignResult, n_story: int, H: float):
    fig, axes = plt.subplots(1, min(5, len(res.modal.mode_shapes)), figsize=(18, 5))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    y = np.linspace(0, H, n_story)
    for i, ax in enumerate(axes):
        phi = np.array(res.modal.mode_shapes[i])
        ax.axvline(0.0, color="#bbbbbb", ls="--", lw=1.0)
        for yy in y:
            ax.plot([-1.05, 1.05], [yy, yy], color="#f0f0f0", lw=0.8)
        ax.plot(phi, y, color="#0b5ed7", lw=2)
        ax.scatter(phi, y, color="#dc3545", s=18)
        ax.set_title(f"Mode {i+1}\nT={res.modal.periods_s[i]:.3f}s", fontsize=10, fontweight="bold")
        ax.set_xlim(-1.1, 1.1)
        if i == 0:
            ax.set_ylabel("Height (m)")
        ax.set_xticks([])
    fig.tight_layout()
    return fig


def plot_plan(inp: BuildingInput, res: DesignResult):
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot([0, inp.plan_x, inp.plan_x, 0, 0], [0, 0, inp.plan_y, inp.plan_y, 0], color="black", lw=1.5)
    for i in range(inp.n_bays_x + 1):
        gx = i * inp.bay_x
        ax.plot([gx, gx], [0, inp.plan_y], color="#d9d9d9", lw=0.8)
    for j in range(inp.n_bays_y + 1):
        gy = j * inp.bay_y
        ax.plot([0, inp.plan_x], [gy, gy], color="#d9d9d9", lw=0.8)

    cc, pc, ic = res.col_dims
    for i in range(inp.n_bays_x + 1):
        for j in range(inp.n_bays_y + 1):
            px = i * inp.bay_x
            py = j * inp.bay_y
            at_lr = i == 0 or i == inp.n_bays_x
            at_bt = j == 0 or j == inp.n_bays_y
            if at_lr and at_bt:
                d, col = cc, "#1f77b4"
            elif at_lr or at_bt:
                d, col = pc, "#2ca02c"
            else:
                d, col = ic, "#ff7f0e"
            ax.add_patch(plt.Rectangle((px - d/2, py - d/2), d, d, color=col, alpha=0.90))

    cx0 = (inp.plan_x - res.core_outer[0]) / 2
    cy0 = (inp.plan_y - res.core_outer[1]) / 2
    ix0 = (inp.plan_x - res.core_opening[0]) / 2
    iy0 = (inp.plan_y - res.core_opening[1]) / 2
    ax.add_patch(plt.Rectangle((cx0, cy0), res.core_outer[0], res.core_outer[1], fill=False, lw=2.5, color="#111827"))
    ax.add_patch(plt.Rectangle((ix0, iy0), res.core_opening[0], res.core_opening[1], fill=False, lw=1.4, ls="--", color="#6b7280"))
    for (p0, p1) in res.core_wall_segments:
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color="#2e8b57", lw=5)

    ax.text(cx0 + res.core_outer[0]/2, cy0 - 1.2, "CORE", ha="center", fontsize=10, fontweight="bold")
    ax.text(ix0 + res.core_opening[0]/2, iy0 + res.core_opening[1]/2, "OPENING", ha="center", va="center", fontsize=9)

    info_x = inp.plan_x + 8
    info_y = inp.plan_y - 2
    info_lines = [
        f"Active core walls = {res.active_core_walls}",
        f"Core outer = {res.core_outer[0]:.2f} x {res.core_outer[1]:.2f} m",
        f"Core opening = {res.core_opening[0]:.2f} x {res.core_opening[1]:.2f} m",
        f"Wall thickness = {res.wall_t:.2f} m",
        f"Corner col = {res.col_dims[0]:.2f} m",
        f"Perim col = {res.col_dims[1]:.2f} m",
        f"Interior col = {res.col_dims[2]:.2f} m",
        f"Beam = {res.beam_b:.2f} x {res.beam_h:.2f} m",
        f"Slab t = {res.slab_t:.2f} m",
    ]
    for k, txt in enumerate(info_lines):
        ax.text(info_x, info_y - 3.5 * k, txt, fontsize=8.8)

    ax.set_xlim(-5, inp.plan_x + 40)
    ax.set_ylim(inp.plan_y + 5, -5)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig


def build_report(inp: BuildingInput, res: DesignResult) -> str:
    q = res.qty
    lines = []
    lines.append("GLOBAL RESPONSE")
    lines.append("-" * 74)
    lines.append(f"Reference period               = {res.T_ref:.3f} s")
    lines.append(f"Design target period           = {res.T_target:.3f} s")
    lines.append(f"Estimated dynamic period       = {res.T_est:.3f} s")
    lines.append(f"Upper limit period             = {res.T_upper:.3f} s")
    lines.append(f"Target position factor beta    = {res.beta:.3f}")
    lines.append(f"Period error                   = {100*res.period_error:.2f} %")
    lines.append(f"Period check                   = {'OK' if res.period_ok else 'NOT OK'}")
    lines.append(f"Total stiffness                = {res.K_est:,.3e} N/m")
    lines.append(f"Top drift                      = {res.drift_m:.3f} m")
    lines.append(f"Drift ratio                    = {res.drift_ratio:.5f}")
    lines.append(f"Drift check                    = {'OK' if res.drift_ok else 'NOT OK'}")
    lines.append("")
    lines.append("PRIMARY MEMBER OUTPUT")
    lines.append("-" * 74)
    lines.append(f"Beam size                      = {res.beam_b:.2f} x {res.beam_h:.2f} m")
    lines.append(f"Slab thickness                 = {res.slab_t:.2f} m")
    lines.append(f"Wall thickness                 = {res.wall_t:.2f} m")
    lines.append(f"Corner column                  = {res.col_dims[0]:.2f} m")
    lines.append(f"Perimeter column               = {res.col_dims[1]:.2f} m")
    lines.append(f"Interior column                = {res.col_dims[2]:.2f} m")
    lines.append(f"Core outer                     = {res.core_outer[0]:.2f} x {res.core_outer[1]:.2f} m")
    lines.append(f"Core opening                   = {res.core_opening[0]:.2f} x {res.core_opening[1]:.2f} m")
    lines.append(f"Active core walls              = {res.active_core_walls}")
    lines.append("")
    lines.append("CONCRETE AND STEEL QUANTITIES")
    lines.append("-" * 74)
    lines.append(f"Wall concrete                  = {q.wall_conc_m3:,.2f} m3")
    lines.append(f"Column concrete                = {q.column_conc_m3:,.2f} m3")
    lines.append(f"Beam concrete                  = {q.beam_conc_m3:,.2f} m3")
    lines.append(f"Slab concrete                  = {q.slab_conc_m3:,.2f} m3")
    lines.append(f"Wall steel                     = {q.wall_steel_kg:,.0f} kg")
    lines.append(f"Column steel                   = {q.column_steel_kg:,.0f} kg")
    lines.append(f"Beam steel                     = {q.beam_steel_kg:,.0f} kg")
    lines.append(f"Slab steel                     = {q.slab_steel_kg:,.0f} kg")
    lines.append(f"Total steel                    = {q.total_steel_kg:,.0f} kg")
    lines.append("")
    lines.append("MODAL MASS PARTICIPATION")
    lines.append("-" * 74)
    for i, (T, f, r, c) in enumerate(zip(res.modal.periods_s, res.modal.frequencies_hz, res.modal.effective_mass_ratios, res.modal.cumulative_effective_mass_ratios), start=1):
        lines.append(f"Mode {i}: T = {T:.4f} s | f = {f:.4f} Hz | mass ratio = {100*r:.2f}% | cumulative = {100*c:.2f}%")
    lines.append("")
    lines.append("ITERATION LOG")
    lines.append("-" * 74)
    for it, cs, cols, te, tt, dr in res.iteration_log:
        lines.append(f"Iter {it:02d}: core_scale={cs:.3f}, col_scale={cols:.3f}, T_est={te:.3f}, T_target={tt:.3f}, drift_ratio={dr:.5f}")
    lines.append("")
    lines.append("GOVERNING ISSUE")
    lines.append("-" * 74)
    lines.append(res.governing_issue)
    lines.append("")
    lines.append("REDESIGN SUGGESTIONS")
    lines.append("-" * 74)
    for s in res.redesign_suggestions:
        lines.append(f"- {s}")
    return "\n".join(lines)


inp = streamlit_input_panel()

if st.button("Analyze"):
    res = run_design(inp)
    H = inp.n_story * inp.story_height
    report = build_report(inp, res)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Reference period (s)", f"{res.T_ref:.3f}")
    c2.metric("Design target (s)", f"{res.T_target:.3f}")
    c3.metric("Estimated dynamic (s)", f"{res.T_est:.3f}")
    c4.metric("Upper limit (s)", f"{res.T_upper:.3f}")

    d1, d2, d3 = st.columns(3)
    d1.metric("Period error (%)", f"{100*res.period_error:.2f}")
    d2.metric("Total stiffness (N/m)", f"{res.K_est:,.2e}")
    d3.metric("Top drift (m)", f"{res.drift_m:.3f}")

    st.caption(f"Target formula: T_target = T_ref + β (T_upper - T_ref), β = {res.beta:.2f}")

    tab1, tab2, tab3, tab4 = st.tabs(["Plan", "Modes", "Quantities", "Report"])

    with tab1:
        st.pyplot(plot_plan(inp, res), use_container_width=True)

    with tab2:
        st.pyplot(plot_modes(res, inp.n_story, H), use_container_width=True)
        rows = []
        for i, (T, f, r, c) in enumerate(zip(res.modal.periods_s, res.modal.frequencies_hz, res.modal.effective_mass_ratios, res.modal.cumulative_effective_mass_ratios), start=1):
            rows.append({"Mode": i, "Period (s)": round(T, 4), "Frequency (Hz)": round(f, 4), "Eff. mass ratio (%)": round(100*r, 2), "Cumulative (%)": round(100*c, 2)})
        st.dataframe(rows, use_container_width=True)

    with tab3:
        q = res.qty
        st.write({
            "Wall concrete (m3)": round(q.wall_conc_m3, 2),
            "Column concrete (m3)": round(q.column_conc_m3, 2),
            "Beam concrete (m3)": round(q.beam_conc_m3, 2),
            "Slab concrete (m3)": round(q.slab_conc_m3, 2),
            "Wall steel (kg)": round(q.wall_steel_kg, 0),
            "Column steel (kg)": round(q.column_steel_kg, 0),
            "Beam steel (kg)": round(q.beam_steel_kg, 0),
            "Slab steel (kg)": round(q.slab_steel_kg, 0),
            "Total steel (kg)": round(q.total_steel_kg, 0),
        })
        st.markdown("### Redesign suggestions")
        for s in res.redesign_suggestions:
            st.write(f"- {s}")
        st.markdown("### Iteration log")
        st.dataframe(
            [{"Iteration": it, "Core scale": round(cs, 3), "Column scale": round(cols, 3), "T_est (s)": round(te, 3), "T_target (s)": round(tt, 3), "Drift ratio": round(dr, 5)}
             for it, cs, cols, te, tt, dr in res.iteration_log],
            use_container_width=True
        )

    with tab4:
        st.text_area("Report", report, height=600)
