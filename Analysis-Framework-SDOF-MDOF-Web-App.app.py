
from __future__ import annotations
from dataclasses import dataclass, field
from math import pi, sqrt
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from scipy.optimize import minimize

st.set_page_config(page_title="Tall Building Structural Analysis", layout="wide", initial_sidebar_state="expanded")

AUTHOR_NAME = "Benyamin"
APP_VERSION = "v2.2"
G = 9.81
STEEL_DENSITY = 7850.0

CORNER_COLOR = "#8b0000"
PERIM_COLOR = "#cc5500"
INTERIOR_COLOR = "#4444aa"
CORE_COLOR = "#2e8b57"
PERIM_WALL_COLOR = "#4caf50"

@dataclass
class ZoneDefinition:
    name: str
    story_start: int
    story_end: int
    @property
    def n_stories(self) -> int:
        return self.story_end - self.story_start + 1

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
    stair_count: int = 2
    elevator_count: int = 4
    elevator_area_each: float = 3.5
    stair_area_each: float = 20.0
    service_area: float = 35.0
    corridor_factor: float = 1.4
    fck: float = 70.0
    Ec: float = 36000.0
    fy: float = 420.0
    DL: float = 3.0
    LL: float = 2.5
    slab_finish_allowance: float = 1.5
    facade_line_load: float = 1.0
    prelim_lateral_force_coeff: float = 0.015
    drift_limit_ratio: float = 1 / 500
    upper_period_limit_factor: float = 1.40
    min_wall_thickness: float = 0.30
    max_wall_thickness: float = 1.20
    min_column_dim: float = 0.70
    max_column_dim: float = 1.80
    min_beam_width: float = 0.40
    min_beam_depth: float = 0.75
    min_slab_thickness: float = 0.22
    max_slab_thickness: float = 0.40
    wall_cracked_factor: float = 0.70
    column_cracked_factor: float = 0.70
    max_story_wall_slenderness: float = 12.0
    wall_rebar_ratio: float = 0.003
    column_rebar_ratio: float = 0.010
    beam_rebar_ratio: float = 0.015
    slab_rebar_ratio: float = 0.0035
    seismic_mass_factor: float = 1.0
    Ct: float = 0.0488
    x_period: float = 0.75
    perimeter_column_factor: float = 1.10
    corner_column_factor: float = 1.30
    lower_zone_wall_count: int = 8
    middle_zone_wall_count: int = 6
    upper_zone_wall_count: int = 4
    basement_retaining_wall_thickness: float = 0.50
    perimeter_shear_wall_ratio: float = 0.20

@dataclass
class ZoneCoreResult:
    zone: ZoneDefinition
    wall_count: int
    wall_lengths: List[float]
    wall_thickness: float
    core_outer_x: float
    core_outer_y: float
    core_opening_x: float
    core_opening_y: float
    Ieq_gross_m4: float
    Ieq_effective_m4: float
    story_slenderness: float
    perimeter_wall_segments: List[Tuple[str, float, float]]
    retaining_wall_active: bool

@dataclass
class ZoneColumnResult:
    zone: ZoneDefinition
    corner_column_x_m: float
    corner_column_y_m: float
    perimeter_column_x_m: float
    perimeter_column_y_m: float
    interior_column_x_m: float
    interior_column_y_m: float
    I_col_group_effective_m4: float

@dataclass
class ReinforcementEstimate:
    wall_concrete_volume_m3: float
    column_concrete_volume_m3: float
    beam_concrete_volume_m3: float
    slab_concrete_volume_m3: float
    wall_steel_kg: float
    column_steel_kg: float
    beam_steel_kg: float
    slab_steel_kg: float
    total_steel_kg: float

@dataclass
class ModalResult:
    n_dof: int
    periods_s: List[float]
    frequencies_hz: List[float]
    mode_shapes: List[List[float]]
    story_masses_kg: List[float]
    story_stiffness_N_per_m: List[float]
    modal_participation_factors: List[float]
    effective_modal_masses_kg: List[float]
    effective_mass_ratios: List[float]
    cumulative_effective_mass_ratios: List[float]

@dataclass
class DesignResult:
    H_m: float
    total_weight_kN: float
    reference_period_s: float
    design_target_period_s: float
    upper_limit_period_s: float
    estimated_period_s: float
    period_error_ratio: float
    period_ok: bool
    drift_ok: bool
    K_estimated_N_per_m: float
    top_drift_m: float
    drift_ratio: float
    zone_core_results: List[ZoneCoreResult]
    zone_column_results: List[ZoneColumnResult]
    slab_thickness_m: float
    beam_width_m: float
    beam_depth_m: float
    reinforcement: ReinforcementEstimate
    optimization_success: bool
    optimization_message: str
    core_scale: float
    column_scale: float
    messages: List[str] = field(default_factory=list)
    redesign_suggestions: List[str] = field(default_factory=list)
    governing_issue: str = ""
    modal_result: ModalResult | None = None

def total_height(inp: BuildingInput) -> float:
    return inp.n_story * inp.story_height

def floor_area(inp: BuildingInput) -> float:
    return 0.5 * inp.plan_x * inp.plan_y if inp.plan_shape == "triangle" else inp.plan_x * inp.plan_y

def slab_thickness_prelim(inp: BuildingInput) -> float:
    return max(inp.min_slab_thickness, min(inp.max_slab_thickness, max(inp.bay_x, inp.bay_y) / 28.0))

def beam_size_prelim(inp: BuildingInput) -> tuple[float, float]:
    span = max(inp.bay_x, inp.bay_y)
    h = max(inp.min_beam_depth, span / 12.0)
    b = max(inp.min_beam_width, 0.45 * h)
    return b, h

def total_weight_kN(inp: BuildingInput, slab_t: float, core_scale: float, col_scale: float) -> float:
    A = floor_area(inp)
    perimeter = 2.0 * (inp.plan_x + inp.plan_y) if inp.plan_shape == "square" else (inp.plan_x + inp.plan_y + (inp.plan_x**2 + inp.plan_y**2)**0.5)
    slab_self = slab_t * 25.0
    floor_load = inp.DL + inp.LL + inp.slab_finish_allowance + slab_self
    base = floor_load * A * inp.n_story + 1.10 * floor_load * A * inp.n_basement + inp.facade_line_load * perimeter * inp.n_story
    # simple structural self weight sensitivity
    structural = 0.10 * base + 0.05 * base * (0.7 * core_scale + 0.3 * col_scale)
    return (base + structural) * inp.seismic_mass_factor

def define_three_zones(n_story: int) -> List[ZoneDefinition]:
    z1 = max(1, round(0.30 * n_story))
    z2 = max(z1 + 1, round(0.70 * n_story))
    return [ZoneDefinition("Lower Zone", 1, z1), ZoneDefinition("Middle Zone", z1 + 1, z2), ZoneDefinition("Upper Zone", z2 + 1, n_story)]

def required_opening_area(inp: BuildingInput) -> float:
    return (inp.elevator_count * inp.elevator_area_each + inp.stair_count * inp.stair_area_each + inp.service_area) * inp.corridor_factor

def opening_dimensions(inp: BuildingInput) -> tuple[float, float]:
    area = required_opening_area(inp)
    oy = sqrt(area / 1.6)
    return 1.6 * oy, oy

def active_wall_count_by_zone(inp: BuildingInput, name: str) -> int:
    return {"Lower Zone": inp.lower_zone_wall_count, "Middle Zone": inp.middle_zone_wall_count, "Upper Zone": inp.upper_zone_wall_count}[name]

def wall_lengths_for_layout(outer_x: float, outer_y: float, wall_count: int) -> List[float]:
    if wall_count == 4:
        return [outer_x, outer_x, outer_y, outer_y]
    if wall_count == 6:
        return [outer_x, outer_x, outer_y, outer_y, 0.45 * outer_x, 0.45 * outer_x]
    return [outer_x, outer_x, outer_y, outer_y, 0.45 * outer_x, 0.45 * outer_x, 0.45 * outer_y, 0.45 * outer_y]

def wall_rect_inertia_about_global_y(length, thickness, x_centroid):
    return length * thickness**3 / 12.0 + length * thickness * x_centroid**2

def wall_rect_inertia_about_global_x(length, thickness, y_centroid):
    return length * thickness**3 / 12.0 + length * thickness * y_centroid**2

def core_equivalent_inertia(outer_x, outer_y, lengths, t, wall_count):
    x_side, y_side = outer_x / 2.0, outer_y / 2.0
    top_len, bot_len, left_len, right_len = lengths[:4]
    I_x = wall_rect_inertia_about_global_x(top_len, t, y_side) + wall_rect_inertia_about_global_x(bot_len, t, -y_side)
    I_y = (t * top_len**3 / 12.0) + (t * bot_len**3 / 12.0)
    I_y += wall_rect_inertia_about_global_y(left_len, t, -x_side) + wall_rect_inertia_about_global_y(right_len, t, x_side)
    I_x += (t * left_len**3 / 12.0) + (t * right_len**3 / 12.0)
    if wall_count >= 6:
        inner_x = 0.22 * outer_x
        l1, l2 = lengths[4], lengths[5]
        I_y += wall_rect_inertia_about_global_y(l1, t, -inner_x) + wall_rect_inertia_about_global_y(l2, t, inner_x)
        I_x += (t * l1**3 / 12.0) + (t * l2**3 / 12.0)
    if wall_count >= 8:
        inner_y = 0.22 * outer_y
        l3, l4 = lengths[6], lengths[7]
        I_x += wall_rect_inertia_about_global_x(l3, t, -inner_y) + wall_rect_inertia_about_global_x(l4, t, inner_y)
        I_y += (t * l3**3 / 12.0) + (t * l4**3 / 12.0)
    return min(I_x, I_y)

def perimeter_wall_segments_for_square(inp: BuildingInput, zone: ZoneDefinition):
    if zone.name == "Lower Zone":
        return [("top", 0.0, inp.plan_x), ("bottom", 0.0, inp.plan_x), ("left", 0.0, inp.plan_y), ("right", 0.0, inp.plan_y)]
    ratio = inp.perimeter_shear_wall_ratio
    lx, ly = inp.plan_x * ratio, inp.plan_y * ratio
    sx, sy = (inp.plan_x - lx) / 2.0, (inp.plan_y - ly) / 2.0
    return [("top", sx, sx + lx), ("bottom", sx, sx + lx), ("left", sy, sy + ly), ("right", sy, sy + ly)]

def design_core_by_zone(inp: BuildingInput, zones: List[ZoneDefinition], core_scale: float) -> List[ZoneCoreResult]:
    opening_x, opening_y = opening_dimensions(inp)
    outer_x = min(max(opening_x + 3.0, 0.24 * inp.plan_x) * core_scale, 0.50 * inp.plan_x)
    outer_y = min(max(opening_y + 3.0, 0.22 * inp.plan_y) * core_scale, 0.50 * inp.plan_y)
    H = total_height(inp)
    out = []
    for zone in zones:
        wall_count = active_wall_count_by_zone(inp, zone.name)
        lengths = wall_lengths_for_layout(outer_x, outer_y, wall_count)
        base_t = max(inp.min_wall_thickness, min(inp.max_wall_thickness, H / 180.0))
        t = {"Lower Zone": base_t, "Middle Zone": max(inp.min_wall_thickness, 0.80 * base_t), "Upper Zone": max(inp.min_wall_thickness, 0.60 * base_t)}[zone.name]
        t = min(inp.max_wall_thickness, max(inp.min_wall_thickness, t * core_scale))
        I_gross = core_equivalent_inertia(outer_x, outer_y, lengths, t, wall_count)
        I_eff = inp.wall_cracked_factor * I_gross
        out.append(ZoneCoreResult(zone, wall_count, lengths, t, outer_x, outer_y, opening_x, opening_y, I_gross, I_eff, inp.story_height / t, perimeter_wall_segments_for_square(inp, zone), zone.name == "Lower Zone"))
    return out

def estimate_zone_column_sizes(inp: BuildingInput, zones: List[ZoneDefinition], slab_t: float, col_scale: float) -> List[ZoneColumnResult]:
    q = inp.DL + inp.LL + inp.slab_finish_allowance + slab_t * 25.0
    sigma_allow = 0.35 * inp.fck * 1000.0
    total_columns = (inp.n_bays_x + 1) * (inp.n_bays_y + 1)
    corner_cols = 4
    perimeter_cols = max(0, 2 * (inp.n_bays_x - 1) + 2 * (inp.n_bays_y - 1))
    interior_cols = max(0, total_columns - corner_cols - perimeter_cols)
    plan_center_x, plan_center_y = inp.plan_x / 2.0, inp.plan_y / 2.0
    r2_sum = 0.0
    for i in range(inp.n_bays_x + 1):
        for j in range(inp.n_bays_y + 1):
            x, y = i * inp.bay_x, j * inp.bay_y
            r2_sum += (x - plan_center_x) ** 2 + (y - plan_center_y) ** 2
    out = []
    for zone in zones:
        n_effective = (inp.n_story - zone.story_start + 1) + 0.70 * inp.n_basement
        P_interior = inp.bay_x * inp.bay_y * q * n_effective * 1.18
        interior = min(inp.max_column_dim, max(inp.min_column_dim, sqrt(P_interior / sigma_allow) * col_scale))
        perim = min(inp.max_column_dim, max(inp.min_column_dim, interior * inp.perimeter_column_factor))
        corner = min(inp.max_column_dim, max(inp.min_column_dim, interior * inp.corner_column_factor))
        Iavg_corner = corner**4 / 12.0
        Iavg_perim = perim**4 / 12.0
        Iavg_inter = interior**4 / 12.0
        I_avg = (corner_cols * Iavg_corner + perimeter_cols * Iavg_perim + interior_cols * Iavg_inter) / max(total_columns, 1)
        A_avg = (corner_cols * corner**2 + perimeter_cols * perim**2 + interior_cols * interior**2) / max(total_columns, 1)
        I_col_group = inp.column_cracked_factor * (I_avg * max(total_columns, 1) + A_avg * r2_sum)
        out.append(ZoneColumnResult(zone, corner, corner, perim, perim, interior, interior, I_col_group))
    return out

def cantilever_tip_stiffness(EI: float, H: float) -> float:
    return 3.0 * EI / max(H**3, 1e-9)

def weighted_core_stiffness(inp: BuildingInput, zone_cores: List[ZoneCoreResult]) -> float:
    H = total_height(inp)
    E = inp.Ec * 1e6
    total_flex = 0.0
    for zc in zone_cores:
        hi = zc.zone.n_stories * inp.story_height
        perimeter_bonus = 1.0 + 0.20 * len(zc.perimeter_wall_segments)
        total_flex += (hi / H) / max(E * zc.Ieq_effective_m4 * perimeter_bonus, 1e-9)
    EI_equiv = 1.0 / max(total_flex, 1e-18)
    return cantilever_tip_stiffness(EI_equiv, H)

def weighted_column_stiffness(inp: BuildingInput, zone_cols: List[ZoneColumnResult]) -> float:
    H = total_height(inp)
    E = inp.Ec * 1e6
    total_flex = 0.0
    for zc in zone_cols:
        hi = zc.zone.n_stories * inp.story_height
        total_flex += (hi / H) / max(E * zc.I_col_group_effective_m4, 1e-9)
    EI_equiv = 1.0 / max(total_flex, 1e-18)
    return cantilever_tip_stiffness(EI_equiv, H)

def estimate_reinforcement(inp: BuildingInput, zone_cores, zone_cols, slab_t, beam_b, beam_h):
    n_levels = inp.n_story + inp.n_basement
    total_area = floor_area(inp) * n_levels
    wall_conc = 0.0
    for zc in zone_cores:
        wall_conc += sum(zc.wall_lengths) * zc.wall_thickness * (zc.zone.n_stories * inp.story_height)
        for _, a, b in zc.perimeter_wall_segments:
            wall_conc += (b - a) * zc.wall_thickness * (zc.zone.n_stories * inp.story_height)
    corner_cols = 4
    perimeter_cols = max(0, 2 * (inp.n_bays_x - 1) + 2 * (inp.n_bays_y - 1))
    total_cols = (inp.n_bays_x + 1) * (inp.n_bays_y + 1)
    interior_cols = max(0, total_cols - corner_cols - perimeter_cols)
    col_conc = 0.0
    for zc in zone_cols:
        zh = zc.zone.n_stories * inp.story_height
        col_conc += corner_cols * zc.corner_column_x_m * zc.corner_column_y_m * zh
        col_conc += perimeter_cols * zc.perimeter_column_x_m * zc.perimeter_column_y_m * zh
        col_conc += interior_cols * zc.interior_column_x_m * zc.interior_column_y_m * zh
    beam_lines = max(1, inp.n_bays_y * (inp.n_bays_x + 1) + inp.n_bays_x * (inp.n_bays_y + 1))
    total_beam_length = beam_lines * 0.5 * (inp.bay_x + inp.bay_y) * n_levels
    beam_conc = beam_b * beam_h * total_beam_length
    slab_conc = total_area * slab_t
    wall_steel = wall_conc * inp.wall_rebar_ratio * STEEL_DENSITY
    col_steel = col_conc * inp.column_rebar_ratio * STEEL_DENSITY
    beam_steel = beam_conc * inp.beam_rebar_ratio * STEEL_DENSITY
    slab_steel = slab_conc * inp.slab_rebar_ratio * STEEL_DENSITY
    return ReinforcementEstimate(wall_conc, col_conc, beam_conc, slab_conc, wall_steel, col_steel, beam_steel, slab_steel, wall_steel + col_steel + beam_steel + slab_steel)

def build_story_masses(inp: BuildingInput, total_weight_kN_value: float) -> List[float]:
    W_story_kN = total_weight_kN_value / max(inp.n_story, 1)
    m_story = (W_story_kN * 1000.0) / G
    return [m_story for _ in range(inp.n_story)]

def build_story_stiffnesses(inp: BuildingInput, K_total: float) -> List[float]:
    n = inp.n_story
    raw = [1.35 - 0.55 * (i / max(n - 1, 1)) for i in range(n)]
    c = K_total * sum(1.0 / a for a in raw)
    return [c * a for a in raw]

def assemble_m_k_matrices(story_masses: List[float], story_stiffness: List[float]) -> tuple[np.ndarray, np.ndarray]:
    n = len(story_masses)
    M = np.diag(story_masses)
    K = np.zeros((n, n), dtype=float)
    for i in range(n):
        ki = story_stiffness[i]
        if i == 0:
            K[i, i] += ki
        else:
            K[i, i] += ki
            K[i, i - 1] -= ki
            K[i - 1, i] -= ki
            K[i - 1, i - 1] += ki
    return M, K

def solve_mdof_modes(inp: BuildingInput, total_weight_kN_value: float, K_total: float, n_modes: int = 5) -> ModalResult:
    masses = build_story_masses(inp, total_weight_kN_value)
    k_stories = build_story_stiffnesses(inp, K_total)
    M, K = assemble_m_k_matrices(masses, k_stories)
    A = np.linalg.inv(M) @ K
    eigvals, eigvecs = np.linalg.eig(A)
    eigvals = np.real(eigvals)
    eigvecs = np.real(eigvecs)
    pos = eigvals > 1e-12
    eigvals = eigvals[pos]
    eigvecs = eigvecs[:, pos]
    order = np.argsort(eigvals)
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    omegas = np.sqrt(eigvals)
    periods = [2.0 * pi / w for w in omegas[:n_modes]]
    freqs = [w / (2.0 * pi) for w in omegas[:n_modes]]
    ones = np.ones((len(masses), 1))
    total_mass = np.sum(np.diag(M)).item()
    gammas = []
    meffs = []
    ratios = []
    cumrat = []
    mode_shapes = []
    cum = 0.0
    for i in range(min(n_modes, eigvecs.shape[1])):
        phi = eigvecs[:, i].reshape(-1, 1)
        denom = (phi.T @ M @ phi).item()
        gamma = ((phi.T @ M @ ones) / denom).item()
        meff = gamma**2 * denom
        ratio = meff / total_mass
        cum += ratio
        phi_plot = phi.flatten().copy()
        if abs(phi_plot[-1]) > 1e-12:
            phi_plot = phi_plot / phi_plot[-1]
        if phi_plot[-1] < 0:
            phi_plot = -phi_plot
        mode_shapes.append(phi_plot.tolist())
        gammas.append(gamma)
        meffs.append(meff)
        ratios.append(ratio)
        cumrat.append(cum)
    return ModalResult(len(masses), periods, freqs, mode_shapes, masses, k_stories, gammas, meffs, ratios, cumrat)

def generate_redesign_suggestions(inp, T_est, T_target, T_limit, drift_ratio, drift_limit, core_scale, column_scale):
    suggestions = []
    governing_issue = "OK"
    if T_est > T_limit:
        governing_issue = "Estimated dynamic period exceeds upper limit"
        suggestions += [
            "Increase lateral stiffness: enlarge the core walls and/or perimeter columns.",
            "Increase the number of active shear walls in middle and upper zones.",
            "Consider a larger core footprint if architectural constraints allow.",
            "Consider adding outriggers or belt trusses for tall-building stiffness enhancement.",
        ]
    elif T_est > 1.10 * T_target:
        governing_issue = "Estimated dynamic period is larger than target"
        suggestions += [
            "Increase wall thickness or activate more internal core wall segments.",
            "Increase corner and perimeter column dimensions.",
            "Reduce excessive flexibility in the lateral system layout.",
        ]
    elif T_est < 0.90 * T_target:
        governing_issue = "Estimated dynamic period is smaller than target"
        suggestions += [
            "The system may be overly stiff and potentially uneconomical.",
            "Consider reducing wall thicknesses or column dimensions where feasible.",
        ]
    if drift_ratio > drift_limit:
        if governing_issue == "OK":
            governing_issue = "Drift exceeds allowable limit"
        suggestions += [
            "Increase global stiffness by enlarging the core and perimeter columns.",
            "Increase the wall-to-floor-area ratio or add perimeter wall segments.",
        ]
    if core_scale >= 1.58:
        suggestions.append("Core scale factor is near its upper bound; core configuration may be insufficient.")
    if column_scale >= 1.58:
        suggestions.append("Column scale factor is near its upper bound; revise column layout or increase column capacity.")
    if not suggestions:
        suggestions.append("Structural system appears preliminarily adequate; no redesign action is required.")
    return governing_issue, suggestions

def evaluate_design(inp: BuildingInput, core_scale: float, col_scale: float) -> dict:
    H = total_height(inp)
    slab_t = slab_thickness_prelim(inp)
    beam_b, beam_h = beam_size_prelim(inp)
    zones = define_three_zones(inp.n_story)
    zone_cores = design_core_by_zone(inp, zones, core_scale)
    zone_cols = estimate_zone_column_sizes(inp, zones, slab_t, col_scale)
    W_total = total_weight_kN(inp, slab_t, core_scale, col_scale)
    K_core = weighted_core_stiffness(inp, zone_cores)
    K_cols = weighted_column_stiffness(inp, zone_cols)
    K_est = K_core + K_cols
    T_est = 2.0 * pi * sqrt(((W_total * 1000.0) / G) / max(K_est, 1e-9))
    F = inp.prelim_lateral_force_coeff * W_total * 1000.0
    top_drift = F / max(K_est, 1e-9)
    drift_ratio = top_drift / max(H, 1e-9)
    modal = solve_mdof_modes(inp, W_total, K_est, 5)
    reinf = estimate_reinforcement(inp, zone_cores, zone_cols, slab_t, beam_b, beam_h)
    return {
        "H": H, "W_total": W_total, "slab_t": slab_t, "beam_b": beam_b, "beam_h": beam_h,
        "zone_cores": zone_cores, "zone_cols": zone_cols, "K_est": K_est,
        "T_est": T_est, "top_drift": top_drift, "drift_ratio": drift_ratio,
        "modal": modal, "reinf": reinf
    }

def run_design(inp: BuildingInput) -> DesignResult:
    T_ref = inp.Ct * (total_height(inp) ** inp.x_period)
    T_target = T_ref
    T_limit = inp.upper_period_limit_factor * T_ref
    drift_allow = inp.drift_limit_ratio

    def objective(x):
        core_scale, col_scale = x
        res = evaluate_design(inp, core_scale, col_scale)
        T_est = res["T_est"]
        drift_ratio = res["drift_ratio"]
        W = res["W_total"]
        J = ((T_est - T_target) / max(T_target, 1e-9))**2
        J += 0.20 * ((W / 1e6))**2
        if T_est > T_limit:
            J += 200.0 * ((T_est - T_limit) / max(T_limit, 1e-9))**2
        if drift_ratio > drift_allow:
            J += 200.0 * ((drift_ratio - drift_allow) / max(drift_allow, 1e-9))**2
        return J

    result = minimize(
        objective,
        x0=np.array([1.0, 1.0]),
        bounds=[(0.70, 1.60), (0.70, 1.60)],
        method="L-BFGS-B",
        options={"maxiter": 120}
    )

    core_scale, col_scale = np.asarray(result.x)
    res = evaluate_design(inp, float(core_scale), float(col_scale))

    T_est = res["T_est"]
    period_error_ratio = abs(T_est - T_target) / max(T_target, 1e-9)
    period_ok = T_est <= T_limit
    drift_ok = res["drift_ratio"] <= drift_allow
    governing_issue, redesign_suggestions = generate_redesign_suggestions(inp, T_est, T_target, T_limit, res["drift_ratio"], drift_allow, float(core_scale), float(col_scale))
    messages = [
        f"Reference period computed from Ct * H^x = {T_ref:.3f} s.",
        f"Upper limit period = upper_period_limit_factor * reference period = {T_limit:.3f} s.",
        f"Optimization objective targets the reference period while minimizing excessive weight and violating penalties.",
    ]

    return DesignResult(
        H_m=res["H"], total_weight_kN=res["W_total"], reference_period_s=T_ref, design_target_period_s=T_target,
        upper_limit_period_s=T_limit, estimated_period_s=T_est, period_error_ratio=period_error_ratio,
        period_ok=period_ok, drift_ok=drift_ok, K_estimated_N_per_m=res["K_est"], top_drift_m=res["top_drift"],
        drift_ratio=res["drift_ratio"], zone_core_results=res["zone_cores"], zone_column_results=res["zone_cols"],
        slab_thickness_m=res["slab_t"], beam_width_m=res["beam_b"], beam_depth_m=res["beam_h"],
        reinforcement=res["reinf"], optimization_success=bool(result.success), optimization_message=str(result.message),
        core_scale=float(core_scale), column_scale=float(col_scale), messages=messages,
        redesign_suggestions=redesign_suggestions, governing_issue=governing_issue, modal_result=res["modal"]
    )

def build_report(result: DesignResult) -> str:
    lines = []
    lines += ["GLOBAL RESPONSE", "-" * 74]
    lines.append(f"Reference period               = {result.reference_period_s:.3f} s")
    lines.append(f"Design target period           = {result.design_target_period_s:.3f} s")
    lines.append(f"Estimated dynamic period       = {result.estimated_period_s:.3f} s")
    lines.append(f"Upper limit period             = {result.upper_limit_period_s:.3f} s")
    lines.append(f"Period error ratio             = {100 * result.period_error_ratio:.2f} %")
    lines.append(f"Period check                   = {'OK' if result.period_ok else 'NOT OK'}")
    lines.append(f"Total stiffness                = {result.K_estimated_N_per_m:,.3e} N/m")
    lines.append(f"Estimated top drift            = {result.top_drift_m:.3f} m")
    lines.append(f"Estimated drift ratio          = {result.drift_ratio:.5f}")
    lines.append(f"Drift check                    = {'OK' if result.drift_ok else 'NOT OK'}")
    lines.append(f"Total structural weight        = {result.total_weight_kN:,.0f} kN")
    lines.append("")
    lines += ["GOVERNING ISSUE", "-" * 74, result.governing_issue, ""]
    lines += ["PRIMARY MEMBER OUTPUT", "-" * 74]
    lines.append(f"Beam size (b x h)              = {result.beam_width_m:.2f} x {result.beam_depth_m:.2f} m")
    lines.append(f"Slab thickness                 = {result.slab_thickness_m:.2f} m")
    lines.append("")
    lines += ["ZONE-BY-ZONE COLUMN DIMENSIONS", "-" * 74]
    for zc in result.zone_column_results:
        lines.append(f"{zc.zone.name}:")
        lines.append(f"  Corner columns              = {zc.corner_column_x_m:.2f} x {zc.corner_column_y_m:.2f} m")
        lines.append(f"  Perimeter columns           = {zc.perimeter_column_x_m:.2f} x {zc.perimeter_column_y_m:.2f} m")
        lines.append(f"  Interior columns            = {zc.interior_column_x_m:.2f} x {zc.interior_column_y_m:.2f} m")
        lines.append(f"  Column group Ieff           = {zc.I_col_group_effective_m4:,.2f} m^4")
    lines.append("")
    lines += ["ZONE-BY-ZONE WALL / CORE OUTPUT", "-" * 74]
    for zc in result.zone_core_results:
        lines.append(f"{zc.zone.name}:")
        lines.append(f"  Core outer                  = {zc.core_outer_x:.2f} x {zc.core_outer_y:.2f} m")
        lines.append(f"  Core opening                = {zc.core_opening_x:.2f} x {zc.core_opening_y:.2f} m")
        lines.append(f"  Wall thickness              = {zc.wall_thickness:.2f} m")
        lines.append(f"  Active core walls           = {zc.wall_count}")
        lines.append(f"  Gross Ieq                   = {zc.Ieq_gross_m4:,.2f} m^4")
        lines.append(f"  Effective Ieq               = {zc.Ieq_effective_m4:,.2f} m^4")
        lines.append(f"  Story slenderness           = {zc.story_slenderness:.2f}")
    lines.append("")
    lines += ["OPTIMIZATION STATUS", "-" * 74]
    lines.append(f"Success                       = {result.optimization_success}")
    lines.append(f"Message                       = {result.optimization_message}")
    lines.append(f"Core scale factor             = {result.core_scale:.3f}")
    lines.append(f"Column scale factor           = {result.column_scale:.3f}")
    lines.append("")
    lines += ["REDESIGN SUGGESTIONS", "-" * 74]
    for s in result.redesign_suggestions:
        lines.append(f"- {s}")
    lines.append("")
    lines += ["MATERIAL / QUANTITY SUMMARY", "-" * 74]
    r = result.reinforcement
    lines.append(f"Wall concrete volume           = {r.wall_concrete_volume_m3:,.2f} m³")
    lines.append(f"Column concrete volume         = {r.column_concrete_volume_m3:,.2f} m³")
    lines.append(f"Beam concrete volume           = {r.beam_concrete_volume_m3:,.2f} m³")
    lines.append(f"Slab concrete volume           = {r.slab_concrete_volume_m3:,.2f} m³")
    lines.append(f"Wall steel                     = {r.wall_steel_kg:,.0f} kg")
    lines.append(f"Column steel                   = {r.column_steel_kg:,.0f} kg")
    lines.append(f"Beam steel                     = {r.beam_steel_kg:,.0f} kg")
    lines.append(f"Slab steel                     = {r.slab_steel_kg:,.0f} kg")
    lines.append(f"Total steel                    = {r.total_steel_kg:,.0f} kg")
    lines.append("")
    lines += ["MODAL ANALYSIS", "-" * 74]
    for i, (T, f, rati, c) in enumerate(zip(result.modal_result.periods_s, result.modal_result.frequencies_hz, result.modal_result.effective_mass_ratios, result.modal_result.cumulative_effective_mass_ratios), start=1):
        lines.append(f"Mode {i}: T = {T:.4f} s | f = {f:.4f} Hz | mass ratio = {100*rati:.2f}% | cumulative = {100*c:.2f}%")
    lines.append("")
    lines += ["MESSAGES", "-" * 74]
    for m in result.messages:
        lines.append(f"- {m}")
    return "\n".join(lines)

def _draw_rect(ax, x, y, w, h, color, fill=True, alpha=1.0, lw=1.0, ls="-", ec=None):
    rect = plt.Rectangle((x, y), w, h, facecolor=color if fill else "none", edgecolor=ec if ec else color, linewidth=lw, linestyle=ls, alpha=alpha)
    ax.add_patch(rect)

def plot_plan(inp: BuildingInput, result: DesignResult, zone_name: str):
    core = next(z for z in result.zone_core_results if z.zone.name == zone_name)
    cols = next(z for z in result.zone_column_results if z.zone.name == zone_name)
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.plot([0, inp.plan_x, inp.plan_x, 0, 0], [0, 0, inp.plan_y, inp.plan_y, 0], color="black", linewidth=1.5)
    for i in range(inp.n_bays_x + 1):
        gx = i * inp.bay_x
        ax.plot([gx, gx], [0, inp.plan_y], color="#d9d9d9", linewidth=0.8)
    for j in range(inp.n_bays_y + 1):
        gy = j * inp.bay_y
        ax.plot([0, inp.plan_x], [gy, gy], color="#d9d9d9", linewidth=0.8)
    for i in range(inp.n_bays_x + 1):
        for j in range(inp.n_bays_y + 1):
            px, py = i * inp.bay_x, j * inp.bay_y
            at_lr = i == 0 or i == inp.n_bays_x
            at_bt = j == 0 or j == inp.n_bays_y
            if at_lr and at_bt:
                dx, dy, c = cols.corner_column_x_m, cols.corner_column_y_m, CORNER_COLOR
            elif at_lr or at_bt:
                dx, dy, c = cols.perimeter_column_x_m, cols.perimeter_column_y_m, PERIM_COLOR
            else:
                dx, dy, c = cols.interior_column_x_m, cols.interior_column_y_m, INTERIOR_COLOR
            _draw_rect(ax, px - dx/2, py - dy/2, dx, dy, c, fill=True, alpha=0.95, lw=0.5)
    cx0 = (inp.plan_x - core.core_outer_x) / 2
    cy0 = (inp.plan_y - core.core_outer_y) / 2
    ix0 = (inp.plan_x - core.core_opening_x) / 2
    iy0 = (inp.plan_y - core.core_opening_y) / 2
    _draw_rect(ax, cx0, cy0, core.core_outer_x, core.core_outer_y, CORE_COLOR, fill=False, lw=2.5)
    _draw_rect(ax, ix0, iy0, core.core_opening_x, core.core_opening_y, "#666666", fill=False, lw=1.3, ls="--")
    t = core.wall_thickness
    _draw_rect(ax, cx0, cy0, core.core_outer_x, t, CORE_COLOR, fill=True, alpha=0.25)
    _draw_rect(ax, cx0, cy0 + core.core_outer_y - t, core.core_outer_x, t, CORE_COLOR, fill=True, alpha=0.25)
    _draw_rect(ax, cx0, cy0, t, core.core_outer_y, CORE_COLOR, fill=True, alpha=0.25)
    _draw_rect(ax, cx0 + core.core_outer_x - t, cy0, t, core.core_outer_y, CORE_COLOR, fill=True, alpha=0.25)
    thickness = inp.basement_retaining_wall_thickness if core.retaining_wall_active else core.wall_thickness
    for side, a, b in core.perimeter_wall_segments:
        if side == "top":
            _draw_rect(ax, a, 0, b-a, thickness, PERIM_WALL_COLOR, fill=True, alpha=0.85)
        elif side == "bottom":
            _draw_rect(ax, a, inp.plan_y-thickness, b-a, thickness, PERIM_WALL_COLOR, fill=True, alpha=0.85)
        elif side == "left":
            _draw_rect(ax, 0, a, thickness, b-a, PERIM_WALL_COLOR, fill=True, alpha=0.85)
        else:
            _draw_rect(ax, inp.plan_x-thickness, a, thickness, b-a, PERIM_WALL_COLOR, fill=True, alpha=0.85)
    ax.annotate("", xy=(0,-4), xytext=(inp.plan_x,-4), arrowprops=dict(arrowstyle="<->", lw=1.2))
    ax.text(inp.plan_x/2, -6.0, f"Plan X = {inp.plan_x:.2f} m", ha="center", va="top", fontsize=10)
    ax.annotate("", xy=(inp.plan_x+4,0), xytext=(inp.plan_x+4, inp.plan_y), arrowprops=dict(arrowstyle="<->", lw=1.2))
    ax.text(inp.plan_x+6.0, inp.plan_y/2, f"Plan Y = {inp.plan_y:.2f} m", rotation=90, va="center", fontsize=10)
    info_x, info_y = inp.plan_x + 10, inp.plan_y - 2
    info_lines = [
        f"Core {core.core_outer_x:.2f} x {core.core_outer_y:.2f} m",
        f"Wall t = {core.wall_thickness:.2f} m",
        f"Corner col = {cols.corner_column_x_m:.2f} x {cols.corner_column_y_m:.2f} m",
        f"Perim col = {cols.perimeter_column_x_m:.2f} x {cols.perimeter_column_y_m:.2f} m",
        f"Interior col = {cols.interior_column_x_m:.2f} x {cols.interior_column_y_m:.2f} m",
        f"Beam = {result.beam_width_m:.2f} x {result.beam_depth_m:.2f} m",
        f"Slab t = {result.slab_thickness_m:.2f} m",
        f"Ieff = {core.Ieq_effective_m4:.1f} m^4",
    ]
    for k, txt in enumerate(info_lines):
        ax.text(info_x, info_y - 4*k, txt, fontsize=9, va="top")
    lx, ly = inp.plan_x * 0.40, inp.plan_y * 0.08
    legend = [(CORNER_COLOR, "Strong corner column"), (PERIM_COLOR, "Perimeter column"), (INTERIOR_COLOR, "Interior column"), (CORE_COLOR, "Core shear wall"), (PERIM_WALL_COLOR, "Perimeter wall / retaining wall")]
    for i, (c, label) in enumerate(legend):
        yy = ly + (4 - i) * 4.0
        _draw_rect(ax, lx, yy, 2.2, 2.2, c, fill=True, alpha=0.95)
        ax.text(lx + 3.0, yy + 1.1, label, va="center", fontsize=9)
    ax.set_title(f"{core.zone.name} - Square plan", fontsize=14, fontweight="bold")
    ax.set_xlim(-8, inp.plan_x + 32)
    ax.set_ylim(inp.plan_y + 8, -12)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig

def plot_modes(result: DesignResult):
    mr = result.modal_result
    n_modes = min(5, len(mr.mode_shapes))
    H = result.H_m
    y = np.linspace(0.0, H, mr.n_dof)
    fig, axes = plt.subplots(1, n_modes, figsize=(18, 6))
    if n_modes == 1:
        axes = [axes]
    for m in range(n_modes):
        ax = axes[m]
        phi = np.array(mr.mode_shapes[m], dtype=float)
        ax.axvline(0.0, color="#bbbbbb", linestyle="--", linewidth=1.0)
        for yi in y:
            ax.plot([-1.05, 1.05], [yi, yi], color="#f0f0f0", linewidth=0.8)
        ax.plot(phi, y, color="#0b5ed7", linewidth=2)
        ax.scatter(phi, y, color="#dc3545", s=18, zorder=3)
        ax.set_title(f"Mode {m+1}\nT = {mr.periods_s[m]:.3f} s", fontsize=11, fontweight="bold")
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(0.0, H)
        if m == 0:
            ax.set_ylabel("Height (m)", fontsize=10)
            ax.set_yticks([0.0, H]); ax.set_yticklabels([f"Base\n0.0", f"Roof\n{H:.1f}"])
        else:
            ax.set_yticks([])
        ax.set_xticks([])
    fig.suptitle("First 5 Mode Shapes (MDOF)", fontsize=15, fontweight="bold")
    fig.tight_layout()
    return fig

def modal_mass_df(result: DesignResult):
    mr = result.modal_result
    return {
        "Mode": list(range(1, len(mr.periods_s) + 1)),
        "Period (s)": mr.periods_s,
        "Freq. (Hz)": mr.frequencies_hz,
        "Mass ratio (%)": [100*x for x in mr.effective_mass_ratios],
        "Cumulative (%)": [100*x for x in mr.cumulative_effective_mass_ratios],
    }

st.markdown("""
<style>
.main .block-container {padding-top:0.6rem; padding-bottom:0.6rem; max-width:100%;}
div[data-testid="stHorizontalBlock"] > div {padding-right:0.35rem; padding-left:0.35rem;}
.stButton button {width:100%; font-weight:700; height:3rem;}
.stNumberInput label, .stSelectbox label, .stRadio label {font-size:0.95rem !important;}
</style>
""", unsafe_allow_html=True)

st.title("Final Tall Building Plan Output Tool + MDOF Modes")
st.caption(f"Prepared by {AUTHOR_NAME} | {APP_VERSION}")

if "result" not in st.session_state: st.session_state.result = None
if "report" not in st.session_state: st.session_state.report = ""
if "view_mode" not in st.session_state: st.session_state.view_mode = "plan"

def input_panel():
    st.markdown("### Plan Shape")
    plan_shape = st.radio(" ", ["square", "triangle"], horizontal=True, label_visibility="collapsed")
    st.markdown("### Geometry")
    c1, c2 = st.columns(2)
    with c1:
        n_story = st.number_input("Above-grade stories", 1, 120, 50)
        basement_height = st.number_input("Basement height (m)", 2.5, 6.0, 3.0)
        plan_x = st.number_input("Plan X (m)", 10.0, 300.0, 40.0)
        n_bays_x = st.number_input("Bays in X", 1, 30, 4)
        bay_x = st.number_input("Bay X (m)", 2.0, 20.0, 10.0)
        stairs = st.number_input("Stairs", 0, 20, 2)
    with c2:
        n_basement = st.number_input("Basement stories", 0, 20, 10)
        story_height = st.number_input("Story height (m)", 2.5, 6.0, 3.2)
        plan_y = st.number_input("Plan Y (m)", 10.0, 300.0, 40.0)
        n_bays_y = st.number_input("Bays in Y", 1, 30, 4)
        bay_y = st.number_input("Bay Y (m)", 2.0, 20.0, 10.0)
        elevators = st.number_input("Elevators", 0, 30, 4)
    st.markdown("### Loads/Materials")
    c3, c4 = st.columns(2)
    with c3:
        elevator_area_each = st.number_input("Elevator area each (m²)", 0.0, 20.0, 3.5)
        service_area = st.number_input("Service area (m²)", 0.0, 200.0, 35.0)
        fck = st.number_input("fck (MPa)", 20.0, 100.0, 70.0)
        fy = st.number_input("fy (MPa)", 200.0, 700.0, 420.0)
        DL = st.number_input("DL (kN/m²)", 0.0, 20.0, 3.0)
        slab_finish_allowance = st.number_input("Slab/fit-out allowance", 0.0, 10.0, 1.5)
        wall_cracked_factor = st.number_input("Wall cracked factor", 0.1, 1.0, 0.7)
        basement_retaining_wall_thickness = st.number_input("Basement retaining wall t (m)", 0.1, 2.0, 0.5)
    with c4:
        stair_area_each = st.number_input("Stair area each (m²)", 0.0, 50.0, 20.0)
        corridor_factor = st.number_input("Core circulation factor", 0.5, 3.0, 1.4)
        Ec = st.number_input("Ec (MPa)", 20000.0, 60000.0, 36000.0)
        LL = st.number_input("LL (kN/m²)", 0.0, 20.0, 2.5)
        facade_line_load = st.number_input("Facade line load (kN/m)", 0.0, 50.0, 1.0)
        column_cracked_factor = st.number_input("Column cracked factor", 0.1, 1.0, 0.7)
    st.markdown("### Controls/Final Options")
    c5, c6 = st.columns(2)
    with c5:
        prelim_coeff = st.number_input("Prelim lateral coeff", 0.001, 0.100, 0.015, format="%.3f")
        upper_factor = st.number_input("Upper period limit factor", 1.0, 3.0, 1.4)
        min_wall_t = st.number_input("Min wall thickness (m)", 0.1, 2.0, 0.3)
        min_col = st.number_input("Min column dimension (m)", 0.1, 3.0, 0.7)
        min_beam_b = st.number_input("Min beam width (m)", 0.1, 3.0, 0.4)
        min_slab_t = st.number_input("Min slab thickness (m)", 0.05, 1.0, 0.22)
        max_slend = st.number_input("Max wall slenderness", 1.0, 50.0, 12.0)
        corner_factor = st.number_input("Corner column factor", 1.0, 3.0, 1.3)
        middle_wc = st.number_input("Middle zone wall count", 4, 8, 6)
        wall_rr = st.number_input("Wall rebar ratio", 0.0, 0.1, 0.003, format="%.4f")
        beam_rr = st.number_input("Beam rebar ratio", 0.0, 0.1, 0.015, format="%.4f")
        seismic_mass_factor = st.number_input("Seismic mass factor", 0.1, 2.0, 1.0)
        Ct = st.number_input("Ct", 0.001, 0.200, 0.0488, format="%.4f")
    with c6:
        drift_den = st.number_input("Drift denominator", 100.0, 2000.0, 500.0)
        max_wall_t = st.number_input("Max wall thickness (m)", 0.1, 3.0, 1.2)
        max_col = st.number_input("Max column dimension (m)", 0.1, 5.0, 1.8)
        min_beam_h = st.number_input("Min beam depth (m)", 0.1, 3.0, 0.75)
        max_slab_t = st.number_input("Max slab thickness (m)", 0.05, 1.0, 0.4)
        perimeter_factor = st.number_input("Perimeter column factor", 1.0, 3.0, 1.1)
        lower_wc = st.number_input("Lower zone wall count", 4, 8, 8)
        upper_wc = st.number_input("Upper zone wall count", 4, 8, 4)
        perim_sw_ratio = st.number_input("Perimeter shear wall ratio", 0.0, 1.0, 0.2, format="%.3f")
        col_rr = st.number_input("Column rebar ratio", 0.0, 0.1, 0.01, format="%.4f")
        slab_rr = st.number_input("Slab rebar ratio", 0.0, 0.1, 0.0035, format="%.4f")
        x_period = st.number_input("x exponent", 0.1, 1.5, 0.75)
    return BuildingInput(
        plan_shape=plan_shape, n_story=int(n_story), n_basement=int(n_basement), story_height=float(story_height), basement_height=float(basement_height),
        plan_x=float(plan_x), plan_y=float(plan_y), n_bays_x=int(n_bays_x), n_bays_y=int(n_bays_y), bay_x=float(bay_x), bay_y=float(bay_y),
        stair_count=int(stairs), elevator_count=int(elevators), elevator_area_each=float(elevator_area_each), stair_area_each=float(stair_area_each),
        service_area=float(service_area), corridor_factor=float(corridor_factor), fck=float(fck), Ec=float(Ec), fy=float(fy), DL=float(DL), LL=float(LL),
        slab_finish_allowance=float(slab_finish_allowance), facade_line_load=float(facade_line_load), prelim_lateral_force_coeff=float(prelim_coeff),
        drift_limit_ratio=1.0/float(drift_den), upper_period_limit_factor=float(upper_factor), min_wall_thickness=float(min_wall_t), max_wall_thickness=float(max_wall_t),
        min_column_dim=float(min_col), max_column_dim=float(max_col), min_beam_width=float(min_beam_b), min_beam_depth=float(min_beam_h),
        min_slab_thickness=float(min_slab_t), max_slab_thickness=float(max_slab_t), wall_cracked_factor=float(wall_cracked_factor), column_cracked_factor=float(column_cracked_factor),
        max_story_wall_slenderness=float(max_slend), wall_rebar_ratio=float(wall_rr), column_rebar_ratio=float(col_rr), beam_rebar_ratio=float(beam_rr), slab_rebar_ratio=float(slab_rr),
        seismic_mass_factor=float(seismic_mass_factor), Ct=float(Ct), x_period=float(x_period), perimeter_column_factor=float(perimeter_factor), corner_column_factor=float(corner_factor),
        lower_zone_wall_count=int(lower_wc), middle_zone_wall_count=int(middle_wc), upper_zone_wall_count=int(upper_wc),
        basement_retaining_wall_thickness=float(basement_retaining_wall_thickness), perimeter_shear_wall_ratio=float(perim_sw_ratio)
    )

left, right = st.columns([1.0, 2.0], gap="medium")
with left:
    inp = input_panel()
    b1, b2, b3 = st.columns(3)
    with b1:
        if st.button("ANALYZE"):
            try:
                st.session_state.result = run_design(inp)
                st.session_state.report = build_report(st.session_state.result)
                st.session_state.view_mode = "plan"
            except Exception as e:
                st.error(f"Analysis failed: {e}")
    with b2:
        if st.button("SHOW 5 MODES"):
            try:
                if st.session_state.result is None:
                    st.session_state.result = run_design(inp)
                    st.session_state.report = build_report(st.session_state.result)
                st.session_state.view_mode = "modes"
            except Exception as e:
                st.error(f"Mode display failed: {e}")
    with b3:
        if st.session_state.report:
            st.download_button("SAVE REPORT", data=st.session_state.report.encode("utf-8"), file_name=f"tall_building_report_{AUTHOR_NAME}.txt", mime="text/plain")
        else:
            st.button("SAVE REPORT", disabled=True)

with right:
    zone_name = st.selectbox("Displayed zone:", ["Lower Zone", "Middle Zone", "Upper Zone"], index=0)
    if st.session_state.result is None:
        st.info("Run ANALYZE first.")
    else:
        res = st.session_state.result
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Reference period (s)", f"{res.reference_period_s:.3f}")
        c2.metric("Design target (s)", f"{res.design_target_period_s:.3f}")
        c3.metric("Estimated dynamic (s)", f"{res.estimated_period_s:.3f}")
        c4.metric("Upper limit (s)", f"{res.upper_limit_period_s:.3f}")
        d1, d2, d3 = st.columns(3)
        d1.metric("Period error (%)", f"{100*res.period_error_ratio:.2f}")
        d2.metric("Total stiffness (N/m)", f"{res.K_estimated_N_per_m:.2e}")
        d3.metric("Top drift (m)", f"{res.top_drift_m:.3f}")
        tabs = st.tabs(["Graphic output", "Mass participation", "Report"])
        with tabs[0]:
            if st.session_state.view_mode == "modes":
                st.pyplot(plot_modes(res), use_container_width=True)
            else:
                st.pyplot(plot_plan(inp, res, zone_name), use_container_width=True)
            st.markdown("### Redesign suggestions")
            for s in res.redesign_suggestions:
                st.write(f"- {s}")
        with tabs[1]:
            st.dataframe(modal_mass_df(res), use_container_width=True)
        with tabs[2]:
            st.text_area("", st.session_state.report, height=520, label_visibility="collapsed")
