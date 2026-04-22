
from __future__ import annotations
from dataclasses import dataclass, field
from math import pi, sqrt
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from scipy.optimize import minimize

# ----------------------------- CONFIG -----------------------------
st.set_page_config(page_title="Tall Building Plan Output Tool + MDOF", layout="wide")

AUTHOR_NAME = "Benyamin"
APP_VERSION = "vFinal-adaptive"

G = 9.81
CONCRETE_UNIT_WEIGHT = 25.0  # kN/m3
STEEL_DENSITY = 7850.0

CORNER_COLOR = "#8b0000"
PERIM_COLOR = "#cc5500"
INTERIOR_COLOR = "#4444aa"
CORE_COLOR = "#2e8b57"
PERIM_WALL_COLOR = "#4caf50"


# ----------------------------- DATA MODELS -----------------------------
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
    stair_count: int
    elevator_count: int
    elevator_area_each: float
    stair_area_each: float
    service_area: float
    corridor_factor: float
    fck: float
    Ec: float
    fy: float
    DL: float
    LL: float
    slab_finish_allowance: float
    facade_line_load: float
    prelim_lateral_force_coeff: float
    drift_limit_ratio: float
    upper_period_factor: float
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
    max_story_wall_slenderness: float
    wall_rebar_ratio: float
    column_rebar_ratio: float
    beam_rebar_ratio: float
    slab_rebar_ratio: float
    seismic_mass_factor: float
    Ct: float
    x_period: float
    perimeter_column_factor: float
    corner_column_factor: float
    lower_zone_wall_count: int
    middle_zone_wall_count: int
    upper_zone_wall_count: int
    basement_retaining_wall_thickness: float
    perimeter_shear_wall_ratio: float
    min_core_scale: float = 0.70
    max_core_scale: float = 1.80
    min_column_scale: float = 0.70
    max_column_scale: float = 1.80


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
    floor_area_m2: float
    total_weight_kN: float
    effective_modal_mass_kg: float
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


# ----------------------------- BASIC FUNCTIONS -----------------------------
def total_height(inp: BuildingInput) -> float:
    return inp.n_story * inp.story_height


def floor_area(inp: BuildingInput) -> float:
    return 0.5 * inp.plan_x * inp.plan_y if inp.plan_shape == "triangle" else inp.plan_x * inp.plan_y


def code_reference_period(inp: BuildingInput) -> float:
    return inp.Ct * total_height(inp) ** inp.x_period


def define_three_zones(n_story: int) -> List[ZoneDefinition]:
    z1 = max(1, round(0.30 * n_story))
    z2 = max(z1 + 1, round(0.70 * n_story))
    return [
        ZoneDefinition("Lower Zone", 1, z1),
        ZoneDefinition("Middle Zone", z1 + 1, z2),
        ZoneDefinition("Upper Zone", z2 + 1, n_story),
    ]


def required_opening_area(inp: BuildingInput) -> float:
    return (
        inp.elevator_count * inp.elevator_area_each
        + inp.stair_count * inp.stair_area_each
        + inp.service_area
    ) * inp.corridor_factor


def opening_dimensions(inp: BuildingInput) -> tuple[float, float]:
    area = required_opening_area(inp)
    aspect = 1.6
    oy = sqrt(area / aspect)
    return aspect * oy, oy


def initial_core_dimensions(inp: BuildingInput, opening_x: float, opening_y: float) -> tuple[float, float]:
    outer_x = max(opening_x + 3.0, 0.24 * inp.plan_x)
    outer_y = max(opening_y + 3.0, 0.22 * inp.plan_y)
    return min(outer_x, 0.42 * inp.plan_x), min(outer_y, 0.42 * inp.plan_y)


def slab_thickness_prelim(inp: BuildingInput) -> float:
    span = max(inp.bay_x, inp.bay_y)
    return max(inp.min_slab_thickness, min(inp.max_slab_thickness, span / 28.0))


def beam_size_prelim(inp: BuildingInput) -> tuple[float, float]:
    span = max(inp.bay_x, inp.bay_y)
    depth = max(inp.min_beam_depth, span / 12.0)
    width = max(inp.min_beam_width, 0.45 * depth)
    return width, depth


def active_wall_count_by_zone(inp: BuildingInput, zone_name: str) -> int:
    if zone_name == "Lower Zone":
        return inp.lower_zone_wall_count
    if zone_name == "Middle Zone":
        return inp.middle_zone_wall_count
    return inp.upper_zone_wall_count


def wall_lengths_for_layout(outer_x: float, outer_y: float, wall_count: int) -> List[float]:
    if wall_count == 4:
        return [outer_x, outer_x, outer_y, outer_y]
    if wall_count == 6:
        return [outer_x, outer_x, outer_y, outer_y, 0.45 * outer_x, 0.45 * outer_x]
    return [outer_x, outer_x, outer_y, outer_y, 0.45 * outer_x, 0.45 * outer_x, 0.45 * outer_y, 0.45 * outer_y]


def wall_rect_inertia_about_global_y(length: float, thickness: float, x_centroid: float) -> float:
    i_local = length * thickness**3 / 12.0
    area = length * thickness
    return i_local + area * x_centroid**2


def wall_rect_inertia_about_global_x(length: float, thickness: float, y_centroid: float) -> float:
    i_local = length * thickness**3 / 12.0
    area = length * thickness
    return i_local + area * y_centroid**2


def core_equivalent_inertia(outer_x: float, outer_y: float, lengths: List[float], t: float, wall_count: int) -> float:
    x_side = outer_x / 2.0
    y_side = outer_y / 2.0
    top_len, bot_len, left_len, right_len = lengths[0], lengths[1], lengths[2], lengths[3]
    I_x = 0.0
    I_y = 0.0

    I_x += wall_rect_inertia_about_global_x(top_len, t, +y_side)
    I_x += wall_rect_inertia_about_global_x(bot_len, t, -y_side)
    I_y += (t * top_len**3 / 12.0) + (t * bot_len**3 / 12.0)

    I_y += wall_rect_inertia_about_global_y(left_len, t, -x_side)
    I_y += wall_rect_inertia_about_global_y(right_len, t, +x_side)
    I_x += (t * left_len**3 / 12.0) + (t * right_len**3 / 12.0)

    if wall_count >= 6:
        inner_x = 0.22 * outer_x
        l1, l2 = lengths[4], lengths[5]
        I_y += wall_rect_inertia_about_global_y(l1, t, -inner_x)
        I_y += wall_rect_inertia_about_global_y(l2, t, +inner_x)
        I_x += (t * l1**3 / 12.0) + (t * l2**3 / 12.0)

    if wall_count >= 8:
        inner_y = 0.22 * outer_y
        l3, l4 = lengths[6], lengths[7]
        I_x += wall_rect_inertia_about_global_x(l3, t, -inner_y)
        I_x += wall_rect_inertia_about_global_x(l4, t, +inner_y)
        I_y += (t * l3**3 / 12.0) + (t * l4**3 / 12.0)

    return min(I_x, I_y)


def perimeter_wall_segments_for_square(inp: BuildingInput, zone: ZoneDefinition) -> List[Tuple[str, float, float]]:
    if zone.name == "Lower Zone":
        return [("top", 0.0, inp.plan_x), ("bottom", 0.0, inp.plan_x), ("left", 0.0, inp.plan_y), ("right", 0.0, inp.plan_y)]
    ratio = inp.perimeter_shear_wall_ratio
    lx = inp.plan_x * ratio
    ly = inp.plan_y * ratio
    sx = (inp.plan_x - lx) / 2.0
    sy = (inp.plan_y - ly) / 2.0
    return [("top", sx, sx + lx), ("bottom", sx, sx + lx), ("left", sy, sy + ly), ("right", sy, sy + ly)]


def perimeter_wall_segments_for_triangle(inp: BuildingInput, zone: ZoneDefinition) -> List[Tuple[str, float, float]]:
    if zone.name == "Lower Zone":
        return [("edge1", 0.0, 1.0), ("edge2", 0.0, 1.0), ("edge3", 0.0, 1.0)]
    ratio = inp.perimeter_shear_wall_ratio
    s = (1.0 - ratio) / 2.0
    return [("edge1", s, s + ratio), ("edge2", s, s + ratio), ("edge3", s, s + ratio)]


def directional_column_dims(base_dim: float, corner_factor: float, perimeter_factor: float, plan_x: float, plan_y: float, col_type: str) -> tuple[float, float]:
    aspect = max(plan_x, plan_y) / max(min(plan_x, plan_y), 1e-9)
    if col_type == "interior":
        nominal = base_dim
    elif col_type == "perimeter":
        nominal = base_dim * perimeter_factor
    else:
        nominal = base_dim * corner_factor
    if aspect <= 1.10:
        return nominal, nominal
    major = nominal * 1.15
    minor = nominal * 0.90
    return (major, minor) if plan_x >= plan_y else (minor, major)


# ----------------------------- DESIGN MODEL WITH SCALES -----------------------------
def build_scaled_design(inp: BuildingInput, core_scale: float, column_scale: float):
    H = total_height(inp)
    A_floor = floor_area(inp)
    zones = define_three_zones(inp.n_story)

    slab_t = slab_thickness_prelim(inp)
    beam_b, beam_h = beam_size_prelim(inp)

    opening_x, opening_y = opening_dimensions(inp)
    outer_x0, outer_y0 = initial_core_dimensions(inp, opening_x, opening_y)

    zone_core_results: List[ZoneCoreResult] = []
    for zone in zones:
        wall_count = active_wall_count_by_zone(inp, zone.name)
        t_base = max(inp.min_wall_thickness, min(inp.max_wall_thickness, H / 180.0))
        if zone.name == "Middle Zone":
            t_base *= 0.80
        elif zone.name == "Upper Zone":
            t_base *= 0.60
        t = max(inp.min_wall_thickness, min(inp.max_wall_thickness, t_base * core_scale))
        outer_x = min(outer_x0 * sqrt(core_scale), 0.48 * inp.plan_x)
        outer_y = min(outer_y0 * sqrt(core_scale), 0.48 * inp.plan_y)
        lengths = wall_lengths_for_layout(outer_x, outer_y, wall_count)
        I_gross = core_equivalent_inertia(outer_x, outer_y, lengths, t, wall_count)
        I_eff = inp.wall_cracked_factor * I_gross
        perim = perimeter_wall_segments_for_triangle(inp, zone) if inp.plan_shape == "triangle" else perimeter_wall_segments_for_square(inp, zone)
        zone_core_results.append(
            ZoneCoreResult(
                zone=zone,
                wall_count=wall_count,
                wall_lengths=lengths,
                wall_thickness=t,
                core_outer_x=outer_x,
                core_outer_y=outer_y,
                core_opening_x=opening_x,
                core_opening_y=opening_y,
                Ieq_gross_m4=I_gross,
                Ieq_effective_m4=I_eff,
                story_slenderness=inp.story_height / max(t, 1e-6),
                perimeter_wall_segments=perim,
                retaining_wall_active=(zone.name == "Lower Zone"),
            )
        )

    q = inp.DL + inp.LL + inp.slab_finish_allowance + slab_t * CONCRETE_UNIT_WEIGHT
    sigma_allow = 0.35 * inp.fck * 1000.0

    if inp.plan_shape == "triangle":
        total_columns = (inp.n_bays_x + 1) * (inp.n_bays_y + 1) // 2 + 3
        corner_cols = 3
        perimeter_cols = max(6, inp.n_bays_x + inp.n_bays_y)
        interior_cols = max(0, total_columns - corner_cols - perimeter_cols)
        r2_sum = 0.35 * inp.plan_x * inp.plan_y * max(total_columns, 1)
    else:
        total_columns = (inp.n_bays_x + 1) * (inp.n_bays_y + 1)
        corner_cols = 4
        perimeter_cols = max(0, 2 * (inp.n_bays_x - 1) + 2 * (inp.n_bays_y - 1))
        interior_cols = max(0, total_columns - corner_cols - perimeter_cols)
        plan_center_x = inp.plan_x / 2.0
        plan_center_y = inp.plan_y / 2.0
        r2_sum = 0.0
        for i in range(inp.n_bays_x + 1):
            for j in range(inp.n_bays_y + 1):
                x = i * inp.bay_x
                y = j * inp.bay_y
                r2_sum += (x - plan_center_x) ** 2 + (y - plan_center_y) ** 2

    zone_column_results: List[ZoneColumnResult] = []
    for zone in zones:
        floors_above = inp.n_story - zone.story_start + 1
        n_effective = floors_above + 0.70 * inp.n_basement
        P_interior = inp.bay_x * inp.bay_y * q * n_effective * 1.18
        base_dim = min(inp.max_column_dim, max(inp.min_column_dim, sqrt(P_interior / sigma_allow)))
        base_dim *= column_scale
        base_dim = min(inp.max_column_dim, max(inp.min_column_dim, base_dim))
        ix, iy = directional_column_dims(base_dim, inp.corner_column_factor, inp.perimeter_column_factor, inp.plan_x, inp.plan_y, "interior")
        px, py = directional_column_dims(base_dim, inp.corner_column_factor, inp.perimeter_column_factor, inp.plan_x, inp.plan_y, "perimeter")
        cx, cy = directional_column_dims(base_dim, inp.corner_column_factor, inp.perimeter_column_factor, inp.plan_x, inp.plan_y, "corner")

        A_corner = cx * cy
        A_perim = px * py
        A_inter = ix * iy
        Iavg_corner = max(cx * cy**3 / 12.0, cy * cx**3 / 12.0)
        Iavg_perim = max(px * py**3 / 12.0, py * px**3 / 12.0)
        Iavg_inter = max(ix * iy**3 / 12.0, iy * ix**3 / 12.0)
        I_avg = (corner_cols * Iavg_corner + perimeter_cols * Iavg_perim + interior_cols * Iavg_inter) / max(total_columns, 1)
        A_avg = (corner_cols * A_corner + perimeter_cols * A_perim + interior_cols * A_inter) / max(total_columns, 1)
        I_col_group = inp.column_cracked_factor * (I_avg * max(total_columns, 1) + A_avg * r2_sum)

        zone_column_results.append(
            ZoneColumnResult(
                zone=zone,
                corner_column_x_m=cx,
                corner_column_y_m=cy,
                perimeter_column_x_m=px,
                perimeter_column_y_m=py,
                interior_column_x_m=ix,
                interior_column_y_m=iy,
                I_col_group_effective_m4=I_col_group,
            )
        )

    # quantities and weight update
    n_total_levels = inp.n_story + inp.n_basement
    total_floor_area = A_floor * n_total_levels
    wall_concrete = 0.0
    for zc in zone_core_results:
        wall_concrete += sum(zc.wall_lengths) * zc.wall_thickness * (zc.zone.n_stories * inp.story_height)
        if inp.plan_shape == "square":
            for _, a, b in zc.perimeter_wall_segments:
                wall_concrete += (b - a) * zc.wall_thickness * (zc.zone.n_stories * inp.story_height)
        else:
            edge_lengths = {"edge1": inp.plan_x, "edge2": inp.plan_y, "edge3": sqrt(inp.plan_x**2 + inp.plan_y**2)}
            for side, a, b in zc.perimeter_wall_segments:
                wall_concrete += edge_lengths[side] * (b - a) * zc.wall_thickness * (zc.zone.n_stories * inp.story_height)

    column_concrete = 0.0
    for zc in zone_column_results:
        zone_height = zc.zone.n_stories * inp.story_height
        column_concrete += (
            corner_cols * zc.corner_column_x_m * zc.corner_column_y_m * zone_height
            + perimeter_cols * zc.perimeter_column_x_m * zc.perimeter_column_y_m * zone_height
            + interior_cols * zc.interior_column_x_m * zc.interior_column_y_m * zone_height
        )

    beam_lines_per_floor = max(1, inp.n_bays_y * (inp.n_bays_x + 1) + inp.n_bays_x * (inp.n_bays_y + 1))
    avg_span = 0.5 * (inp.bay_x + inp.bay_y)
    total_beam_length = beam_lines_per_floor * avg_span * n_total_levels
    beam_concrete = beam_b * beam_h * total_beam_length
    slab_concrete = total_floor_area * slab_t

    wall_steel = wall_concrete * inp.wall_rebar_ratio * STEEL_DENSITY
    column_steel = column_concrete * inp.column_rebar_ratio * STEEL_DENSITY
    beam_steel = beam_concrete * inp.beam_rebar_ratio * STEEL_DENSITY
    slab_steel = slab_concrete * inp.slab_rebar_ratio * STEEL_DENSITY
    total_steel = wall_steel + column_steel + beam_steel + slab_steel

    reinforcement = ReinforcementEstimate(
        wall_concrete_volume_m3=wall_concrete,
        column_concrete_volume_m3=column_concrete,
        beam_concrete_volume_m3=beam_concrete,
        slab_concrete_volume_m3=slab_concrete,
        wall_steel_kg=wall_steel,
        column_steel_kg=column_steel,
        beam_steel_kg=beam_steel,
        slab_steel_kg=slab_steel,
        total_steel_kg=total_steel,
    )

    perimeter = (
        2.0 * (inp.plan_x + inp.plan_y)
        if inp.plan_shape == "square"
        else (inp.plan_x + inp.plan_y + sqrt(inp.plan_x**2 + inp.plan_y**2))
    )
    service_weight = (inp.DL + inp.LL + inp.slab_finish_allowance) * A_floor * n_total_levels
    concrete_weight = CONCRETE_UNIT_WEIGHT * (wall_concrete + column_concrete + beam_concrete + slab_concrete)
    facade_weight = inp.facade_line_load * perimeter * inp.n_story
    total_weight_kN = (service_weight + concrete_weight + facade_weight) * inp.seismic_mass_factor

    H = total_height(inp)
    E = inp.Ec * 1e6
    total_flex_factor_core = 0.0
    for zc in zone_core_results:
        hi = zc.zone.n_stories * inp.story_height
        perimeter_bonus = 1.0 + 0.20 * len(zc.perimeter_wall_segments)
        total_flex_factor_core += (hi / H) / max(E * zc.Ieq_effective_m4 * perimeter_bonus, 1e-9)
    EI_core = 1.0 / max(total_flex_factor_core, 1e-18)
    K_core = 3.0 * EI_core / H**3

    total_flex_factor_col = 0.0
    for zc in zone_column_results:
        hi = zc.zone.n_stories * inp.story_height
        total_flex_factor_col += (hi / H) / max(E * zc.I_col_group_effective_m4, 1e-9)
    EI_col = 1.0 / max(total_flex_factor_col, 1e-18)
    K_col = 3.0 * EI_col / H**3

    K_est = K_core + K_col
    return zone_core_results, zone_column_results, reinforcement, slab_t, beam_b, beam_h, total_weight_kN, K_est


def build_story_masses(inp: BuildingInput, total_weight_kN_value: float) -> List[float]:
    W_story_kN = total_weight_kN_value / max(inp.n_story, 1)
    m_story = (W_story_kN * 1000.0) / G
    return [m_story for _ in range(inp.n_story)]


def build_story_stiffnesses(inp: BuildingInput, K_total: float) -> List[float]:
    n = inp.n_story
    raw = []
    for i in range(n):
        r = i / max(n - 1, 1)
        raw.append(1.35 - 0.55 * r)
    inv_sum = sum(1.0 / a for a in raw)
    c = K_total * inv_sum
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
    gammas, meffs, ratios, cumulative = [], [], [], []
    cum = 0.0
    mode_shapes = []

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
        cumulative.append(cum)

    return ModalResult(
        n_dof=len(masses),
        periods_s=periods,
        frequencies_hz=freqs,
        mode_shapes=mode_shapes,
        story_masses_kg=masses,
        story_stiffness_N_per_m=k_stories,
        modal_participation_factors=gammas,
        effective_modal_masses_kg=meffs,
        effective_mass_ratios=ratios,
        cumulative_effective_mass_ratios=cumulative,
    )


def adaptive_target_period(T_ref: float, T_upper: float, eta1: float) -> float:
    alpha = 0.35 + 0.25 * (1.0 - eta1)
    alpha = max(0.20, min(0.80, alpha))
    return T_ref + alpha * (T_upper - T_ref)


def generate_redesign_suggestions(inp: BuildingInput, T_est: float, T_target: float, T_limit: float, drift_ratio: float, drift_limit: float, core_scale: float, column_scale: float):
    suggestions = []
    governing_issue = "OK"
    period_ratio = T_est / max(T_target, 1e-9)

    if T_est > T_limit:
        governing_issue = "Estimated period exceeds the upper limit."
        suggestions += [
            "Increase global lateral stiffness by enlarging the core walls.",
            "Increase perimeter and corner column dimensions.",
            "Increase the number of active shear walls in middle and upper zones.",
            "Consider outriggers or belt trusses for tall-building stiffness enhancement.",
        ]
    elif period_ratio > 1.10:
        governing_issue = "Estimated period is larger than the design target."
        suggestions += [
            "Increase wall thicknesses and/or the effective core footprint.",
            "Increase corner and perimeter columns.",
            "Reduce excessive flexibility in the structural layout.",
        ]
    elif period_ratio < 0.90:
        governing_issue = "Estimated period is smaller than the design target."
        suggestions += [
            "The system may be overly stiff and uneconomical.",
            "Reduce wall thicknesses or column dimensions if drift remains acceptable.",
        ]

    if drift_ratio > drift_limit:
        governing_issue = "Drift exceeds allowable limit."
        suggestions += [
            "Increase the lower-story stiffness.",
            "Increase core stiffness and perimeter column stiffness.",
            "Check the wall distribution along the height.",
        ]

    if core_scale >= inp.max_core_scale * 0.98:
        suggestions.append("Core scale is near its upper bound.")
    if column_scale >= inp.max_column_scale * 0.98:
        suggestions.append("Column scale is near its upper bound.")
    if not suggestions:
        suggestions.append("Structural system appears preliminarily adequate.")

    return governing_issue, suggestions


# ----------------------------- OPTIMIZATION -----------------------------
def run_design(inp: BuildingInput) -> DesignResult:
    H = total_height(inp)
    A_floor = floor_area(inp)
    T_ref = code_reference_period(inp)
    T_upper = inp.upper_period_factor * T_ref

    # initial guess
    x0 = np.array([1.0, 1.0])

    def evaluate(scales):
        core_scale, col_scale = float(scales[0]), float(scales[1])
        (
            zone_cores, zone_cols, reinforcement,
            slab_t, beam_b, beam_h, W_total, K_est
        ) = build_scaled_design(inp, core_scale, col_scale)

        modal = solve_mdof_modes(inp, W_total, K_est, n_modes=5)
        eta1 = modal.effective_mass_ratios[0] if modal.effective_mass_ratios else 0.7
        T_target = adaptive_target_period(T_ref, T_upper, eta1)
        T_est = modal.periods_s[0]
        period_error = abs(T_est - T_target) / max(T_target, 1e-9)
        top_drift = inp.prelim_lateral_force_coeff * W_total * 1000.0 / max(K_est, 1e-9)
        drift_ratio = top_drift / max(H, 1e-9)
        return {
            "zone_cores": zone_cores,
            "zone_cols": zone_cols,
            "reinforcement": reinforcement,
            "slab_t": slab_t,
            "beam_b": beam_b,
            "beam_h": beam_h,
            "W_total": W_total,
            "K_est": K_est,
            "modal": modal,
            "eta1": eta1,
            "T_target": T_target,
            "T_est": T_est,
            "period_error": period_error,
            "top_drift": top_drift,
            "drift_ratio": drift_ratio,
        }

    def objective(scales):
        res = evaluate(scales)
        weight_norm = res["W_total"] / 1e6
        period_term = 200.0 * res["period_error"]**2
        drift_term = 100.0 * (res["drift_ratio"] / max(inp.drift_limit_ratio, 1e-9))**2
        return weight_norm + period_term + 0.05 * drift_term

    cons = [
        {"type": "ineq", "fun": lambda s: inp.drift_limit_ratio - evaluate(s)["drift_ratio"]},
        {"type": "ineq", "fun": lambda s: T_upper - evaluate(s)["T_est"]},
        {"type": "ineq", "fun": lambda s: 0.08 - evaluate(s)["period_error"]},  # 8% tolerance
    ]
    bnds = [(inp.min_core_scale, inp.max_core_scale), (inp.min_column_scale, inp.max_column_scale)]

    opt = minimize(objective, x0, method="SLSQP", bounds=bnds, constraints=cons, options={"maxiter": 80, "ftol": 1e-6})
    if not opt.success:
        # fallback: best feasible-ish point from optimization result or x0
        scales = np.clip(opt.x if hasattr(opt, "x") else x0, [b[0] for b in bnds], [b[1] for b in bnds])
    else:
        scales = opt.x

    res = evaluate(scales)
    core_scale = float(scales[0])
    col_scale = float(scales[1])

    period_ok = res["T_est"] <= T_upper and res["period_error"] <= 0.08
    drift_ok = res["drift_ratio"] <= inp.drift_limit_ratio

    messages = []
    if period_ok:
        messages.append("Dynamic period is within the target tolerance and the upper limit.")
    else:
        messages.append("Dynamic period does not satisfy the target tolerance and/or the upper limit.")
    if drift_ok:
        messages.append("Drift check is acceptable.")
    else:
        messages.append("Drift exceeds the allowable limit.")

    governing_issue, redesign_suggestions = generate_redesign_suggestions(
        inp, res["T_est"], res["T_target"], T_upper, res["drift_ratio"], inp.drift_limit_ratio, core_scale, col_scale
    )

    return DesignResult(
        H_m=H,
        floor_area_m2=A_floor,
        total_weight_kN=res["W_total"],
        effective_modal_mass_kg=sum(res["modal"].effective_modal_masses_kg),
        reference_period_s=T_ref,
        design_target_period_s=res["T_target"],
        upper_limit_period_s=T_upper,
        estimated_period_s=res["T_est"],
        period_error_ratio=res["period_error"],
        period_ok=period_ok,
        drift_ok=drift_ok,
        K_estimated_N_per_m=res["K_est"],
        top_drift_m=res["top_drift"],
        drift_ratio=res["drift_ratio"],
        zone_core_results=res["zone_cores"],
        zone_column_results=res["zone_cols"],
        slab_thickness_m=res["slab_t"],
        beam_width_m=res["beam_b"],
        beam_depth_m=res["beam_h"],
        reinforcement=res["reinforcement"],
        optimization_success=bool(opt.success),
        optimization_message=str(opt.message),
        core_scale=core_scale,
        column_scale=col_scale,
        messages=messages,
        redesign_suggestions=redesign_suggestions,
        governing_issue=governing_issue,
        modal_result=res["modal"],
    )


# ----------------------------- REPORT -----------------------------
def build_report(result: DesignResult) -> str:
    lines = []
    lines.append("GLOBAL RESPONSE")
    lines.append("-" * 74)
    lines.append(f"Reference period               = {result.reference_period_s:.3f} s")
    lines.append(f"Design target period           = {result.design_target_period_s:.3f} s")
    lines.append(f"Estimated dynamic period       = {result.estimated_period_s:.3f} s")
    lines.append(f"Upper limit period             = {result.upper_limit_period_s:.3f} s")
    lines.append(f"Period error ratio             = {100*result.period_error_ratio:.2f} %")
    lines.append(f"Period check                   = {'OK' if result.period_ok else 'NOT OK'}")
    lines.append(f"Total stiffness                = {result.K_estimated_N_per_m:,.3e} N/m")
    lines.append(f"Estimated top drift            = {result.top_drift_m:.3f} m")
    lines.append(f"Estimated drift ratio          = {result.drift_ratio:.5f}")
    lines.append(f"Drift check                    = {'OK' if result.drift_ok else 'NOT OK'}")
    lines.append(f"Total structural weight        = {result.total_weight_kN:,.0f} kN")
    lines.append("")
    lines.append("PRIMARY MEMBER OUTPUT")
    lines.append("-" * 74)
    lines.append(f"Beam size (b x h)              = {result.beam_width_m:.2f} x {result.beam_depth_m:.2f} m")
    lines.append(f"Slab thickness                 = {result.slab_thickness_m:.2f} m")
    lines.append("")
    lines.append("ZONE-BY-ZONE COLUMN DIMENSIONS")
    lines.append("-" * 74)
    for zc in result.zone_column_results:
        lines.append(f"{zc.zone.name}:")
        lines.append(f"  Corner columns              = {zc.corner_column_x_m:.2f} x {zc.corner_column_y_m:.2f} m")
        lines.append(f"  Perimeter columns           = {zc.perimeter_column_x_m:.2f} x {zc.perimeter_column_y_m:.2f} m")
        lines.append(f"  Interior columns            = {zc.interior_column_x_m:.2f} x {zc.interior_column_y_m:.2f} m")
        lines.append(f"  Column group Ieff           = {zc.I_col_group_effective_m4:,.2f} m^4")
    lines.append("")
    lines.append("ZONE-BY-ZONE WALL / CORE OUTPUT")
    lines.append("-" * 74)
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
    lines.append("OPTIMIZATION STATUS")
    lines.append("-" * 74)
    lines.append(f"Success                       = {result.optimization_success}")
    lines.append(f"Message                       = {result.optimization_message}")
    lines.append(f"Core scale factor             = {result.core_scale:.3f}")
    lines.append(f"Column scale factor           = {result.column_scale:.3f}")
    lines.append("")
    lines.append("REDESIGN SUGGESTIONS")
    lines.append("-" * 74)
    for s in result.redesign_suggestions:
        lines.append(f"- {s}")
    lines.append("")
    lines.append("MODAL ANALYSIS")
    lines.append("-" * 74)
    for i, (T, f, rati, c) in enumerate(
        zip(
            result.modal_result.periods_s,
            result.modal_result.frequencies_hz,
            result.modal_result.effective_mass_ratios,
            result.modal_result.cumulative_effective_mass_ratios,
        ),
        start=1,
    ):
        lines.append(f"Mode {i}: T = {T:.4f} s | f = {f:.4f} Hz | mass ratio = {100*rati:.2f}% | cumulative = {100*c:.2f}%")
    lines.append("")
    lines.append("MESSAGES")
    lines.append("-" * 74)
    for m in result.messages:
        lines.append(f"- {m}")
    return "\n".join(lines)


# ----------------------------- PLOTTING -----------------------------
def plot_mode_shapes(result: DesignResult):
    mr = result.modal_result
    n_modes = min(5, len(mr.mode_shapes))
    H = result.H_m
    y = np.linspace(0.0, H, mr.n_dof)
    fig, axes = plt.subplots(1, n_modes, figsize=(18, 6))
    if n_modes == 1:
        axes = [axes]
    for i in range(n_modes):
        ax = axes[i]
        phi = np.array(mr.mode_shapes[i], dtype=float)
        phi = phi / max(np.max(np.abs(phi)), 1e-9)
        if phi[-1] < 0:
            phi = -phi
        ax.axvline(0.0, color="#bbbbbb", linestyle="--", linewidth=1.0)
        for yi in y:
            ax.plot([-1.05, 1.05], [yi, yi], color="#f0f0f0", linewidth=0.8)
        ax.plot(phi, y, color="#0b5ed7", linewidth=2)
        ax.scatter(phi, y, color="#dc3545", s=18, zorder=3)
        ax.set_title(f"Mode {i+1}\nT = {mr.periods_s[i]:.3f} s", fontsize=11, fontweight="bold")
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(0.0, H)
        if i == 0:
            ax.set_ylabel("Height (m)")
            ax.set_yticks([0.0, H])
            ax.set_yticklabels([f"Base\n0.0", f"Roof\n{H:.1f}"])
        else:
            ax.set_yticks([])
        ax.set_xticks([])
    fig.suptitle("First 5 Mode Shapes (MDOF)", fontsize=15, fontweight="bold")
    fig.tight_layout()
    return fig


def _draw_rect(ax, x, y, w, h, color, fill=True, alpha=1.0, lw=1.0, ls="-", ec=None):
    rect = plt.Rectangle((x, y), w, h, facecolor=color if fill else "none", edgecolor=ec if ec else color, linewidth=lw, linestyle=ls, alpha=alpha)
    ax.add_patch(rect)


def plot_plan(inp: BuildingInput, result: DesignResult, zone_name: str):
    core = next(z for z in result.zone_core_results if z.zone.name == zone_name)
    cols = next(z for z in result.zone_column_results if z.zone.name == zone_name)

    fig, ax = plt.subplots(figsize=(13, 7))
    if inp.plan_shape == "square":
        ax.plot([0, inp.plan_x, inp.plan_x, 0, 0], [0, 0, inp.plan_y, inp.plan_y, 0], color="black", linewidth=1.5)
        for i in range(inp.n_bays_x + 1):
            gx = i * inp.bay_x
            ax.plot([gx, gx], [0, inp.plan_y], color="#d9d9d9", linewidth=0.8)
        for j in range(inp.n_bays_y + 1):
            gy = j * inp.bay_y
            ax.plot([0, inp.plan_x], [gy, gy], color="#d9d9d9", linewidth=0.8)

        for i in range(inp.n_bays_x + 1):
            for j in range(inp.n_bays_y + 1):
                px = i * inp.bay_x
                py = j * inp.bay_y
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
        ax.text(cx0 + core.core_outer_x / 2, cy0 - 1.0, "CORE", ha="center", fontsize=10, fontweight="bold")
        ax.text(ix0 + core.core_opening_x / 2, iy0 + core.core_opening_y / 2, "OPENING", ha="center", va="center", fontsize=9)

        thickness = inp.basement_retaining_wall_thickness if core.retaining_wall_active else core.wall_thickness
        for side, a, b in core.perimeter_wall_segments:
            if side == "top":
                _draw_rect(ax, a, 0, b-a, thickness, PERIM_WALL_COLOR, fill=True, alpha=0.85)
            elif side == "bottom":
                _draw_rect(ax, a, inp.plan_y - thickness, b-a, thickness, PERIM_WALL_COLOR, fill=True, alpha=0.85)
            elif side == "left":
                _draw_rect(ax, 0, a, thickness, b-a, PERIM_WALL_COLOR, fill=True, alpha=0.85)
            else:
                _draw_rect(ax, inp.plan_x - thickness, a, thickness, b-a, PERIM_WALL_COLOR, fill=True, alpha=0.85)

        ax.annotate("", xy=(0, -4), xytext=(inp.plan_x, -4), arrowprops=dict(arrowstyle="<->", lw=1.2))
        ax.text(inp.plan_x / 2, -6.0, f"Plan X = {inp.plan_x:.2f} m", ha="center", va="top", fontsize=10)
        ax.annotate("", xy=(inp.plan_x + 4, 0), xytext=(inp.plan_x + 4, inp.plan_y), arrowprops=dict(arrowstyle="<->", lw=1.2))
        ax.text(inp.plan_x + 6.0, inp.plan_y / 2, f"Plan Y = {inp.plan_y:.2f} m", rotation=90, va="center", fontsize=10)

        info_x = inp.plan_x + 10
        info_y = inp.plan_y - 2
        info_lines = [
            f"Core outer = {core.core_outer_x:.2f} x {core.core_outer_y:.2f} m",
            f"Core opening = {core.core_opening_x:.2f} x {core.core_opening_y:.2f} m",
            f"Wall thickness = {core.wall_thickness:.2f} m",
            f"Corner col = {cols.corner_column_x_m:.2f} x {cols.corner_column_y_m:.2f} m",
            f"Perim col = {cols.perimeter_column_x_m:.2f} x {cols.perimeter_column_y_m:.2f} m",
            f"Interior col = {cols.interior_column_x_m:.2f} x {cols.interior_column_y_m:.2f} m",
            f"Beam = {result.beam_width_m:.2f} x {result.beam_depth_m:.2f} m",
            f"Slab t = {result.slab_thickness_m:.2f} m",
        ]
        for k, txt in enumerate(info_lines):
            ax.text(info_x, info_y - 4 * k, txt, fontsize=9, va="top")

        lx = inp.plan_x * 0.42
        ly = inp.plan_y * 0.12
        legend = [
            (CORNER_COLOR, "Strong corner column"),
            (PERIM_COLOR, "Perimeter column"),
            (INTERIOR_COLOR, "Interior column"),
            (CORE_COLOR, "Core shear wall"),
            (PERIM_WALL_COLOR, "Perimeter wall / retaining wall"),
        ]
        for i, (c, label) in enumerate(legend):
            yy = ly + (4 - i) * 4.0
            _draw_rect(ax, lx, yy, 2.2, 2.2, c, fill=True, alpha=0.95)
            ax.text(lx + 3.0, yy + 1.1, label, va="center", fontsize=9)

        ax.set_title(f"{core.zone.name} - Square plan", fontsize=14, fontweight="bold")
        ax.set_xlim(-8, inp.plan_x + 34)
        ax.set_ylim(inp.plan_y + 8, -12)
    else:
        pts = np.array([[0, inp.plan_y], [inp.plan_x / 2, 0], [inp.plan_x, inp.plan_y], [0, inp.plan_y]])
        ax.plot(pts[:, 0], pts[:, 1], color="black", linewidth=1.5)
        ax.set_title(f"{core.zone.name} - Triangular plan", fontsize=14, fontweight="bold")
        ax.set_xlim(-8, inp.plan_x + 34)
        ax.set_ylim(inp.plan_y + 8, -12)

    ax.set_aspect("equal")
    ax.axis("off")
    return fig


# ----------------------------- UI -----------------------------
def input_panel() -> BuildingInput:
    st.markdown("### Plan Shape")
    plan_shape = st.radio(" ", ["square", "triangle"], horizontal=True, label_visibility="collapsed")

    st.markdown("### Geometry")
    c1, c2 = st.columns(2)
    with c1:
        n_story = st.number_input("Above-grade stories", 1, 120, 50)
        basement_height = st.number_input("Basement height (m)", 2.5, 6.0, 3.0)
        plan_x = st.number_input("Plan X (m)", 10.0, 300.0, 80.0)
        n_bays_x = st.number_input("Bays in X", 1, 30, 8)
        bay_x = st.number_input("Bay X (m)", 2.0, 20.0, 10.0)
        stair_count = st.number_input("Stairs", 0, 20, 2)
    with c2:
        n_basement = st.number_input("Basement stories", 0, 20, 10)
        story_height = st.number_input("Story height (m)", 2.5, 6.0, 3.2)
        plan_y = st.number_input("Plan Y (m)", 10.0, 300.0, 80.0)
        n_bays_y = st.number_input("Bays in Y", 1, 30, 8)
        bay_y = st.number_input("Bay Y (m)", 2.0, 20.0, 10.0)
        elevator_count = st.number_input("Elevators", 0, 30, 4)

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
        prelim_lateral_force_coeff = st.number_input("Prelim lateral coeff", 0.001, 0.100, 0.015, format="%.3f")
        upper_period_factor = st.number_input("Upper period factor", 1.0, 3.0, 2.0, format="%.2f")
        min_wall_thickness = st.number_input("Min wall thickness (m)", 0.1, 2.0, 0.3)
        min_column_dim = st.number_input("Min column dimension (m)", 0.1, 3.0, 0.7)
        min_beam_width = st.number_input("Min beam width (m)", 0.1, 3.0, 0.4)
        min_slab_thickness = st.number_input("Min slab thickness (m)", 0.05, 1.0, 0.22)
        max_story_wall_slenderness = st.number_input("Max wall slenderness", 1.0, 50.0, 12.0)
        corner_column_factor = st.number_input("Corner column factor", 1.0, 3.0, 1.3)
        middle_zone_wall_count = st.number_input("Middle zone wall count", 4, 8, 6)
        wall_rebar_ratio = st.number_input("Wall rebar ratio", 0.0, 0.1, 0.003, format="%.4f")
        beam_rebar_ratio = st.number_input("Beam rebar ratio", 0.0, 0.1, 0.015, format="%.4f")
        seismic_mass_factor = st.number_input("Seismic mass factor", 0.1, 2.0, 1.0)
        Ct = st.number_input("Ct", 0.001, 0.200, 0.0488, format="%.4f")
    with c6:
        drift_denominator = st.number_input("Drift denominator", 100.0, 2000.0, 500.0)
        max_wall_thickness = st.number_input("Max wall thickness (m)", 0.1, 3.0, 1.2)
        max_column_dim = st.number_input("Max column dimension (m)", 0.1, 5.0, 1.8)
        min_beam_depth = st.number_input("Min beam depth (m)", 0.1, 3.0, 0.75)
        max_slab_thickness = st.number_input("Max slab thickness (m)", 0.05, 1.0, 0.4)
        perimeter_column_factor = st.number_input("Perimeter column factor", 1.0, 3.0, 1.1)
        lower_zone_wall_count = st.number_input("Lower zone wall count", 4, 8, 8)
        upper_zone_wall_count = st.number_input("Upper zone wall count", 4, 8, 4)
        perimeter_shear_wall_ratio = st.number_input("Perimeter shear wall ratio", 0.0, 1.0, 0.2, format="%.3f")
        column_rebar_ratio = st.number_input("Column rebar ratio", 0.0, 0.1, 0.01, format="%.4f")
        slab_rebar_ratio = st.number_input("Slab rebar ratio", 0.0, 0.1, 0.0035, format="%.4f")
        x_period = st.number_input("x exponent", 0.1, 1.5, 0.75)

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
        corridor_factor=float(corridor_factor),
        fck=float(fck),
        Ec=float(Ec),
        fy=float(fy),
        DL=float(DL),
        LL=float(LL),
        slab_finish_allowance=float(slab_finish_allowance),
        facade_line_load=float(facade_line_load),
        prelim_lateral_force_coeff=float(prelim_lateral_force_coeff),
        drift_limit_ratio=1.0 / float(drift_denominator),
        upper_period_factor=float(upper_period_factor),
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
        max_story_wall_slenderness=float(max_story_wall_slenderness),
        wall_rebar_ratio=float(wall_rebar_ratio),
        column_rebar_ratio=float(column_rebar_ratio),
        beam_rebar_ratio=float(beam_rebar_ratio),
        slab_rebar_ratio=float(slab_rebar_ratio),
        seismic_mass_factor=float(seismic_mass_factor),
        Ct=float(Ct),
        x_period=float(x_period),
        perimeter_column_factor=float(perimeter_column_factor),
        corner_column_factor=float(corner_column_factor),
        lower_zone_wall_count=int(lower_zone_wall_count),
        middle_zone_wall_count=int(middle_zone_wall_count),
        upper_zone_wall_count=int(upper_zone_wall_count),
        basement_retaining_wall_thickness=float(basement_retaining_wall_thickness),
        perimeter_shear_wall_ratio=float(perimeter_shear_wall_ratio),
    )


def mass_participation_dataframe(mr: ModalResult):
    import pandas as pd
    return pd.DataFrame(
        {
            "Mode": np.arange(1, len(mr.periods_s) + 1),
            "Period (s)": mr.periods_s,
            "Frequency (Hz)": mr.frequencies_hz,
            "Mass ratio (%)": [100 * r for r in mr.effective_mass_ratios],
            "Cumulative (%)": [100 * c for c in mr.cumulative_effective_mass_ratios],
        }
    )


st.markdown(
    """
    <style>
    .main .block-container {padding-top: 0.6rem; max-width: 100%;}
    .stButton button {width: 100%; font-weight: 700; height: 3rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Final Tall Building Plan Output Tool + MDOF Modes")
st.caption(f"Prepared by {AUTHOR_NAME} | {APP_VERSION}")

if "result" not in st.session_state:
    st.session_state.result = None
if "report" not in st.session_state:
    st.session_state.report = ""
if "view_mode" not in st.session_state:
    st.session_state.view_mode = "plan"

left_col, right_col = st.columns([1.0, 2.2], gap="medium")

with left_col:
    inp = input_panel()
    b1, b2, b3 = st.columns(3)
    with b1:
        if st.button("ANALYZE"):
            try:
                res = run_design(inp)
                st.session_state.result = res
                st.session_state.report = build_report(res)
                st.session_state.view_mode = "plan"
            except Exception as e:
                st.error(f"Analysis failed: {e}")
    with b2:
        if st.button("SHOW 5 MODES"):
            try:
                if st.session_state.result is None:
                    res = run_design(inp)
                    st.session_state.result = res
                    st.session_state.report = build_report(res)
                st.session_state.view_mode = "modes"
            except Exception as e:
                st.error(f"Mode display failed: {e}")
    with b3:
        if st.session_state.report:
            st.download_button("SAVE REPORT", data=st.session_state.report.encode("utf-8"), file_name="tall_building_report.txt", mime="text/plain")
        else:
            st.button("SAVE REPORT", disabled=True)

with right_col:
    zone_name = st.selectbox("Displayed zone:", ["Lower Zone", "Middle Zone", "Upper Zone"], index=0)
    if st.session_state.result is not None:
        r = st.session_state.result
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Reference period (s)", f"{r.reference_period_s:.3f}")
        c2.metric("Design target (s)", f"{r.design_target_period_s:.3f}")
        c3.metric("Estimated dynamic (s)", f"{r.estimated_period_s:.3f}")
        c4.metric("Upper limit (s)", f"{r.upper_limit_period_s:.3f}")
        d1, d2, d3 = st.columns(3)
        d1.metric("Period error (%)", f"{100*r.period_error_ratio:.2f}")
        d2.metric("Total stiffness (N/m)", f"{r.K_estimated_N_per_m:,.2e}")
        d3.metric("Top drift (m)", f"{r.top_drift_m:.3f}")

        tabs = st.tabs(["Graphic output", "Mass participation", "Report"])
        with tabs[0]:
            if st.session_state.view_mode == "modes":
                st.pyplot(plot_mode_shapes(r), use_container_width=True)
            else:
                st.pyplot(plot_plan(inp, r, zone_name), use_container_width=True)
            st.markdown("### Redesign suggestions")
            for s in r.redesign_suggestions:
                st.write(f"- {s}")
        with tabs[1]:
            st.dataframe(mass_participation_dataframe(r.modal_result), use_container_width=True)
        with tabs[2]:
            st.text_area("", st.session_state.report, height=520, label_visibility="collapsed")
    else:
        st.info("Run the analysis first.")
