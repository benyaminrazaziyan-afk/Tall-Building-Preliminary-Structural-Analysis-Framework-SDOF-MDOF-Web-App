from __future__ import annotations

from dataclasses import dataclass, field, asdict
from math import pi, sqrt
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from scipy.optimize import minimize

# ----------------------------- CONFIG -----------------------------
st.set_page_config(
    page_title="Tall Building Preliminary Structural Analysis",
    layout="wide",
    initial_sidebar_state="expanded",
)

AUTHOR_NAME = "Benyamin"
APP_VERSION = "v3.1-MDOF-Iterative-Complete"
G = 9.81
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

    stair_count: int = 2
    elevator_count: int = 4
    elevator_area_each: float = 3.5
    stair_area_each: float = 20.0
    service_area: float = 35.0
    corridor_factor: float = 1.40

    fck: float = 70.0
    Ec: float = 36000.0
    fy: float = 420.0

    DL: float = 3.0
    LL: float = 2.5
    slab_finish_allowance: float = 1.5
    facade_line_load: float = 1.0

    prelim_lateral_force_coeff: float = 0.015
    drift_limit_ratio: float = 1 / 500

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
    upper_period_factor: float = 1.20
    target_position_factor: float = 0.85

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
    P_corner_kN: float
    P_perimeter_kN: float
    P_interior_kN: float
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
    effective_mass_ratios: List[float]
    cumulative_effective_mass_ratios: List[float]


@dataclass
class IterationLog:
    iteration: int
    core_scale: float
    column_scale: float
    T_estimated: float
    T_target: float
    error_percent: float
    total_weight_kN: float
    K_total_N_m: float


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
    iteration_history: List[IterationLog] = field(default_factory=list)


# ----------------------------- BASIC CALC -----------------------------
def total_height(inp: BuildingInput) -> float:
    return inp.n_story * inp.story_height


def total_model_height(inp: BuildingInput) -> float:
    return inp.n_story * inp.story_height + inp.n_basement * inp.basement_height


def floor_area(inp: BuildingInput) -> float:
    return 0.5 * inp.plan_x * inp.plan_y if inp.plan_shape == "triangle" else inp.plan_x * inp.plan_y


def code_type_period(H: float, Ct: float, x_period: float) -> float:
    return Ct * (H ** x_period)


def preliminary_lateral_force_N(inp: BuildingInput, W_total_kN: float) -> float:
    return inp.prelim_lateral_force_coeff * W_total_kN * 1000.0


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


def active_wall_count_by_zone(inp: BuildingInput, zone_name: str) -> int:
    return {
        "Lower Zone": inp.lower_zone_wall_count,
        "Middle Zone": inp.middle_zone_wall_count,
        "Upper Zone": inp.upper_zone_wall_count,
    }[zone_name]


def wall_lengths_for_layout(outer_x: float, outer_y: float, wall_count: int) -> List[float]:
    if wall_count == 4:
        return [outer_x, outer_x, outer_y, outer_y]
    if wall_count == 6:
        return [outer_x, outer_x, outer_y, outer_y, 0.45 * outer_x, 0.45 * outer_x]
    return [outer_x, outer_x, outer_y, outer_y, 0.45 * outer_x, 0.45 * outer_x, 0.45 * outer_y, 0.45 * outer_y]


def wall_rect_inertia_about_global_y(length: float, thickness: float, x_centroid: float) -> float:
    I_local = length * thickness**3 / 12.0
    area = length * thickness
    return I_local + area * x_centroid**2


def wall_rect_inertia_about_global_x(length: float, thickness: float, y_centroid: float) -> float:
    I_local = length * thickness**3 / 12.0
    area = length * thickness
    return I_local + area * y_centroid**2


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
        return [
            ("top", 0.0, inp.plan_x), ("bottom", 0.0, inp.plan_x),
            ("left", 0.0, inp.plan_y), ("right", 0.0, inp.plan_y),
        ]
    ratio = inp.perimeter_shear_wall_ratio
    lx = inp.plan_x * ratio
    ly = inp.plan_y * ratio
    sx = (inp.plan_x - lx) / 2.0
    sy = (inp.plan_y - ly) / 2.0
    return [
        ("top", sx, sx + lx), ("bottom", sx, sx + lx),
        ("left", sy, sy + ly), ("right", sy, sy + ly),
    ]


def wall_thickness_by_zone(inp: BuildingInput, H: float, zone: ZoneDefinition, core_scale: float) -> float:
    base_t = max(inp.min_wall_thickness, min(inp.max_wall_thickness, H / 180.0))
    zone_factor = {"Lower Zone": 1.00, "Middle Zone": 0.80, "Upper Zone": 0.60}[zone.name]
    t = base_t * zone_factor * core_scale
    return max(inp.min_wall_thickness, min(inp.max_wall_thickness, t))


def slab_thickness_prelim(inp: BuildingInput, column_scale: float) -> float:
    span = max(inp.bay_x, inp.bay_y)
    base = span / 28.0
    t = base * (0.90 + 0.10 * column_scale)
    return max(inp.min_slab_thickness, min(inp.max_slab_thickness, t))


def beam_size_prelim(inp: BuildingInput, column_scale: float) -> tuple[float, float]:
    span = max(inp.bay_x, inp.bay_y)
    depth = max(inp.min_beam_depth, min(2.0, (span / 12.0) * (0.90 + 0.15 * column_scale)))
    width = max(inp.min_beam_width, 0.45 * depth)
    return width, depth


def directional_dims(base_dim: float, plan_x: float, plan_y: float) -> tuple[float, float]:
    aspect = max(plan_x, plan_y) / max(min(plan_x, plan_y), 1e-9)
    if aspect <= 1.10:
        return base_dim, base_dim
    major = base_dim * 1.15
    minor = base_dim * 0.90
    return (major, minor) if plan_x >= plan_y else (minor, major)


def build_zone_results(inp: BuildingInput, core_scale: float, column_scale: float, slab_t: float) -> tuple[List[ZoneCoreResult], List[ZoneColumnResult]]:
    H = total_height(inp)
    zones = define_three_zones(inp.n_story)
    opening_x, opening_y = opening_dimensions(inp)
    outer_x, outer_y = initial_core_dimensions(inp, opening_x, opening_y)

    zone_cores: List[ZoneCoreResult] = []
    zone_cols: List[ZoneColumnResult] = []

    q = inp.DL + inp.LL + inp.slab_finish_allowance + slab_t * 25.0
    sigma_allow = 0.35 * inp.fck * 1000.0

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

    for zone in zones:
        wall_count = active_wall_count_by_zone(inp, zone.name)
        lengths = wall_lengths_for_layout(outer_x, outer_y, wall_count)
        t = wall_thickness_by_zone(inp, H, zone, core_scale)
        I_gross = core_equivalent_inertia(outer_x, outer_y, lengths, t, wall_count)
        I_eff = inp.wall_cracked_factor * I_gross
        perim = perimeter_wall_segments_for_square(inp, zone)
        zone_cores.append(
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
                story_slenderness=inp.story_height / max(t, 1e-9),
                perimeter_wall_segments=perim,
                retaining_wall_active=(zone.name == "Lower Zone"),
            )
        )

        floors_above = inp.n_story - zone.story_start + 1
        n_effective = floors_above + 0.70 * inp.n_basement
        tributary_interior = inp.bay_x * inp.bay_y
        tributary_perimeter = 0.50 * inp.bay_x * inp.bay_y
        tributary_corner = 0.25 * inp.bay_x * inp.bay_y

        P_interior = tributary_interior * q * n_effective * 1.18
        interior_dim = min(inp.max_column_dim, max(inp.min_column_dim, sqrt(max(P_interior, 1e-9) / sigma_allow))) * column_scale
        perimeter_dim = min(inp.max_column_dim, max(inp.min_column_dim, interior_dim * inp.perimeter_column_factor))
        corner_dim = min(inp.max_column_dim, max(inp.min_column_dim, interior_dim * inp.corner_column_factor))
        perimeter_dim = min(inp.max_column_dim, perimeter_dim)
        corner_dim = min(inp.max_column_dim, corner_dim)

        interior_x, interior_y = directional_dims(interior_dim, inp.plan_x, inp.plan_y)
        perimeter_x, perimeter_y = directional_dims(perimeter_dim, inp.plan_x, inp.plan_y)
        corner_x, corner_y = directional_dims(corner_dim, inp.plan_x, inp.plan_y)

        A_corner = corner_x * corner_y
        A_perim = perimeter_x * perimeter_y
        A_inter = interior_x * interior_y
        Iavg_corner = max(corner_x * corner_y**3 / 12.0, corner_y * corner_x**3 / 12.0)
        Iavg_perim = max(perimeter_x * perimeter_y**3 / 12.0, perimeter_y * perimeter_x**3 / 12.0)
        Iavg_inter = max(interior_x * interior_y**3 / 12.0, interior_y * interior_x**3 / 12.0)
        I_avg = (corner_cols * Iavg_corner + perimeter_cols * Iavg_perim + interior_cols * Iavg_inter) / max(total_columns, 1)
        A_avg = (corner_cols * A_corner + perimeter_cols * A_perim + interior_cols * A_inter) / max(total_columns, 1)
        I_col_group = inp.column_cracked_factor * (I_avg * max(total_columns, 1) + A_avg * r2_sum)

        zone_cols.append(
            ZoneColumnResult(
                zone=zone,
                corner_column_x_m=corner_x,
                corner_column_y_m=corner_y,
                perimeter_column_x_m=perimeter_x,
                perimeter_column_y_m=perimeter_y,
                interior_column_x_m=interior_x,
                interior_column_y_m=interior_y,
                P_corner_kN=tributary_corner * q * n_effective * 1.18,
                P_perimeter_kN=tributary_perimeter * q * n_effective * 1.18,
                P_interior_kN=P_interior,
                I_col_group_effective_m4=I_col_group,
            )
        )

    return zone_cores, zone_cols


def weighted_core_stiffness(inp: BuildingInput, zone_cores: List[ZoneCoreResult]) -> float:
    H = total_height(inp)
    E = inp.Ec * 1e6
    total_flex_factor = 0.0
    for zc in zone_cores:
        hi = zc.zone.n_stories * inp.story_height
        total_flex_factor += (hi / H) / max(E * zc.Ieq_effective_m4 * (1.0 + 0.20 * len(zc.perimeter_wall_segments)), 1e-9)
    EI_equiv = 1.0 / max(total_flex_factor, 1e-18)
    return 3.0 * EI_equiv / (H**3)


def weighted_column_stiffness(inp: BuildingInput, zone_cols: List[ZoneColumnResult]) -> float:
    H = total_height(inp)
    E = inp.Ec * 1e6
    total_flex_factor = 0.0
    for zc in zone_cols:
        hi = zc.zone.n_stories * inp.story_height
        total_flex_factor += (hi / H) / max(E * zc.I_col_group_effective_m4, 1e-9)
    EI_equiv = 1.0 / max(total_flex_factor, 1e-18)
    return 3.0 * EI_equiv / (H**3)


def estimate_reinforcement(inp: BuildingInput, zone_cores: List[ZoneCoreResult], zone_cols: List[ZoneColumnResult], slab_t: float, beam_b: float, beam_h: float) -> ReinforcementEstimate:
    n_total_levels = inp.n_story + inp.n_basement
    total_floor_area = floor_area(inp) * n_total_levels

    wall_concrete = 0.0
    for zc in zone_cores:
        zone_height = zc.zone.n_stories * inp.story_height
        wall_concrete += sum(zc.wall_lengths) * zc.wall_thickness * zone_height
        for _, a, b in zc.perimeter_wall_segments:
            wall_concrete += (b - a) * zc.wall_thickness * zone_height

    corner_cols = 4
    perimeter_cols = max(0, 2 * (inp.n_bays_x - 1) + 2 * (inp.n_bays_y - 1))
    total_cols = (inp.n_bays_x + 1) * (inp.n_bays_y + 1)
    interior_cols = max(0, total_cols - corner_cols - perimeter_cols)

    column_concrete = 0.0
    for zc in zone_cols:
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

    return ReinforcementEstimate(
        wall_concrete_volume_m3=wall_concrete,
        column_concrete_volume_m3=column_concrete,
        beam_concrete_volume_m3=beam_concrete,
        slab_concrete_volume_m3=slab_concrete,
        wall_steel_kg=wall_steel,
        column_steel_kg=column_steel,
        beam_steel_kg=beam_steel,
        slab_steel_kg=slab_steel,
        total_steel_kg=wall_steel + column_steel + beam_steel + slab_steel,
    )


def total_weight_kN_from_quantities(inp: BuildingInput, reinf: ReinforcementEstimate) -> float:
    concrete_vol = (
        reinf.wall_concrete_volume_m3
        + reinf.column_concrete_volume_m3
        + reinf.beam_concrete_volume_m3
        + reinf.slab_concrete_volume_m3
    )
    concrete_weight = concrete_vol * 25.0
    steel_weight = reinf.total_steel_kg * G / 1000.0
    A = floor_area(inp)
    superimposed = (inp.DL + inp.LL + inp.slab_finish_allowance) * A * (inp.n_story + inp.n_basement)
    facade = inp.facade_line_load * (2 * (inp.plan_x + inp.plan_y)) * inp.n_story
    return (concrete_weight + steel_weight + superimposed + facade) * inp.seismic_mass_factor


def build_story_masses(inp: BuildingInput, total_weight_kN_value: float) -> List[float]:
    # include basements + superstructure so sensitivity to n_basement is preserved
    n = inp.n_story + inp.n_basement
    if n <= 0:
        return []
    weights = []
    for i in range(inp.n_basement):
        weights.append(1.10)
    for i in range(inp.n_story):
        weights.append(1.00)
    weights = np.array(weights, dtype=float)
    weights = weights / weights.sum() * total_weight_kN_value
    return [(w * 1000.0) / G for w in weights.tolist()]


def build_story_stiffnesses(inp: BuildingInput, K_total: float) -> List[float]:
    n = inp.n_story + inp.n_basement
    raw = []
    for i in range(n):
        if i < inp.n_basement:
            raw.append(1.55 - 0.05 * i)
        else:
            j = i - inp.n_basement
            r = j / max(inp.n_story - 1, 1)
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
    mass_ratios = []
    cumulative = []
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
        mass_ratios.append(ratio)
        cumulative.append(cum)

    return ModalResult(
        n_dof=len(masses),
        periods_s=periods,
        frequencies_hz=freqs,
        mode_shapes=mode_shapes,
        story_masses_kg=masses,
        story_stiffness_N_per_m=k_stories,
        effective_mass_ratios=mass_ratios,
        cumulative_effective_mass_ratios=cumulative,
    )


def generate_redesign_suggestions(inp: BuildingInput, T_est: float, T_target: float, T_limit: float, drift_ratio: float, drift_limit: float, core_scale: float, column_scale: float):
    suggestions = []
    governing_issue = "OK"
    if T_est > T_limit:
        governing_issue = "Period exceeds upper limit"
        suggestions.extend([
            "Increase lateral stiffness: enlarge core walls and/or perimeter columns.",
            "Increase active wall count in middle and upper zones.",
            "Consider larger core footprint if architecture permits.",
        ])
    elif T_est > 1.10 * T_target:
        governing_issue = "Period above target"
        suggestions.extend([
            "System is softer than target.",
            "Increase wall thickness or internal core wall engagement.",
            "Increase corner and perimeter columns.",
        ])
    elif T_est < 0.90 * T_target:
        governing_issue = "Period below target"
        suggestions.extend([
            "System is stiffer than target and may be uneconomical.",
            "Reduce wall thicknesses or column sizes where feasible.",
        ])
    if drift_ratio > drift_limit:
        governing_issue = "Drift exceeds allowable limit"
        suggestions.extend([
            "Increase global stiffness by enlarging the core and perimeter columns.",
            "Increase wall ratio or add perimeter wall segments.",
        ])
    if core_scale >= 1.55:
        suggestions.append("Core scale factor is near its upper bound.")
    if column_scale >= 1.55:
        suggestions.append("Column scale factor is near its upper bound.")
    if not suggestions:
        suggestions.append("Structural system appears preliminarily adequate.")
    return governing_issue, suggestions


# ----------------------------- OPTIMIZATION -----------------------------
def evaluate_design(inp: BuildingInput, core_scale: float, column_scale: float, beta: float):
    H = total_height(inp)
    T_ref = code_type_period(H, inp.Ct, inp.x_period)
    T_upper = inp.upper_period_factor * T_ref
    T_target = T_ref + beta * (T_upper - T_ref)

    slab_t = slab_thickness_prelim(inp, column_scale)
    beam_b, beam_h = beam_size_prelim(inp, column_scale)
    zone_cores, zone_cols = build_zone_results(inp, core_scale, column_scale, slab_t)
    reinf = estimate_reinforcement(inp, zone_cores, zone_cols, slab_t, beam_b, beam_h)
    W_total = total_weight_kN_from_quantities(inp, reinf)
    M_eff = W_total * 1000.0 / G

    K_core = weighted_core_stiffness(inp, zone_cores)
    K_cols = weighted_column_stiffness(inp, zone_cols)
    K_est = K_core + K_cols

    modal = solve_mdof_modes(inp, W_total, K_est, n_modes=5)
    T_est = modal.periods_s[0] if modal.periods_s else 2.0 * pi * sqrt(M_eff / max(K_est, 1e-9))

    top_drift = preliminary_lateral_force_N(inp, W_total) / max(K_est, 1e-9)
    drift_ratio = top_drift / max(total_model_height(inp), 1e-9)
    period_error = abs(T_est - T_target) / max(T_target, 1e-9)

    return {
        "T_ref": T_ref,
        "T_upper": T_upper,
        "T_target": T_target,
        "T_est": T_est,
        "period_error": period_error,
        "W_total": W_total,
        "M_eff": M_eff,
        "K_est": K_est,
        "top_drift": top_drift,
        "drift_ratio": drift_ratio,
        "modal": modal,
        "zone_cores": zone_cores,
        "zone_cols": zone_cols,
        "slab_t": slab_t,
        "beam_b": beam_b,
        "beam_h": beam_h,
        "reinf": reinf,
    }


def optimize_scales(inp: BuildingInput, beta: float, x0: np.ndarray | None = None):
    def objective(x):
        core_scale, col_scale = float(x[0]), float(x[1])
        ev = evaluate_design(inp, core_scale, col_scale, beta)
        period_term = 900.0 * ev["period_error"]**2
        upper_term = 5000.0 * max(ev["T_est"] / ev["T_upper"] - 1.0, 0.0) ** 2
        drift_term = 3500.0 * max(ev["drift_ratio"] / inp.drift_limit_ratio - 1.0, 0.0) ** 2
        weight_term = 0.20 * (ev["W_total"] / 1e6)
        balance_term = 2.0 * (core_scale - col_scale) ** 2
        return period_term + upper_term + drift_term + weight_term + balance_term

    bounds = [(0.55, 1.60), (0.55, 1.60)]
    if x0 is None:
        x0 = np.array([1.0, 1.0], dtype=float)
    res = minimize(objective, np.asarray(x0, dtype=float), bounds=bounds, method="L-BFGS-B")
    core_scale, col_scale = float(res.x[0]), float(res.x[1])
    ev = evaluate_design(inp, core_scale, col_scale, beta)
    return res, core_scale, col_scale, ev


def run_iterative_design(inp: BuildingInput) -> DesignResult:
    beta = inp.target_position_factor
    max_iterations = 20
    tolerance = 0.02

    core_scale = 1.0
    column_scale = 1.0

    iteration_history: List[IterationLog] = []
    best_result = None
    best_error = float("inf")

    for iteration in range(1, max_iterations + 1):
        ev = evaluate_design(inp, core_scale, column_scale, beta)

        T_est = ev["T_est"]
        T_target = ev["T_target"]
        T_upper = ev["T_upper"]
        error = abs(T_est - T_target) / T_target
        error_percent = error * 100

        iteration_history.append(
            IterationLog(
                iteration=iteration,
                core_scale=core_scale,
                column_scale=column_scale,
                T_estimated=T_est,
                T_target=T_target,
                error_percent=error_percent,
                total_weight_kN=ev["W_total"],
                K_total_N_m=ev["K_est"],
            )
        )

        if error < best_error and T_est <= T_upper and ev["drift_ratio"] <= inp.drift_limit_ratio:
            best_error = error
            best_result = (core_scale, column_scale, ev)

        if error <= tolerance and T_est <= T_upper and ev["drift_ratio"] <= inp.drift_limit_ratio:
            break

        period_ratio = T_est / T_target
        alpha = 0.5
        scale_factor = (1.0 / period_ratio) ** alpha
        new_core_scale = max(0.55, min(1.60, core_scale * scale_factor))
        new_column_scale = max(0.55, min(1.60, column_scale * scale_factor))

        if abs(new_core_scale - core_scale) < 1e-4 and abs(new_column_scale - column_scale) < 1e-4:
            break
        core_scale, column_scale = new_core_scale, new_column_scale

    if best_result is None or best_error > tolerance:
        res, core_scale, column_scale, ev = optimize_scales(inp, beta)
        opt_success = bool(res.success)
        opt_msg = str(res.message)
    else:
        core_scale, column_scale, ev = best_result
        opt_success = True
        opt_msg = "Iterative MDOF convergence"

    T_ref = ev["T_ref"]
    T_upper = ev["T_upper"]
    T_target = ev["T_target"]
    T_est = ev["T_est"]
    drift_ratio = ev["drift_ratio"]

    period_ok = T_est <= T_upper
    drift_ok = drift_ratio <= inp.drift_limit_ratio
    governing_issue, redesign_suggestions = generate_redesign_suggestions(
        inp, T_est, T_target, T_upper, drift_ratio, inp.drift_limit_ratio, core_scale, column_scale
    )

    messages = [
        f"Target formula: T_target = T_ref + beta*(T_upper - T_ref)",
        f"beta = {beta:.3f}",
        f"T_ref = {T_ref:.3f} s",
        f"T_upper = {T_upper:.3f} s",
        f"T_target = {T_target:.3f} s",
        f"Final T_est (MDOF) = {T_est:.3f} s",
        f"Iterations used = {len(iteration_history)}",
    ]
    if ev["modal"].effective_mass_ratios:
        messages.append(f"MDOF Mode 1 mass participation = {100*ev['modal'].effective_mass_ratios[0]:.1f}%")
    messages.append(f"Upper period check = {'OK' if period_ok else 'NOT OK'}")
    messages.append(f"Drift check = {'OK' if drift_ok else 'NOT OK'}")

    return DesignResult(
        H_m=total_height(inp),
        floor_area_m2=floor_area(inp),
        total_weight_kN=ev["W_total"],
        effective_modal_mass_kg=ev["M_eff"],
        reference_period_s=T_ref,
        design_target_period_s=T_target,
        upper_limit_period_s=T_upper,
        estimated_period_s=T_est,
        period_error_ratio=ev["period_error"],
        period_ok=period_ok,
        drift_ok=drift_ok,
        K_estimated_N_per_m=ev["K_est"],
        top_drift_m=ev["top_drift"],
        drift_ratio=drift_ratio,
        zone_core_results=ev["zone_cores"],
        zone_column_results=ev["zone_cols"],
        slab_thickness_m=ev["slab_t"],
        beam_width_m=ev["beam_b"],
        beam_depth_m=ev["beam_h"],
        reinforcement=ev["reinf"],
        optimization_success=opt_success,
        optimization_message=opt_msg,
        core_scale=core_scale,
        column_scale=column_scale,
        messages=messages,
        redesign_suggestions=redesign_suggestions,
        governing_issue=governing_issue,
        modal_result=ev["modal"],
        iteration_history=iteration_history,
    )


def run_design(inp: BuildingInput) -> DesignResult:
    return run_iterative_design(inp)


def build_report(result: DesignResult) -> str:
    lines = []
    lines.append("=" * 74)
    lines.append("TALL BUILDING PRELIMINARY DESIGN REPORT")
    lines.append("=" * 74)
    lines.append("")
    lines.append("GLOBAL RESPONSE")
    lines.append("-" * 74)
    lines.append(f"Reference period                = {result.reference_period_s:.3f} s")
    lines.append(f"Design target period            = {result.design_target_period_s:.3f} s")
    lines.append(f"Estimated dynamic period (MDOF) = {result.estimated_period_s:.3f} s")
    lines.append(f"Upper limit period              = {result.upper_limit_period_s:.3f} s")
    lines.append(f"Period error ratio              = {100*result.period_error_ratio:.2f} %")
    lines.append(f"Period check                    = {'OK' if result.period_ok else 'NOT OK'}")
    lines.append(f"Total stiffness                 = {result.K_estimated_N_per_m:,.3e} N/m")
    lines.append(f"Estimated top drift             = {result.top_drift_m:.3f} m")
    lines.append(f"Estimated drift ratio           = {result.drift_ratio:.5f}")
    lines.append(f"Drift check                     = {'OK' if result.drift_ok else 'NOT OK'}")
    lines.append(f"Total structural weight         = {result.total_weight_kN:,.0f} kN")
    lines.append(f"Core scale factor               = {result.core_scale:.3f}")
    lines.append(f"Column scale factor             = {result.column_scale:.3f}")
    lines.append("")
    lines.append("ITERATION HISTORY (MDOF LOOP)")
    lines.append("-" * 74)
    lines.append(f"{'Iter':>4} {'CoreSc':>8} {'ColSc':>8} {'T_est':>10} {'T_target':>10} {'Err%':>8} {'Weight':>12} {'K_total':>12}")
    lines.append("-" * 74)
    for log in result.iteration_history:
        lines.append(f"{log.iteration:>4} {log.core_scale:>8.3f} {log.column_scale:>8.3f} {log.T_estimated:>10.3f} {log.T_target:>10.3f} {log.error_percent:>8.2f} {log.total_weight_kN:>12,.0f} {log.K_total_N_m:>12.3e}")
    lines.append("")
    lines.append("PRIMARY MEMBER OUTPUT")
    lines.append("-" * 74)
    lines.append(f"Beam size (b x h)               = {result.beam_width_m:.2f} x {result.beam_depth_m:.2f} m")
    lines.append(f"Slab thickness                  = {result.slab_thickness_m:.2f} m")
    lines.append("")
    lines.append("ZONE-BY-ZONE COLUMN DIMENSIONS")
    lines.append("-" * 74)
    for zc in result.zone_column_results:
        lines.append(f"{zc.zone.name}:")
        lines.append(f"  Corner columns               = {zc.corner_column_x_m:.2f} x {zc.corner_column_y_m:.2f} m")
        lines.append(f"  Perimeter columns            = {zc.perimeter_column_x_m:.2f} x {zc.perimeter_column_y_m:.2f} m")
        lines.append(f"  Interior columns             = {zc.interior_column_x_m:.2f} x {zc.interior_column_y_m:.2f} m")
        lines.append(f"  Column group Ieff            = {zc.I_col_group_effective_m4:,.2f} m^4")
    lines.append("")
    lines.append("ZONE-BY-ZONE WALL / CORE OUTPUT")
    lines.append("-" * 74)
    for zc in result.zone_core_results:
        lines.append(f"{zc.zone.name}:")
        lines.append(f"  Core outer                   = {zc.core_outer_x:.2f} x {zc.core_outer_y:.2f} m")
        lines.append(f"  Core opening                 = {zc.core_opening_x:.2f} x {zc.core_opening_y:.2f} m")
        lines.append(f"  Wall thickness               = {zc.wall_thickness:.2f} m")
        lines.append(f"  Active core walls            = {zc.wall_count}")
        lines.append(f"  Effective Ieq                = {zc.Ieq_effective_m4:,.2f} m^4")
    lines.append("")
    lines.append("MATERIAL / QUANTITY SUMMARY")
    lines.append("-" * 74)
    r = result.reinforcement
    lines.append(f"Wall concrete volume            = {r.wall_concrete_volume_m3:,.2f} m³")
    lines.append(f"Column concrete volume          = {r.column_concrete_volume_m3:,.2f} m³")
    lines.append(f"Beam concrete volume            = {r.beam_concrete_volume_m3:,.2f} m³")
    lines.append(f"Slab concrete volume            = {r.slab_concrete_volume_m3:,.2f} m³")
    lines.append(f"Wall steel                      = {r.wall_steel_kg:,.0f} kg")
    lines.append(f"Column steel                    = {r.column_steel_kg:,.0f} kg")
    lines.append(f"Beam steel                      = {r.beam_steel_kg:,.0f} kg")
    lines.append(f"Slab steel                      = {r.slab_steel_kg:,.0f} kg")
    lines.append(f"Total steel                     = {r.total_steel_kg:,.0f} kg")
    lines.append("")
    lines.append("REDESIGN SUGGESTIONS")
    lines.append("-" * 74)
    for s in result.redesign_suggestions:
        lines.append(f"- {s}")
    lines.append("")
    lines.append("MODAL ANALYSIS")
    lines.append("-" * 74)
    for i, (T, f, mr, cum) in enumerate(zip(result.modal_result.periods_s, result.modal_result.frequencies_hz, result.modal_result.effective_mass_ratios, result.modal_result.cumulative_effective_mass_ratios), start=1):
        lines.append(f"Mode {i}: T = {T:.4f} s | f = {f:.4f} Hz | mass ratio = {100*mr:.2f}% | cumulative = {100*cum:.2f}%")
    lines.append("")
    lines.append("MESSAGES")
    lines.append("-" * 74)
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
            _draw_rect(ax, px - dx / 2, py - dy / 2, dx, dy, c, fill=True, alpha=0.95, lw=0.5)

    cx0 = (inp.plan_x - core.core_outer_x) / 2
    cy0 = (inp.plan_y - core.core_outer_y) / 2
    cx1 = cx0 + core.core_outer_x
    cy1 = cy0 + core.core_outer_y
    ix0 = (inp.plan_x - core.core_opening_x) / 2
    iy0 = (inp.plan_y - core.core_opening_y) / 2

    _draw_rect(ax, cx0, cy0, core.core_outer_x, core.core_outer_y, CORE_COLOR, fill=False, lw=2.5)
    _draw_rect(ax, ix0, iy0, core.core_opening_x, core.core_opening_y, "#666666", fill=False, lw=1.3, ls="--")
    ax.text(cx0 + core.core_outer_x/2, cy0 - 1.0, "CORE", ha="center", fontsize=10, fontweight="bold")
    ax.text(ix0 + core.core_opening_x/2, iy0 + core.core_opening_y/2, "OPENING", ha="center", va="center", fontsize=9)

    t = core.wall_thickness
    _draw_rect(ax, cx0, cy0, core.core_outer_x, t, CORE_COLOR, fill=True, alpha=0.25)
    _draw_rect(ax, cx0, cy1 - t, core.core_outer_x, t, CORE_COLOR, fill=True, alpha=0.25)
    _draw_rect(ax, cx0, cy0, t, core.core_outer_y, CORE_COLOR, fill=True, alpha=0.25)
    _draw_rect(ax, cx1 - t, cy0, t, core.core_outer_y, CORE_COLOR, fill=True, alpha=0.25)

    thickness = inp.basement_retaining_wall_thickness if core.retaining_wall_active else core.wall_thickness
    for side, a, b in core.perimeter_wall_segments:
        if side == "top":
            _draw_rect(ax, a, 0, b - a, thickness, PERIM_WALL_COLOR, fill=True, alpha=0.85)
        elif side == "bottom":
            _draw_rect(ax, a, inp.plan_y - thickness, b - a, thickness, PERIM_WALL_COLOR, fill=True, alpha=0.85)
        elif side == "left":
            _draw_rect(ax, 0, a, thickness, b - a, PERIM_WALL_COLOR, fill=True, alpha=0.85)
        else:
            _draw_rect(ax, inp.plan_x - thickness, a, thickness, b - a, PERIM_WALL_COLOR, fill=True, alpha=0.85)

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
        f"Ieff = {core.Ieq_effective_m4:.1f} m^4",
    ]
    for k, txt in enumerate(info_lines):
        ax.text(info_x, info_y - 4 * k, txt, fontsize=9, va="top")

    lx = inp.plan_x * 0.40
    ly = inp.plan_y * 0.08
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

    ax.set_title(f"{core.zone.name} - {inp.plan_shape.title()} plan", fontsize=14, fontweight="bold")
    ax.set_xlim(-8, inp.plan_x + 32)
    ax.set_ylim(inp.plan_y + 8, -12)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig


def plot_mode_shapes(result: DesignResult):
    mr = result.modal_result
    n_modes = min(5, len(mr.mode_shapes))
    H = total_model_height(st.inputs)
    y = np.linspace(0.0, H, mr.n_dof)
    fig, axes = plt.subplots(1, n_modes, figsize=(18, 6))
    if n_modes == 1:
        axes = [axes]
    for m in range(n_modes):
        ax = axes[m]
        phi = np.array(mr.mode_shapes[m], dtype=float)
        phi = phi / max(np.max(np.abs(phi)), 1e-9)
        if phi[-1] < 0:
            phi = -phi
        ax.axvline(0.0, color="#bbbbbb", linestyle="--", linewidth=1.0)
        for yi in y:
            ax.plot([-1.05, 1.05], [yi, yi], color="#f0f0f0", linewidth=0.8)
        ax.plot(phi, y, color="#0b5ed7", linewidth=2)
        ax.scatter(phi, y, color="#dc3545", s=18, zorder=3)
        ax.set_title(f"Mode {m+1}\nT = {mr.periods_s[m]:.3f} s", fontsize=11, fontweight="bold")
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(0.0, H)
        if m == 0:
            ax.set_ylabel("Height (m)")
        else:
            ax.set_yticks([])
        ax.set_xticks([])
    fig.suptitle("First 5 Mode Shapes (MDOF)", fontsize=15, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_iteration_history(result: DesignResult):
    if not result.iteration_history:
        return None
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    iters = [log.iteration for log in result.iteration_history]
    t_est = [log.T_estimated for log in result.iteration_history]
    t_target = [log.T_target for log in result.iteration_history]
    errors = [log.error_percent for log in result.iteration_history]
    core_scales = [log.core_scale for log in result.iteration_history]
    col_scales = [log.column_scale for log in result.iteration_history]
    weights = [log.total_weight_kN / 1000 for log in result.iteration_history]

    ax = axes[0, 0]
    ax.plot(iters, t_est, 'b-o', label='T_estimated (MDOF)', linewidth=2, markersize=6)
    ax.axhline(y=t_target[0], color='r', linestyle='--', label=f'T_target = {t_target[0]:.3f}s')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Period (s)')
    ax.set_title('Period Convergence')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(iters, errors, 'g-s', linewidth=2, markersize=6)
    ax.axhline(y=2.0, color='r', linestyle='--', label='Tolerance (2%)')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Error (%)')
    ax.set_title('Period Error Convergence')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(iters, core_scales, 'm-o', label='Core Scale', linewidth=2, markersize=6)
    ax.plot(iters, col_scales, 'c-s', label='Column Scale', linewidth=2, markersize=6)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Scale Factor')
    ax.set_title('Section Scale Factors')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(iters, weights, 'k-d', linewidth=2, markersize=6)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Total Weight (MN)')
    ax.set_title('Structural Weight Evolution')
    ax.grid(True, alpha=0.3)

    fig.suptitle("MDOF Iterative Convergence History", fontsize=15, fontweight="bold")
    fig.tight_layout()
    return fig


def make_zone_tables(result: DesignResult) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    core_rows = []
    col_rows = []
    modal_rows = []
    for zc in result.zone_core_results:
        core_rows.append({
            "Zone": zc.zone.name,
            "Stories": f"{zc.zone.story_start}-{zc.zone.story_end}",
            "Wall count": zc.wall_count,
            "Wall t (m)": zc.wall_thickness,
            "Core outer X (m)": zc.core_outer_x,
            "Core outer Y (m)": zc.core_outer_y,
            "Opening X (m)": zc.core_opening_x,
            "Opening Y (m)": zc.core_opening_y,
            "Ieq eff (m4)": zc.Ieq_effective_m4,
            "Story slenderness": zc.story_slenderness,
        })
    for zc in result.zone_column_results:
        col_rows.append({
            "Zone": zc.zone.name,
            "Corner col X (m)": zc.corner_column_x_m,
            "Corner col Y (m)": zc.corner_column_y_m,
            "Perimeter col X (m)": zc.perimeter_column_x_m,
            "Perimeter col Y (m)": zc.perimeter_column_y_m,
            "Interior col X (m)": zc.interior_column_x_m,
            "Interior col Y (m)": zc.interior_column_y_m,
            "P corner (kN)": zc.P_corner_kN,
            "P perimeter (kN)": zc.P_perimeter_kN,
            "P interior (kN)": zc.P_interior_kN,
            "Ieff group (m4)": zc.I_col_group_effective_m4,
        })
    if result.modal_result:
        for i, (T, f, mr, cum) in enumerate(zip(result.modal_result.periods_s, result.modal_result.frequencies_hz, result.modal_result.effective_mass_ratios, result.modal_result.cumulative_effective_mass_ratios), start=1):
            modal_rows.append({
                "Mode": i,
                "Period (s)": T,
                "Frequency (Hz)": f,
                "Effective mass ratio": mr,
                "Cumulative mass ratio": cum,
            })
    return pd.DataFrame(core_rows), pd.DataFrame(col_rows), pd.DataFrame(modal_rows)


# ----------------------------- STREAMLIT LAYOUT -----------------------------
def main():
    st.markdown(f"## Tall Building Preliminary Structural Analysis")
    st.caption(f"{APP_VERSION} • {AUTHOR_NAME}")
    st.info("This app performs a preliminary single-direction MDOF-based period-targeting study with zone-based core and column sizing. It is for preliminary calibration, not final code design.")

    with st.sidebar:
        st.header("Inputs")
        inp = streamlit_input_panel()
        st.inputs = inp  # simple handle for plot function
        run = st.button("Run analysis", type="primary", use_container_width=True)

    if not run:
        st.stop()

    result = run_design(inp)
    core_df, col_df, modal_df = make_zone_tables(result)
    report_text = build_report(result)

    # Summary metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("T_ref (s)", f"{result.reference_period_s:.3f}")
    c2.metric("T_target (s)", f"{result.design_target_period_s:.3f}")
    c3.metric("T_est (s)", f"{result.estimated_period_s:.3f}")
    c4.metric("Error (%)", f"{100*result.period_error_ratio:.2f}")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("K_total (N/m)", f"{result.K_estimated_N_per_m:,.2e}")
    c6.metric("Top drift (m)", f"{result.top_drift_m:.3f}")
    c7.metric("Weight (kN)", f"{result.total_weight_kN:,.0f}")
    c8.metric("Mode 1 mass (%)", f"{100*result.modal_result.effective_mass_ratios[0]:.1f}" if result.modal_result and result.modal_result.effective_mass_ratios else "-")

    status_col1, status_col2 = st.columns(2)
    status_col1.success("Period upper-limit check passed" if result.period_ok else "Period upper-limit check failed")
    status_col2.success("Drift check passed" if result.drift_ok else "Drift check failed")

    st.markdown("### Messages")
    for msg in result.messages:
        st.write(f"- {msg}")

    st.markdown("### Redesign suggestions")
    for msg in result.redesign_suggestions:
        st.write(f"- {msg}")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Global report",
        "Zone core table",
        "Zone column table",
        "Modal table",
        "Plan plots",
        "Mode shapes",
    ])

    with tab1:
        st.text(report_text)
        st.download_button(
            "Download report as TXT",
            data=report_text,
            file_name="tall_building_preliminary_report.txt",
            mime="text/plain",
            use_container_width=True,
        )

    with tab2:
        st.dataframe(core_df, use_container_width=True)

    with tab3:
        st.dataframe(col_df, use_container_width=True)

    with tab4:
        st.dataframe(modal_df, use_container_width=True)
        st.pyplot(plot_iteration_history(result), use_container_width=True)

    with tab5:
        zone_names = [z.zone.name for z in result.zone_core_results]
        selected_zone = st.selectbox("Zone to plot", zone_names, index=0)
        st.pyplot(plot_plan(inp, result, selected_zone), use_container_width=True)

    with tab6:
        st.pyplot(plot_mode_shapes(result), use_container_width=True)

    st.markdown("### Input snapshot")
    st.json(asdict(inp), expanded=False)


if __name__ == "__main__":
    main()
