from __future__ import annotations
from dataclasses import dataclass, field, replace
from math import pi, sqrt
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
import streamlit as st

# ----------------------------- CONFIG -----------------------------
st.set_page_config(page_title="Tall Building Structural Analysis", layout="wide", initial_sidebar_state="expanded")

DEFAULT_AUTHOR_NAME = "Benyamin"
APP_VERSION = "v3.0-optimized"

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
    elevator_count: int = 8
    elevator_area_each: float = 3.5
    stair_area_each: float = 14.0
    service_area: float = 35.0
    corridor_factor: float = 1.40

    fck: float = 60.0
    Ec: float = 36000.0
    fy: float = 420.0

    DL: float = 6.5
    LL: float = 2.5
    slab_finish_allowance: float = 1.5
    facade_line_load: float = 14.0

    prelim_lateral_force_coeff: float = 0.015
    drift_limit_ratio: float = 1 / 500
    target_period_factor: float = 0.95
    max_period_factor_over_target: float = 1.25

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
    effective_modal_mass_ratio: float = 0.80

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
    corner_column_m: float
    perimeter_column_m: float
    interior_column_m: float
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
    modal_participation_factors: List[float]
    effective_modal_masses_kg: List[float]
    effective_mass_ratios: List[float]
    cumulative_effective_mass_ratios: List[float]


@dataclass
class OptimizationResult:
    lambda_core: float
    lambda_col: float
    objective_value: float
    success: bool
    message: str
    n_iterations: int


@dataclass
class DesignResult:
    H_m: float
    floor_area_m2: float
    total_weight_kN: float
    effective_modal_mass_kg: float
    T_code_s: float
    T_limit_s: float
    period_ok: bool
    T_target_s: float
    T_est_s: float
    K_required_N_per_m: float
    K_core_N_per_m: float
    K_columns_N_per_m: float
    K_estimated_N_per_m: float
    top_drift_m: float
    drift_ratio: float
    zone_core_results: List[ZoneCoreResult]
    zone_column_results: List[ZoneColumnResult]
    slab_thickness_m: float
    beam_width_m: float
    beam_depth_m: float
    reinforcement: ReinforcementEstimate
    system_assessment: str
    optimization: OptimizationResult
    messages: List[str] = field(default_factory=list)
    modal_result: ModalResult | None = None


# ----------------------------- BASIC FUNCTIONS -----------------------------
def total_height(inp: BuildingInput) -> float:
    return inp.n_story * inp.story_height


def floor_area(inp: BuildingInput) -> float:
    if inp.plan_shape == "triangle":
        return 0.5 * inp.plan_x * inp.plan_y
    return inp.plan_x * inp.plan_y


def slab_thickness_prelim(inp: BuildingInput) -> float:
    span = max(inp.bay_x, inp.bay_y)
    return max(inp.min_slab_thickness, min(inp.max_slab_thickness, span / 28.0))


def beam_size_prelim(inp: BuildingInput) -> tuple[float, float]:
    span = max(inp.bay_x, inp.bay_y)
    depth = max(inp.min_beam_depth, span / 12.0)
    width = max(inp.min_beam_width, 0.45 * depth)
    return width, depth


def code_type_period(H: float, Ct: float, x_period: float) -> float:
    return Ct * (H ** x_period)


def total_weight_kN(inp: BuildingInput, slab_t: float) -> float:
    A = floor_area(inp)
    perimeter = (
        2.0 * (inp.plan_x + inp.plan_y)
        if inp.plan_shape == "square"
        else (inp.plan_x + inp.plan_y + (inp.plan_x**2 + inp.plan_y**2) ** 0.5)
    )
    slab_self_weight = slab_t * 25.0
    floor_load = inp.DL + inp.LL + inp.slab_finish_allowance + slab_self_weight
    above_grade = floor_load * A * inp.n_story
    basement = 1.10 * floor_load * A * inp.n_basement
    facade = inp.facade_line_load * perimeter * inp.n_story
    structure_allowance = 0.12 * (above_grade + basement)
    return (above_grade + basement + facade + structure_allowance) * inp.seismic_mass_factor


def effective_modal_mass(total_weight_kN_value: float, ratio: float) -> float:
    return ratio * total_weight_kN_value * 1000.0 / G


def required_stiffness(M_eff: float, T_target: float) -> float:
    return 4.0 * pi**2 * M_eff / (T_target**2)


def preliminary_lateral_force_N(inp: BuildingInput, W_total_kN: float) -> float:
    return inp.prelim_lateral_force_coeff * W_total_kN * 1000.0


def cantilever_tip_stiffness(EI: float, H: float) -> float:
    return 3.0 * EI / (H**3)


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
    if wall_count == 8:
        return [outer_x, outer_x, outer_y, outer_y, 0.45 * outer_x, 0.45 * outer_x, 0.45 * outer_y, 0.45 * outer_y]
    raise ValueError("wall_count must be one of 4, 6, 8")


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


def wall_thickness_by_zone(inp: BuildingInput, H: float, zone: ZoneDefinition) -> float:
    base_t = max(inp.min_wall_thickness, min(inp.max_wall_thickness, H / 180.0))
    if zone.name == "Lower Zone":
        return base_t
    if zone.name == "Middle Zone":
        return max(inp.min_wall_thickness, 0.80 * base_t)
    return max(inp.min_wall_thickness, 0.60 * base_t)


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


# ----------------------------- PRELIMINARY SIZING -----------------------------
def design_core_by_zone(inp: BuildingInput, zones: List[ZoneDefinition]) -> List[ZoneCoreResult]:
    opening_x, opening_y = opening_dimensions(inp)
    outer_x, outer_y = initial_core_dimensions(inp, opening_x, opening_y)
    H = total_height(inp)
    results = []
    for zone in zones:
        wall_count = active_wall_count_by_zone(inp, zone.name)
        lengths = wall_lengths_for_layout(outer_x, outer_y, wall_count)
        t = wall_thickness_by_zone(inp, H, zone)
        I_gross = core_equivalent_inertia(outer_x, outer_y, lengths, t, wall_count)
        I_eff = inp.wall_cracked_factor * I_gross
        perim = perimeter_wall_segments_for_triangle(inp, zone) if inp.plan_shape == "triangle" else perimeter_wall_segments_for_square(inp, zone)
        results.append(ZoneCoreResult(zone=zone, wall_count=wall_count, wall_lengths=lengths, wall_thickness=t,
                                      core_outer_x=outer_x, core_outer_y=outer_y, core_opening_x=opening_x,
                                      core_opening_y=opening_y, Ieq_gross_m4=I_gross, Ieq_effective_m4=I_eff,
                                      story_slenderness=inp.story_height / t, perimeter_wall_segments=perim,
                                      retaining_wall_active=(zone.name == "Lower Zone")))
    return results


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


def estimate_zone_column_sizes(inp: BuildingInput, zones: List[ZoneDefinition], slab_t: float) -> List[ZoneColumnResult]:
    q = inp.DL + inp.LL + inp.slab_finish_allowance + slab_t * 25.0
    sigma_allow = 0.35 * inp.fck * 1000.0
    results = []

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

    for zone in zones:
        floors_above = inp.n_story - zone.story_start + 1
        n_effective = floors_above + 0.70 * inp.n_basement
        tributary_interior = inp.bay_x * inp.bay_y
        tributary_perimeter = 0.50 * inp.bay_x * inp.bay_y
        tributary_corner = 0.25 * inp.bay_x * inp.bay_y

        P_interior = tributary_interior * q * n_effective * 1.18
        interior_dim = min(inp.max_column_dim, max(inp.min_column_dim, sqrt(P_interior / sigma_allow)))
        perimeter_dim = min(inp.max_column_dim, max(inp.min_column_dim, interior_dim * inp.perimeter_column_factor))
        corner_dim = min(inp.max_column_dim, max(inp.min_column_dim, interior_dim * inp.corner_column_factor))
        P_perimeter = tributary_perimeter * q * n_effective * 1.18
        P_corner = tributary_corner * q * n_effective * 1.18

        interior_x, interior_y = directional_column_dims(interior_dim, inp.corner_column_factor, inp.perimeter_column_factor, inp.plan_x, inp.plan_y, "interior")
        perimeter_x, perimeter_y = directional_column_dims(interior_dim, inp.corner_column_factor, inp.perimeter_column_factor, inp.plan_x, inp.plan_y, "perimeter")
        corner_x, corner_y = directional_column_dims(interior_dim, inp.corner_column_factor, inp.perimeter_column_factor, inp.plan_x, inp.plan_y, "corner")

        Iavg_corner = max(corner_x * corner_y**3 / 12.0, corner_y * corner_x**3 / 12.0)
        Iavg_perim = max(perimeter_x * perimeter_y**3 / 12.0, perimeter_y * perimeter_x**3 / 12.0)
        Iavg_inter = max(interior_x * interior_y**3 / 12.0, interior_y * interior_x**3 / 12.0)
        A_avg = (corner_cols * corner_x * corner_y + perimeter_cols * perimeter_x * perimeter_y + interior_cols * interior_x * interior_y) / max(total_columns, 1)
        I_avg = (corner_cols * Iavg_corner + perimeter_cols * Iavg_perim + interior_cols * Iavg_inter) / max(total_columns, 1)
        I_col_group = inp.column_cracked_factor * (I_avg * max(total_columns, 1) + A_avg * r2_sum)

        results.append(ZoneColumnResult(zone=zone, corner_column_m=corner_dim, perimeter_column_m=perimeter_dim, interior_column_m=interior_dim,
                                        corner_column_x_m=corner_x, corner_column_y_m=corner_y,
                                        perimeter_column_x_m=perimeter_x, perimeter_column_y_m=perimeter_y,
                                        interior_column_x_m=interior_x, interior_column_y_m=interior_y,
                                        P_corner_kN=P_corner, P_perimeter_kN=P_perimeter, P_interior_kN=P_interior,
                                        I_col_group_effective_m4=I_col_group))
    return results


def count_columns(inp: BuildingInput) -> tuple[int, int, int, int, float]:
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
    return total_columns, corner_cols, perimeter_cols, interior_cols, r2_sum


# ----------------------------- SCALING + OPTIMIZATION -----------------------------
def scale_core_results(inp: BuildingInput, base_cores: List[ZoneCoreResult], lambda_core: float) -> List[ZoneCoreResult]:
    scaled = []
    for z in base_cores:
        t = min(inp.max_wall_thickness, max(inp.min_wall_thickness, z.wall_thickness * lambda_core))
        I_gross = core_equivalent_inertia(z.core_outer_x, z.core_outer_y, z.wall_lengths, t, z.wall_count)
        I_eff = inp.wall_cracked_factor * I_gross
        scaled.append(replace(z, wall_thickness=t, Ieq_gross_m4=I_gross, Ieq_effective_m4=I_eff, story_slenderness=inp.story_height / t))
    return scaled


def scale_column_results(inp: BuildingInput, base_cols: List[ZoneColumnResult], lambda_col: float) -> List[ZoneColumnResult]:
    total_columns, corner_cols, perimeter_cols, interior_cols, r2_sum = count_columns(inp)
    scaled = []
    for z in base_cols:
        corner_x = min(inp.max_column_dim, max(inp.min_column_dim, z.corner_column_x_m * lambda_col))
        corner_y = min(inp.max_column_dim, max(inp.min_column_dim, z.corner_column_y_m * lambda_col))
        perimeter_x = min(inp.max_column_dim, max(inp.min_column_dim, z.perimeter_column_x_m * lambda_col))
        perimeter_y = min(inp.max_column_dim, max(inp.min_column_dim, z.perimeter_column_y_m * lambda_col))
        interior_x = min(inp.max_column_dim, max(inp.min_column_dim, z.interior_column_x_m * lambda_col))
        interior_y = min(inp.max_column_dim, max(inp.min_column_dim, z.interior_column_y_m * lambda_col))

        Iavg_corner = max(corner_x * corner_y**3 / 12.0, corner_y * corner_x**3 / 12.0)
        Iavg_perim = max(perimeter_x * perimeter_y**3 / 12.0, perimeter_y * perimeter_x**3 / 12.0)
        Iavg_inter = max(interior_x * interior_y**3 / 12.0, interior_y * interior_x**3 / 12.0)
        A_avg = (corner_cols * corner_x * corner_y + perimeter_cols * perimeter_x * perimeter_y + interior_cols * interior_x * interior_y) / max(total_columns, 1)
        I_avg = (corner_cols * Iavg_corner + perimeter_cols * Iavg_perim + interior_cols * Iavg_inter) / max(total_columns, 1)
        I_col_group = inp.column_cracked_factor * (I_avg * max(total_columns, 1) + A_avg * r2_sum)

        scaled.append(replace(z,
                              corner_column_m=max(corner_x, corner_y),
                              perimeter_column_m=max(perimeter_x, perimeter_y),
                              interior_column_m=max(interior_x, interior_y),
                              corner_column_x_m=corner_x, corner_column_y_m=corner_y,
                              perimeter_column_x_m=perimeter_x, perimeter_column_y_m=perimeter_y,
                              interior_column_x_m=interior_x, interior_column_y_m=interior_y,
                              I_col_group_effective_m4=I_col_group))
    return scaled


def weighted_core_stiffness(inp: BuildingInput, zone_cores: List[ZoneCoreResult]) -> float:
    H = total_height(inp)
    E = inp.Ec * 1e6
    total_flex_factor = 0.0
    for zc in zone_cores:
        hi = zc.zone.n_stories * inp.story_height
        perimeter_bonus = 1.0 + 0.20 * len(zc.perimeter_wall_segments)
        total_flex_factor += (hi / H) / max(E * zc.Ieq_effective_m4 * perimeter_bonus, 1e-9)
    EI_equiv = 1.0 / max(total_flex_factor, 1e-18)
    return cantilever_tip_stiffness(EI_equiv, H)


def weighted_column_stiffness(inp: BuildingInput, zone_cols: List[ZoneColumnResult]) -> float:
    H = total_height(inp)
    E = inp.Ec * 1e6
    total_flex_factor = 0.0
    for zc in zone_cols:
        hi = zc.zone.n_stories * inp.story_height
        total_flex_factor += (hi / H) / max(E * zc.I_col_group_effective_m4, 1e-9)
    EI_equiv = 1.0 / max(total_flex_factor, 1e-18)
    return cantilever_tip_stiffness(EI_equiv, H)


def estimate_reinforcement(inp: BuildingInput, zone_cores: List[ZoneCoreResult], zone_cols: List[ZoneColumnResult], slab_t: float, beam_b: float, beam_h: float) -> ReinforcementEstimate:
    n_total_levels = inp.n_story + inp.n_basement
    total_floor_area = floor_area(inp) * n_total_levels
    wall_concrete = 0.0

    for zc in zone_cores:
        wall_concrete += sum(zc.wall_lengths) * zc.wall_thickness * (zc.zone.n_stories * inp.story_height)
        if inp.plan_shape == "square":
            for _, a, b in zc.perimeter_wall_segments:
                wall_concrete += (b - a) * zc.wall_thickness * (zc.zone.n_stories * inp.story_height)
        else:
            perim_lengths = {"edge1": inp.plan_x, "edge2": inp.plan_y, "edge3": (inp.plan_x**2 + inp.plan_y**2) ** 0.5}
            for side, a, b in zc.perimeter_wall_segments:
                wall_concrete += perim_lengths[side] * (b - a) * zc.wall_thickness * (zc.zone.n_stories * inp.story_height)

    total_cols, corner_cols, perimeter_cols, interior_cols, _ = count_columns(inp)
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
    total_steel = wall_steel + column_steel + beam_steel + slab_steel

    return ReinforcementEstimate(
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


def evaluate_design(inp: BuildingInput, base_cores: List[ZoneCoreResult], base_cols: List[ZoneColumnResult], slab_t: float, beam_b: float, beam_h: float, M_eff: float, T_target: float, T_limit: float, W_total: float, lambda_core: float, lambda_col: float):
    zone_cores = scale_core_results(inp, base_cores, lambda_core)
    zone_cols = scale_column_results(inp, base_cols, lambda_col)

    K_core = weighted_core_stiffness(inp, zone_cores)
    K_cols = weighted_column_stiffness(inp, zone_cols)
    K_est = K_core + K_cols
    T_est = 2.0 * pi * sqrt(M_eff / K_est)
    drift = preliminary_lateral_force_N(inp, W_total) / K_est
    drift_ratio = drift / total_height(inp)
    K_req = required_stiffness(M_eff, T_target)
    reinforcement = estimate_reinforcement(inp, zone_cores, zone_cols, slab_t, beam_b, beam_h)

    objective = (reinforcement.wall_concrete_volume_m3 + reinforcement.column_concrete_volume_m3 + reinforcement.beam_concrete_volume_m3 + reinforcement.slab_concrete_volume_m3) + reinforcement.total_steel_kg / 1000.0

    return {
        "zone_cores": zone_cores,
        "zone_cols": zone_cols,
        "K_core": K_core,
        "K_cols": K_cols,
        "K_est": K_est,
        "K_req": K_req,
        "T_est": T_est,
        "drift": drift,
        "drift_ratio": drift_ratio,
        "reinforcement": reinforcement,
        "objective": objective,
        "period_ok": T_est <= T_limit,
    }


def optimize_design(inp: BuildingInput, base_cores: List[ZoneCoreResult], base_cols: List[ZoneColumnResult], slab_t: float, beam_b: float, beam_h: float, M_eff: float, T_target: float, T_limit: float, W_total: float) -> tuple[OptimizationResult, dict]:
    def obj(x):
        ev = evaluate_design(inp, base_cores, base_cols, slab_t, beam_b, beam_h, M_eff, T_target, T_limit, W_total, x[0], x[1])
        return ev["objective"]

    def c_stiff(x):
        ev = evaluate_design(inp, base_cores, base_cols, slab_t, beam_b, beam_h, M_eff, T_target, T_limit, W_total, x[0], x[1])
        return ev["K_est"] - ev["K_req"]

    def c_drift(x):
        ev = evaluate_design(inp, base_cores, base_cols, slab_t, beam_b, beam_h, M_eff, T_target, T_limit, W_total, x[0], x[1])
        return inp.drift_limit_ratio - ev["drift_ratio"]

    def c_period_limit(x):
        ev = evaluate_design(inp, base_cores, base_cols, slab_t, beam_b, beam_h, M_eff, T_target, T_limit, W_total, x[0], x[1])
        return T_limit - ev["T_est"]

    def c_wall_slenderness(x):
        ev = evaluate_design(inp, base_cores, base_cols, slab_t, beam_b, beam_h, M_eff, T_target, T_limit, W_total, x[0], x[1])
        return min(inp.max_story_wall_slenderness - z.story_slenderness for z in ev["zone_cores"])

    bounds = [(0.60, 1.60), (0.60, 1.60)]
    x0 = np.array([1.0, 1.0])
    constraints = [
        {"type": "ineq", "fun": c_stiff},
        {"type": "ineq", "fun": c_drift},
        {"type": "ineq", "fun": c_period_limit},
        {"type": "ineq", "fun": c_wall_slenderness},
    ]

    res = opt.minimize(obj, x0, method="SLSQP", bounds=bounds, constraints=constraints, options={"maxiter": 80, "ftol": 1e-6})

    if not res.success:
        # fallback to feasible base sizing
        x_best = x0
        message = f"Optimizer fallback used: {res.message}"
        success = False
        nit = int(getattr(res, "nit", 0))
    else:
        x_best = res.x
        message = str(res.message)
        success = True
        nit = int(getattr(res, "nit", 0))

    ev = evaluate_design(inp, base_cores, base_cols, slab_t, beam_b, beam_h, M_eff, T_target, T_limit, W_total, float(x_best[0]), float(x_best[1]))
    opt_result = OptimizationResult(lambda_core=float(x_best[0]), lambda_col=float(x_best[1]), objective_value=float(ev["objective"]), success=success, message=message, n_iterations=nit)
    return opt_result, ev


# ----------------------------- MDOF + MASS PARTICIPATION -----------------------------
def build_story_masses(inp: BuildingInput, total_weight_kN_value: float) -> List[float]:
    if inp.n_story <= 0:
        return []
    W_story_kN = total_weight_kN_value / max(inp.n_story, 1)
    m_story = (W_story_kN * 1000.0) / G
    return [m_story for _ in range(inp.n_story)]


def build_story_stiffnesses(inp: BuildingInput, K_total: float) -> List[float]:
    n = inp.n_story
    if n <= 0:
        return []
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
    total_mass = float(ones.T @ M @ ones)

    mode_shapes = []
    gammas = []
    eff_masses = []
    ratios = []

    for i in range(min(n_modes, eigvecs.shape[1])):
        phi = eigvecs[:, i].reshape(-1, 1)
        gamma = float((phi.T @ M @ ones) / (phi.T @ M @ phi))
        m_eff = float((gamma ** 2) * (phi.T @ M @ phi))
        ratio = m_eff / total_mass if total_mass > 0 else 0.0

        phi_plot = phi.flatten().copy()
        if abs(phi_plot[-1]) > 1e-12:
            phi_plot = phi_plot / phi_plot[-1]
        if phi_plot[-1] < 0:
            phi_plot = -phi_plot

        mode_shapes.append(phi_plot.tolist())
        gammas.append(gamma)
        eff_masses.append(m_eff)
        ratios.append(ratio)

    cumulative = list(np.cumsum(ratios))

    return ModalResult(
        n_dof=len(masses),
        periods_s=periods,
        frequencies_hz=freqs,
        mode_shapes=mode_shapes,
        story_masses_kg=masses,
        story_stiffness_N_per_m=k_stories,
        modal_participation_factors=gammas,
        effective_modal_masses_kg=eff_masses,
        effective_mass_ratios=ratios,
        cumulative_effective_mass_ratios=cumulative,
    )


# ----------------------------- MAIN DESIGN -----------------------------
def run_design(inp: BuildingInput) -> DesignResult:
    H = total_height(inp)
    A = floor_area(inp)
    zones = define_three_zones(inp.n_story)
    slab_t = slab_thickness_prelim(inp)
    beam_b, beam_h = beam_size_prelim(inp)
    W_total = total_weight_kN(inp, slab_t)
    M_eff = effective_modal_mass(W_total, inp.effective_modal_mass_ratio)

    T_code = code_type_period(H, inp.Ct, inp.x_period)
    T_limit = 1.40 * T_code
    T_target = inp.target_period_factor * T_code

    base_cores = design_core_by_zone(inp, zones)
    base_cols = estimate_zone_column_sizes(inp, zones, slab_t)

    opt_result, ev = optimize_design(inp, base_cores, base_cols, slab_t, beam_b, beam_h, M_eff, T_target, T_limit, W_total)
    zone_cores = ev["zone_cores"]
    zone_cols = ev["zone_cols"]
    K_core = ev["K_core"]
    K_cols = ev["K_cols"]
    K_est = ev["K_est"]
    K_req = ev["K_req"]
    T_est = ev["T_est"]
    drift_ratio = ev["drift_ratio"]
    top_drift = ev["drift"]
    reinforcement = ev["reinforcement"]
    period_ok = ev["period_ok"]

    modal = solve_mdof_modes(inp, W_total, K_est, n_modes=5)

    messages = []
    messages.append(f"Optimization status: {'SUCCESS' if opt_result.success else 'FALLBACK'} | {opt_result.message}")
    messages.append(f"Optimized lambda_core = {opt_result.lambda_core:.3f}")
    messages.append(f"Optimized lambda_col  = {opt_result.lambda_col:.3f}")
    if K_est < K_req:
        messages.append("Estimated total stiffness is lower than required stiffness from the design target period.")
    if drift_ratio > inp.drift_limit_ratio:
        messages.append("Estimated top drift exceeds selected preliminary drift limit.")
    if period_ok:
        messages.append(f"Period check OK: T_est = {T_est:.3f} s <= 1.40*T_code = {T_limit:.3f} s.")
    else:
        messages.append(f"Period check NOT OK: T_est = {T_est:.3f} s > 1.40*T_code = {T_limit:.3f} s.")
    for zc in zone_cores:
        if zc.story_slenderness > inp.max_story_wall_slenderness:
            messages.append(f"{zc.zone.name}: wall slenderness h/t exceeds selected preliminary limit.")
    messages.append("Code empirical period uses T_code = Ct * H^x.")
    messages.append("Design target period drives optimized section scaling.")
    messages.append("Modal mass participation ratios are computed from M and modal vectors.")

    assessment = (
        "System appears preliminarily adequate and optimized."
        if (K_est >= K_req and drift_ratio <= inp.drift_limit_ratio and period_ok)
        else "System is not yet adequate; revise constraints, minimum sizes, or the target period."
    )

    return DesignResult(
        H_m=H,
        floor_area_m2=A,
        total_weight_kN=W_total,
        effective_modal_mass_kg=M_eff,
        T_code_s=T_code,
        T_limit_s=T_limit,
        period_ok=period_ok,
        T_target_s=T_target,
        T_est_s=T_est,
        K_required_N_per_m=K_req,
        K_core_N_per_m=K_core,
        K_columns_N_per_m=K_cols,
        K_estimated_N_per_m=K_est,
        top_drift_m=top_drift,
        drift_ratio=drift_ratio,
        zone_core_results=zone_cores,
        zone_column_results=zone_cols,
        slab_thickness_m=slab_t,
        beam_width_m=beam_b,
        beam_depth_m=beam_h,
        reinforcement=reinforcement,
        system_assessment=assessment,
        optimization=opt_result,
        messages=messages,
        modal_result=modal,
    )


# ----------------------------- REPORTING -----------------------------
def build_report(result: DesignResult) -> str:
    lines = []
    lines.append("GLOBAL RESPONSE")
    lines.append("-" * 74)
    lines.append(f"Estimated period from M/K      = {result.T_est_s:.3f} s")
    lines.append(f"Code empirical period          = {result.T_code_s:.3f} s")
    lines.append(f"TBDY period upper limit        = {result.T_limit_s:.3f} s")
    lines.append(f"Design target period           = {result.T_target_s:.3f} s")
    lines.append(f"TBDY period check              = {'OK' if result.period_ok else 'NOT OK'}")
    lines.append(f"Required stiffness             = {result.K_required_N_per_m:,.3e} N/m")
    lines.append(f"Core stiffness                 = {result.K_core_N_per_m:,.3e} N/m")
    lines.append(f"Column stiffness contribution  = {result.K_columns_N_per_m:,.3e} N/m")
    lines.append(f"Total estimated stiffness      = {result.K_estimated_N_per_m:,.3e} N/m")
    lines.append(f"Estimated top drift            = {result.top_drift_m:.3f} m")
    lines.append(f"Total structural weight        = {result.total_weight_kN:,.0f} kN")
    lines.append(f"Effective modal mass (input)   = {result.effective_modal_mass_kg:,.0f} kg")
    lines.append(f"Estimated drift ratio          = {result.drift_ratio:.5f}")
    lines.append(f"Beam size (b x h)              = {result.beam_width_m:.2f} x {result.beam_depth_m:.2f} m")
    lines.append(f"Slab thickness                 = {result.slab_thickness_m:.2f} m")
    lines.append("")
    lines.append("OPTIMIZATION")
    lines.append("-" * 74)
    lines.append(f"lambda_core                    = {result.optimization.lambda_core:.3f}")
    lines.append(f"lambda_col                     = {result.optimization.lambda_col:.3f}")
    lines.append(f"objective                      = {result.optimization.objective_value:.3f}")
    lines.append(f"success                        = {result.optimization.success}")
    lines.append(f"message                        = {result.optimization.message}")
    lines.append(f"iterations                     = {result.optimization.n_iterations}")
    lines.append("")
    lines.append("MODAL MASS PARTICIPATION")
    lines.append("-" * 74)
    if result.modal_result is not None:
        mr = result.modal_result
        for i in range(len(mr.periods_s)):
            lines.append(
                f"Mode {i+1}: T={mr.periods_s[i]:.4f} s | f={mr.frequencies_hz[i]:.4f} Hz | "
                f"Gamma={mr.modal_participation_factors[i]:.4f} | "
                f"Meff ratio={100*mr.effective_mass_ratios[i]:.2f}% | "
                f"Cumulative={100*mr.cumulative_effective_mass_ratios[i]:.2f}%"
            )
    lines.append("")
    lines.append("SYSTEM ASSESSMENT")
    lines.append("-" * 74)
    lines.append(result.system_assessment)
    lines.append("")
    lines.append("MESSAGES")
    lines.append("-" * 74)
    for m in result.messages:
        lines.append(f"- {m}")
    return "\n".join(lines)


# ----------------------------- DISPLAY -----------------------------
def plot_mode_shapes(result: DesignResult):
    mr = result.modal_result
    if mr is None:
        raise ValueError("Modal result is not available.")
    n_modes = min(5, len(mr.mode_shapes))
    n_story = mr.n_dof
    H = result.H_m
    fig, axes = plt.subplots(1, n_modes, figsize=(18, 6))
    if n_modes == 1:
        axes = [axes]
    y = np.linspace(0.0, H, n_story)
    for m in range(n_modes):
        ax = axes[m]
        phi = np.array(mr.mode_shapes[m], dtype=float)
        max_abs = max(np.max(np.abs(phi)), 1e-9)
        phi = phi / max_abs
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
            ax.set_ylabel("Height (m)", fontsize=10)
            ax.set_yticks([0.0, H])
            ax.set_yticklabels([f"Base\n0.0", f"Roof\n{H:.1f}"])
        else:
            ax.set_yticks([])
        ax.set_xticks([])
        for spine in ax.spines.values():
            spine.set_color("#999999")
            spine.set_linewidth(1.0)
    fig.suptitle("First 5 Mode Shapes (MDOF)", fontsize=15, fontweight="bold")
    fig.tight_layout()
    return fig


def _draw_rect(ax, x, y, w, h, color, fill=True, alpha=1.0, lw=1.0, ls="-", ec=None):
    rect = plt.Rectangle((x, y), w, h, facecolor=color if fill else "none", edgecolor=ec if ec else color, linewidth=lw, linestyle=ls, alpha=alpha)
    ax.add_patch(rect)


def _draw_square_plan(ax, inp: BuildingInput, core: ZoneCoreResult, cols: ZoneColumnResult, result: DesignResult):
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
                dx, dy, color = cols.corner_column_x_m, cols.corner_column_y_m, CORNER_COLOR
            elif at_lr or at_bt:
                dx, dy, color = cols.perimeter_column_x_m, cols.perimeter_column_y_m, PERIM_COLOR
            else:
                dx, dy, color = cols.interior_column_x_m, cols.interior_column_y_m, INTERIOR_COLOR
            _draw_rect(ax, px - dx / 2, py - dy / 2, dx, dy, color, fill=True, alpha=0.95, lw=0.5)
    cx0 = (inp.plan_x - core.core_outer_x) / 2
    cy0 = (inp.plan_y - core.core_outer_y) / 2
    ix0 = (inp.plan_x - core.core_opening_x) / 2
    iy0 = (inp.plan_y - core.core_opening_y) / 2
    _draw_rect(ax, cx0, cy0, core.core_outer_x, core.core_outer_y, CORE_COLOR, fill=False, lw=2.5)
    _draw_rect(ax, ix0, iy0, core.core_opening_x, core.core_opening_y, "#666666", fill=False, lw=1.3, ls="--")
    ax.set_title(f"{core.zone.name} - Square plan", fontsize=14, fontweight="bold")
    ax.set_xlim(-8, inp.plan_x + 8)
    ax.set_ylim(inp.plan_y + 8, -8)
    ax.set_aspect("equal")
    ax.axis("off")


def _draw_triangle_plan(ax, inp: BuildingInput, core: ZoneCoreResult, cols: ZoneColumnResult):
    pts = np.array([[0, inp.plan_y], [inp.plan_x / 2, 0], [inp.plan_x, inp.plan_y], [0, inp.plan_y]])
    ax.plot(pts[:, 0], pts[:, 1], color="black", linewidth=1.5)
    corner_pts = [(0, inp.plan_y), (inp.plan_x / 2, 0), (inp.plan_x, inp.plan_y)]
    for x, y in corner_pts:
        _draw_rect(ax, x - cols.corner_column_x_m / 2, y - cols.corner_column_y_m / 2, cols.corner_column_x_m, cols.corner_column_y_m, CORNER_COLOR, fill=True, alpha=0.95)
    cx0 = (inp.plan_x - core.core_outer_x) / 2
    cy0 = inp.plan_y * 0.42
    _draw_rect(ax, cx0, cy0, core.core_outer_x, core.core_outer_y, CORE_COLOR, fill=False, lw=2.5)
    ax.set_title(f"{core.zone.name} - Triangular plan", fontsize=14, fontweight="bold")
    ax.set_xlim(-8, inp.plan_x + 8)
    ax.set_ylim(inp.plan_y + 8, -8)
    ax.set_aspect("equal")
    ax.axis("off")


def plot_plan(inp: BuildingInput, result: DesignResult, zone_name: str):
    core = next(z for z in result.zone_core_results if z.zone.name == zone_name)
    cols = next(z for z in result.zone_column_results if z.zone.name == zone_name)
    fig, ax = plt.subplots(figsize=(12, 7))
    if inp.plan_shape == "triangle":
        _draw_triangle_plan(ax, inp, core, cols)
    else:
        _draw_square_plan(ax, inp, core, cols, result)
    return fig


def streamlit_input_panel() -> BuildingInput:
    st.markdown("### Plan Shape")
    plan_shape = st.radio(" ", ["square", "triangle"], horizontal=True, label_visibility="collapsed")
    st.markdown("### Geometry")
    c1, c2 = st.columns(2)
    with c1:
        n_story = st.number_input("Above-grade stories", 1, 120, 50)
        n_basement = st.number_input("Basement stories", 0, 20, 10)
        plan_x = st.number_input("Plan X (m)", 10.0, 300.0, 80.0)
        n_bays_x = st.number_input("Bays in X", 1, 30, 8)
        bay_x = st.number_input("Bay X (m)", 2.0, 20.0, 10.0)
    with c2:
        story_height = st.number_input("Story height (m)", 2.5, 6.0, 3.2)
        basement_height = st.number_input("Basement height (m)", 2.5, 6.0, 3.0)
        plan_y = st.number_input("Plan Y (m)", 10.0, 300.0, 80.0)
        n_bays_y = st.number_input("Bays in Y", 1, 30, 8)
        bay_y = st.number_input("Bay Y (m)", 2.0, 20.0, 10.0)
    st.markdown("### Loads / Materials")
    c3, c4 = st.columns(2)
    with c3:
        DL = st.number_input("DL (kN/mÂ²)", 0.0, 20.0, 6.5)
        slab_finish_allowance = st.number_input("Slab/fit-out allowance", 0.0, 10.0, 1.5)
        fck = st.number_input("fck (MPa)", 20.0, 100.0, 60.0)
        wall_cracked_factor = st.number_input("Wall cracked factor", 0.1, 1.0, 0.7)
        target_period_factor = st.number_input("Target period factor", 0.1, 2.0, 0.95)
    with c4:
        LL = st.number_input("LL (kN/mÂ²)", 0.0, 20.0, 2.5)
        facade_line_load = st.number_input("Facade line load (kN/m)", 0.0, 50.0, 14.0)
        Ec = st.number_input("Ec (MPa)", 20000.0, 60000.0, 36000.0)
        column_cracked_factor = st.number_input("Column cracked factor", 0.1, 1.0, 0.7)
        effective_modal_mass_ratio = st.number_input("Effective modal mass ratio", 0.1, 1.0, 0.80)
    return BuildingInput(
        plan_shape=plan_shape,
        n_story=int(n_story), n_basement=int(n_basement), story_height=float(story_height), basement_height=float(basement_height),
        plan_x=float(plan_x), plan_y=float(plan_y), n_bays_x=int(n_bays_x), n_bays_y=int(n_bays_y), bay_x=float(bay_x), bay_y=float(bay_y),
        DL=float(DL), LL=float(LL), slab_finish_allowance=float(slab_finish_allowance), facade_line_load=float(facade_line_load),
        fck=float(fck), Ec=float(Ec), wall_cracked_factor=float(wall_cracked_factor), column_cracked_factor=float(column_cracked_factor),
        target_period_factor=float(target_period_factor), effective_modal_mass_ratio=float(effective_modal_mass_ratio)
    )


# ----------------------------- UI -----------------------------
st.title("Tall Building Structural Analysis â Optimized")
author_name = st.text_input("Author name", value=DEFAULT_AUTHOR_NAME)
st.caption(f"App version {APP_VERSION}")

if "result" not in st.session_state:
    st.session_state.result = None
if "report" not in st.session_state:
    st.session_state.report = ""
if "view_mode" not in st.session_state:
    st.session_state.view_mode = "plan"

left_col, right_col = st.columns([1.0, 2.1], gap="medium")

with left_col:
    inp = streamlit_input_panel()
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
            st.download_button("SAVE REPORT", data=st.session_state.report.encode("utf-8"), file_name=f"tall_building_report_{author_name}.txt", mime="text/plain")
        else:
            st.button("SAVE REPORT", disabled=True)

with right_col:
    zone_name = st.selectbox("Displayed zone:", ["Lower Zone", "Middle Zone", "Upper Zone"], index=0)
    if st.session_state.result is None:
        st.info("Run ANALYZE to optimize sections and display the plan, or SHOW 5 MODES to display modal shapes.")
    else:
        r = st.session_state.result
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Code period (s)", f"{r.T_code_s:.3f}")
        c2.metric("TBDY limit (s)", f"{r.T_limit_s:.3f}")
        c3.metric("Estimated period (s)", f"{r.T_est_s:.3f}")
        c4.metric("Top drift (m)", f"{r.top_drift_m:.3f}")
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Design target (s)", f"{r.T_target_s:.3f}")
        d2.metric("Core scale", f"{r.optimization.lambda_core:.3f}")
        d3.metric("Column scale", f"{r.optimization.lambda_col:.3f}")
        d4.metric("TBDY check", "OK" if r.period_ok else "NOT OK")

        if st.session_state.view_mode == "modes":
            st.pyplot(plot_mode_shapes(r), use_container_width=True)
            if r.modal_result is not None:
                st.markdown("**Mass participation**")
                rows = []
                for i in range(len(r.modal_result.periods_s)):
                    rows.append({
                        "Mode": i + 1,
                        "Period (s)": round(r.modal_result.periods_s[i], 4),
                        "Freq. (Hz)": round(r.modal_result.frequencies_hz[i], 4),
                        "Gamma": round(r.modal_result.modal_participation_factors[i], 4),
                        "Eff. mass ratio (%)": round(100 * r.modal_result.effective_mass_ratios[i], 2),
                        "Cumulative (%)": round(100 * r.modal_result.cumulative_effective_mass_ratios[i], 2),
                    })
                st.dataframe(rows, use_container_width=True)
        else:
            st.pyplot(plot_plan(inp, r, zone_name), use_container_width=True)

        st.text_area("", st.session_state.report, height=380, label_visibility="collapsed")
