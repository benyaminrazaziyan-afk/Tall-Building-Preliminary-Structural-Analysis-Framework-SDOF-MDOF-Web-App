from __future__ import annotations

from dataclasses import dataclass, field, asdict
from math import pi, sqrt
from typing import List, Tuple, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.optimize import minimize

# ----------------------------- CONFIG -----------------------------
st.set_page_config(
    page_title="Tall Building Preliminary Structural Analysis",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_VERSION = "v4.0-Principled-Storywise"
AUTHOR_NAME = "Benyamin"

G = 9.81
STEEL_DENSITY = 7850.0
CONCRETE_UNIT_WEIGHT = 25.0  # kN/m3

CORNER_COLOR = "#8b0000"
PERIM_COLOR = "#cc5500"
INTERIOR_COLOR = "#4444aa"
CORE_COLOR = "#2e8b57"
PERIM_WALL_COLOR = "#4caf50"


# ----------------------------- DATA MODELS -----------------------------
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
    beam_frame_factor: float = 1.15
    max_story_wall_slenderness: float = 12.0

    wall_rebar_ratio: float = 0.0030
    column_rebar_ratio: float = 0.0100
    beam_rebar_ratio: float = 0.0150
    slab_rebar_ratio: float = 0.0035

    seismic_mass_factor: float = 1.0
    Ct: float = 0.0488
    x_period: float = 0.75
    upper_period_factor: float = 1.40
    target_position_factor: float = 0.85

    perimeter_column_factor: float = 1.10
    corner_column_factor: float = 1.30

    lower_zone_wall_count: int = 8
    middle_zone_wall_count: int = 6
    upper_zone_wall_count: int = 4

    basement_retaining_wall_thickness: float = 0.50
    perimeter_shear_wall_ratio: float = 0.20


@dataclass
class StoryResult:
    story: int
    zone: str
    floors_supported: float
    slab_thickness_m: float
    beam_width_m: float
    beam_depth_m: float
    wall_thickness_m: float
    wall_count: int
    core_outer_x_m: float
    core_outer_y_m: float
    core_opening_x_m: float
    core_opening_y_m: float
    core_wall_total_length_m: float
    perimeter_wall_total_length_m: float
    corner_column_x_m: float
    corner_column_y_m: float
    perimeter_column_x_m: float
    perimeter_column_y_m: float
    interior_column_x_m: float
    interior_column_y_m: float
    p_corner_kN: float
    p_perimeter_kN: float
    p_interior_kN: float
    k_wall_x_N_per_m: float
    k_wall_y_N_per_m: float
    k_column_x_N_per_m: float
    k_column_y_N_per_m: float
    k_story_x_N_per_m: float
    k_story_y_N_per_m: float
    k_story_eq_N_per_m: float
    story_weight_kN: float


@dataclass
class ModalResult:
    periods_s: List[float]
    frequencies_hz: List[float]
    mode_shapes: List[List[float]]
    effective_mass_ratios: List[float]
    cumulative_effective_mass_ratios: List[float]
    story_masses_kg: List[float]
    story_stiffness_N_per_m: List[float]


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
class DesignResult:
    H_m: float
    floor_area_m2: float
    total_weight_kN: float
    K_total_x_N_per_m: float
    K_total_y_N_per_m: float
    K_total_eq_N_per_m: float
    reference_period_s: float
    design_target_period_s: float
    upper_limit_period_s: float
    estimated_period_s: float
    period_error_ratio: float
    period_ok: bool
    drift_ok: bool
    top_drift_m: float
    drift_ratio: float
    core_scale: float
    column_scale: float
    story_results: List[StoryResult]
    modal_result: ModalResult
    reinforcement: ReinforcementEstimate
    messages: List[str] = field(default_factory=list)
    redesign_suggestions: List[str] = field(default_factory=list)


# ----------------------------- BASIC GEOMETRY -----------------------------
def total_height(inp: BuildingInput) -> float:
    return inp.n_story * inp.story_height


def floor_area(inp: BuildingInput) -> float:
    if inp.plan_shape == "triangle":
        return 0.5 * inp.plan_x * inp.plan_y
    return inp.plan_x * inp.plan_y


def code_type_period(H: float, Ct: float, x_period: float) -> float:
    return Ct * (H ** x_period)


def preliminary_lateral_force_N(inp: BuildingInput, W_total_kN: float) -> float:
    return inp.prelim_lateral_force_coeff * W_total_kN * 1000.0


def required_opening_area(inp: BuildingInput) -> float:
    return (
        inp.elevator_count * inp.elevator_area_each
        + inp.stair_count * inp.stair_area_each
        + inp.service_area
    ) * inp.corridor_factor


def opening_dimensions(inp: BuildingInput) -> tuple[float, float]:
    area = required_opening_area(inp)
    aspect = 1.6
    oy = sqrt(max(area / aspect, 1e-9))
    return aspect * oy, oy


def initial_core_dimensions(inp: BuildingInput, opening_x: float, opening_y: float) -> tuple[float, float]:
    outer_x = max(opening_x + 3.0, 0.24 * inp.plan_x)
    outer_y = max(opening_y + 3.0, 0.22 * inp.plan_y)
    return min(outer_x, 0.42 * inp.plan_x), min(outer_y, 0.42 * inp.plan_y)


def zone_name_for_story(inp: BuildingInput, story: int) -> str:
    z1 = max(1, round(0.30 * inp.n_story))
    z2 = max(z1 + 1, round(0.70 * inp.n_story))
    if story <= z1:
        return "Lower Zone"
    if story <= z2:
        return "Middle Zone"
    return "Upper Zone"


def active_wall_count_by_zone(inp: BuildingInput, zone_name: str) -> int:
    return {
        "Lower Zone": inp.lower_zone_wall_count,
        "Middle Zone": inp.middle_zone_wall_count,
        "Upper Zone": inp.upper_zone_wall_count,
    }[zone_name]


def zone_factor(zone_name: str) -> float:
    return {"Lower Zone": 1.00, "Middle Zone": 0.82, "Upper Zone": 0.65}[zone_name]


def directional_dims(base_dim: float, plan_x: float, plan_y: float) -> tuple[float, float]:
    aspect = max(plan_x, plan_y) / max(min(plan_x, plan_y), 1e-9)
    if aspect <= 1.10:
        return base_dim, base_dim
    major = base_dim * 1.12
    minor = base_dim * 0.92
    return (major, minor) if plan_x >= plan_y else (minor, major)


# ----------------------------- WALL LAYOUT -----------------------------
def core_wall_segments(outer_x: float, outer_y: float, wall_count: int) -> List[Dict[str, float | str]]:
    segs: List[Dict[str, float | str]] = [
        {"name": "top", "orientation": "x", "length": outer_x},
        {"name": "bottom", "orientation": "x", "length": outer_x},
        {"name": "left", "orientation": "y", "length": outer_y},
        {"name": "right", "orientation": "y", "length": outer_y},
    ]
    if wall_count >= 6:
        segs.extend([
            {"name": "inner_left", "orientation": "y", "length": 0.45 * outer_y},
            {"name": "inner_right", "orientation": "y", "length": 0.45 * outer_y},
        ])
    if wall_count >= 8:
        segs.extend([
            {"name": "inner_top", "orientation": "x", "length": 0.45 * outer_x},
            {"name": "inner_bottom", "orientation": "x", "length": 0.45 * outer_x},
        ])
    return segs


def perimeter_wall_segments(inp: BuildingInput, zone: str) -> List[Dict[str, float | str]]:
    if zone == "Lower Zone":
        return [
            {"name": "perim_top", "orientation": "x", "length": inp.plan_x},
            {"name": "perim_bottom", "orientation": "x", "length": inp.plan_x},
            {"name": "perim_left", "orientation": "y", "length": inp.plan_y},
            {"name": "perim_right", "orientation": "y", "length": inp.plan_y},
        ]
    ratio = inp.perimeter_shear_wall_ratio
    return [
        {"name": "perim_top", "orientation": "x", "length": inp.plan_x * ratio},
        {"name": "perim_bottom", "orientation": "x", "length": inp.plan_x * ratio},
        {"name": "perim_left", "orientation": "y", "length": inp.plan_y * ratio},
        {"name": "perim_right", "orientation": "y", "length": inp.plan_y * ratio},
    ]


def wall_inplane_I(length: float, thickness: float) -> float:
    return thickness * (length ** 3) / 12.0


def wall_story_stiffness(E: float, h: float, length: float, thickness: float, cracked_factor: float) -> float:
    I = wall_inplane_I(length, thickness)
    return cracked_factor * 12.0 * E * I / max(h ** 3, 1e-12)


def column_story_stiffness(E: float, h: float, dim_x: float, dim_y: float, cracked_factor: float, count: int, direction: str) -> float:
    if direction == "x":
        I = dim_y * dim_x ** 3 / 12.0
    else:
        I = dim_x * dim_y ** 3 / 12.0
    return count * cracked_factor * 12.0 * E * I / max(h ** 3, 1e-12)


# ----------------------------- STORYWISE SIZING -----------------------------
def slab_thickness_prelim(inp: BuildingInput, column_scale: float, story: int) -> float:
    span = max(inp.bay_x, inp.bay_y)
    upper_relief = 1.0 - 0.10 * ((story - 1) / max(inp.n_story - 1, 1))
    base = span / 28.0
    t = base * (0.92 + 0.10 * column_scale) * upper_relief
    return max(inp.min_slab_thickness, min(inp.max_slab_thickness, t))


def beam_size_prelim(inp: BuildingInput, column_scale: float, story: int) -> tuple[float, float]:
    span = max(inp.bay_x, inp.bay_y)
    upper_relief = 1.0 - 0.12 * ((story - 1) / max(inp.n_story - 1, 1))
    depth = (span / 12.0) * (0.92 + 0.15 * column_scale) * upper_relief
    depth = max(inp.min_beam_depth, min(2.0, depth))
    width = max(inp.min_beam_width, 0.45 * depth)
    return width, depth


def wall_thickness_story(inp: BuildingInput, core_scale: float, story: int, zone: str) -> float:
    H = total_height(inp)
    base_t = max(inp.min_wall_thickness, min(inp.max_wall_thickness, H / 180.0))
    taper = 1.0 - 0.18 * ((story - 1) / max(inp.n_story - 1, 1))
    t = base_t * zone_factor(zone) * taper * core_scale
    return max(inp.min_wall_thickness, min(inp.max_wall_thickness, t))


def story_gravity_load_kN(inp: BuildingInput, slab_t: float, beam_b: float, beam_h: float, wall_t: float, zone: str) -> float:
    area = floor_area(inp)
    slab_self = slab_t * CONCRETE_UNIT_WEIGHT * area
    superimposed = (inp.DL + inp.LL + inp.slab_finish_allowance) * area
    beam_lines = inp.n_bays_y * (inp.n_bays_x + 1) + inp.n_bays_x * (inp.n_bays_y + 1)
    avg_span = 0.5 * (inp.bay_x + inp.bay_y)
    beam_self = beam_lines * avg_span * beam_b * beam_h * CONCRETE_UNIT_WEIGHT
    opening_x, opening_y = opening_dimensions(inp)
    outer_x, outer_y = initial_core_dimensions(inp, opening_x, opening_y)
    core_len = sum(float(s["length"]) for s in core_wall_segments(outer_x, outer_y, active_wall_count_by_zone(inp, zone)))
    perim_len = sum(float(s["length"]) for s in perimeter_wall_segments(inp, zone))
    wall_self = (core_len + perim_len) * wall_t * inp.story_height * CONCRETE_UNIT_WEIGHT
    facade = inp.facade_line_load * 2.0 * (inp.plan_x + inp.plan_y)
    return slab_self + superimposed + beam_self + wall_self + facade


def story_column_sizes(inp: BuildingInput, column_scale: float, story: int, slab_t: float, beam_b: float, beam_h: float) -> dict:
    floors_supported = (inp.n_story - story + 1) + 0.7 * inp.n_basement
    q_floor = story_gravity_load_kN(inp, slab_t, beam_b, beam_h, inp.min_wall_thickness, zone_name_for_story(inp, story)) / max(floor_area(inp), 1e-9)
    sigma_allow = 0.35 * inp.fck * 1000.0  # kN/m2

    tributary_interior = inp.bay_x * inp.bay_y
    tributary_perimeter = 0.50 * tributary_interior
    tributary_corner = 0.25 * tributary_interior

    P_interior = 1.18 * q_floor * tributary_interior * floors_supported
    P_perimeter = 1.18 * q_floor * tributary_perimeter * floors_supported
    P_corner = 1.18 * q_floor * tributary_corner * floors_supported

    interior_dim = sqrt(max(P_interior / max(sigma_allow, 1e-9), 1e-9)) * column_scale
    interior_dim = max(inp.min_column_dim, min(inp.max_column_dim, interior_dim))
    perimeter_dim = max(inp.min_column_dim, min(inp.max_column_dim, interior_dim * inp.perimeter_column_factor))
    corner_dim = max(inp.min_column_dim, min(inp.max_column_dim, interior_dim * inp.corner_column_factor))

    ix, iy = directional_dims(interior_dim, inp.plan_x, inp.plan_y)
    px, py = directional_dims(perimeter_dim, inp.plan_x, inp.plan_y)
    cx, cy = directional_dims(corner_dim, inp.plan_x, inp.plan_y)

    return {
        "floors_supported": floors_supported,
        "P_corner": P_corner,
        "P_perimeter": P_perimeter,
        "P_interior": P_interior,
        "corner_x": cx,
        "corner_y": cy,
        "perim_x": px,
        "perim_y": py,
        "interior_x": ix,
        "interior_y": iy,
    }


def build_story_results(inp: BuildingInput, core_scale: float, column_scale: float) -> List[StoryResult]:
    E = inp.Ec * 1e6
    opening_x, opening_y = opening_dimensions(inp)
    outer_x, outer_y = initial_core_dimensions(inp, opening_x, opening_y)

    total_columns = (inp.n_bays_x + 1) * (inp.n_bays_y + 1)
    corner_cols = 4
    perimeter_cols = max(0, 2 * (inp.n_bays_x - 1) + 2 * (inp.n_bays_y - 1))
    interior_cols = max(0, total_columns - corner_cols - perimeter_cols)

    results: List[StoryResult] = []
    for story in range(1, inp.n_story + 1):
        zone = zone_name_for_story(inp, story)
        wall_count = active_wall_count_by_zone(inp, zone)
        slab_t = slab_thickness_prelim(inp, column_scale, story)
        beam_b, beam_h = beam_size_prelim(inp, column_scale, story)
        wall_t = wall_thickness_story(inp, core_scale, story, zone)
        col = story_column_sizes(inp, column_scale, story, slab_t, beam_b, beam_h)

        core_segs = core_wall_segments(outer_x, outer_y, wall_count)
        perim_segs = perimeter_wall_segments(inp, zone)
        core_wall_total_length = sum(float(s["length"]) for s in core_segs)
        perim_wall_total_length = sum(float(s["length"]) for s in perim_segs)

        k_wall_x = sum(
            wall_story_stiffness(E, inp.story_height, float(s["length"]), wall_t, inp.wall_cracked_factor)
            for s in core_segs + perim_segs if s["orientation"] == "x"
        )
        k_wall_y = sum(
            wall_story_stiffness(E, inp.story_height, float(s["length"]), wall_t, inp.wall_cracked_factor)
            for s in core_segs + perim_segs if s["orientation"] == "y"
        )

        k_col_x = inp.beam_frame_factor * (
            column_story_stiffness(E, inp.story_height, col["corner_x"], col["corner_y"], inp.column_cracked_factor, corner_cols, "x")
            + column_story_stiffness(E, inp.story_height, col["perim_x"], col["perim_y"], inp.column_cracked_factor, perimeter_cols, "x")
            + column_story_stiffness(E, inp.story_height, col["interior_x"], col["interior_y"], inp.column_cracked_factor, interior_cols, "x")
        )
        k_col_y = inp.beam_frame_factor * (
            column_story_stiffness(E, inp.story_height, col["corner_x"], col["corner_y"], inp.column_cracked_factor, corner_cols, "y")
            + column_story_stiffness(E, inp.story_height, col["perim_x"], col["perim_y"], inp.column_cracked_factor, perimeter_cols, "y")
            + column_story_stiffness(E, inp.story_height, col["interior_x"], col["interior_y"], inp.column_cracked_factor, interior_cols, "y")
        )

        k_story_x = k_wall_x + k_col_x
        k_story_y = k_wall_y + k_col_y
        k_story_eq = sqrt(max(k_story_x * k_story_y, 1e-12))
        story_weight = story_gravity_load_kN(inp, slab_t, beam_b, beam_h, wall_t, zone)

        results.append(
            StoryResult(
                story=story,
                zone=zone,
                floors_supported=col["floors_supported"],
                slab_thickness_m=slab_t,
                beam_width_m=beam_b,
                beam_depth_m=beam_h,
                wall_thickness_m=wall_t,
                wall_count=wall_count,
                core_outer_x_m=outer_x,
                core_outer_y_m=outer_y,
                core_opening_x_m=opening_x,
                core_opening_y_m=opening_y,
                core_wall_total_length_m=core_wall_total_length,
                perimeter_wall_total_length_m=perim_wall_total_length,
                corner_column_x_m=col["corner_x"],
                corner_column_y_m=col["corner_y"],
                perimeter_column_x_m=col["perim_x"],
                perimeter_column_y_m=col["perim_y"],
                interior_column_x_m=col["interior_x"],
                interior_column_y_m=col["interior_y"],
                p_corner_kN=col["P_corner"],
                p_perimeter_kN=col["P_perimeter"],
                p_interior_kN=col["P_interior"],
                k_wall_x_N_per_m=k_wall_x,
                k_wall_y_N_per_m=k_wall_y,
                k_column_x_N_per_m=k_col_x,
                k_column_y_N_per_m=k_col_y,
                k_story_x_N_per_m=k_story_x,
                k_story_y_N_per_m=k_story_y,
                k_story_eq_N_per_m=k_story_eq,
                story_weight_kN=story_weight,
            )
        )
    return results


# ----------------------------- QUANTITIES / MODAL -----------------------------
def equivalent_series_stiffness(k_story_values: List[float]) -> float:
    return 1.0 / max(sum(1.0 / max(k, 1e-12) for k in k_story_values), 1e-18)


def build_story_masses_kg(inp: BuildingInput, story_results: List[StoryResult]) -> List[float]:
    return [sr.story_weight_kN * 1000.0 * inp.seismic_mass_factor / G for sr in story_results]


def assemble_m_k_matrices(story_masses: List[float], story_stiffness: List[float]) -> tuple[np.ndarray, np.ndarray]:
    n = len(story_masses)
    M = np.diag(story_masses)
    K = np.zeros((n, n), dtype=float)
    for i, ki in enumerate(story_stiffness):
        if i == 0:
            K[i, i] += ki
        else:
            K[i, i] += ki
            K[i, i - 1] -= ki
            K[i - 1, i] -= ki
            K[i - 1, i - 1] += ki
    return M, K


def solve_mdof_modes(story_masses: List[float], story_stiffness: List[float], n_modes: int = 5) -> ModalResult:
    M, K = assemble_m_k_matrices(story_masses, story_stiffness)
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

    periods = [2.0 * pi / w for w in omegas[:n_modes]]
    freqs = [w / (2.0 * pi) for w in omegas[:n_modes]]

    ones = np.ones((len(story_masses), 1))
    total_mass = float(np.sum(np.diag(M)))
    mass_ratios = []
    cumulative = []
    mode_shapes = []
    cum = 0.0
    for i in range(min(n_modes, eigvecs.shape[1])):
        phi = eigvecs[:, i].reshape(-1, 1)
        denom = float((phi.T @ M @ phi).item())
        gamma = float(((phi.T @ M @ ones) / denom).item())
        meff = gamma ** 2 * denom
        ratio = meff / max(total_mass, 1e-12)
        cum += ratio
        shape = phi.flatten().copy()
        if abs(shape[-1]) > 1e-12:
            shape = shape / shape[-1]
        if shape[-1] < 0:
            shape = -shape
        mode_shapes.append(shape.tolist())
        mass_ratios.append(ratio)
        cumulative.append(cum)

    return ModalResult(
        periods_s=periods,
        frequencies_hz=freqs,
        mode_shapes=mode_shapes,
        effective_mass_ratios=mass_ratios,
        cumulative_effective_mass_ratios=cumulative,
        story_masses_kg=story_masses,
        story_stiffness_N_per_m=story_stiffness,
    )


def estimate_reinforcement(inp: BuildingInput, story_results: List[StoryResult]) -> ReinforcementEstimate:
    corner_cols = 4
    perimeter_cols = max(0, 2 * (inp.n_bays_x - 1) + 2 * (inp.n_bays_y - 1))
    total_cols = (inp.n_bays_x + 1) * (inp.n_bays_y + 1)
    interior_cols = max(0, total_cols - corner_cols - perimeter_cols)
    beam_lines = inp.n_bays_y * (inp.n_bays_x + 1) + inp.n_bays_x * (inp.n_bays_y + 1)
    avg_span = 0.5 * (inp.bay_x + inp.bay_y)
    floorA = floor_area(inp)

    wall_concrete = 0.0
    column_concrete = 0.0
    beam_concrete = 0.0
    slab_concrete = 0.0
    for sr in story_results:
        story_h = inp.story_height
        wall_concrete += (sr.core_wall_total_length_m + sr.perimeter_wall_total_length_m) * sr.wall_thickness_m * story_h
        column_concrete += story_h * (
            corner_cols * sr.corner_column_x_m * sr.corner_column_y_m
            + perimeter_cols * sr.perimeter_column_x_m * sr.perimeter_column_y_m
            + interior_cols * sr.interior_column_x_m * sr.interior_column_y_m
        )
        beam_concrete += beam_lines * avg_span * sr.beam_width_m * sr.beam_depth_m
        slab_concrete += floorA * sr.slab_thickness_m

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


def total_weight_kN(inp: BuildingInput, story_results: List[StoryResult], reinf: ReinforcementEstimate) -> float:
    superstructure = sum(sr.story_weight_kN for sr in story_results)
    basement_extra = inp.n_basement * floor_area(inp) * (inp.DL + 0.5 * inp.LL + inp.slab_finish_allowance)
    steel_weight = reinf.total_steel_kg * G / 1000.0
    return (superstructure + basement_extra + steel_weight) * inp.seismic_mass_factor


# ----------------------------- DESIGN EVALUATION -----------------------------
def redesign_suggestions(T_est: float, T_target: float, T_upper: float, drift_ratio: float, drift_limit: float) -> List[str]:
    notes: List[str] = []
    if T_est > T_upper:
        notes.append("Period exceeds upper limit; increase core wall thickness and/or wall count.")
    elif T_est > 1.08 * T_target:
        notes.append("System is too flexible; prioritize wall stiffness before enlarging interior columns.")
    elif T_est < 0.92 * T_target:
        notes.append("System is too stiff; reduce over-conservative wall/column sizes if economical design is desired.")
    if drift_ratio > drift_limit:
        notes.append("Drift exceeds limit; increase story stiffness in lower and middle stories.")
    if not notes:
        notes.append("Preliminary system is within the chosen period and drift targets.")
    return notes


def evaluate_design(inp: BuildingInput, core_scale: float, column_scale: float) -> DesignResult:
    H = total_height(inp)
    T_ref = code_type_period(H, inp.Ct, inp.x_period)
    T_upper = inp.upper_period_factor * T_ref
    T_target = T_ref + inp.target_position_factor * (T_upper - T_ref)

    story_results = build_story_results(inp, core_scale, column_scale)
    reinf = estimate_reinforcement(inp, story_results)
    W_total = total_weight_kN(inp, story_results, reinf)

    kx = [sr.k_story_x_N_per_m for sr in story_results]
    ky = [sr.k_story_y_N_per_m for sr in story_results]
    keq = [sr.k_story_eq_N_per_m for sr in story_results]

    K_total_x = equivalent_series_stiffness(kx)
    K_total_y = equivalent_series_stiffness(ky)
    K_total_eq = equivalent_series_stiffness(keq)

    masses = build_story_masses_kg(inp, story_results)
    modal = solve_mdof_modes(masses, keq, n_modes=5)
    T_est = modal.periods_s[0]

    V = preliminary_lateral_force_N(inp, W_total)
    top_drift = V / max(K_total_eq, 1e-12)
    drift_ratio = top_drift / max(H, 1e-12)
    period_error = abs(T_est - T_target) / max(T_target, 1e-12)

    wall_share = sum(sr.k_wall_x_N_per_m + sr.k_wall_y_N_per_m for sr in story_results)
    col_share = sum(sr.k_column_x_N_per_m + sr.k_column_y_N_per_m for sr in story_results)
    total_share = max(wall_share + col_share, 1e-12)

    messages = [
        f"APP_VERSION = {APP_VERSION}",
        f"Reference period T_ref = {T_ref:.3f} s",
        f"Target period T_target = {T_target:.3f} s",
        f"Estimated period T_est = {T_est:.3f} s",
        f"Total equivalent stiffness = {K_total_eq:,.3e} N/m",
        f"Wall stiffness share = {100.0 * wall_share / total_share:.1f}%",
        f"Column/frame stiffness share = {100.0 * col_share / total_share:.1f}%",
        f"Mode-1 mass participation = {100.0 * modal.effective_mass_ratios[0]:.1f}%",
    ]

    return DesignResult(
        H_m=H,
        floor_area_m2=floor_area(inp),
        total_weight_kN=W_total,
        K_total_x_N_per_m=K_total_x,
        K_total_y_N_per_m=K_total_y,
        K_total_eq_N_per_m=K_total_eq,
        reference_period_s=T_ref,
        design_target_period_s=T_target,
        upper_limit_period_s=T_upper,
        estimated_period_s=T_est,
        period_error_ratio=period_error,
        period_ok=(T_est <= T_upper),
        drift_ok=(drift_ratio <= inp.drift_limit_ratio),
        top_drift_m=top_drift,
        drift_ratio=drift_ratio,
        core_scale=core_scale,
        column_scale=column_scale,
        story_results=story_results,
        modal_result=modal,
        reinforcement=reinf,
        messages=messages,
        redesign_suggestions=redesign_suggestions(T_est, T_target, T_upper, drift_ratio, inp.drift_limit_ratio),
    )


def optimize_design(inp: BuildingInput) -> DesignResult:
    def objective(x: np.ndarray) -> float:
        core_scale = float(x[0])
        column_scale = float(x[1])
        result = evaluate_design(inp, core_scale, column_scale)
        obj = 900.0 * result.period_error_ratio ** 2
        obj += 4000.0 * max(result.estimated_period_s / result.upper_limit_period_s - 1.0, 0.0) ** 2
        obj += 4000.0 * max(result.drift_ratio / inp.drift_limit_ratio - 1.0, 0.0) ** 2
        obj += 0.15 * (result.total_weight_kN / 1e6)
        obj += 2.0 * (core_scale - column_scale) ** 2
        return obj

    res = minimize(objective, np.array([1.0, 1.0]), bounds=[(0.25, 2.50), (0.25, 2.50)], method="L-BFGS-B")
    return evaluate_design(inp, float(res.x[0]), float(res.x[1]))


# ----------------------------- REPORTING -----------------------------
def story_schedule_dataframe(story_results: List[StoryResult]) -> pd.DataFrame:
    df = pd.DataFrame([asdict(sr) for sr in story_results])
    return df.sort_values("story", ascending=False).reset_index(drop=True)


def build_report(result: DesignResult) -> str:
    lines = []
    lines.append("=" * 88)
    lines.append("TALL BUILDING PRELIMINARY DESIGN REPORT - PRINCIPLED STORYWISE MODEL")
    lines.append("=" * 88)
    lines.append(f"Total height H                         = {result.H_m:.2f} m")
    lines.append(f"Floor area                            = {result.floor_area_m2:.2f} m2")
    lines.append(f"Total seismic weight                  = {result.total_weight_kN:,.0f} kN")
    lines.append(f"Reference period T_ref                = {result.reference_period_s:.3f} s")
    lines.append(f"Target period T_target                = {result.design_target_period_s:.3f} s")
    lines.append(f"Estimated period T_est                = {result.estimated_period_s:.3f} s")
    lines.append(f"Upper period limit                    = {result.upper_limit_period_s:.3f} s")
    lines.append(f"Period error                          = {100.0 * result.period_error_ratio:.2f} %")
    lines.append(f"Period check                          = {'OK' if result.period_ok else 'NOT OK'}")
    lines.append(f"Drift ratio                           = {result.drift_ratio:.5f}")
    lines.append(f"Drift check                           = {'OK' if result.drift_ok else 'NOT OK'}")
    lines.append(f"K_total_x                             = {result.K_total_x_N_per_m:,.3e} N/m")
    lines.append(f"K_total_y                             = {result.K_total_y_N_per_m:,.3e} N/m")
    lines.append(f"K_total_eq                            = {result.K_total_eq_N_per_m:,.3e} N/m")
    lines.append(f"Core scale                            = {result.core_scale:.3f}")
    lines.append(f"Column scale                          = {result.column_scale:.3f}")
    lines.append("")
    lines.append("FIRST STORY / ROOF STORY SNAPSHOT")
    lines.append("-" * 88)
    roof = result.story_results[-1]
    base = result.story_results[0]
    for label, sr in [("Base Story", base), ("Roof Story", roof)]:
        lines.append(f"{label}: story {sr.story}")
        lines.append(f"  Wall thickness                      = {sr.wall_thickness_m:.3f} m")
        lines.append(f"  Corner column                       = {sr.corner_column_x_m:.3f} x {sr.corner_column_y_m:.3f} m")
        lines.append(f"  Perimeter column                    = {sr.perimeter_column_x_m:.3f} x {sr.perimeter_column_y_m:.3f} m")
        lines.append(f"  Interior column                     = {sr.interior_column_x_m:.3f} x {sr.interior_column_y_m:.3f} m")
        lines.append(f"  k_wall_x / k_col_x                  = {sr.k_wall_x_N_per_m:,.3e} / {sr.k_column_x_N_per_m:,.3e} N/m")
        lines.append(f"  k_wall_y / k_col_y                  = {sr.k_wall_y_N_per_m:,.3e} / {sr.k_column_y_N_per_m:,.3e} N/m")
    lines.append("")
    lines.append("MODAL SUMMARY")
    lines.append("-" * 88)
    for i, (T, f, mr, cmr) in enumerate(zip(
        result.modal_result.periods_s,
        result.modal_result.frequencies_hz,
        result.modal_result.effective_mass_ratios,
        result.modal_result.cumulative_effective_mass_ratios,
    ), start=1):
        lines.append(f"Mode {i}: T = {T:.4f} s | f = {f:.4f} Hz | mass ratio = {100*mr:.2f}% | cumulative = {100*cmr:.2f}%")
    lines.append("")
    lines.append("QUANTITY SUMMARY")
    lines.append("-" * 88)
    r = result.reinforcement
    lines.append(f"Wall concrete                         = {r.wall_concrete_volume_m3:,.2f} m3")
    lines.append(f"Column concrete                       = {r.column_concrete_volume_m3:,.2f} m3")
    lines.append(f"Beam concrete                         = {r.beam_concrete_volume_m3:,.2f} m3")
    lines.append(f"Slab concrete                         = {r.slab_concrete_volume_m3:,.2f} m3")
    lines.append(f"Total steel                           = {r.total_steel_kg:,.0f} kg")
    lines.append("")
    lines.append("MESSAGES")
    lines.append("-" * 88)
    for m in result.messages:
        lines.append(f"- {m}")
    lines.append("")
    lines.append("REDESIGN SUGGESTIONS")
    lines.append("-" * 88)
    for s in result.redesign_suggestions:
        lines.append(f"- {s}")
    return "\n".join(lines)


def plot_mode_shapes(result: DesignResult):
    modal = result.modal_result
    n_modes = min(5, len(modal.mode_shapes))
    H = result.H_m
    y = np.linspace(inp.story_height, H, len(result.story_results))
    fig, axes = plt.subplots(1, n_modes, figsize=(18, 6))
    if n_modes == 1:
        axes = [axes]
    for i in range(n_modes):
        ax = axes[i]
        shape = np.array(modal.mode_shapes[i], dtype=float)
        ax.axvline(0.0, color="#bbbbbb", linestyle="--", linewidth=1.0)
        ax.plot(shape, y, linewidth=2)
        ax.scatter(shape, y, s=16)
        ax.set_title(f"Mode {i+1}\nT = {modal.periods_s[i]:.3f} s")
        ax.set_xlabel("Normalized displacement")
        if i == 0:
            ax.set_ylabel("Height (m)")
    fig.tight_layout()
    return fig


def plot_story_stiffness(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.plot(df["k_wall_x_N_per_m"], df["story"], label="Wall X")
    ax.plot(df["k_column_x_N_per_m"], df["story"], label="Column X")
    ax.plot(df["k_story_x_N_per_m"], df["story"], label="Total X", linewidth=2)
    ax.set_xlabel("Stiffness (N/m)")
    ax.set_ylabel("Story")
    ax.set_title("Story Stiffness Distribution - X Direction")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return fig


# ----------------------------- UI -----------------------------
def streamlit_input_panel() -> BuildingInput:
    st.markdown("### Plan Shape")
    plan_shape = st.radio(" ", ["square", "triangle"], horizontal=True, label_visibility="collapsed")

    st.markdown("### Geometry")
    c1, c2 = st.columns(2)
    with c1:
        n_story = st.number_input("Above-grade stories", min_value=1, max_value=120, value=50, step=1)
        basement_height = st.number_input("Basement height (m)", min_value=2.5, max_value=6.0, value=3.0)
        plan_x = st.number_input("Plan X (m)", min_value=10.0, max_value=300.0, value=80.0)
        n_bays_x = st.number_input("Bays in X", min_value=1, max_value=30, value=8, step=1)
        bay_x = st.number_input("Bay X (m)", min_value=2.0, max_value=20.0, value=10.0)
    with c2:
        n_basement = st.number_input("Basement stories", min_value=0, max_value=20, value=10, step=1)
        story_height = st.number_input("Story height (m)", min_value=2.5, max_value=6.0, value=3.2)
        plan_y = st.number_input("Plan Y (m)", min_value=10.0, max_value=300.0, value=80.0)
        n_bays_y = st.number_input("Bays in Y", min_value=1, max_value=30, value=8, step=1)
        bay_y = st.number_input("Bay Y (m)", min_value=2.0, max_value=20.0, value=10.0)

    st.markdown("### Core and Service Areas")
    c3, c4 = st.columns(2)
    with c3:
        stair_count = st.number_input("Stairs", min_value=0, max_value=20, value=2, step=1)
        stair_area_each = st.number_input("Stair area each (m²)", min_value=0.0, max_value=50.0, value=20.0)
        service_area = st.number_input("Service area (m²)", min_value=0.0, max_value=200.0, value=35.0)
    with c4:
        elevator_count = st.number_input("Elevators", min_value=0, max_value=30, value=4, step=1)
        elevator_area_each = st.number_input("Elevator area each (m²)", min_value=0.0, max_value=20.0, value=3.5)
        corridor_factor = st.number_input("Core circulation factor", min_value=0.5, max_value=3.0, value=1.4)

    st.markdown("### Material / Load / Model Parameters")
    c5, c6 = st.columns(2)
    with c5:
        fck = st.number_input("fck (MPa)", min_value=20.0, max_value=100.0, value=70.0)
        Ec = st.number_input("Ec (MPa)", min_value=20000.0, max_value=60000.0, value=36000.0)
        fy = st.number_input("fy (MPa)", min_value=200.0, max_value=700.0, value=420.0)
        DL = st.number_input("DL (kN/m²)", min_value=0.0, max_value=20.0, value=3.0)
        LL = st.number_input("LL (kN/m²)", min_value=0.0, max_value=20.0, value=2.5)
        slab_finish_allowance = st.number_input("Fit-out allowance (kN/m²)", min_value=0.0, max_value=10.0, value=1.5)
        facade_line_load = st.number_input("Facade line load (kN/m)", min_value=0.0, max_value=50.0, value=1.0)
        wall_cracked_factor = st.number_input("Wall cracked factor", min_value=0.1, max_value=1.0, value=0.70)
        column_cracked_factor = st.number_input("Column cracked factor", min_value=0.1, max_value=1.0, value=0.70)
    with c6:
        prelim_lateral_force_coeff = st.number_input("Prelim lateral coefficient", min_value=0.001, max_value=0.100, value=0.015)
        drift_denominator = st.number_input("Drift denominator", min_value=100.0, max_value=2000.0, value=500.0)
        Ct = st.number_input("Ct", min_value=0.001, max_value=0.200, value=0.0488, format="%.4f")
        x_period = st.number_input("x exponent", min_value=0.1, max_value=1.5, value=0.75)
        upper_period_factor = st.number_input("Upper period factor", min_value=1.0, max_value=3.0, value=1.40)
        target_position_factor = st.number_input("Target position factor", min_value=0.10, max_value=0.95, value=0.85)
        beam_frame_factor = st.number_input("Beam-frame factor", min_value=0.8, max_value=2.0, value=1.15)

    st.markdown("### Size Bounds")
    c7, c8 = st.columns(2)
    with c7:
        min_wall_thickness = st.number_input("Min wall thickness (m)", min_value=0.1, max_value=2.0, value=0.30)
        max_wall_thickness = st.number_input("Max wall thickness (m)", min_value=0.1, max_value=3.0, value=1.20)
        min_column_dim = st.number_input("Min column dimension (m)", min_value=0.1, max_value=3.0, value=0.70)
        max_column_dim = st.number_input("Max column dimension (m)", min_value=0.1, max_value=5.0, value=1.80)
        min_beam_width = st.number_input("Min beam width (m)", min_value=0.1, max_value=3.0, value=0.40)
        min_beam_depth = st.number_input("Min beam depth (m)", min_value=0.1, max_value=3.0, value=0.75)
    with c8:
        min_slab_thickness = st.number_input("Min slab thickness (m)", min_value=0.05, max_value=1.0, value=0.22)
        max_slab_thickness = st.number_input("Max slab thickness (m)", min_value=0.05, max_value=1.0, value=0.40)
        perimeter_column_factor = st.number_input("Perimeter column factor", min_value=1.0, max_value=3.0, value=1.10)
        corner_column_factor = st.number_input("Corner column factor", min_value=1.0, max_value=3.0, value=1.30)
        lower_zone_wall_count = st.number_input("Lower zone wall count", min_value=4, max_value=8, value=8, step=1)
        middle_zone_wall_count = st.number_input("Middle zone wall count", min_value=4, max_value=8, value=6, step=1)
        upper_zone_wall_count = st.number_input("Upper zone wall count", min_value=4, max_value=8, value=4, step=1)

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
        beam_frame_factor=float(beam_frame_factor),
        perimeter_column_factor=float(perimeter_column_factor),
        corner_column_factor=float(corner_column_factor),
        lower_zone_wall_count=int(lower_zone_wall_count),
        middle_zone_wall_count=int(middle_zone_wall_count),
        upper_zone_wall_count=int(upper_zone_wall_count),
        Ct=float(Ct),
        x_period=float(x_period),
        upper_period_factor=float(upper_period_factor),
        target_position_factor=float(target_position_factor),
    )


# ----------------------------- MAIN APP -----------------------------
st.title("Tall Building Preliminary Structural Analysis")
st.caption(f"{APP_VERSION} | Author: {AUTHOR_NAME}")

inp = streamlit_input_panel()

run_opt = st.sidebar.checkbox("Optimize scales", value=True)
if st.button("Run Analysis"):
    result = optimize_design(inp) if run_opt else evaluate_design(inp, 1.0, 1.0)
    df = story_schedule_dataframe(result.story_results)
    report = build_report(result)

    tab1, tab2, tab3, tab4 = st.tabs(["Summary", "Story Schedule", "Charts", "Report"])
    with tab1:
        c1, c2, c3 = st.columns(3)
        c1.metric("Estimated period (s)", f"{result.estimated_period_s:.3f}")
        c2.metric("Drift ratio", f"{result.drift_ratio:.5f}")
        c3.metric("Total weight (kN)", f"{result.total_weight_kN:,.0f}")
        st.write("Messages")
        for m in result.messages:
            st.write(f"- {m}")
        st.write("Redesign suggestions")
        for s in result.redesign_suggestions:
            st.write(f"- {s}")
    with tab2:
        st.dataframe(df, use_container_width=True)
        st.download_button(
            "Download story schedule CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="story_schedule.csv",
            mime="text/csv",
        )
    with tab3:
        st.pyplot(plot_story_stiffness(df))
    with tab4:
        st.text(report)
        st.download_button(
            "Download report TXT",
            data=report.encode("utf-8"),
            file_name="design_report.txt",
            mime="text/plain",
        )
