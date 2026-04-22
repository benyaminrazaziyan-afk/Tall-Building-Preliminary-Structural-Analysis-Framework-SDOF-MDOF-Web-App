
from __future__ import annotations

from dataclasses import dataclass
from math import pi, sqrt
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.optimize import minimize

# ----------------------------- CONFIG -----------------------------
st.set_page_config(page_title="Tall Building Preliminary Design", layout="wide")

APP_AUTHOR = "Benyamin"
APP_VERSION = "v3.0"

G = 9.81
CONCRETE_UNIT_WEIGHT = 25.0  # kN/m3
STEEL_DENSITY = 7850.0  # kg/m3


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
    upper_period_limit_factor: float
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


@dataclass
class ZoneDefinition:
    name: str
    story_start: int
    story_end: int

    @property
    def n_stories(self) -> int:
        return self.story_end - self.story_start + 1


@dataclass
class ZoneCoreResult:
    zone: ZoneDefinition
    wall_count: int
    wall_thickness: float
    core_outer_x: float
    core_outer_y: float
    core_opening_x: float
    core_opening_y: float
    Ieq_effective_m4: float
    story_slenderness: float
    perimeter_wall_segments: List[Tuple[str, float, float]]
    retaining_wall_active: bool


@dataclass
class ZoneColumnResult:
    zone: ZoneDefinition
    corner_x_m: float
    corner_y_m: float
    perimeter_x_m: float
    perimeter_y_m: float
    interior_x_m: float
    interior_y_m: float
    I_col_group_effective_m4: float


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
    reference_period_s: float
    design_target_period_s: float
    upper_limit_period_s: float
    estimated_period_s: float
    period_error_ratio: float
    period_within_limit: bool
    total_stiffness_N_per_m: float
    required_stiffness_N_per_m: float
    top_drift_m: float
    drift_ratio: float
    total_weight_kN: float
    slab_thickness_m: float
    beam_width_m: float
    beam_depth_m: float
    lambda_core: float
    lambda_col: float
    zone_cores: List[ZoneCoreResult]
    zone_cols: List[ZoneColumnResult]
    modal: ModalResult
    assessment: str
    messages: List[str]


# ----------------------------- BASIC HELPERS -----------------------------
def total_height(inp: BuildingInput) -> float:
    return inp.n_story * inp.story_height


def floor_area(inp: BuildingInput) -> float:
    return 0.5 * inp.plan_x * inp.plan_y if inp.plan_shape == "triangle" else inp.plan_x * inp.plan_y


def perimeter_length(inp: BuildingInput) -> float:
    if inp.plan_shape == "triangle":
        return inp.plan_x + inp.plan_y + (inp.plan_x**2 + inp.plan_y**2) ** 0.5
    return 2.0 * (inp.plan_x + inp.plan_y)


def define_three_zones(n_story: int) -> List[ZoneDefinition]:
    z1 = max(1, round(0.30 * n_story))
    z2 = max(z1 + 1, round(0.70 * n_story))
    return [
        ZoneDefinition("Lower Zone", 1, z1),
        ZoneDefinition("Middle Zone", z1 + 1, z2),
        ZoneDefinition("Upper Zone", z2 + 1, n_story),
    ]


def code_reference_period(inp: BuildingInput) -> float:
    return inp.Ct * (total_height(inp) ** inp.x_period)


def opening_dimensions(inp: BuildingInput) -> tuple[float, float]:
    area = (
        inp.elevator_count * inp.elevator_area_each
        + inp.stair_count * inp.stair_area_each
        + inp.service_area
    ) * inp.corridor_factor
    aspect = 1.6
    oy = sqrt(max(area, 1e-6) / aspect)
    return aspect * oy, oy


def initial_core_dimensions(inp: BuildingInput, opening_x: float, opening_y: float) -> tuple[float, float]:
    outer_x = max(opening_x + 3.0, 0.24 * inp.plan_x)
    outer_y = max(opening_y + 3.0, 0.22 * inp.plan_y)
    return min(outer_x, 0.42 * inp.plan_x), min(outer_y, 0.42 * inp.plan_y)


def active_wall_count(inp: BuildingInput, zone_name: str) -> int:
    if zone_name == "Lower Zone":
        return inp.lower_zone_wall_count
    if zone_name == "Middle Zone":
        return inp.middle_zone_wall_count
    return inp.upper_zone_wall_count


def slab_thickness_prelim(inp: BuildingInput) -> float:
    span = max(inp.bay_x, inp.bay_y)
    return max(inp.min_slab_thickness, min(inp.max_slab_thickness, span / 28.0))


def beam_size_prelim(inp: BuildingInput) -> tuple[float, float]:
    span = max(inp.bay_x, inp.bay_y)
    depth = max(inp.min_beam_depth, span / 12.0)
    width = max(inp.min_beam_width, 0.45 * depth)
    return width, depth


# ----------------------------- SECTION STATE -----------------------------
def section_state(inp: BuildingInput, lam_core: float, lam_col: float):
    zones = define_three_zones(inp.n_story)
    opening_x, opening_y = opening_dimensions(inp)
    outer_x, outer_y = initial_core_dimensions(inp, opening_x, opening_y)
    H = total_height(inp)

    slab_t = slab_thickness_prelim(inp)
    beam_b, beam_h = beam_size_prelim(inp)

    core_results: List[ZoneCoreResult] = []
    col_results: List[ZoneColumnResult] = []

    # column layout counts
    if inp.plan_shape == "triangle":
        total_cols = (inp.n_bays_x + 1) * (inp.n_bays_y + 1) // 2 + 3
        corner_cols = 3
        perim_cols = max(6, inp.n_bays_x + inp.n_bays_y)
        interior_cols = max(0, total_cols - corner_cols - perim_cols)
    else:
        total_cols = (inp.n_bays_x + 1) * (inp.n_bays_y + 1)
        corner_cols = 4
        perim_cols = max(0, 2 * (inp.n_bays_x - 1) + 2 * (inp.n_bays_y - 1))
        interior_cols = max(0, total_cols - corner_cols - perim_cols)

    for zone in zones:
        base_t = max(inp.min_wall_thickness, min(inp.max_wall_thickness, H / 180.0))
        if zone.name == "Middle Zone":
            base_t *= 0.80
        elif zone.name == "Upper Zone":
            base_t *= 0.60
        t = min(inp.max_wall_thickness, max(inp.min_wall_thickness, base_t * lam_core))

        wcount = active_wall_count(inp, zone.name)
        side_factor = 1.0 if zone.name == "Lower Zone" else (0.85 if zone.name == "Middle Zone" else 0.75)
        cx = outer_x * side_factor
        cy = outer_y * side_factor
        cx = max(opening_x + 2.0, cx)
        cy = max(opening_y + 2.0, cy)

        # very simplified effective inertia
        I_ring_x = cx * t * (cy / 2.0) ** 2 * 2.0 + cy * t**3 / 12.0 * 2.0
        I_ring_y = cy * t * (cx / 2.0) ** 2 * 2.0 + cx * t**3 / 12.0 * 2.0
        I_base = min(I_ring_x, I_ring_y)
        if wcount >= 6:
            I_base *= 1.20
        if wcount >= 8:
            I_base *= 1.15
        I_eff = inp.wall_cracked_factor * I_base

        if inp.plan_shape == "square":
            if zone.name == "Lower Zone":
                perim_segments = [
                    ("top", 0.0, inp.plan_x), ("bottom", 0.0, inp.plan_x),
                    ("left", 0.0, inp.plan_y), ("right", 0.0, inp.plan_y)
                ]
            else:
                ratio = inp.perimeter_shear_wall_ratio
                lx = inp.plan_x * ratio
                ly = inp.plan_y * ratio
                sx = (inp.plan_x - lx) / 2.0
                sy = (inp.plan_y - ly) / 2.0
                perim_segments = [
                    ("top", sx, sx + lx), ("bottom", sx, sx + lx),
                    ("left", sy, sy + ly), ("right", sy, sy + ly)
                ]
        else:
            if zone.name == "Lower Zone":
                perim_segments = [("edge1", 0.0, 1.0), ("edge2", 0.0, 1.0), ("edge3", 0.0, 1.0)]
            else:
                ratio = inp.perimeter_shear_wall_ratio
                s = (1.0 - ratio) / 2.0
                perim_segments = [("edge1", s, s + ratio), ("edge2", s, s + ratio), ("edge3", s, s + ratio)]

        core_results.append(
            ZoneCoreResult(
                zone=zone,
                wall_count=wcount,
                wall_thickness=t,
                core_outer_x=cx,
                core_outer_y=cy,
                core_opening_x=opening_x,
                core_opening_y=opening_y,
                Ieq_effective_m4=I_eff,
                story_slenderness=inp.story_height / max(t, 1e-6),
                perimeter_wall_segments=perim_segments,
                retaining_wall_active=(zone.name == "Lower Zone"),
            )
        )

        # zone-based column sizes
        q = inp.DL + inp.LL + inp.slab_finish_allowance + slab_t * CONCRETE_UNIT_WEIGHT
        floors_above = inp.n_story - zone.story_start + 1
        n_effective = floors_above + 0.70 * inp.n_basement
        tributary_interior = inp.bay_x * inp.bay_y
        sigma_allow = 0.35 * inp.fck * 1000.0

        base_col = sqrt(max(tributary_interior * q * n_effective * 1.10 / max(sigma_allow, 1e-6), 1e-6))
        base_col = min(inp.max_column_dim, max(inp.min_column_dim, base_col * lam_col))

        interior_x = interior_y = base_col
        perim_x = perim_y = min(inp.max_column_dim, max(inp.min_column_dim, base_col * inp.perimeter_column_factor))
        corner_x = corner_y = min(inp.max_column_dim, max(inp.min_column_dim, base_col * inp.corner_column_factor))

        I_i = interior_x * interior_y**3 / 12.0
        I_p = perim_x * perim_y**3 / 12.0
        I_c = corner_x * corner_y**3 / 12.0
        I_group = inp.column_cracked_factor * (
            interior_cols * I_i + perim_cols * I_p + corner_cols * I_c
        )

        col_results.append(
            ZoneColumnResult(
                zone=zone,
                corner_x_m=corner_x,
                corner_y_m=corner_y,
                perimeter_x_m=perim_x,
                perimeter_y_m=perim_y,
                interior_x_m=interior_x,
                interior_y_m=interior_y,
                I_col_group_effective_m4=I_group,
            )
        )

    return core_results, col_results, slab_t, beam_b, beam_h


# ----------------------------- WEIGHT / STIFFNESS -----------------------------
def estimate_total_weight(inp: BuildingInput, core_results: List[ZoneCoreResult], col_results: List[ZoneColumnResult],
                          slab_t: float, beam_b: float, beam_h: float) -> float:
    A = floor_area(inp)
    n_total = inp.n_story + inp.n_basement

    # base floor loads
    floor_load = inp.DL + inp.LL + inp.slab_finish_allowance + slab_t * CONCRETE_UNIT_WEIGHT
    W_floor = floor_load * A * inp.n_story
    W_basement = 1.10 * floor_load * A * inp.n_basement
    W_facade = inp.facade_line_load * perimeter_length(inp) * inp.n_story

    # slab self already included above; now add variable structural weight
    wall_vol = 0.0
    for z in core_results:
        wall_ring_area = 2.0 * z.wall_thickness * (z.core_outer_x + z.core_outer_y)
        wall_vol += wall_ring_area * z.zone.n_stories * inp.story_height

    # columns
    if inp.plan_shape == "triangle":
        total_cols = (inp.n_bays_x + 1) * (inp.n_bays_y + 1) // 2 + 3
        corner_cols = 3
        perim_cols = max(6, inp.n_bays_x + inp.n_bays_y)
        interior_cols = max(0, total_cols - corner_cols - perim_cols)
    else:
        total_cols = (inp.n_bays_x + 1) * (inp.n_bays_y + 1)
        corner_cols = 4
        perim_cols = max(0, 2 * (inp.n_bays_x - 1) + 2 * (inp.n_bays_y - 1))
        interior_cols = max(0, total_cols - corner_cols - perim_cols)

    col_vol = 0.0
    for z in col_results:
        h = z.zone.n_stories * inp.story_height
        col_vol += corner_cols * z.corner_x_m * z.corner_y_m * h
        col_vol += perim_cols * z.perimeter_x_m * z.perimeter_y_m * h
        col_vol += interior_cols * z.interior_x_m * z.interior_y_m * h

    beam_lines_per_floor = max(1, inp.n_bays_y * (inp.n_bays_x + 1) + inp.n_bays_x * (inp.n_bays_y + 1))
    avg_span = 0.5 * (inp.bay_x + inp.bay_y)
    beam_vol = beam_b * beam_h * beam_lines_per_floor * avg_span * n_total

    W_struct = CONCRETE_UNIT_WEIGHT * (wall_vol + col_vol + beam_vol)
    W_total = (W_floor + W_basement + W_facade + W_struct) * inp.seismic_mass_factor
    return W_total


def weighted_core_stiffness(inp: BuildingInput, zone_cores: List[ZoneCoreResult]) -> float:
    H = total_height(inp)
    E = inp.Ec * 1e6
    total_flex_factor = 0.0
    for zc in zone_cores:
        hi = zc.zone.n_stories * inp.story_height
        total_flex_factor += (hi / H) / max(E * zc.Ieq_effective_m4, 1e-9)
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


def preliminary_lateral_force_N(inp: BuildingInput, W_total_kN: float) -> float:
    return inp.prelim_lateral_force_coeff * W_total_kN * 1000.0


# ----------------------------- MDOF -----------------------------
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


def assemble_m_k_matrices(story_masses: List[float], story_stiffness: List[float]):
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
    modal_participation_factors = []
    effective_modal_masses = []
    effective_mass_ratios = []
    cumulative = []
    cum = 0.0
    total_mass = np.sum(np.diag(M)).item()

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
        modal_participation_factors.append(gamma)
        effective_modal_masses.append(meff)
        effective_mass_ratios.append(ratio)
        cumulative.append(cum)

    return ModalResult(
        n_dof=len(masses),
        periods_s=periods,
        frequencies_hz=freqs,
        mode_shapes=mode_shapes,
        story_masses_kg=masses,
        story_stiffness_N_per_m=k_stories,
        modal_participation_factors=modal_participation_factors,
        effective_modal_masses_kg=effective_modal_masses,
        effective_mass_ratios=effective_mass_ratios,
        cumulative_effective_mass_ratios=cumulative,
    )


# ----------------------------- OPTIMIZATION -----------------------------
def optimize_sections(inp: BuildingInput):
    T_ref = code_reference_period(inp)
    T_target = T_ref
    T_limit = inp.upper_period_limit_factor * T_ref
    H = total_height(inp)

    # reference weight for normalization
    zc0, zl0, slab_t0, beam_b0, beam_h0 = section_state(inp, 1.0, 1.0)
    W_ref = estimate_total_weight(inp, zc0, zl0, slab_t0, beam_b0, beam_h0)

    def evaluate(lams):
        lam_core = float(np.asarray(lams[0]).item())
        lam_col = float(np.asarray(lams[1]).item())

        zone_cores, zone_cols, slab_t, beam_b, beam_h = section_state(inp, lam_core, lam_col)
        W_total = estimate_total_weight(inp, zone_cores, zone_cols, slab_t, beam_b, beam_h)

        M_eff = (W_total * 1000.0 / G)  # full seismic mass for stiffness/period relation
        K_core = weighted_core_stiffness(inp, zone_cores)
        K_cols = weighted_column_stiffness(inp, zone_cols)
        K_est = K_core + K_cols

        T_est = 2.0 * pi * sqrt(M_eff / max(K_est, 1e-9))
        top_drift = preliminary_lateral_force_N(inp, W_total) / max(K_est, 1e-9)
        drift_ratio = top_drift / H

        return {
            "zone_cores": zone_cores,
            "zone_cols": zone_cols,
            "slab_t": slab_t,
            "beam_b": beam_b,
            "beam_h": beam_h,
            "W_total": W_total,
            "K_est": K_est,
            "T_est": T_est,
            "top_drift": top_drift,
            "drift_ratio": drift_ratio,
            "T_ref": T_ref,
            "T_target": T_target,
            "T_limit": T_limit,
        }

    def objective(lams):
        data = evaluate(lams)

        T_err = (data["T_est"] - data["T_target"]) / max(data["T_target"], 1e-9)
        W_err = (data["W_total"] - W_ref) / max(W_ref, 1e-9)
        drift_util = data["drift_ratio"] / max(inp.drift_limit_ratio, 1e-12)

        # balanced objective
        obj = (
            2500.0 * T_err**2
            + 20.0 * max(W_err, -0.8)**2
            + 200.0 * max(drift_util - 1.0, 0.0)**2
        )

        # penalties
        if data["T_est"] > data["T_limit"]:
            obj += 4000.0 * ((data["T_est"] - data["T_limit"]) / data["T_limit"]) ** 2

        for z in data["zone_cores"]:
            if z.story_slenderness > inp.max_story_wall_slenderness:
                obj += 500.0 * ((z.story_slenderness - inp.max_story_wall_slenderness) / inp.max_story_wall_slenderness) ** 2

        return obj

    res = minimize(
        objective,
        x0=np.array([1.0, 1.0]),
        bounds=[(0.65, 1.60), (0.65, 1.60)],
        method="L-BFGS-B",
    )

    final = evaluate(res.x)
    final["lambda_core"] = float(np.asarray(res.x[0]).item())
    final["lambda_col"] = float(np.asarray(res.x[1]).item())
    return final


# ----------------------------- REPORT / PLOTS -----------------------------
def build_report(result: DesignResult) -> str:
    lines = []
    lines.append("GLOBAL RESPONSE")
    lines.append("-" * 74)
    lines.append(f"Reference period               = {result.reference_period_s:.3f} s")
    lines.append(f"Design target period           = {result.design_target_period_s:.3f} s")
    lines.append(f"Estimated dynamic period       = {result.estimated_period_s:.3f} s")
    lines.append(f"Upper limit period             = {result.upper_limit_period_s:.3f} s")
    lines.append(f"Period error (%)               = {100.0 * result.period_error_ratio:.2f}")
    lines.append(f"Period check                   = {'OK' if result.period_within_limit else 'NOT OK'}")
    lines.append(f"Required stiffness             = {result.required_stiffness_N_per_m:,.3e} N/m")
    lines.append(f"Estimated stiffness            = {result.total_stiffness_N_per_m:,.3e} N/m")
    lines.append(f"Top drift                      = {result.top_drift_m:.3f} m")
    lines.append(f"Drift ratio                    = {result.drift_ratio:.6f}")
    lines.append(f"Total structural weight        = {result.total_weight_kN:,.0f} kN")
    lines.append(f"Slab thickness                 = {result.slab_thickness_m:.2f} m")
    lines.append(f"Beam size (b x h)              = {result.beam_width_m:.2f} x {result.beam_depth_m:.2f} m")
    lines.append(f"Optimized core scale           = {result.lambda_core:.3f}")
    lines.append(f"Optimized column scale         = {result.lambda_col:.3f}")
    lines.append("")
    lines.append("SYSTEM ASSESSMENT")
    lines.append("-" * 74)
    lines.append(result.assessment)
    lines.append("")
    lines.append("ZONE-BY-ZONE CORE / WALLS")
    lines.append("-" * 74)
    for z in result.zone_cores:
        lines.append(f"{z.zone.name}:")
        lines.append(f"  Core outer       = {z.core_outer_x:.2f} x {z.core_outer_y:.2f} m")
        lines.append(f"  Core opening     = {z.core_opening_x:.2f} x {z.core_opening_y:.2f} m")
        lines.append(f"  Wall thickness   = {z.wall_thickness:.2f} m")
        lines.append(f"  Active core walls= {z.wall_count}")
        lines.append(f"  Effective Ieq    = {z.Ieq_effective_m4:,.2f} m^4")
        lines.append(f"  Story slenderness= {z.story_slenderness:.2f}")
    lines.append("")
    lines.append("ZONE-BY-ZONE COLUMN DIMENSIONS")
    lines.append("-" * 74)
    for z in result.zone_cols:
        lines.append(f"{z.zone.name}:")
        lines.append(f"  Corner columns   = {z.corner_x_m:.2f} x {z.corner_y_m:.2f} m")
        lines.append(f"  Perimeter cols   = {z.perimeter_x_m:.2f} x {z.perimeter_y_m:.2f} m")
        lines.append(f"  Interior cols    = {z.interior_x_m:.2f} x {z.interior_y_m:.2f} m")
        lines.append(f"  Column group Ieff= {z.I_col_group_effective_m4:,.2f} m^4")
    lines.append("")
    lines.append("MODAL ANALYSIS")
    lines.append("-" * 74)
    for i, (T, f, r, c) in enumerate(zip(
        result.modal.periods_s,
        result.modal.frequencies_hz,
        result.modal.effective_mass_ratios,
        result.modal.cumulative_effective_mass_ratios
    ), start=1):
        lines.append(f"Mode {i}: T={T:.4f} s | f={f:.4f} Hz | mass ratio={100*r:.2f}% | cumulative={100*c:.2f}%")
    lines.append("")
    lines.append("MESSAGES")
    lines.append("-" * 74)
    for m in result.messages:
        lines.append(f"- {m}")
    return "\n".join(lines)


def plot_mode_shapes(result: DesignResult):
    mr = result.modal
    n_modes = min(5, len(mr.mode_shapes))
    H = total_height_from_result(result)
    n_story = mr.n_dof
    y = np.linspace(0.0, H, n_story)

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
            ax.set_ylabel("Height (m)")
            ax.set_yticks([0.0, H])
            ax.set_yticklabels([f"Base\n0.0", f"Roof\n{H:.1f}"])
        else:
            ax.set_yticks([])
        ax.set_xticks([])
    fig.suptitle("First 5 Mode Shapes (MDOF)", fontsize=15, fontweight="bold")
    fig.tight_layout()
    return fig


def total_height_from_result(result: DesignResult) -> float:
    # inferred from drift ratio = drift/H
    return result.top_drift_m / max(result.drift_ratio, 1e-12)


def _draw_rect(ax, x, y, w, h, color, fill=True, alpha=1.0, lw=1.0, ls="-", ec=None):
    rect = plt.Rectangle((x, y), w, h,
                         facecolor=color if fill else "none",
                         edgecolor=ec if ec else color,
                         linewidth=lw, linestyle=ls, alpha=alpha)
    ax.add_patch(rect)


def plot_plan(inp: BuildingInput, result: DesignResult, zone_name: str):
    core = next(z for z in result.zone_cores if z.zone.name == zone_name)
    cols = next(z for z in result.zone_cols if z.zone.name == zone_name)

    fig, ax = plt.subplots(figsize=(13, 7))
    if inp.plan_shape == "triangle":
        pts = np.array([[0, inp.plan_y], [inp.plan_x / 2, 0], [inp.plan_x, inp.plan_y], [0, inp.plan_y]])
        ax.plot(pts[:, 0], pts[:, 1], color="black", linewidth=1.5)
    else:
        ax.plot([0, inp.plan_x, inp.plan_x, 0, 0], [0, 0, inp.plan_y, inp.plan_y, 0], color="black", linewidth=1.5)
        for i in range(inp.n_bays_x + 1):
            gx = i * inp.bay_x
            ax.plot([gx, gx], [0, inp.plan_y], color="#d9d9d9", linewidth=0.8)
        for j in range(inp.n_bays_y + 1):
            gy = j * inp.bay_y
            ax.plot([0, inp.plan_x], [gy, gy], color="#d9d9d9", linewidth=0.8)

    # columns
    if inp.plan_shape == "square":
        for i in range(inp.n_bays_x + 1):
            for j in range(inp.n_bays_y + 1):
                px = i * inp.bay_x
                py = j * inp.bay_y
                at_lr = i == 0 or i == inp.n_bays_x
                at_bt = j == 0 or j == inp.n_bays_y
                if at_lr and at_bt:
                    dx, dy, c = cols.corner_x_m, cols.corner_y_m, "#8b0000"
                elif at_lr or at_bt:
                    dx, dy, c = cols.perimeter_x_m, cols.perimeter_y_m, "#cc5500"
                else:
                    dx, dy, c = cols.interior_x_m, cols.interior_y_m, "#4444aa"
                _draw_rect(ax, px - dx/2, py - dy/2, dx, dy, c, fill=True, alpha=0.95, lw=0.5)

    # core
    cx0 = (inp.plan_x - core.core_outer_x) / 2
    cy0 = (inp.plan_y - core.core_outer_y) / 2
    _draw_rect(ax, cx0, cy0, core.core_outer_x, core.core_outer_y, "#2e8b57", fill=False, lw=2.5)
    ix0 = (inp.plan_x - core.core_opening_x) / 2
    iy0 = (inp.plan_y - core.core_opening_y) / 2
    _draw_rect(ax, ix0, iy0, core.core_opening_x, core.core_opening_y, "#666666", fill=False, lw=1.3, ls="--")

    t = core.wall_thickness
    thickness = inp.basement_retaining_wall_thickness if core.retaining_wall_active else t
    for side, a, b in core.perimeter_wall_segments:
        if inp.plan_shape == "square":
            if side == "top":
                _draw_rect(ax, a, 0, b-a, thickness, "#4caf50", fill=True, alpha=0.85)
            elif side == "bottom":
                _draw_rect(ax, a, inp.plan_y-thickness, b-a, thickness, "#4caf50", fill=True, alpha=0.85)
            elif side == "left":
                _draw_rect(ax, 0, a, thickness, b-a, "#4caf50", fill=True, alpha=0.85)
            elif side == "right":
                _draw_rect(ax, inp.plan_x-thickness, a, thickness, b-a, "#4caf50", fill=True, alpha=0.85)

    ax.set_title(f"{zone_name} - {inp.plan_shape.capitalize()} plan", fontsize=14, fontweight="bold")
    ax.set_xlim(-8, inp.plan_x + 25)
    ax.set_ylim(inp.plan_y + 8, -12)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig


def build_result(inp: BuildingInput) -> DesignResult:
    opt = optimize_sections(inp)

    zone_cores = opt["zone_cores"]
    zone_cols = opt["zone_cols"]
    slab_t = opt["slab_t"]
    beam_b = opt["beam_b"]
    beam_h = opt["beam_h"]
    W_total = opt["W_total"]
    K_est = opt["K_est"]
    T_est = opt["T_est"]
    T_ref = opt["T_ref"]
    T_target = opt["T_target"]
    T_limit = opt["T_limit"]
    top_drift = opt["top_drift"]
    drift_ratio = opt["drift_ratio"]

    modal = solve_mdof_modes(inp, W_total, K_est, n_modes=5)
    period_error = (T_est - T_target) / max(T_target, 1e-9)
    required_K = 4.0 * pi**2 * (W_total * 1000.0 / G) / max(T_target**2, 1e-9)
    period_ok = T_est <= T_limit

    messages = []
    if period_ok:
        messages.append("Estimated dynamic period is within the selected upper limit.")
    else:
        messages.append("Estimated dynamic period exceeds the selected upper limit.")
    if drift_ratio <= inp.drift_limit_ratio:
        messages.append("Estimated drift is within the selected drift limit.")
    else:
        messages.append("Estimated drift exceeds the selected drift limit.")
    messages.append("Reference period is computed from T_ref = Ct * H^x.")
    messages.append("Design target period is set equal to the reference period.")
    messages.append("Upper limit period is computed as limit_factor * T_ref.")
    messages.append("Mass participation ratios are computed from the MDOF eigenvectors and mass matrix.")

    assessment = (
        "System appears preliminarily adequate."
        if (period_ok and drift_ratio <= inp.drift_limit_ratio and abs(period_error) <= 0.08)
        else "System requires revision: period target, upper limit, or drift condition is not sufficiently satisfied."
    )

    return DesignResult(
        reference_period_s=T_ref,
        design_target_period_s=T_target,
        upper_limit_period_s=T_limit,
        estimated_period_s=T_est,
        period_error_ratio=period_error,
        period_within_limit=period_ok,
        total_stiffness_N_per_m=K_est,
        required_stiffness_N_per_m=required_K,
        top_drift_m=top_drift,
        drift_ratio=drift_ratio,
        total_weight_kN=W_total,
        slab_thickness_m=slab_t,
        beam_width_m=beam_b,
        beam_depth_m=beam_h,
        lambda_core=opt["lambda_core"],
        lambda_col=opt["lambda_col"],
        zone_cores=zone_cores,
        zone_cols=zone_cols,
        modal=modal,
        assessment=assessment,
        messages=messages,
    )


# ----------------------------- UI -----------------------------
def input_panel() -> BuildingInput:
    st.markdown("### Plan Shape")
    plan_shape = st.radio(" ", ["square", "triangle"], horizontal=True, label_visibility="collapsed")

    st.markdown("### Geometry")
    c1, c2 = st.columns(2)
    with c1:
        n_story = st.number_input("Above-grade stories", 1, 120, 50, 1)
        basement_height = st.number_input("Basement height (m)", 2.5, 6.0, 3.0)
        plan_x = st.number_input("Plan X (m)", 10.0, 300.0, 80.0)
        n_bays_x = st.number_input("Bays in X", 1, 30, 8, 1)
        bay_x = st.number_input("Bay X (m)", 2.0, 20.0, 10.0)
        stair_count = st.number_input("Stairs", 0, 20, 2, 1)
    with c2:
        n_basement = st.number_input("Basement stories", 0, 20, 10, 1)
        story_height = st.number_input("Story height (m)", 2.5, 6.0, 3.2)
        plan_y = st.number_input("Plan Y (m)", 10.0, 300.0, 80.0)
        n_bays_y = st.number_input("Bays in Y", 1, 30, 8, 1)
        bay_y = st.number_input("Bay Y (m)", 2.0, 20.0, 10.0)
        elevator_count = st.number_input("Elevators", 0, 30, 4, 1)

    st.markdown("### Loads / Materials")
    c3, c4 = st.columns(2)
    with c3:
        elevator_area_each = st.number_input("Elevator area each (m²)", 0.0, 20.0, 3.5)
        service_area = st.number_input("Service area (m²)", 0.0, 200.0, 35.0)
        fck = st.number_input("fck (MPa)", 20.0, 100.0, 70.0)
        fy = st.number_input("fy (MPa)", 200.0, 700.0, 420.0)
        DL = st.number_input("DL (kN/m²)", 0.0, 20.0, 3.0)
        slab_finish_allowance = st.number_input("Slab / fit-out allowance", 0.0, 10.0, 1.5)
        wall_cracked_factor = st.number_input("Wall cracked factor", 0.1, 1.0, 0.7)
        basement_retaining_wall_thickness = st.number_input("Basement retaining wall t (m)", 0.1, 2.0, 0.5)
    with c4:
        stair_area_each = st.number_input("Stair area each (m²)", 0.0, 50.0, 20.0)
        corridor_factor = st.number_input("Core circulation factor", 0.5, 3.0, 1.4)
        Ec = st.number_input("Ec (MPa)", 20000.0, 60000.0, 36000.0)
        LL = st.number_input("LL (kN/m²)", 0.0, 20.0, 2.5)
        facade_line_load = st.number_input("Facade line load (kN/m)", 0.0, 50.0, 1.0)
        column_cracked_factor = st.number_input("Column cracked factor", 0.1, 1.0, 0.7)

    st.markdown("### Controls")
    c5, c6 = st.columns(2)
    with c5:
        prelim_lateral_force_coeff = st.number_input("Prelim lateral coeff", 0.001, 0.100, 0.015, format="%.3f")
        upper_period_limit_factor = st.number_input("Upper period limit factor", 1.0, 3.0, 1.4, format="%.2f")
        min_wall_thickness = st.number_input("Min wall thickness (m)", 0.1, 2.0, 0.3)
        min_column_dim = st.number_input("Min column dimension (m)", 0.1, 3.0, 0.7)
        min_beam_width = st.number_input("Min beam width (m)", 0.1, 3.0, 0.4)
        min_slab_thickness = st.number_input("Min slab thickness (m)", 0.05, 1.0, 0.22)
        max_story_wall_slenderness = st.number_input("Max wall slenderness", 1.0, 50.0, 12.0)
        corner_column_factor = st.number_input("Corner column factor", 1.0, 3.0, 1.3)
        middle_zone_wall_count = st.number_input("Middle zone wall count", 4, 8, 6, 1)
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
        lower_zone_wall_count = st.number_input("Lower zone wall count", 4, 8, 8, 1)
        upper_zone_wall_count = st.number_input("Upper zone wall count", 4, 8, 4, 1)
        perimeter_shear_wall_ratio = st.number_input("Perimeter shear wall ratio", 0.0, 1.0, 0.2, format="%.3f")
        column_rebar_ratio = st.number_input("Column rebar ratio", 0.0, 0.1, 0.010, format="%.4f")
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
        upper_period_limit_factor=float(upper_period_limit_factor),
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


def modal_table(result: DesignResult) -> pd.DataFrame:
    mr = result.modal
    return pd.DataFrame({
        "Mode": np.arange(1, len(mr.periods_s) + 1),
        "Period (s)": mr.periods_s,
        "Frequency (Hz)": mr.frequencies_hz,
        "Gamma": mr.modal_participation_factors,
        "Eff. mass ratio": mr.effective_mass_ratios,
        "Cum. mass ratio": mr.cumulative_effective_mass_ratios,
    })


st.markdown(
    """
    <style>
    .main .block-container {
        padding-top: 0.6rem;
        padding-bottom: 0.6rem;
        max-width: 100%;
    }
    .stButton button {
        width: 100%;
        font-weight: 700;
        height: 3rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Final Tall Building Plan Output Tool + MDOF Modes")
st.caption(f"Prepared by {APP_AUTHOR} | version {APP_VERSION}")

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
                res = build_result(inp)
                st.session_state.result = res
                st.session_state.report = build_report(res)
                st.session_state.view_mode = "plan"
            except Exception as e:
                st.error(f"Analysis failed: {e}")

    with b2:
        if st.button("SHOW 5 MODES"):
            try:
                if st.session_state.result is None:
                    res = build_result(inp)
                    st.session_state.result = res
                    st.session_state.report = build_report(res)
                st.session_state.view_mode = "modes"
            except Exception as e:
                st.error(f"Mode display failed: {e}")

    with b3:
        if st.session_state.report:
            st.download_button(
                "SAVE REPORT",
                data=st.session_state.report.encode("utf-8"),
                file_name=f"tall_building_report_{APP_AUTHOR}.txt",
                mime="text/plain",
            )
        else:
            st.button("SAVE REPORT", disabled=True)

with right_col:
    zone_name = st.selectbox("Displayed zone:", ["Lower Zone", "Middle Zone", "Upper Zone"], index=0)

    if st.session_state.result is None:
        st.info("Click ANALYZE to display the plan and report, or SHOW 5 MODES to display modal shapes.")
    else:
        r = st.session_state.result
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Reference period (s)", f"{r.reference_period_s:.3f}")
        m2.metric("Design target (s)", f"{r.design_target_period_s:.3f}")
        m3.metric("Estimated dynamic (s)", f"{r.estimated_period_s:.3f}")
        m4.metric("Upper limit (s)", f"{r.upper_limit_period_s:.3f}")

        n1, n2, n3 = st.columns(3)
        n1.metric("Period error (%)", f"{100*r.period_error_ratio:.2f}")
        n2.metric("Total stiffness (N/m)", f"{r.total_stiffness_N_per_m:,.2e}")
        n3.metric("Top drift (m)", f"{r.top_drift_m:.3f}")

        tab1, tab2, tab3 = st.tabs(["Graphic output", "Mass participation", "Report"])

        with tab1:
            if st.session_state.view_mode == "modes":
                st.pyplot(plot_mode_shapes(r), use_container_width=True)
            else:
                st.pyplot(plot_plan(inp, r, zone_name), use_container_width=True)

        with tab2:
            st.dataframe(modal_table(r), use_container_width=True)

        with tab3:
            st.text_area("", st.session_state.report, height=420, label_visibility="collapsed")
