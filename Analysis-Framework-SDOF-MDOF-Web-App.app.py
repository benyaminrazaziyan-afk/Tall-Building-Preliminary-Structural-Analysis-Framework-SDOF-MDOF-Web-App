# ----------------------------- PLOTTING + UI -----------------------------
import io

AUTHOR_NAME = "Benyamin"
APP_VERSION = "v2.2-display-matched"

CORNER_COLOR = "#8b0000"
PERIM_COLOR = "#cc5500"
INTERIOR_COLOR = "#4444aa"
CORE_COLOR = "#2e8b57"
PERIM_WALL_COLOR = "#4caf50"


def plot_mode_shapes_like_original(result: DesignResult):
    mr = result.modal_result
    if mr is None:
        raise ValueError("Modal result is not available.")

    n_modes = min(5, len(mr.mode_shapes))
    n_story = mr.n_dof

    fig, axes = plt.subplots(1, n_modes, figsize=(18, 6))
    if n_modes == 1:
        axes = [axes]

    story_y = np.linspace(0, n_story - 1, n_story)

    for m in range(n_modes):
        ax = axes[m]
        phi = np.array(mr.mode_shapes[m], dtype=float)

        max_abs = max(np.max(np.abs(phi)), 1e-9)
        phi = phi / max_abs

        ax.axvline(0.0, color="#bbbbbb", linestyle="--", linewidth=1.0)
        for i in range(n_story):
            ax.plot([-1.05, 1.05], [story_y[i], story_y[i]], color="#f0f0f0", linewidth=0.8)

        ax.plot(phi, story_y, color="#0b5ed7", linewidth=2)
        ax.scatter(phi, story_y, color="#dc3545", s=18, zorder=3)

        ax.set_title(f"Mode {m+1}\nT = {mr.periods_s[m]:.3f} s", fontsize=11, fontweight="bold")
        ax.set_xlim(-1.1, 1.1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.invert_yaxis()
        for spine in ax.spines.values():
            spine.set_color("#999999")
            spine.set_linewidth(1.0)

    fig.suptitle("First 5 Mode Shapes (MDOF)", fontsize=15, fontweight="bold")
    fig.tight_layout()
    return fig


def _draw_rect(ax, x, y, w, h, color, fill=True, alpha=1.0, lw=1.0, ls="-", ec=None):
    rect = plt.Rectangle(
        (x, y), w, h,
        facecolor=color if fill else "none",
        edgecolor=ec if ec else color,
        linewidth=lw,
        linestyle=ls,
        alpha=alpha
    )
    ax.add_patch(rect)


def _draw_square_plan_like_original(ax, inp: BuildingInput, core: ZoneCoreResult, cols: ZoneColumnResult, result: DesignResult):
    # outer boundary
    ax.plot([0, inp.plan_x, inp.plan_x, 0, 0], [0, 0, inp.plan_y, inp.plan_y, 0], color="black", linewidth=1.5)

    # grid
    for i in range(inp.n_bays_x + 1):
        gx = i * inp.bay_x
        ax.plot([gx, gx], [0, inp.plan_y], color="#d9d9d9", linewidth=0.8)
    for j in range(inp.n_bays_y + 1):
        gy = j * inp.bay_y
        ax.plot([0, inp.plan_x], [gy, gy], color="#d9d9d9", linewidth=0.8)

    # columns
    for i in range(inp.n_bays_x + 1):
        for j in range(inp.n_bays_y + 1):
            px = i * inp.bay_x
            py = j * inp.bay_y
            at_lr = i == 0 or i == inp.n_bays_x
            at_bt = j == 0 or j == inp.n_bays_y

            if at_lr and at_bt:
                dx = cols.corner_column_x_m
                dy = cols.corner_column_y_m
                color = CORNER_COLOR
            elif at_lr or at_bt:
                dx = cols.perimeter_column_x_m
                dy = cols.perimeter_column_y_m
                color = PERIM_COLOR
            else:
                dx = cols.interior_column_x_m
                dy = cols.interior_column_y_m
                color = INTERIOR_COLOR

            _draw_rect(ax, px - dx/2, py - dy/2, dx, dy, color, fill=True, alpha=0.95, lw=0.5)

    # core
    cx0 = (inp.plan_x - core.core_outer_x) / 2
    cy0 = (inp.plan_y - core.core_outer_y) / 2
    cx1 = cx0 + core.core_outer_x
    cy1 = cy0 + core.core_outer_y

    ix0 = (inp.plan_x - core.core_opening_x) / 2
    iy0 = (inp.plan_y - core.core_opening_y) / 2

    # outer core boundary
    _draw_rect(ax, cx0, cy0, core.core_outer_x, core.core_outer_y, CORE_COLOR, fill=False, lw=2.5)
    # opening
    _draw_rect(ax, ix0, iy0, core.core_opening_x, core.core_opening_y, "#666666", fill=False, lw=1.3, ls="--")

    # wall thickness strips
    t = core.wall_thickness
    _draw_rect(ax, cx0, cy0, core.core_outer_x, t, CORE_COLOR, fill=True, alpha=0.25)
    _draw_rect(ax, cx0, cy1 - t, core.core_outer_x, t, CORE_COLOR, fill=True, alpha=0.25)
    _draw_rect(ax, cx0, cy0, t, core.core_outer_y, CORE_COLOR, fill=True, alpha=0.25)
    _draw_rect(ax, cx1 - t, cy0, t, core.core_outer_y, CORE_COLOR, fill=True, alpha=0.25)

    # internal core walls
    if core.wall_count >= 6:
        inner_x1 = (inp.plan_x / 2) - 0.22 * core.core_outer_x
        inner_x2 = (inp.plan_x / 2) + 0.22 * core.core_outer_x - t
        wlen = 0.45 * core.core_outer_x
        ymid0 = (inp.plan_y - wlen) / 2
        _draw_rect(ax, inner_x1, ymid0, t, wlen, CORE_COLOR, fill=True, alpha=0.85)
        _draw_rect(ax, inner_x2, ymid0, t, wlen, CORE_COLOR, fill=True, alpha=0.85)

    if core.wall_count >= 8:
        inner_y1 = (inp.plan_y / 2) - 0.22 * core.core_outer_y
        inner_y2 = (inp.plan_y / 2) + 0.22 * core.core_outer_y - t
        wlen = 0.45 * core.core_outer_y
        xmid0 = (inp.plan_x - wlen) / 2
        _draw_rect(ax, xmid0, inner_y1, wlen, t, CORE_COLOR, fill=True, alpha=0.85)
        _draw_rect(ax, xmid0, inner_y2, wlen, t, CORE_COLOR, fill=True, alpha=0.85)

    # perimeter wall segments
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

    # dimensions
    ax.annotate("", xy=(0, -4), xytext=(inp.plan_x, -4), arrowprops=dict(arrowstyle="<->", lw=1.2))
    ax.text(inp.plan_x / 2, -6.0, f"Plan X = {inp.plan_x:.2f} m", ha="center", va="top", fontsize=10)

    ax.annotate("", xy=(inp.plan_x + 4, 0), xytext=(inp.plan_x + 4, inp.plan_y), arrowprops=dict(arrowstyle="<->", lw=1.2))
    ax.text(inp.plan_x + 6.0, inp.plan_y / 2, f"Plan Y = {inp.plan_y:.2f} m", rotation=90, va="center", fontsize=10)

    # legend/info block
    info_x = inp.plan_x + 10
    info_y = inp.plan_y - 2
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
        ax.text(info_x, info_y - 4 * k, txt, fontsize=9, va="top")

    # legend
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

    ax.set_title(f"{core.zone.name} - Square plan", fontsize=14, fontweight="bold")
    ax.set_xlim(-8, inp.plan_x + 32)
    ax.set_ylim(inp.plan_y + 8, -12)
    ax.set_aspect("equal")
    ax.axis("off")


def _draw_triangle_plan_like_original(ax, inp: BuildingInput, core: ZoneCoreResult, cols: ZoneColumnResult, result: DesignResult):
    pts = np.array([[0, inp.plan_y], [inp.plan_x / 2, 0], [inp.plan_x, inp.plan_y], [0, inp.plan_y]])
    ax.plot(pts[:, 0], pts[:, 1], color="black", linewidth=1.5)

    corner_pts = [(0, inp.plan_y), (inp.plan_x / 2, 0), (inp.plan_x, inp.plan_y)]
    for x, y in corner_pts:
        _draw_rect(ax, x - cols.corner_column_x_m / 2, y - cols.corner_column_y_m / 2,
                   cols.corner_column_x_m, cols.corner_column_y_m, CORNER_COLOR, fill=True, alpha=0.95)

    # side/perimeter columns
    for i in range(1, inp.n_bays_x):
        x = inp.plan_x * i / inp.n_bays_x
        y = inp.plan_y
        _draw_rect(ax, x - cols.perimeter_column_x_m/2, y - cols.perimeter_column_y_m/2,
                   cols.perimeter_column_x_m, cols.perimeter_column_y_m, PERIM_COLOR, fill=True, alpha=0.95)

    for i in range(1, inp.n_bays_y):
        x = (inp.plan_x / 2) * (1 - i / inp.n_bays_y)
        y = inp.plan_y * (i / inp.n_bays_y)
        _draw_rect(ax, x - cols.perimeter_column_x_m/2, y - cols.perimeter_column_y_m/2,
                   cols.perimeter_column_x_m, cols.perimeter_column_y_m, PERIM_COLOR, fill=True, alpha=0.95)

    for i in range(1, inp.n_bays_y):
        x = inp.plan_x / 2 + (inp.plan_x / 2) * (i / inp.n_bays_y)
        y = inp.plan_y * (i / inp.n_bays_y)
        _draw_rect(ax, x - cols.perimeter_column_x_m/2, y - cols.perimeter_column_y_m/2,
                   cols.perimeter_column_x_m, cols.perimeter_column_y_m, PERIM_COLOR, fill=True, alpha=0.95)

    # one interior
    _draw_rect(ax, inp.plan_x / 2 - cols.interior_column_x_m/2, 0.65 * inp.plan_y - cols.interior_column_y_m/2,
               cols.interior_column_x_m, cols.interior_column_y_m, INTERIOR_COLOR, fill=True, alpha=0.95)

    # core
    cx0 = (inp.plan_x - core.core_outer_x) / 2
    cy0 = inp.plan_y * 0.42
    _draw_rect(ax, cx0, cy0, core.core_outer_x, core.core_outer_y, CORE_COLOR, fill=False, lw=2.5)
    ix0 = (inp.plan_x - core.core_opening_x) / 2
    iy0 = cy0 + (core.core_outer_y - core.core_opening_y) / 2
    _draw_rect(ax, ix0, iy0, core.core_opening_x, core.core_opening_y, "#666666", fill=False, lw=1.3, ls="--")

    ax.set_title(f"{core.zone.name} - Triangular plan", fontsize=14, fontweight="bold")
    ax.set_xlim(-8, inp.plan_x + 32)
    ax.set_ylim(inp.plan_y + 8, -12)
    ax.set_aspect("equal")
    ax.axis("off")


def plot_plan_like_original(inp: BuildingInput, result: DesignResult, zone_name: str):
    core = next(z for z in result.zone_core_results if z.zone.name == zone_name)
    cols = next(z for z in result.zone_column_results if z.zone.name == zone_name)

    fig, ax = plt.subplots(figsize=(13, 7))
    fig.patch.set_facecolor("white")

    if inp.plan_shape == "triangle":
        _draw_triangle_plan_like_original(ax, inp, core, cols, result)
    else:
        _draw_square_plan_like_original(ax, inp, core, cols, result)

    return fig


def streamlit_input_panel():
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
        stair_count = st.number_input("Stairs", min_value=0, max_value=20, value=2, step=1)
    with c2:
        n_basement = st.number_input("Basement stories", min_value=0, max_value=20, value=10, step=1)
        story_height = st.number_input("Story height (m)", min_value=2.5, max_value=6.0, value=3.2)
        plan_y = st.number_input("Plan Y (m)", min_value=10.0, max_value=300.0, value=80.0)
        n_bays_y = st.number_input("Bays in Y", min_value=1, max_value=30, value=8, step=1)
        bay_y = st.number_input("Bay Y (m)", min_value=2.0, max_value=20.0, value=10.0)
        elevator_count = st.number_input("Elevators", min_value=0, max_value=30, value=4, step=1)

    st.markdown("### Loads/Materials")
    c3, c4 = st.columns(2)
    with c3:
        elevator_area_each = st.number_input("Elevator area each (m²)", min_value=0.0, max_value=20.0, value=3.5)
        service_area = st.number_input("Service area (m²)", min_value=0.0, max_value=200.0, value=35.0)
        fck = st.number_input("fck (MPa)", min_value=20.0, max_value=100.0, value=70.0)
        fy = st.number_input("fy (MPa)", min_value=200.0, max_value=700.0, value=420.0)
        DL = st.number_input("DL (kN/m²)", min_value=0.0, max_value=20.0, value=3.0)
        slab_finish_allowance = st.number_input("Slab/fit-out allowance", min_value=0.0, max_value=10.0, value=1.5)
        wall_cracked_factor = st.number_input("Wall cracked factor", min_value=0.1, max_value=1.0, value=0.7)
        basement_retaining_wall_thickness = st.number_input("Basement retaining wall t (m)", min_value=0.1, max_value=2.0, value=0.5)
    with c4:
        stair_area_each = st.number_input("Stair area each (m²)", min_value=0.0, max_value=50.0, value=20.0)
        corridor_factor = st.number_input("Core circulation factor", min_value=0.5, max_value=3.0, value=1.4)
        Ec = st.number_input("Ec (MPa)", min_value=20000.0, max_value=60000.0, value=36000.0)
        LL = st.number_input("LL (kN/m²)", min_value=0.0, max_value=20.0, value=2.5)
        facade_line_load = st.number_input("Facade line load (kN/m)", min_value=0.0, max_value=50.0, value=1.0)
        column_cracked_factor = st.number_input("Column cracked factor", min_value=0.1, max_value=1.0, value=0.7)

    st.markdown("### Controls/Final Options")
    c5, c6 = st.columns(2)
    with c5:
        prelim_lateral_force_coeff = st.number_input("Prelim lateral coeff", min_value=0.001, max_value=0.100, value=0.015)
        target_period_factor = st.number_input("Target period factor", min_value=0.1, max_value=2.0, value=0.95)
        min_wall_thickness = st.number_input("Min wall thickness (m)", min_value=0.1, max_value=2.0, value=0.3)
        min_column_dim = st.number_input("Min column dimension (m)", min_value=0.1, max_value=3.0, value=0.7)
        min_beam_width = st.number_input("Min beam width (m)", min_value=0.1, max_value=3.0, value=0.4)
        min_slab_thickness = st.number_input("Min slab thickness (m)", min_value=0.05, max_value=1.0, value=0.22)
        max_story_wall_slenderness = st.number_input("Max wall slenderness", min_value=1.0, max_value=50.0, value=12.0)
        corner_column_factor = st.number_input("Corner column factor", min_value=1.0, max_value=3.0, value=1.3)
        middle_zone_wall_count = st.number_input("Middle zone wall count", min_value=4, max_value=8, value=6, step=1)
        wall_rebar_ratio = st.number_input("Wall rebar ratio", min_value=0.0, max_value=0.1, value=0.003, format="%.4f")
        beam_rebar_ratio = st.number_input("Beam rebar ratio", min_value=0.0, max_value=0.1, value=0.015, format="%.4f")
        seismic_mass_factor = st.number_input("Seismic mass factor", min_value=0.1, max_value=2.0, value=1.0)
        Ct = st.number_input("Ct", min_value=0.001, max_value=0.200, value=0.0488, format="%.4f")
    with c6:
        drift_denominator = st.number_input("Drift denominator", min_value=100.0, max_value=2000.0, value=500.0)
        max_period_factor_over_target = st.number_input("Max period/target", min_value=0.5, max_value=3.0, value=1.25)
        max_wall_thickness = st.number_input("Max wall thickness (m)", min_value=0.1, max_value=3.0, value=1.2)
        max_column_dim = st.number_input("Max column dimension (m)", min_value=0.1, max_value=5.0, value=1.8)
        min_beam_depth = st.number_input("Min beam depth (m)", min_value=0.1, max_value=3.0, value=0.75)
        max_slab_thickness = st.number_input("Max slab thickness (m)", min_value=0.05, max_value=1.0, value=0.4)
        perimeter_column_factor = st.number_input("Perimeter column factor", min_value=1.0, max_value=3.0, value=1.1)
        lower_zone_wall_count = st.number_input("Lower zone wall count", min_value=4, max_value=8, value=8, step=1)
        upper_zone_wall_count = st.number_input("Upper zone wall count", min_value=4, max_value=8, value=4, step=1)
        perimeter_shear_wall_ratio = st.number_input("Perimeter shear wall ratio", min_value=0.0, max_value=1.0, value=0.2, format="%.3f")
        column_rebar_ratio = st.number_input("Column rebar ratio", min_value=0.0, max_value=0.1, value=0.01, format="%.4f")
        slab_rebar_ratio = st.number_input("Slab rebar ratio", min_value=0.0, max_value=0.1, value=0.0035, format="%.4f")
        effective_modal_mass_ratio = st.number_input("Effective modal mass ratio", min_value=0.1, max_value=1.0, value=0.8)
        x_period = st.number_input("x exponent", min_value=0.1, max_value=1.5, value=0.75)

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
        target_period_factor=float(target_period_factor),
        max_period_factor_over_target=float(max_period_factor_over_target),
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
        effective_modal_mass_ratio=float(effective_modal_mass_ratio),
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


def download_report_bytes(report_text: str):
    return report_text.encode("utf-8")


# ----------------------------- STREAMLIT LAYOUT -----------------------------
st.markdown(
    f"""
    <style>
    .main .block-container {{
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: 100%;
    }}
    div[data-testid="stHorizontalBlock"] > div {{
        padding-right: 0.5rem;
        padding-left: 0.5rem;
    }}
    .stButton button {{
        width: 100%;
        font-weight: 700;
        height: 3rem;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Final Tall Building Plan Output Tool + MDOF Modes")
st.caption(f"Prepared by {AUTHOR_NAME}")

if "result" not in st.session_state:
    st.session_state.result = None
if "report" not in st.session_state:
    st.session_state.report = ""
if "view_mode" not in st.session_state:
    st.session_state.view_mode = "plan"

left_col, right_col = st.columns([1.0, 1.9], gap="medium")

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
            st.download_button(
                "SAVE REPORT",
                data=download_report_bytes(st.session_state.report),
                file_name=f"tall_building_report_{AUTHOR_NAME}.txt",
                mime="text/plain"
            )
        else:
            st.button("SAVE REPORT", disabled=True)

with right_col:
    zone_name = st.selectbox("Displayed zone:", ["Lower Zone", "Middle Zone", "Upper Zone"], index=0)

    if st.session_state.result is None:
        st.info("Click ANALYZE to display the plan and report, or SHOW 5 MODES to display modal shapes.")
    else:
        if st.session_state.view_mode == "modes":
            fig_modes = plot_mode_shapes_like_original(st.session_state.result)
            st.pyplot(fig_modes, use_container_width=True)
        else:
            fig_plan = plot_plan_like_original(inp, st.session_state.result, zone_name)
            st.pyplot(fig_plan, use_container_width=True)

        st.text_area(
            "",
            st.session_state.report,
            height=420,
            label_visibility="collapsed"
        )
