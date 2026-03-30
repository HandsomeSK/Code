import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

# ………………………………………………………………………………………………………………
# User settings
# ………………………………………………………………………………………………………………
SMART_HOME_CSV = "caseA_smart_home_30min_summer.csv"
EV_EVENTS_CSV = "caseA_ev_events.csv"

DT_H = 0.5                 # 30 min = 0.5 h
E_BAT = 5.0                # usable battery capacity (kWh)
P_CH_MAX = 2.5             # max battery charge power (kW)
P_DIS_MAX = 2.5            # max battery discharge power (kW)
ETA_CH = 0.95
ETA_DIS = 0.95
SOC0 = 0.5 * E_BAT         # initial SOC = 50%
TERMINAL_LOOKBACK = 48     # last 48 steps = last 24 hours
TOL = 1e-9
SMALL_REG = 1e-6           # tiny regularisation to avoid degenerate cycling


# ………………………………………………………………………………………………………………
# Data loading
# ………………………………………………………………………………………………………………
def load_smart_home_data(csv_path=SMART_HOME_CSV):
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    required = [
        "timestamp",
        "pv_kw",
        "base_load_kw",
        "import_tariff_gbp_per_kwh",
        "export_price_gbp_per_kwh",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in smart-home CSV: {missing}")

    df = df.sort_values("timestamp").reset_index(drop=True)

    if df[required[1:]].isna().any().any():
        raise ValueError("Smart-home data contains missing values.")

    if (df["pv_kw"] < -TOL).any() or (df["base_load_kw"] < -TOL).any():
        raise ValueError("pv_kw or base_load_kw contains negative values.")

    dt = df["timestamp"].diff().dropna().dt.total_seconds() / 3600.0
    if not np.allclose(dt.to_numpy(), DT_H, atol=1e-9):
        raise ValueError("Timestamp spacing is not consistently 30 minutes.")

    return df


def load_ev_events(csv_path=EV_EVENTS_CSV):
    ev = pd.read_csv(csv_path)
    ev["arrival_time"] = pd.to_datetime(ev["arrival_time"])
    ev["departure_time"] = pd.to_datetime(ev["departure_time"])

    required = [
        "arrival_time",
        "departure_time",
        "required_energy_kwh",
        "max_charge_power_kw",
    ]
    missing = [c for c in required if c not in ev.columns]
    if missing:
        raise ValueError(f"Missing columns in EV CSV: {missing}")

    ev = ev.sort_values("arrival_time").reset_index(drop=True)

    if ev[required[2:]].isna().any().any():
        raise ValueError("EV event data contains missing values.")

    if (ev["required_energy_kwh"] < -TOL).any() or (ev["max_charge_power_kw"] < -TOL).any():
        raise ValueError("EV event data contains negative values.")

    if not (ev["departure_time"] > ev["arrival_time"]).all():
        raise ValueError("Every EV event must have departure_time > arrival_time.")

    return ev


# ………………………………………………………………………………………………………………
# EV preprocessing
# ………………………………………………………………………………………………………………
def build_ev_event_info(df, ev):
    """
    Convert EV arrival/departure events into:
      - per-slot availability fractions
      - per-slot EV max charging power
      - per-event masks for energy constraints/checks

    Assumption:
      Events do not overlap. The provided file should satisfy this.
    """
    n = len(df)
    slot_start = df["timestamp"].to_numpy(dtype="datetime64[ns]")
    slot_end = (df["timestamp"] + pd.Timedelta(hours=DT_H)).to_numpy(dtype="datetime64[ns]")

    event_info = []
    active_count = np.zeros(n, dtype=int)

    for j, row in ev.iterrows():
        arr = np.datetime64(row["arrival_time"])
        dep = np.datetime64(row["departure_time"])
        req = float(row["required_energy_kwh"])
        pmax = float(row["max_charge_power_kw"])

        frac = np.zeros(n)
        for t in range(n):
            start = slot_start[t]
            end = slot_end[t]
            overlap_h = (min(end, dep) - max(start, arr)) / np.timedelta64(1, "h")
            if overlap_h > 0:
                frac[t] = float(overlap_h / DT_H)

        mask = frac > 0
        max_energy_possible = float((frac * pmax * DT_H).sum())

        if max_energy_possible + 1e-9 < req:
            raise ValueError(
                f"EV event {j + 1} is infeasible after time-grid mapping. "
                f"Required = {req:.4f} kWh, max possible = {max_energy_possible:.4f} kWh"
            )

        active_count += mask.astype(int)

        event_info.append({
            "event_id": j + 1,
            "arrival_time": row["arrival_time"],
            "departure_time": row["departure_time"],
            "required_energy_kwh": req,
            "max_charge_power_kw": pmax,
            "slot_fraction": frac,
            "slot_mask": mask,
            "slot_max_kw": frac * pmax,
            "available_hours": float(frac.sum() * DT_H),
            "max_energy_possible_kwh": max_energy_possible,
        })

    if active_count.max() > 1:
        raise ValueError(
            "This script assumes EV events do not overlap. "
            "The provided file should have non-overlapping events."
        )

    ev_slot_max_kw = np.zeros(n)
    for info in event_info:
        ev_slot_max_kw = np.maximum(ev_slot_max_kw, info["slot_max_kw"])

    return {
        "event_info": event_info,
        "ev_slot_max_kw": ev_slot_max_kw,
        "n_events": len(event_info),
    }


# ……………………………………………………………………………………………………………………………………
# Terminal SOC repair (for simulation policies only)
# ……………………………………………………………………………………………………………………………………
def repair_terminal_soc(df, soc, p_imp, p_ch, p_dis, lookback_steps=TERMINAL_LOOKBACK):
    """
    If final SOC is below SOC0, add grid charging near the end of horizon,
    prioritising cheapest import periods while respecting:
      - battery charge power limit
      - battery energy capacity limit
      - no simultaneous charge/discharge in the same slot
    """
    gap = SOC0 - soc[-1]
    if gap <= TOL:
        return

    start = max(0, len(df) - lookback_steps)
    candidate_idx = list(range(start, len(df)))
    candidate_idx.sort(key=lambda t: df.loc[t, "import_tariff_gbp_per_kwh"])

    for t in candidate_idx:
        gap = SOC0 - soc[-1]
        if gap <= TOL:
            break

        # Do not add charging in a step that already discharges
        if p_dis[t] > TOL:
            continue

        power_headroom = P_CH_MAX - p_ch[t]
        if power_headroom <= TOL:
            continue

        future_headroom = np.min(E_BAT - soc[t + 1:])
        if future_headroom <= TOL:
            continue

        max_delta_soc = min(gap, future_headroom)
        add_ch = min(power_headroom, max_delta_soc / (ETA_CH * DT_H))

        if add_ch <= TOL:
            continue

        p_ch[t] += add_ch
        p_imp[t] += add_ch
        soc[t + 1:] += ETA_CH * add_ch * DT_H


# ………………………………………………………………………………………………………………
# Policy 1A: Base simulation
# ………………………………………………………………………………………………………………
def simulate_base_self_consumption(df):
    n = len(df)

    soc = np.zeros(n + 1)
    p_ch = np.zeros(n)
    p_dis = np.zeros(n)
    p_imp = np.zeros(n)
    p_exp = np.zeros(n)
    p_ev = np.zeros(n)

    soc[0] = SOC0

    for t in range(n):
        pv = df.loc[t, "pv_kw"]
        load = df.loc[t, "base_load_kw"]
        soc_now = soc[t]

        if pv >= load:
            surplus = pv - load
            ch_limit_soc = max(0.0, (E_BAT - soc_now) / (ETA_CH * DT_H))
            p_ch[t] = min(surplus, P_CH_MAX, ch_limit_soc)
            p_exp[t] = surplus - p_ch[t]
        else:
            deficit = load - pv
            dis_limit_soc = max(0.0, soc_now * ETA_DIS / DT_H)
            p_dis[t] = min(deficit, P_DIS_MAX, dis_limit_soc)
            p_imp[t] = deficit - p_dis[t]

        soc[t + 1] = soc_now + ETA_CH * p_ch[t] * DT_H - (p_dis[t] * DT_H) / ETA_DIS

    repair_terminal_soc(df, soc, p_imp, p_ch, p_dis)

    return package_results(
        df=df,
        soc=soc,
        p_ch=p_ch,
        p_dis=p_dis,
        p_imp=p_imp,
        p_exp=p_exp,
        p_ev=p_ev,
        ev_bundle=None,
        case_name="Base case",
        policy_name="Policy 1: Simulation (self-consumption first)",
        extra_meta={"implementation_style": "Simulation", "extension": "None"},
    )


# ………………………………………………………………………………………………………………
# Policy 1B: EV extension simulation
# ………………………………………………………………………………………………………………
def simulate_ev_immediate_charging(df, ev_bundle):
    """
    Extension simulation rule:
      EV charges immediately whenever connected, subject to slot availability
      and remaining required energy for the current event.
    Then the house+battery dispatch still follows self-consumption-first.
    """
    n = len(df)

    soc = np.zeros(n + 1)
    p_ch = np.zeros(n)
    p_dis = np.zeros(n)
    p_imp = np.zeros(n)
    p_exp = np.zeros(n)
    p_ev = np.zeros(n)

    soc[0] = SOC0

    # Immediate EV charging schedule
    for info in ev_bundle["event_info"]:
        remaining = info["required_energy_kwh"]
        for t in np.where(info["slot_mask"])[0]:
            if remaining <= TOL:
                break
            slot_max_kw = info["slot_max_kw"][t]
            ev_power = min(slot_max_kw, remaining / DT_H)
            p_ev[t] += ev_power
            remaining -= ev_power * DT_H

        if remaining > 1e-6:
            raise RuntimeError(
                f"Immediate-charging simulation could not meet EV event {info['event_id']}."
            )

    for t in range(n):
        pv = df.loc[t, "pv_kw"]
        total_load = df.loc[t, "base_load_kw"] + p_ev[t]
        soc_now = soc[t]

        if pv >= total_load:
            surplus = pv - total_load
            ch_limit_soc = max(0.0, (E_BAT - soc_now) / (ETA_CH * DT_H))
            p_ch[t] = min(surplus, P_CH_MAX, ch_limit_soc)
            p_exp[t] = surplus - p_ch[t]
        else:
            deficit = total_load - pv
            dis_limit_soc = max(0.0, soc_now * ETA_DIS / DT_H)
            p_dis[t] = min(deficit, P_DIS_MAX, dis_limit_soc)
            p_imp[t] = deficit - p_dis[t]

        soc[t + 1] = soc_now + ETA_CH * p_ch[t] * DT_H - (p_dis[t] * DT_H) / ETA_DIS

    repair_terminal_soc(df, soc, p_imp, p_ch, p_dis)

    return package_results(
        df=df,
        soc=soc,
        p_ch=p_ch,
        p_dis=p_dis,
        p_imp=p_imp,
        p_exp=p_exp,
        p_ev=p_ev,
        ev_bundle=ev_bundle,
        case_name="EV extension",
        policy_name="Policy 1 + EV: Simulation (immediate EV charging)",
        extra_meta={"implementation_style": "Simulation", "extension": "EV charging"},
    )


# ………………………………………………………………………………………………………………
# CVXPY helpers
# ………………………………………………………………………………………………………………
def choose_cvxpy_solver(cp):
    installed = cp.installed_solvers()
    preferred = ["CLARABEL", "ECOS", "OSQP", "SCS"]
    for name in preferred:
        if name in installed:
            return name
    return None


def solve_cvxpy_problem(problem, cp):
    solver = choose_cvxpy_solver(cp)
    if solver is None:
        problem.solve(verbose=False)
        return "CVXPY default"
    problem.solve(solver=solver, verbose=False)
    return solver


# ………………………………………………………………………………………………………………
# Policy 2A: Base optimisation
# ………………………………………………………………………………………………………………
def optimise_base_full_cost_cvxpy(df):
    try:
        import cvxpy as cp
    except ImportError as e:
        raise ImportError("cvxpy is not installed. Run: pip install cvxpy") from e

    n = len(df)

    pv = df["pv_kw"].to_numpy()
    load = df["base_load_kw"].to_numpy()
    buy = df["import_tariff_gbp_per_kwh"].to_numpy()
    sell = df["export_price_gbp_per_kwh"].to_numpy()

    p_ch = cp.Variable(n, nonneg=True)
    p_dis = cp.Variable(n, nonneg=True)
    p_imp = cp.Variable(n, nonneg=True)
    p_exp = cp.Variable(n, nonneg=True)
    soc = cp.Variable(n + 1)
    p_ev = np.zeros(n)

    constraints = []
    constraints += [soc[0] == SOC0]
    constraints += [pv + p_dis + p_imp == load + p_ch + p_exp]
    constraints += [soc[1:] == soc[:-1] + ETA_CH * DT_H * p_ch - (DT_H / ETA_DIS) * p_dis]
    constraints += [soc >= 0, soc <= E_BAT]
    constraints += [p_ch <= P_CH_MAX, p_dis <= P_DIS_MAX]
    constraints += [soc[-1] >= SOC0]

    objective = cp.Minimize(
        DT_H * cp.sum(cp.multiply(buy, p_imp) - cp.multiply(sell, p_exp))
        + SMALL_REG * cp.sum(p_ch + p_dis + p_imp + p_exp)
    )

    problem = cp.Problem(objective, constraints)
    solver_used = solve_cvxpy_problem(problem, cp)

    if problem.status not in ["optimal", "optimal_inaccurate"]:
        raise RuntimeError(f"Base optimisation failed. Status: {problem.status}")

    return package_results(
        df=df,
        soc=np.asarray(soc.value).flatten(),
        p_ch=np.maximum(np.asarray(p_ch.value).flatten(), 0.0),
        p_dis=np.maximum(np.asarray(p_dis.value).flatten(), 0.0),
        p_imp=np.maximum(np.asarray(p_imp.value).flatten(), 0.0),
        p_exp=np.maximum(np.asarray(p_exp.value).flatten(), 0.0),
        p_ev=p_ev,
        ev_bundle=None,
        case_name="Base case",
        policy_name="Policy 2: Optimisation (CVXPY full cost optimisation)",
        extra_meta={
            "implementation_style": "Optimisation",
            "extension": "None",
            "solver_status": problem.status,
            "solver_used": solver_used,
            "objective_value_gbp": float(problem.value),
        },
    )


# ………………………………………………………………………………………………………………
# Policy 2B: EV extension optimisation
# ………………………………………………………………………………………………………………
def optimise_ev_full_cost_cvxpy(df, ev_bundle):
    try:
        import cvxpy as cp
    except ImportError as e:
        raise ImportError("cvxpy is not installed. Run: pip install cvxpy") from e

    n = len(df)

    pv = df["pv_kw"].to_numpy()
    load = df["base_load_kw"].to_numpy()
    buy = df["import_tariff_gbp_per_kwh"].to_numpy()
    sell = df["export_price_gbp_per_kwh"].to_numpy()

    p_ch = cp.Variable(n, nonneg=True)
    p_dis = cp.Variable(n, nonneg=True)
    p_imp = cp.Variable(n, nonneg=True)
    p_exp = cp.Variable(n, nonneg=True)
    p_ev = cp.Variable(n, nonneg=True)
    soc = cp.Variable(n + 1)

    constraints = []
    constraints += [soc[0] == SOC0]
    constraints += [pv + p_dis + p_imp == load + p_ev + p_ch + p_exp]
    constraints += [soc[1:] == soc[:-1] + ETA_CH * DT_H * p_ch - (DT_H / ETA_DIS) * p_dis]
    constraints += [soc >= 0, soc <= E_BAT]
    constraints += [p_ch <= P_CH_MAX, p_dis <= P_DIS_MAX]
    constraints += [soc[-1] >= SOC0]

    # EV slot availability
    constraints += [p_ev <= ev_bundle["ev_slot_max_kw"]]

    # EV energy requirement for each event
    for info in ev_bundle["event_info"]:
        idx = np.where(info["slot_mask"])[0]
        constraints += [DT_H * cp.sum(p_ev[idx]) >= info["required_energy_kwh"]]

    objective = cp.Minimize(
        DT_H * cp.sum(cp.multiply(buy, p_imp) - cp.multiply(sell, p_exp))
        + SMALL_REG * cp.sum(p_ch + p_dis + p_imp + p_exp + p_ev)
    )

    problem = cp.Problem(objective, constraints)
    solver_used = solve_cvxpy_problem(problem, cp)

    if problem.status not in ["optimal", "optimal_inaccurate"]:
        raise RuntimeError(f"EV optimisation failed. Status: {problem.status}")

    return package_results(
        df=df,
        soc=np.asarray(soc.value).flatten(),
        p_ch=np.maximum(np.asarray(p_ch.value).flatten(), 0.0),
        p_dis=np.maximum(np.asarray(p_dis.value).flatten(), 0.0),
        p_imp=np.maximum(np.asarray(p_imp.value).flatten(), 0.0),
        p_exp=np.maximum(np.asarray(p_exp.value).flatten(), 0.0),
        p_ev=np.maximum(np.asarray(p_ev.value).flatten(), 0.0),
        ev_bundle=ev_bundle,
        case_name="EV extension",
        policy_name="Policy 2 + EV: Optimisation (CVXPY full cost optimisation)",
        extra_meta={
            "implementation_style": "Optimisation",
            "extension": "EV charging",
            "solver_status": problem.status,
            "solver_used": solver_used,
            "objective_value_gbp": float(problem.value),
        },
    )


# ………………………………………………………………………………………………………………
# Result packaging
# ………………………………………………………………………………………………………………
def package_results(df, soc, p_ch, p_dis, p_imp, p_exp, p_ev, ev_bundle, case_name, policy_name, extra_meta=None):
    out = df.copy()

    # Clip tiny numerical noise
    soc = np.clip(np.asarray(soc).flatten(), 0.0, E_BAT)
    p_ch = np.maximum(np.asarray(p_ch).flatten(), 0.0)
    p_dis = np.maximum(np.asarray(p_dis).flatten(), 0.0)
    p_imp = np.maximum(np.asarray(p_imp).flatten(), 0.0)
    p_exp = np.maximum(np.asarray(p_exp).flatten(), 0.0)
    p_ev = np.maximum(np.asarray(p_ev).flatten(), 0.0)

    out["p_ev_kw"] = p_ev
    out["total_load_kw"] = out["base_load_kw"] + out["p_ev_kw"]
    out["p_batt_charge_kw"] = p_ch
    out["p_batt_discharge_kw"] = p_dis
    out["p_grid_import_kw"] = p_imp
    out["p_grid_export_kw"] = p_exp
    out["soc_kwh"] = soc[:-1]
    out["soc_next_kwh"] = soc[1:]

    # Costs
    out["import_cost_gbp"] = out["p_grid_import_kw"] * DT_H * out["import_tariff_gbp_per_kwh"]
    out["export_revenue_gbp"] = out["p_grid_export_kw"] * DT_H * out["export_price_gbp_per_kwh"]
    out["net_cost_gbp"] = out["import_cost_gbp"] - out["export_revenue_gbp"]

    # -----------------------------------------------------
    # Flow decomposition on the combined AC bus
    # -----------------------------------------------------
    pv = out["pv_kw"].to_numpy()
    total_load = out["total_load_kw"].to_numpy()
    dis = out["p_batt_discharge_kw"].to_numpy()
    ch = out["p_batt_charge_kw"].to_numpy()
    imp = out["p_grid_import_kw"].to_numpy()
    exp = out["p_grid_export_kw"].to_numpy()

    # 1) Load supply split
    pv_to_load = np.minimum(pv, total_load)
    load_after_pv = np.maximum(total_load - pv_to_load, 0.0)

    battery_to_load = np.minimum(dis, load_after_pv)
    grid_to_load = np.maximum(load_after_pv - battery_to_load, 0.0)
    grid_to_load = np.minimum(grid_to_load, imp)

    # 2) Battery charge split
    pv_surplus = np.maximum(pv - pv_to_load, 0.0)
    pv_to_battery = np.minimum(ch, pv_surplus)
    grid_to_battery = np.maximum(ch - pv_to_battery, 0.0)

    # 3) Export split
    battery_to_export = np.maximum(dis - battery_to_load, 0.0)
    battery_to_export = np.minimum(battery_to_export, exp)
    pv_export = np.maximum(exp - battery_to_export, 0.0)

    out["pv_direct_to_load_kw"] = pv_to_load
    out["pv_to_battery_kw"] = pv_to_battery
    out["grid_to_battery_kw"] = grid_to_battery
    out["battery_to_load_kw"] = battery_to_load
    out["battery_to_export_kw"] = battery_to_export
    out["grid_to_load_kw"] = grid_to_load
    out["pv_export_kw"] = pv_export

    ev_summary = summarise_ev_delivery(out, ev_bundle)
    summary = make_summary(out, ev_summary)
    verification = verify_results(out, ev_bundle, ev_summary)

    result = {
        "case_name": case_name,
        "policy": policy_name,
        "timeseries": out,
        "summary": summary,
        "verification": verification,
        "ev_event_summary": ev_summary["event_table"],
    }

    if extra_meta is not None:
        result["meta"] = extra_meta

    return result


# ………………………………………………………………………………………………………………
# EV summary
# ………………………………………………………………………………………………………………
def summarise_ev_delivery(out, ev_bundle):
    if ev_bundle is None:
        event_table = pd.DataFrame(columns=[
            "event_id", "arrival_time", "departure_time",
            "required_energy_kwh", "delivered_energy_kwh", "margin_kwh"
        ])
        return {
            "total_ev_energy_kwh": 0.0,
            "n_events": 0,
            "min_event_margin_kwh": np.nan,
            "event_table": event_table,
        }

    rows = []
    delivered_total = 0.0
    margins = []

    for info in ev_bundle["event_info"]:
        idx = np.where(info["slot_mask"])[0]
        delivered = float((out.loc[idx, "p_ev_kw"] * DT_H).sum())
        margin = delivered - info["required_energy_kwh"]
        delivered_total += delivered
        margins.append(margin)

        rows.append({
            "event_id": info["event_id"],
            "arrival_time": info["arrival_time"],
            "departure_time": info["departure_time"],
            "required_energy_kwh": info["required_energy_kwh"],
            "delivered_energy_kwh": delivered,
            "margin_kwh": margin,
        })

    event_table = pd.DataFrame(rows)

    return {
        "total_ev_energy_kwh": delivered_total,
        "n_events": len(rows),
        "min_event_margin_kwh": float(np.min(margins)) if margins else np.nan,
        "event_table": event_table,
    }


# ………………………………………………………………………………………………………………
# Summary KPIs
# ………………………………………………………………………………………………………………
def make_summary(out, ev_summary):
    total_combined_load_kwh = float((out["total_load_kw"] * DT_H).sum())
    total_pv_kwh = float((out["pv_kw"] * DT_H).sum())
    pv_direct_to_load_kwh = float((out["pv_direct_to_load_kw"] * DT_H).sum())
    pv_to_battery_kwh = float((out["pv_to_battery_kw"] * DT_H).sum())
    grid_to_load_kwh = float((out["grid_to_load_kw"] * DT_H).sum())

    summary = {
        "total_base_load_kwh": float((out["base_load_kw"] * DT_H).sum()),
        "total_ev_energy_kwh": float(ev_summary["total_ev_energy_kwh"]),
        "total_combined_load_kwh": total_combined_load_kwh,
        "total_pv_kwh": total_pv_kwh,
        "grid_import_kwh": float((out["p_grid_import_kw"] * DT_H).sum()),
        "grid_export_kwh": float((out["p_grid_export_kw"] * DT_H).sum()),
        "battery_charge_kwh": float((out["p_batt_charge_kw"] * DT_H).sum()),
        "battery_discharge_kwh": float((out["p_batt_discharge_kw"] * DT_H).sum()),
        "pv_direct_to_load_kwh": pv_direct_to_load_kwh,
        "pv_to_battery_kwh": pv_to_battery_kwh,
        "grid_to_battery_kwh": float((out["grid_to_battery_kw"] * DT_H).sum()),
        "battery_to_load_kwh": float((out["battery_to_load_kw"] * DT_H).sum()),
        "battery_to_export_kwh": float((out["battery_to_export_kw"] * DT_H).sum()),
        "grid_to_load_kwh": grid_to_load_kwh,
        "pv_export_kwh": float((out["pv_export_kw"] * DT_H).sum()),
        "import_cost_gbp": float(out["import_cost_gbp"].sum()),
        "export_revenue_gbp": float(out["export_revenue_gbp"].sum()),
        "net_cost_gbp": float(out["net_cost_gbp"].sum()),
        "initial_soc_kwh": float(out["soc_kwh"].iloc[0]),
        "final_soc_kwh": float(out["soc_next_kwh"].iloc[-1]),
        "min_soc_kwh": float(min(out["soc_kwh"].min(), out["soc_next_kwh"].min())),
        "max_soc_kwh": float(max(out["soc_kwh"].max(), out["soc_next_kwh"].max())),
        "self_consumption_ratio": (
            (pv_direct_to_load_kwh + pv_to_battery_kwh) / total_pv_kwh
            if total_pv_kwh > TOL else np.nan
        ),
        "self_sufficiency_ratio": (
            1.0 - grid_to_load_kwh / total_combined_load_kwh
            if total_combined_load_kwh > TOL else np.nan
        ),
    }
    return pd.Series(summary)


# ………………………………………………………………………………………………………………
# Verification
# ………………………………………………………………………………………………………………
def verify_results(out, ev_bundle, ev_summary):
    power_balance_residual = (
        out["pv_kw"]
        + out["p_batt_discharge_kw"]
        + out["p_grid_import_kw"]
        - out["base_load_kw"]
        - out["p_ev_kw"]
        - out["p_batt_charge_kw"]
        - out["p_grid_export_kw"]
    )

    soc_consistency_residual = (
        out["soc_next_kwh"]
        - out["soc_kwh"]
        - ETA_CH * DT_H * out["p_batt_charge_kw"]
        + (DT_H / ETA_DIS) * out["p_batt_discharge_kw"]
    )

    max_simul_charge_discharge = float(
        np.minimum(out["p_batt_charge_kw"], out["p_batt_discharge_kw"]).max()
    )
    max_simul_import_export = float(
        np.minimum(out["p_grid_import_kw"], out["p_grid_export_kw"]).max()
    )

    pv_split_residual = (
        out["pv_kw"]
        - out["pv_direct_to_load_kw"]
        - out["pv_to_battery_kw"]
        - out["pv_export_kw"]
    )

    load_supply_residual = (
        out["total_load_kw"]
        - out["pv_direct_to_load_kw"]
        - out["battery_to_load_kw"]
        - out["grid_to_load_kw"]
    )

    export_split_residual = (
        out["p_grid_export_kw"]
        - out["pv_export_kw"]
        - out["battery_to_export_kw"]
    )

    ver = {
        "max_abs_power_balance_error_kw": float(np.abs(power_balance_residual).max()),
        "max_abs_soc_consistency_error_kwh": float(np.abs(soc_consistency_residual).max()),
        "soc_min_ok": bool((out["soc_kwh"] >= -1e-8).all() and (out["soc_next_kwh"] >= -1e-8).all()),
        "soc_max_ok": bool((out["soc_kwh"] <= E_BAT + 1e-8).all() and (out["soc_next_kwh"] <= E_BAT + 1e-8).all()),
        "charge_power_ok": bool((out["p_batt_charge_kw"] <= P_CH_MAX + 1e-8).all()),
        "discharge_power_ok": bool((out["p_batt_discharge_kw"] <= P_DIS_MAX + 1e-8).all()),
        "terminal_soc_ok": bool(out["soc_next_kwh"].iloc[-1] >= SOC0 - 1e-8),
        "ev_nonnegative_ok": bool((out["p_ev_kw"] >= -1e-8).all()),
        "max_simultaneous_charge_discharge_kw": max_simul_charge_discharge,
        "max_simultaneous_import_export_kw": max_simul_import_export,
        "no_simultaneous_charge_discharge": bool(max_simul_charge_discharge <= 1e-6),
        "no_simultaneous_import_export": bool(max_simul_import_export <= 1e-6),
        "pv_split_ok": bool(np.allclose(pv_split_residual.to_numpy(), 0.0, atol=1e-8)),
        "load_supply_ok": bool(np.allclose(load_supply_residual.to_numpy(), 0.0, atol=1e-8)),
        "export_split_ok": bool(np.allclose(export_split_residual.to_numpy(), 0.0, atol=1e-8)),
    }

    if ev_bundle is not None:
        ev_slot_max = ev_bundle["ev_slot_max_kw"]
        ver["ev_slot_power_ok"] = bool((out["p_ev_kw"] <= ev_slot_max + 1e-8).all())
        ver["ev_all_events_met"] = bool(ev_summary["min_event_margin_kwh"] >= -1e-6)
        ver["ev_min_event_margin_kwh"] = float(ev_summary["min_event_margin_kwh"])
        ver["ev_number_of_events"] = int(ev_summary["n_events"])

    return pd.Series(ver)


# ………………………………………………………………………………………………………………
# Printing
# ………………………………………………………………………………………………………………
def print_report(result):
    print("\n" + "=" * 95)
    print(result["case_name"])
    print(result["policy"])
    print("=" * 95)

    if "meta" in result:
        print("\nMeta:")
        for k, v in result["meta"].items():
            print(f"{k}: {v}")

    print("\nSummary:")
    print(result["summary"].round(6).to_string())

    print("\nVerification:")
    print(result["verification"].to_string())


# ………………………………………………………………………………………………………………
# Saving
# ………………………………………………………………………………………………………………
def save_outputs(results):
    for key, result in results.items():
        safe_key = key.lower().replace(" ", "_")
        result["timeseries"].to_csv(f"{safe_key}_timeseries.csv", index=False)
        if not result["ev_event_summary"].empty:
            result["ev_event_summary"].to_csv(f"{safe_key}_ev_event_summary.csv", index=False)

        if "meta" in result:
            pd.Series(result["meta"], name=f"{key} meta").to_csv(f"{safe_key}_meta.csv", header=True)

    summary_table = pd.DataFrame({k: v["summary"] for k, v in results.items()})
    verification_table = pd.DataFrame({k: v["verification"] for k, v in results.items()})

    summary_table.to_csv("all_cases_summary_comparison.csv")
    verification_table.to_csv("all_cases_verification_comparison.csv")

    print("\nSaved files:")
    for key, result in results.items():
        safe_key = key.lower().replace(" ", "_")
        print(f" - {safe_key}_timeseries.csv")
        print(f" - {safe_key}_ev_event_summary.csv")
        if "meta" in result:
            print(f" - {safe_key}_meta.csv")
    print(" - all_cases_summary_comparison.csv")
    print(" - all_cases_verification_comparison.csv")


# ………………………………………………………………………………………………………………
# Plotting
# ………………………………………………………………………………………………………………
def plot_base_comparison(base_sim, base_opt):
    df1 = base_sim["timeseries"]
    df2 = base_opt["timeseries"]

    plt.figure(figsize=(12, 4))
    plt.plot(df1["timestamp"], df1["pv_kw"], label="PV (kW)")
    plt.plot(df1["timestamp"], df1["base_load_kw"], label="Base load (kW)")
    plt.title("Base case: PV generation and household load")
    plt.xlabel("Time")
    plt.ylabel("kW")
    plt.legend()
    plt.tight_layout()
    plt.savefig("plot_base_1_pv_load.png", dpi=300)
    plt.close()

    plt.figure(figsize=(12, 4))
    plt.plot(df1["timestamp"], df1["soc_kwh"], label="Base simulation")
    plt.plot(df2["timestamp"], df2["soc_kwh"], label="Base optimisation")
    plt.title("Base case: battery SOC comparison")
    plt.xlabel("Time")
    plt.ylabel("kWh")
    plt.legend()
    plt.tight_layout()
    plt.savefig("plot_base_2_soc.png", dpi=300)
    plt.close()

    base_energy = pd.DataFrame({
        "Base simulation": [
            base_sim["summary"]["pv_direct_to_load_kwh"],
            base_sim["summary"]["pv_to_battery_kwh"],
            base_sim["summary"]["grid_to_battery_kwh"],
            base_sim["summary"]["battery_to_load_kwh"],
            base_sim["summary"]["battery_to_export_kwh"],
            base_sim["summary"]["grid_to_load_kwh"],
            base_sim["summary"]["pv_export_kwh"],
        ],
        "Base optimisation": [
            base_opt["summary"]["pv_direct_to_load_kwh"],
            base_opt["summary"]["pv_to_battery_kwh"],
            base_opt["summary"]["grid_to_battery_kwh"],
            base_opt["summary"]["battery_to_load_kwh"],
            base_opt["summary"]["battery_to_export_kwh"],
            base_opt["summary"]["grid_to_load_kwh"],
            base_opt["summary"]["pv_export_kwh"],
        ],
    }, index=["PV -> Load", "PV -> Battery", "Grid -> Battery", "Battery -> Load", "Battery -> Export", "Grid -> Load", "PV -> Export"])

    ax = base_energy.T.plot(kind="bar", figsize=(11, 5))
    ax.set_title("Base case: energy breakdown comparison")
    ax.set_ylabel("kWh")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig("plot_base_3_energy.png", dpi=300)
    plt.close()

    base_cost = pd.DataFrame({
        "Import cost (GBP)": [base_sim["summary"]["import_cost_gbp"], base_opt["summary"]["import_cost_gbp"]],
        "Export revenue (GBP)": [base_sim["summary"]["export_revenue_gbp"], base_opt["summary"]["export_revenue_gbp"]],
        "Net cost (GBP)": [base_sim["summary"]["net_cost_gbp"], base_opt["summary"]["net_cost_gbp"]],
    }, index=["Base simulation", "Base optimisation"])

    ax = base_cost.plot(kind="bar", figsize=(10, 5))
    ax.set_title("Base case: cost breakdown comparison")
    ax.set_ylabel("GBP")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig("plot_base_4_cost.png", dpi=300)
    plt.close()


def plot_extension_comparison(ev_sim, ev_opt):
    df1 = ev_sim["timeseries"]
    df2 = ev_opt["timeseries"]

    plt.figure(figsize=(12, 4))
    plt.plot(df1["timestamp"], df1["p_ev_kw"], label="EV charging - simulation")
    plt.plot(df2["timestamp"], df2["p_ev_kw"], label="EV charging - optimisation")
    plt.title("EV extension: charging schedule comparison")
    plt.xlabel("Time")
    plt.ylabel("kW")
    plt.legend()
    plt.tight_layout()
    plt.savefig("plot_ev_1_charging_schedule.png", dpi=300)
    plt.close()

    plt.figure(figsize=(12, 4))
    plt.plot(df1["timestamp"], df1["soc_kwh"], label="EV simulation")
    plt.plot(df2["timestamp"], df2["soc_kwh"], label="EV optimisation")
    plt.title("EV extension: battery SOC comparison")
    plt.xlabel("Time")
    plt.ylabel("kWh")
    plt.legend()
    plt.tight_layout()
    plt.savefig("plot_ev_2_soc.png", dpi=300)
    plt.close()

    ev_energy = pd.DataFrame({
        "EV simulation": [
            ev_sim["summary"]["pv_direct_to_load_kwh"],
            ev_sim["summary"]["pv_to_battery_kwh"],
            ev_sim["summary"]["grid_to_battery_kwh"],
            ev_sim["summary"]["battery_to_load_kwh"],
            ev_sim["summary"]["battery_to_export_kwh"],
            ev_sim["summary"]["grid_to_load_kwh"],
            ev_sim["summary"]["pv_export_kwh"],
        ],
        "EV optimisation": [
            ev_opt["summary"]["pv_direct_to_load_kwh"],
            ev_opt["summary"]["pv_to_battery_kwh"],
            ev_opt["summary"]["grid_to_battery_kwh"],
            ev_opt["summary"]["battery_to_load_kwh"],
            ev_opt["summary"]["battery_to_export_kwh"],
            ev_opt["summary"]["grid_to_load_kwh"],
            ev_opt["summary"]["pv_export_kwh"],
        ],
    }, index=["PV -> Load", "PV -> Battery", "Grid -> Battery", "Battery -> Load", "Battery -> Export", "Grid -> Load", "PV -> Export"])

    ax = ev_energy.T.plot(kind="bar", figsize=(11, 5))
    ax.set_title("EV extension: energy breakdown comparison")
    ax.set_ylabel("kWh")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig("plot_ev_3_energy.png", dpi=300)
    plt.close()

    ev_cost = pd.DataFrame({
        "Import cost (GBP)": [ev_sim["summary"]["import_cost_gbp"], ev_opt["summary"]["import_cost_gbp"]],
        "Export revenue (GBP)": [ev_sim["summary"]["export_revenue_gbp"], ev_opt["summary"]["export_revenue_gbp"]],
        "Net cost (GBP)": [ev_sim["summary"]["net_cost_gbp"], ev_opt["summary"]["net_cost_gbp"]],
    }, index=["EV simulation", "EV optimisation"])

    ax = ev_cost.plot(kind="bar", figsize=(10, 5))
    ax.set_title("EV extension: cost breakdown comparison")
    ax.set_ylabel("GBP")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig("plot_ev_4_cost.png", dpi=300)
    plt.close()

    print("\nSaved plot files:")
    print(" - plot_base_1_pv_load.png")
    print(" - plot_base_2_soc.png")
    print(" - plot_base_3_energy.png")
    print(" - plot_base_4_cost.png")
    print(" - plot_ev_1_charging_schedule.png")
    print(" - plot_ev_2_soc.png")
    print(" - plot_ev_3_energy.png")
    print(" - plot_ev_4_cost.png")


# ………………………………………………………………………………………………………………
# Main
# ………………………………………………………………………………………………………………
def main():
    df = load_smart_home_data(SMART_HOME_CSV)
    ev = load_ev_events(EV_EVENTS_CSV)
    ev_bundle = build_ev_event_info(df, ev)

    base_sim = simulate_base_self_consumption(df)
    base_opt = optimise_base_full_cost_cvxpy(df)

    ev_sim = simulate_ev_immediate_charging(df, ev_bundle)
    ev_opt = optimise_ev_full_cost_cvxpy(df, ev_bundle)

    results = {
        "Base Simulation": base_sim,
        "Base Optimisation": base_opt,
        "EV Simulation": ev_sim,
        "EV Optimisation": ev_opt,
    }

    for result in results.values():
        print_report(result)

    save_outputs(results)
    plot_base_comparison(base_sim, base_opt)
    plot_extension_comparison(ev_sim, ev_opt)


if __name__ == "__main__":
    main()