"""Phase 2 operations layer: allocation under scarcity, overflow routing, intervention simulator.

Everything here is plain pandas/numpy plus the saved Phase 1 model: 100% local, deterministic,
no network and no LLM (Requirement 5 / offline kill-switch).
"""
import math
import numpy as np
import pandas as pd

TIERS = ["Low", "Medium", "High", "Critical"]
# Supply ratios per predicted case. Fitted from the onsite dataset (sum(supply)/sum(cases)); these are the
# fallback values if an uploaded CSV has no supply columns.
DEFAULT_RATIOS = {"saline": 2.32, "kits": 1.10, "beds": 0.25}
SUPPLY_COLS = {"saline": "saline_bags_needed", "kits": "diagnostic_kits_needed", "beds": "hospital_beds_required"}

# Vector-control assumptions (stated in ARCHITECTURE.md): one crew fully covers a ward of up to 25,000 people/km2
# (dense wards need more crews, max 3); full coverage removes 60% of the rain-driven breeding signal; the effect
# builds up over the forecast window (week 2: 60%, week 3: 85%, week 4: 100%).
CREW_DENSITY_UNIT = 25000
MAX_CREWS_PER_WARD = 3
MAX_BREEDING_CUT = 0.60
RAMP = {2: 0.60, 3: 0.85, 4: 1.00}


def fit_ratios(df: pd.DataFrame) -> dict:
    """supply-per-case ratios from history (falls back to defaults when columns are absent)."""
    if all(c in df.columns for c in SUPPLY_COLS.values()) and df["dengue_cases"].sum() > 0:
        tot = df["dengue_cases"].sum()
        return {k: float(df[c].sum() / tot) for k, c in SUPPLY_COLS.items()}
    return dict(DEFAULT_RATIOS)


def week_frame(feats: pd.DataFrame, preds: pd.DataFrame, date) -> pd.DataFrame:
    """One row per ward for the chosen forecast week, with density and the model forecast."""
    m = feats["date"] == date
    f = feats.loc[m, ["ward_id", "population_density_sqkm"]].reset_index(drop=True)
    p = preds.loc[m].reset_index(drop=True)
    out = p.merge(f, on="ward_id")
    out["tier_idx"] = out["risk_tier"].map({t: i for i, t in enumerate(TIERS)})
    return out


def add_demand(fr: pd.DataFrame, ratios: dict) -> pd.DataFrame:
    fr = fr.copy()
    for k, r in ratios.items():
        fr[f"{k}_need"] = np.ceil(fr["predicted_weekly_cases"] * r - 1e-9).astype(int)
    return fr


# ---------------------------------------------------------------- 1. constrained allocation
def allocate(fr: pd.DataFrame, stock_pct: float, ratios: dict) -> pd.DataFrame:
    """Priority allocation under scarcity.

    Regional stock of each item = stock_pct x total weekly demand. Wards are served strictly in priority order:
    risk tier first (Critical > High > Medium > Low), then predicted cases x density index (dense, high-burden
    wards first). Each ward gets min(need, what is left). Anything short is flagged as a deficit.
    """
    fr = add_demand(fr, ratios)
    dens_idx = fr["population_density_sqkm"] / fr["population_density_sqkm"].median()
    fr["priority_score"] = (fr["predicted_weekly_cases"] * dens_idx).round(3)
    fr = fr.sort_values(["tier_idx", "priority_score"], ascending=False).reset_index(drop=True)
    fr["priority_rank"] = np.arange(1, len(fr) + 1)
    for k in ratios:
        pool = int(math.floor(stock_pct * fr[f"{k}_need"].sum() + 1e-9))
        left, alloc = pool, []
        for need in fr[f"{k}_need"]:
            give = int(min(need, left))
            alloc.append(give)
            left -= give
        fr[f"{k}_alloc"] = alloc
        fr[f"{k}_short"] = fr[f"{k}_need"] - fr[f"{k}_alloc"]
    short = (fr[[f"{k}_short" for k in ratios]] > 0).any(axis=1)
    fr["deficit_level"] = np.where(short & (fr["tier_idx"] >= 2), "CRITICAL DEFICIT",
                                   np.where(short, "Deficit", "Fully supplied"))
    need_tot = sum(fr[f"{k}_need"] for k in ratios)
    alloc_tot = sum(fr[f"{k}_alloc"] for k in ratios)
    fr["fulfilment_pct"] = np.where(need_tot > 0, 100 * alloc_tot / need_tot.where(need_tot > 0, 1), 100).round(0)
    return fr


def pool_summary(fr: pd.DataFrame, ratios: dict) -> pd.DataFrame:
    rows = []
    for k in ratios:
        need, alloc = int(fr[f"{k}_need"].sum()), int(fr[f"{k}_alloc"].sum())
        rows.append({"item": SUPPLY_COLS[k], "needed": need, "allocated": alloc, "shortfall": need - alloc,
                     "wards_short": int((fr[f"{k}_short"] > 0).sum())})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 2. hospital surge and overflow routing
def bed_capacity(hist: pd.DataFrame, ratios: dict, factor: float = 0.9, capacity_csv=None) -> pd.Series:
    """Local bed capacity per ward. The dataset has NO capacity column, so we use one of two sources:
    (1) a user-supplied table (ward_id, bed_capacity), the real path; or
    (2) a reproducible planning assumption: the ward's 95th-percentile weekly bed need in the training years
        (before 2023) x a fixed pseudo-random facility-size multiplier in [0.5, 1.7] seeded by ward_id (real wards
        differ a lot in hospital access) x `factor` (the officer's slider). Replace with a real bed census when available."""
    if capacity_csv is not None:
        t = pd.read_csv(capacity_csv) if not isinstance(capacity_csv, pd.DataFrame) else capacity_csv
        return t.set_index("ward_id")["bed_capacity"].astype(int)
    import zlib
    h = hist[hist["date"] < "2023-01-01"]
    beds = h["hospital_beds_required"] if "hospital_beds_required" in h.columns else np.ceil(h["dengue_cases"] * ratios["beds"])
    p95 = beds.groupby(h["ward_id"]).quantile(0.95)
    size = pd.Series({w: 0.5 + 1.2 * ((zlib.crc32(w.encode()) % 1000) / 999) for w in p95.index})
    return np.maximum(1, np.ceil(p95 * size * factor)).astype(int)


def route_overflow(fr: pd.DataFrame, capacity: pd.Series):
    """Route patients from wards over capacity to wards with headroom in the SAME city only.

    Donors are tried in order of most headroom first (lowest utilisation). Returns (ward table, transfers,
    city summary). Patients that cannot be placed anywhere in the city are reported as unplaced."""
    w = fr[["city", "ward_id", "risk_tier", "beds_need"]].copy()
    w["local_capacity"] = w["ward_id"].map(capacity).fillna(1).astype(int)
    w["net_deficit"] = (w["beds_need"] - w["local_capacity"]).clip(lower=0)
    w["headroom"] = (w["local_capacity"] - w["beds_need"]).clip(lower=0)
    w["sent_out"], w["received"] = 0, 0
    transfers = []
    for city, g in w.groupby("city"):
        room = g.set_index("ward_id")["headroom"].to_dict()
        for ward, deficit in g[g.net_deficit > 0].sort_values("net_deficit", ascending=False)[["ward_id", "net_deficit"]].itertuples(index=False):
            left = int(deficit)
            while left > 0:
                donors = [(h, d) for d, h in room.items() if h > 0]
                if not donors:
                    break
                h, d = max(donors, key=lambda t: (t[0], t[1]))
                move = min(left, h)
                room[d] -= move
                left -= move
                transfers.append({"city": city, "from_ward": ward, "to_ward": d, "patients": move})
                w.loc[w.ward_id == ward, "sent_out"] += move
                w.loc[w.ward_id == d, "received"] += move
    w["final_load"] = w["beds_need"] - w["sent_out"] + w["received"]
    w["unplaced"] = (w["net_deficit"] - w["sent_out"]).clip(lower=0)
    w["utilisation_pct"] = (100 * w["final_load"] / w["local_capacity"]).round(0)
    w["status"] = np.where(w["unplaced"] > 0, "OVERFLOW - city full",
                           np.where(w["sent_out"] > 0, "Overflow routed out",
                                    np.where(w["received"] > 0, "Receiving overflow", "Within capacity")))
    tr = pd.DataFrame(transfers, columns=["city", "from_ward", "to_ward", "patients"])
    if len(tr):
        tr = tr.groupby(["city", "from_ward", "to_ward"], as_index=False)["patients"].sum()
    c = w.groupby("city").agg(bed_capacity=("local_capacity", "sum"), projected_demand=("beds_need", "sum"),
                              overflow_routed=("sent_out", "sum"), unplaced=("unplaced", "sum")).reset_index()
    c["utilisation_pct"] = (100 * (c.projected_demand - c.unplaced) / c.bed_capacity).round(0)
    c["remaining_headroom"] = (c.bed_capacity - (c.projected_demand - c.unplaced)).clip(lower=0)
    return w, tr, c


# ---------------------------------------------------------------- 3. what-if vector control simulator
def crews_full(density) -> np.ndarray:
    return np.clip(np.ceil(np.asarray(density) / CREW_DENSITY_UNIT), 1, MAX_CREWS_PER_WARD).astype(int)


def _predict_cases(X: pd.DataFrame, bundle) -> np.ndarray:
    return bundle["model"].predict(X[bundle["cols"]]).clip(0)


def counterfactual(feats_week: pd.DataFrame, bundle, crews: np.ndarray) -> np.ndarray:
    """Model-driven counterfactual: crews cut the rain-driven breeding signal fed to the saved model, then the
    model is re-run. Returns weekly cases at full effect per ward."""
    cov = np.minimum(1.0, np.asarray(crews, float) / crews_full(feats_week["population_density_sqkm"].values))
    X = feats_week.copy()
    mult = (1 - MAX_BREEDING_CUT * cov)
    for c in [c for c in bundle["cols"] if c.startswith("rain_")]:
        X[c] = X[c].values * mult
    return _predict_cases(X, bundle)


def prevented_by_crews(feats_week: pd.DataFrame, bundle, max_c: int = MAX_CREWS_PER_WARD) -> pd.DataFrame:
    """Cases prevented over weeks 2-4 for 0..max_c crews in every ward (rows: wards, cols: crews)."""
    base = _predict_cases(feats_week, bundle)
    full = crews_full(feats_week["population_density_sqkm"].values)
    ramp_sum = sum(RAMP.values())
    out = {0: np.zeros(len(base))}
    for c in range(1, max_c + 1):
        cc = np.minimum(c, full)
        out[c] = np.maximum(0, base - counterfactual(feats_week, bundle, cc)) * ramp_sum
    res = pd.DataFrame(out)
    res.insert(0, "ward_id", feats_week["ward_id"].values)
    return res


def plan_crews(feats_week: pd.DataFrame, bundle, total_crews: int) -> pd.DataFrame:
    """Greedy deployment by marginal ROI: each crew goes where it prevents the most extra cases.
    Returns per-ward crews and ROI (cases prevented per crew), ranked."""
    tab = prevented_by_crews(feats_week, bundle)
    full = crews_full(feats_week["population_density_sqkm"].values)
    n = len(tab)
    assigned = np.zeros(n, int)
    gains = tab[[0, 1, 2, 3]].values
    for _ in range(int(total_crews)):
        marg = np.full(n, -1.0)
        for i in range(n):
            if assigned[i] < full[i]:
                marg[i] = gains[i, assigned[i] + 1] - gains[i, assigned[i]]
        j = int(marg.argmax())
        if marg[j] <= 1e-9:
            break
        assigned[j] += 1
    prevented = np.array([gains[i, assigned[i]] for i in range(n)])
    out = pd.DataFrame({"ward_id": tab["ward_id"], "crews_needed_full": full, "crews_assigned": assigned,
                        "cases_prevented": prevented.round(1)})
    out["roi_cases_per_crew"] = np.where(assigned > 0, prevented / np.maximum(assigned, 1), 0).round(2)
    # single-crew ROI for ranking every ward, even those not funded
    out["roi_first_crew"] = gains[:, 1].round(2)
    return out.sort_values(["roi_first_crew"], ascending=False).reset_index(drop=True)


def simulate(feats_week: pd.DataFrame, bundle, crews_by_ward: dict, ratios: dict):
    """Run the what-if for a chosen {ward_id: crews}. Returns (summary by week, per-ward table)."""
    fw = feats_week.reset_index(drop=True)
    crews = fw["ward_id"].map(crews_by_ward).fillna(0).astype(int).values
    crews = np.minimum(crews, crews_full(fw["population_density_sqkm"].values))
    base = _predict_cases(fw, bundle)
    cut = np.maximum(0, base - counterfactual(fw, bundle, crews))
    rows = []
    for wk, r in RAMP.items():
        after = base - cut * r
        rows.append({"week_ahead": wk, "cases_without": base.sum(), "cases_with": after.sum(),
                     "cases_prevented": (cut * r).sum(),
                     "beds_without": base.sum() * ratios["beds"], "beds_with": after.sum() * ratios["beds"],
                     "saline_without": base.sum() * ratios["saline"], "saline_with": after.sum() * ratios["saline"]})
    s = pd.DataFrame(rows)
    per = pd.DataFrame({"ward_id": fw["ward_id"], "crews": crews, "cases_without_wk": base.round(2),
                        "cases_prevented_3wk": (cut * sum(RAMP.values())).round(1)})
    per["roi_cases_per_crew"] = np.where(per.crews > 0, per.cases_prevented_3wk / per.crews.clip(lower=1), 0).round(2)
    return s, per[per.crews > 0].sort_values("roi_cases_per_crew", ascending=False).reset_index(drop=True)

def assign_crews_by_priority(fr_alloc: pd.DataFrame, total_crews: int) -> pd.DataFrame:
    """Module 1 crew assignment: crews follow the same priority order as supplies (risk tier, then cases x density).
    Crews wanted: Critical/High = full coverage for the ward's density, Medium = 1, Low = 0.
    (Module 3 separately ranks wards by intervention ROI.)"""
    full = pd.Series(crews_full(fr_alloc["population_density_sqkm"].values), index=fr_alloc["ward_id"])
    want = fr_alloc.set_index("ward_id")["tier_idx"].map(lambda t: {0: 0, 1: 1}.get(t, None))
    out, left = [], int(total_crews)
    for r in fr_alloc.sort_values("priority_rank").itertuples():
        w = full[r.ward_id] if r.tier_idx >= 2 else (1 if r.tier_idx == 1 else 0)
        give = min(int(w), left)
        left -= give
        out.append((r.ward_id, int(w), give))
    return pd.DataFrame(out, columns=["ward_id", "crews_needed_full", "crews_assigned"])

