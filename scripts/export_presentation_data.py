#!/usr/bin/env python3
"""
Export ALL data the AEMCCO presentation needs into a single pre-baked JSON.

Rationale: in mlv-tools the simulations run live (Pyodide). In a talk we want
ZERO live compute, so everything the deck widgets show is baked here once:
  - per-model MSE x100 samples (for histograms / the auto-animated overlap widget)
  - the pairwise P(swap) vs N curves (MonteCarlo + Bootstrap), parsed from the
    tables already produced by src/02_analyze.py
  - a deterministic draw order for the auto-slider (reproducible animation)

Run:  python3 scripts/export_presentation_data.py
"""
import os, re, glob, json, math, random, csv

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DECOMP_YAML = "/home/quique/eeha/web-pages-tools/mlv-tools/raw_data/cpc18_decomp_mse_data_raw.yaml"
DECOMP_CSV  = os.path.join(PROJ, "analysis_results", "csv", "results_factorial_decomp.csv")
SHAREDPOOL_R2_CSV = os.path.join(PROJ, "analysis_results", "csv", "results_sharedpool_r2.csv")
TABLES_DIR  = os.path.join(PROJ, "analysis_results", "tables")
RANDOMSEED_GLOB = os.path.join(PROJ, "output_data", "*", "randomseed_training_metrics.yaml")
OUT_DIR  = "/home/quique/eeha/web-pages-tools/presentations/2026_07_8_AEMCCO_ALMERIA/data"
OUT_FILE = os.path.join(OUT_DIR, "presentation.json")

# ---------------------------------------------------------------- helpers
def stats(x):
    n = len(x)
    if n == 0:
        return {"n": 0}
    m = sum(x) / n
    var = sum((v - m) ** 2 for v in x) / n
    sd = math.sqrt(var)
    sk = sum(((v - m) / sd) ** 3 for v in x) / n if sd > 0 else 0.0
    return {"n": n, "mean": round(m, 5), "sd": round(sd, 5),
            "skew": round(sk, 3), "min": round(min(x), 5), "max": round(max(x), 5)}

def overlap_coef(a, b, bins=40):
    """Histogram overlap coefficient in [0,1] (1 = identical)."""
    lo = min(min(a), min(b)); hi = max(max(a), max(b))
    if hi <= lo:
        return 1.0
    w = (hi - lo) / bins
    def hist(z):
        h = [0] * bins
        for v in z:
            k = min(bins - 1, int((v - lo) / w))
            h[k] += 1
        return [c / len(z) for c in h]
    ha, hb = hist(a), hist(b)
    return round(sum(min(x, y) for x, y in zip(ha, hb)), 3)

# ---------------------------------------------------------------- 1) samples
def load_decomp():
    """init x order factorial runs (negated mse_x100) -> {model: [mse_x100,...]}."""
    try:
        import yaml
        d = yaml.safe_load(open(DECOMP_YAML))
    except Exception as e:
        print("  [decomp] no disponible:", e)
        return {}
    out = {}
    for model, vals in d.items():
        if isinstance(vals, list) and vals:
            out[model] = [round(-float(v), 5) for v in vals]  # de-negate -> mse_x100
    return out

def load_sharedpool_r2():
    """Caso 1: ejecución shared-seed-pool r2 -> {model: [mse_x100,...]} (escala leaderboard).
       Esta es la fuente correcta para los histogramas del Caso 1 (800 filas/modelo)."""
    if not os.path.exists(SHAREDPOOL_R2_CSV):
        print("  [sharedpool_r2] no disponible:", SHAREDPOOL_R2_CSV)
        return {}
    out = {}
    for r in csv.DictReader(open(SHAREDPOOL_R2_CSV)):
        try:
            out.setdefault(r["model"], []).append(round(float(r["mse_x100"]), 5))
        except (ValueError, KeyError):
            pass
    return out

def load_randomseed():
    try:
        import yaml
    except Exception:
        return {}
    out = {}
    for f in sorted(glob.glob(RANDOMSEED_GLOB)):
        model = f.split(os.sep)[-2]
        try:
            d = yaml.safe_load(open(f))
            vals = [round(float(r["mse"]) * 100, 5) for r in d.values()
                    if isinstance(r, dict) and "mse" in r]
            if vals:
                out[model] = vals
        except Exception as e:
            print(f"  [randomseed] {model}: {e}")
    return out

# ---------------------------------------------------------------- 2) P(swap) vs N
def parse_switch_tables():
    """Parse switched_probability combined tables -> {pair: {N, montecarlo, bootstrap}}."""
    out = {}
    for f in glob.glob(os.path.join(TABLES_DIR, "switched_probability_-_combined_-_*")):
        name = os.path.basename(f)
        pair = re.sub(r"^switched_probability_-_combined_-_", "", name)
        pair = re.sub(r"\.(tex|txt)$", "", pair)
        rows = []
        for line in open(f):
            m = re.findall(r"[-+]?\d*\.?\d+", line)
            # combined table rows: N, P_montecarlo, P_bootstrap
            if "&" in line and len(m) >= 3:
                try:
                    N = int(float(m[0])); mc = float(m[1]); bs = float(m[2])
                    rows.append((N, mc, bs))
                except ValueError:
                    pass
        if rows:
            rows.sort()
            out[pair] = {
                "N":          [r[0] for r in rows],
                "montecarlo": [round(r[1], 4) for r in rows],
                "bootstrap":  [round(r[2], 4) for r in rows],
            }
    return out

# ---------------------------------------------------------------- 3) rejilla 35x35 init x order
def build_grids():
    """{model: {n_init, n_order, z: 35x35 mse_x100 (row=init, col=order)}}."""
    if not os.path.exists(DECOMP_CSV):
        print("  [grid] no disponible:", DECOMP_CSV); return {}
    cells = {}  # model -> {(init,order): mse_x100}
    for r in csv.DictReader(open(DECOMP_CSV)):
        try:
            cells.setdefault(r["model"], {})[(int(r["init_seed"]), int(r["order_seed"]))] = float(r["mse_x100"])
        except (ValueError, KeyError):
            pass
    grids = {}
    for m, c in cells.items():
        inits  = sorted({i for (i, o) in c})
        orders = sorted({o for (i, o) in c})
        z = [[round(c.get((i, o), float("nan")), 3) for o in orders] for i in inits]
        grids[m] = {"n_init": len(inits), "n_order": len(orders), "z": z}
    return grids

# ---------------------------------------------------------------- 4) árbol de ablación
CSV_B1 = os.path.join(PROJ, "analysis_results", "csv", "results_ablation_b1.csv")
CSV_B3 = os.path.join(PROJ, "analysis_results", "csv", "results_ablation_b3.csv")

def _fmt_level(field, s):
    return ("%g" % float(s)) if field == "learning_rate" else str(s)

def build_ablation_tree(csv_path, row_field, col_field, case_id, model_label, row_label, col_label):
    """Bakes the greedy one-at-a-time (coordinate descent) decision-path tree per case.
       P(reach optimum) = fraction of the per-seed grids whose greedy endpoint == mean-best cell."""
    from collections import defaultdict
    if not os.path.exists(csv_path):
        return None
    rows = list(csv.DictReader(open(csv_path, newline="")))
    row_levels = sorted({r[row_field] for r in rows}, key=lambda x: float(x))
    col_levels = sorted({r[col_field] for r in rows}, key=lambda x: float(x))
    ri = {v: i for i, v in enumerate(row_levels)}; ci = {v: i for i, v in enumerate(col_levels)}
    nR, nC = len(row_levels), len(col_levels)
    vbs = defaultdict(lambda: [[None]*nC for _ in range(nR)])
    sg = [[0.0]*nC for _ in range(nR)]; cg = [[0]*nC for _ in range(nR)]
    ssq = [[0.0]*nC for _ in range(nR)]
    for r in rows:
        a, b = ri[r[row_field]], ci[r[col_field]]; v = float(r["mse_x100"])
        vbs[r["seed"]][a][b] = v; sg[a][b] += v; ssq[a][b] += v*v; cg[a][b] += 1
    gmean = [[sg[a][b]/cg[a][b] for b in range(nC)] for a in range(nR)]
    gstd = [[max(0.0, ssq[a][b]/cg[a][b]-gmean[a][b]**2)**0.5 for b in range(nC)] for a in range(nR)]
    seeds = sorted(s for s, g in vbs.items() if all(g[a][b] is not None for a in range(nR) for b in range(nC)))
    best = min((gmean[a][b], a, b) for a in range(nR) for b in range(nC)); gopt = (best[1], best[2])
    def endpoint(grid, axis, idx):
        if axis == "row":
            a = idx; b = min(range(nC), key=lambda j: grid[a][j]); s1 = (a, b)
            a = min(range(nR), key=lambda i: grid[i][b]); return a, b, s1
        b = idx; a = min(range(nR), key=lambda i: grid[i][b]); s1 = (a, b)
        b = min(range(nC), key=lambda j: grid[a][j]); return a, b, s1
    starts = []; tot_r = tot_n = 0
    for axis, field, label, levels in (("row", row_field, row_label, row_levels), ("col", col_field, col_label, col_levels)):
        for idx, lvl in enumerate(levels):
            ea, eb, s1 = endpoint(gmean, axis, idx)
            cnt = [[0]*nC for _ in range(nR)]; reached = 0
            for s in seeds:
                ga, gb, _ = endpoint(vbs[s], axis, idx); cnt[ga][gb] += 1
                if (ga, gb) == gopt: reached += 1
            tot_r += reached; tot_n += len(seeds)
            starts.append({"id": "%s=%s" % (label, _fmt_level(field, lvl)), "start_axis": axis,
                "start_index": idx, "start_label": "%s=%s" % (label, _fmt_level(field, lvl)),
                "path": [[s1[0], s1[1]], [ea, eb]], "reaches_optimum_mean": (ea, eb) == gopt,
                "p_optimum": round(reached/len(seeds), 4), "endpoint_counts": [c for row in cnt for c in row]})

    # ---- reach-tree probabilities (Monte Carlo + Bootstrap) --------------
    # Full decision tree rendered in the talk: root -> each col level (L, step 1)
    # -> each row level (B/E, step 2 leaf) = nC*nR leaves. P(reach) of a node =
    # fraction of greedy ablations ("fix a start row, sweep L, then sweep the
    # row factor") whose endpoint lands in it, aggregated over the nR possible
    # starts. Mirrors mlv-tools _simulate_decision_paths / _draw_sample: per
    # trial each cell is resampled INDEPENDENTLY -> MC = Normal(mean,std),
    # Bootstrap = empirical draw from that cell's per-seed values.
    cell_vals = [[[vbs[s][a][b] for s in seeds] for b in range(nC)] for a in range(nR)]

    def _greedy_col_first(g, r0):
        c1 = min(range(nC), key=lambda j: g[r0][j])    # step 1: best col (L) at fixed start row r0
        r2 = min(range(nR), key=lambda i: g[i][c1])    # step 2: best row (B/E) inside that L
        return c1, r2

    def _sim_reach(method, trials, seed0=20260706):
        rng = random.Random(seed0)
        s1 = [0.0] * nC
        leaf = [[0.0] * nR for _ in range(nC)]         # leaf[col][row]
        for _ in range(trials):
            if method == "mc":
                g = [[rng.gauss(gmean[a][b], gstd[a][b]) for b in range(nC)] for a in range(nR)]
            else:
                g = [[rng.choice(cell_vals[a][b]) for b in range(nC)] for a in range(nR)]
            for r0 in range(nR):
                c1, r2 = _greedy_col_first(g, r0)
                s1[c1] += 1.0
                leaf[c1][r2] += 1.0
        denom = float(trials * nR)
        s1 = [round(x / denom, 4) for x in s1]
        leaf = [[round(leaf[cl][rr] / denom, 4) for rr in range(nR)] for cl in range(nC)]
        return s1, leaf

    TRIALS = 2000
    mc_s1, mc_leaf = _sim_reach("mc", TRIALS)
    bs_s1, bs_leaf = _sim_reach("bs", TRIALS)
    reach_tree = {
        "trials": TRIALS, "fixed": "col", "swept": "row",
        "step1": [{"mc": mc_s1[c], "bs": bs_s1[c]} for c in range(nC)],
        "leaf":  [[{"mc": mc_leaf[c][r], "bs": bs_leaf[c][r]} for r in range(nR)] for c in range(nC)],
    }

    # ---- PER-START reach tree (fix the start on the ROW factor: 3 starts) ----
    # Faithful to mlv-tools _simulate_decision_paths: FIX the start factor
    # (row = B), STEP 1 = sweep the col factor (L) to its best, STEP 2 =
    # re-optimize the start factor (B) inside the STEP-1 WINNING L. Per start we
    # bake, under BOTH methods:
    #   step1[c] = P(greedy step-1 lands on L=c | start)              [marginal]
    #   step2[r] = P(greedy step-2 lands on B=r | start, step1=winL)  [conditional]
    # where winL = modal step-1 pick (matches mlv-tools p_step2 = c2/step1_count
    # and pReach = p_step1 * p_step2). Levels stay in a STABLE order across the 3
    # starts so the widget can tween the percentages in place while it cycles.
    def _sim_by_start(method, trials, seed0=20260707):
        rng = random.Random(seed0)
        s1 = [[0] * nC for _ in range(nR)]                                 # s1[r0][c1]
        s2 = [[[0] * nR for _ in range(nC)] for _ in range(nR)]           # s2[r0][c1][r2]
        for _ in range(trials):
            if method == "mc":
                g = [[rng.gauss(gmean[a][b], gstd[a][b]) for b in range(nC)] for a in range(nR)]
            else:
                g = [[rng.choice(cell_vals[a][b]) for b in range(nC)] for a in range(nR)]
            for r0 in range(nR):
                c1 = min(range(nC), key=lambda j: g[r0][j])                # step 1: best L at fixed start B=r0
                r2 = min(range(nR), key=lambda i: g[i][c1])                # step 2: best B inside that L
                s1[r0][c1] += 1
                s2[r0][c1][r2] += 1
        return s1, s2

    mcs1, mcs2 = _sim_by_start("mc", TRIALS)
    bss1, bss2 = _sim_by_start("bs", TRIALS)
    by_start = []
    for r0 in range(nR):
        winL = max(range(nC), key=lambda c: mcs1[r0][c])                   # modal step-1 branch (green)
        step1 = [{"label": "%s=%s" % (col_label, _fmt_level(col_field, col_levels[c])),
                  "mc": round(mcs1[r0][c] / TRIALS, 4), "bs": round(bss1[r0][c] / TRIALS, 4),
                  "best": c == winL} for c in range(nC)]
        mcden = sum(mcs2[r0][winL]) or 1
        bsden = sum(bss2[r0][winL]) or 1
        winB = max(range(nR), key=lambda i: mcs2[r0][winL][i])            # modal step-2 pick (green)
        step2 = [{"label": "%s=%s" % (row_label, _fmt_level(row_field, row_levels[i])),
                  "mc": round(mcs2[r0][winL][i] / mcden, 4), "bs": round(bss2[r0][winL][i] / bsden, 4),
                  "best": i == winB} for i in range(nR)]
        c1m = min(range(nC), key=lambda j: gmean[r0][j])                   # greedy on the mean grid
        r2m = min(range(nR), key=lambda i: gmean[i][c1m])
        # HOJAS COMPLETAS: 3 B bajo CADA L = 9 nodos. leaf[c][r] = P(una búsqueda voraz
        # desde este arranque termine en (L=c, B=r)) = frecuencia conjunta (suma 9 = 1).
        leaf = [[{"mc": round(mcs2[r0][c][r] / float(TRIALS), 4),
                  "bs": round(bss2[r0][c][r] / float(TRIALS), 4)}
                 for r in range(nR)] for c in range(nC)]
        by_start.append({
            "start_label": "%s=%s" % (row_label, _fmt_level(row_field, row_levels[r0])),
            "start_axis": "row", "start_index": r0, "win_col": winL, "win_col_label": step1[winL]["label"],
            "win_row": winB, "step1": step1, "step2": step2, "opt_reached": (r2m, c1m) == gopt,
            "leaf": leaf,
        })
    reach_tree["by_start"] = by_start

    return {"case_id": case_id, "model": model_label, "metric_label": "MSE×100", "lower_is_better": True,
        "reach_tree": reach_tree,
        "n_seeds": len(seeds),
        "factors": {"row": {"field": row_field, "label": row_label, "levels": [_fmt_level(row_field, v) for v in row_levels]},
                    "col": {"field": col_field, "label": col_label, "levels": [_fmt_level(col_field, v) for v in col_levels]}},
        "grid_mean": [[round(v, 4) for v in r] for r in gmean], "grid_std": [[round(v, 4) for v in r] for r in gstd],
        "global_optimum": {"row": gopt[0], "col": gopt[1], "value": round(gmean[gopt[0]][gopt[1]], 4),
            "label": "%s=%s | %s=%s" % (row_label, _fmt_level(row_field, row_levels[gopt[0]]), col_label, _fmt_level(col_field, col_levels[gopt[1]]))},
        "starts": starts, "p_optimum_overall": round(tot_r/tot_n, 4)}

def build_tree_all():
    out = {}
    b1 = build_ablation_tree(CSV_B1, "batch_size", "learning_rate", "b1", "BourginMLP", "B", "L")
    b3 = build_ablation_tree(CSV_B3, "epsilon", "learning_rate", "b3", "SparseMLP", "E", "L")
    if b1: out["b1"] = b1
    if b3: out["b3"] = b3
    return out

# ---------------------------------------------------------------- 4b) ANOVA / descomposición de varianza
#  Dos análisis distintos y honestos:
#    · anova_ablation (caso b1): ANOVA de dos factores B×L de mse_x100 con las semillas
#      como repeticiones (800 por celda, diseño equilibrado 3×3). Es INFERENCIAL:
#      hay repetición real dentro de cada celda -> término de error válido -> F y p.
#    · anova_variance (init×orden): rejilla 35×35 con UNA observación por celda. Sin
#      repetición NO hay error puro: la interacción y el residual quedan confundidos
#      y NO existe F/p inferencial. Se hornea SOLO la descomposición factorial de la
#      varianza (η² = SS_fuente / SS_total) por modelo. No inventamos p-valores.

def _betacf(a, b, x):
    """Fracción continua de la beta incompleta (Numerical Recipes)."""
    MAXIT, EPS, FPMIN = 300, 3e-16, 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < FPMIN: d = FPMIN
    d = 1.0 / d; h = d
    for m in range(1, MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN: d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN: c = FPMIN
        d = 1.0 / d; h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN: d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN: c = FPMIN
        d = 1.0 / d; de = d * c; h *= de
        if abs(de - 1.0) < EPS: break
    return h

def _betai(a, b, x):
    """Beta incompleta regularizada I_x(a,b)."""
    if x <= 0.0: return 0.0
    if x >= 1.0: return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    bt = math.exp(lbeta + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b

def f_sf(F, d1, d2):
    """P(X > F) para F(d1,d2). scipy si está; si no, beta incompleta a mano."""
    if F <= 0: return 1.0
    try:
        from scipy import stats
        return float(stats.f.sf(F, d1, d2))
    except Exception:
        return _betai(d2 / 2.0, d1 / 2.0, d2 / (d2 + d1 * F))

def _pdisp(p):
    """Cadena de display para un p-valor (evita '0.0' cuando hay underflow)."""
    if p is None: return None
    if p <= 0 or p < 1e-16: return "<1e-16"
    if p < 1e-4: return "%.1e" % p
    if p < 0.001: return "<0.001"
    return "%.4f" % p

def _p_method():
    try:
        import scipy  # noqa
        return "scipy.stats.f.sf"
    except Exception:
        return "beta-incompleta (Numerical Recipes)"

def build_anova_ablation(csv_path=CSV_B1, row_field="batch_size", col_field="learning_rate",
                         row_label="B", col_label="L", model_label="BourginMLP", metric="mse_x100",
                         subj_field="seed"):
    """RM-ANOVA de dos factores INTRA-SUJETO (B×L) sobre mse_x100; la semilla es el sujeto.
       Cada semilla se mide en las a×b condiciones → cada efecto se contrasta contra su propio
       término de error efecto×sujeto (B vs B×S, L vs L×S, B×L vs B×L×S)."""
    if not os.path.exists(csv_path):
        return None
    rows = list(csv.DictReader(open(csv_path, newline="")))
    A = sorted({r[row_field] for r in rows}, key=lambda x: float(x))
    B = sorted({r[col_field] for r in rows}, key=lambda x: float(x))
    a, b = len(A), len(B)
    ai = {v: i for i, v in enumerate(A)}; bi = {v: i for i, v in enumerate(B)}
    from collections import defaultdict
    X = defaultdict(lambda: [[None] * b for _ in range(a)])            # X[seed][i][j]
    for r in rows:
        try:
            v = float(r[metric])
        except (ValueError, KeyError):
            continue
        sd = r.get(subj_field)
        if sd is None:
            continue
        X[sd][ai[r[row_field]]][bi[r[col_field]]] = v
    # sujetos COMPLETOS (medidos en las a×b condiciones) → diseño equilibrado intra-sujeto
    seeds = [s for s, g in X.items() if all(g[i][j] is not None for i in range(a) for j in range(b))]
    n = len(seeds)
    if n < 2:
        return None
    xv = lambda s, i, j: X[s][i][j]
    allv = [xv(s, i, j) for s in seeds for i in range(a) for j in range(b)]
    N = len(allv); grand = sum(allv) / N
    cmean = [[sum(xv(s, i, j) for s in seeds) / n for j in range(b)] for i in range(a)]   # AC_ij (medias por celda)
    def _sd(vs, mu):
        k = len(vs)
        return (sum((x - mu) ** 2 for x in vs) / (k - 1)) ** 0.5 if k > 1 else 0.0
    cse = [[_sd([xv(s, i, j) for s in seeds], cmean[i][j]) / (n ** 0.5) for j in range(b)] for i in range(a)]
    Amean = [sum(xv(s, i, j) for s in seeds for j in range(b)) / (n * b) for i in range(a)]
    Cmean = [sum(xv(s, i, j) for s in seeds for i in range(a)) / (n * a) for j in range(b)]
    Smean = {s: sum(xv(s, i, j) for i in range(a) for j in range(b)) / (a * b) for s in seeds}
    AS = {(s, i): sum(xv(s, i, j) for j in range(b)) / b for s in seeds for i in range(a)}   # A×S
    CS = {(s, j): sum(xv(s, i, j) for i in range(a)) / a for s in seeds for j in range(b)}   # C×S
    SS_total = sum((v - grand) ** 2 for v in allv)
    SS_A  = b * n * sum((Amean[i] - grand) ** 2 for i in range(a))
    SS_C  = a * n * sum((Cmean[j] - grand) ** 2 for j in range(b))
    SS_AC = n * sum((cmean[i][j] - Amean[i] - Cmean[j] + grand) ** 2 for i in range(a) for j in range(b))
    SS_S  = a * b * sum((Smean[s] - grand) ** 2 for s in seeds)
    SS_AS = b * sum((AS[(s, i)] - Amean[i] - Smean[s] + grand) ** 2 for s in seeds for i in range(a))
    SS_CS = a * sum((CS[(s, j)] - Cmean[j] - Smean[s] + grand) ** 2 for s in seeds for j in range(b))
    SS_ACS = max(0.0, SS_total - SS_A - SS_C - SS_AC - SS_S - SS_AS - SS_CS)   # error de la interacción
    dfA, dfC, dfAC, dfS = a - 1, b - 1, (a - 1) * (b - 1), n - 1
    dfAS, dfCS, dfACS = (a - 1) * (n - 1), (b - 1) * (n - 1), (a - 1) * (b - 1) * (n - 1)
    def _eff(name, SS, df, SSe, dfe):
        MS, MSe = SS / df, SSe / dfe
        F = MS / MSe if MSe > 0 else float("inf")
        p = f_sf(F, df, dfe)
        return {"source": name, "df": df, "df_err": dfe, "SS": round(SS, 4), "MS": round(MS, 5),
                "F": round(F, 2), "p": p, "p_disp": _pdisp(p),
                "eta2": round(SS / SS_total, 4),
                "partial_eta2": round(SS / (SS + SSe), 4) if (SS + SSe) > 0 else 0.0}
    table = [
        _eff(row_label, SS_A, dfA, SS_AS, dfAS),           # B  vs B×Sujeto
        _eff(col_label, SS_C, dfC, SS_CS, dfCS),           # L  vs L×Sujeto
        _eff(row_label + "×" + col_label, SS_AC, dfAC, SS_ACS, dfACS),  # B×L vs B×L×Sujeto
    ]
    return {
        "case_id": "b1", "model": model_label, "metric_label": "MSE×100", "lower_is_better": True,
        "test": f"ANOVA de medidas repetidas ({row_label} × {col_label} intra-sujeto)",
        "note": "%d semillas (sujetos), cada una en las %d condiciones; cada efecto vs su error efecto×sujeto." % (n, a * b),
        "repeated_measures": True, "n_subjects": n, "n_per_cell": n, "balanced": (len(X) == n), "N": N,
        "grand_mean": round(grand, 5), "ss_total": round(SS_total, 4),
        "p_method": _p_method(),
        "factors": {"B": {"label": row_label, "field": row_field, "levels": A},
                    "L": {"label": col_label, "field": col_field, "levels": B}},
        "cell_mean": [[round(cmean[i][j], 4) for j in range(b)] for i in range(a)],  # [B][L]
        "cell_se":   [[round(cse[i][j], 5) for j in range(b)] for i in range(a)],
        "table": table,
    }

def build_anova_variance(csv_path=DECOMP_CSV):
    """DESCOMPOSICIÓN DE VARIANZA DESCRIPTIVA init×orden — NO es un ANOVA inferencial.
       Rejilla balanceada 35×35, UNA ejecución o entrenamiento por celda. El entrenamiento es DETERMINISTA dadas
       (init, orden): repetir una celda da MSE bit-idéntico (verificado en sameseed_*.yaml), así
       que no hay error de medida y el residual ES la interacción init×orden (estructura, no ruido).
       Con 1 obs/celda interacción y error puro estarían confundidos -> NO se hacen F-tests ni p.
       Se reporta solo η² = SS_fuente / SS_total (share de la varianza observada)."""
    if not os.path.exists(csv_path):
        return None
    per = {}
    for r in csv.DictReader(open(csv_path)):
        try:
            i = int(r["init_seed"]); o = int(r["order_seed"]); v = float(r["mse_x100"]); m = r["model"]
        except (ValueError, KeyError):
            continue
        per.setdefault(m, {})[(i, o)] = v
    if not per:
        return None
    models = sorted(per.keys())
    varpct = {}; anova = {}
    for m in models:
        cells = per[m]
        inits = sorted({i for (i, o) in cells}); orders = sorted({o for (i, o) in cells})
        ni, no = len(inits), len(orders)
        vals = list(cells.values()); Nm = len(vals); grand = sum(vals) / Nm
        imean = {i: sum(cells[(i, o)] for o in orders if (i, o) in cells) /
                    sum(1 for o in orders if (i, o) in cells) for i in inits}
        omean = {o: sum(cells[(i, o)] for i in inits if (i, o) in cells) /
                    sum(1 for i in inits if (i, o) in cells) for o in orders}
        SS_total = sum((v - grand) ** 2 for v in vals)
        SS_init = no * sum((imean[i] - grand) ** 2 for i in inits)   # equilibrado: no obs por init
        SS_ord = ni * sum((omean[o] - grand) ** 2 for o in orders)
        SS_int = SS_total - SS_init - SS_ord                         # residual = interacción (sin ruido)
        df_init, df_ord, df_int = ni - 1, no - 1, (ni - 1) * (no - 1)
        # DESCRIPTIVO: η² = SS_fuente / SS_total (share de la varianza observada). SIN F/p:
        # con 1 obs/celda no hay error puro separable de la interacción -> no procede inferencia.
        eta_init = SS_init / SS_total if SS_total else 0.0
        eta_ord = SS_ord / SS_total if SS_total else 0.0
        eta_int = SS_int / SS_total if SS_total else 0.0
        varpct[m] = {"init": round(eta_init, 4), "order": round(eta_ord, 4), "inter": round(eta_int, 4)}
        anova[m] = {
            "table": [
                {"source": "Inicialización", "df": df_init, "SS": round(SS_init, 4), "pct": round(eta_init, 4)},
                {"source": "Orden",          "df": df_ord,  "SS": round(SS_ord, 4),  "pct": round(eta_ord, 4)},
                {"source": "Interacción",    "df": df_int,  "SS": round(SS_int, 4),  "pct": round(eta_int, 4)},
            ],
            "ss_total": round(SS_total, 4), "grand_mean": round(grand, 5), "n": Nm,
            "balanced": (len(cells) == ni * no),
        }
    return {
        "test": "Descomposición de varianza (η² observado · init × orden)",
        "inferential": False, "descriptive": True,
        "note": "Rejilla balanceada 35×35, 1 entrenamiento/celda; entrenamiento DETERMINISTA dadas las "
                "semillas (MSE bit-idéntico al repetir → verificado). η² = SS_fuente/SS_total; sin F "
                "ni p (con 1 obs/celda no hay error puro separable de la interacción → no procede inferencia).",
        "metric_label": "MSE×100", "lower_is_better": True,
        "n_init": 35, "n_order": 35,
        "sources": [{"key": "init",  "label": "Inicialización"},
                    {"key": "order", "label": "Orden"},
                    {"key": "inter", "label": "Interacción"}],
        "models": models,
        "varpct": varpct,
        "anova": anova,
    }

# ---------------------------------------------------------------- 5) Baldominos (MNIST: banda de una red vs. review publicada)
MNIST_REPO = "/home/quique/umh/mnist-learning-playground"

def _load_error_rates():
    import ast
    src = os.path.join(MNIST_REPO, "src", "04_check_distances.py")
    if not os.path.exists(src):
        return None
    tree = ast.parse(open(src, encoding="utf-8").read())
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id == "error_rates":
                    return ast.literal_eval(node.value)
    return None

def build_baldominos(model="CNN_14L"):
    import yaml
    err = _load_error_rates()
    yml = os.path.join(MNIST_REPO, "output_data", model, "randomseed_training_metrics.yaml")
    if err is None or not os.path.exists(yml):
        return None
    d = yaml.safe_load(open(yml, encoding="utf-8"))
    seeds = [round(float(e["accuracy"]) * 100.0, 4) for e in d.values() if isinstance(e, dict) and "accuracy" in e]
    if not seeds:
        return None
    n = len(seeds); lo, hi = min(seeds), max(seeds); mean = sum(seeds) / n
    std = (sum((v - mean) ** 2 for v in seeds) / n) ** 0.5
    reg = {"no_data_augmentation": "no_data_augmentation", "data_augmentated": "data_augmented"}
    published = []
    for src_regime, table in err.items():
        for name, e in table.items():
            acc = round(100.0 - float(e), 4)
            published.append({"name": name.strip(), "value": acc,
                              "regime": reg.get(src_regime, src_regime), "in_band": bool(lo <= acc <= hi)})
    # Valor REPORTADO por los autores de la arquitectura que replicamos (TRAZABILIDAD):
    # CNN_14L = SimpleNet / SimpNet_V1 (variante NoDropout) — Hasanpour, Rouhani, Fayyaz, Sabokrou
    # (2016), "Lets keep it simple, Using simple architectures to outperform deeper and more complex
    # architectures", arXiv:1608.06037 (repo Coderx7/SimpleNet). MNIST: 0.25 % error = 99.75 %
    # accuracy, SIN data augmentation. Fuente confirmada en el README oficial del repo y verificada
    # contra la cabecera de mnist-learning-playground/src/models/CNN_14L.py ("# Model from: ...").
    reported = ({"value": 99.75, "label": "SimpleNet reportó",
                 "source": "Hasanpour et al. 2016, SimpNet_V1 NoDropout (arXiv:1608.06037); MNIST 99.75 % (0.25 % error), sin augmentation"}
                if model == "CNN_14L" else None)
    return {"metric": "precisión", "unit": "%", "higher_is_better": True,
            "source": "Baldominos, Sáez, Isasi 2019, Applied Sciences 9(15):3169",
            "model": {"name": model, "band": [round(lo, 4), round(hi, 4)], "mean": round(mean, 4),
                      "std": round(std, 4), "n_seeds": n, "seeds": seeds, "reported": reported},
            "published": published, "overlap_count": sum(1 for p in published if p["in_band"]),
            "n_published": len(published)}

# ---------------------------------------------------------------- main
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("Cargando muestras Caso 1 (shared-seed-pool r2)...")
    samples = load_sharedpool_r2()
    source = "sharedpool_r2"
    if not samples:
        print("Fallback a decomp init x order...")
        samples = load_decomp()
        source = "decomp_init_x_order"

    models = {m: {"mse_x100": v, **stats(v)} for m, v in samples.items()}
    print(f"  {len(models)} modelos desde '{source}':")
    for m, s in models.items():
        print(f"    {m:<16} n={s['n']:>5} mean={s['mean']:.3f} sd={s['sd']:.3f} "
              f"skew={s['skew']:+.2f} range=[{s['min']:.3f},{s['max']:.3f}]")

    # pareja con mas solape (candidatos para el widget 1)
    names = list(samples.keys())
    pairs_overlap = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            oc = overlap_coef(samples[names[i]], samples[names[j]])
            pairs_overlap.append((oc, names[i], names[j]))
    pairs_overlap.sort(reverse=True)
    print("\n  Solape entre parejas (top 5):")
    for oc, a, b in pairs_overlap[:5]:
        print(f"    {a} vs {b}: {oc}")
    overlap_pair = list(pairs_overlap[0][1:]) if pairs_overlap else names[:2]

    print("\nConstruyendo rejillas 35x35 (init x order)...")
    grids = build_grids()
    for m, g in grids.items():
        flat = [v for row in g["z"] for v in row if v == v]
        print(f"    {m:<16} {g['n_init']}x{g['n_order']}  rango=[{min(flat):.3f},{max(flat):.3f}]")

    print("\nConstruyendo árbol de ablación (b1/b3)...")
    tree = build_tree_all()
    for cid, c in tree.items():
        print(f"    {cid} {c['model']:<14} óptimo {c['global_optimum']['label']}  "
              f"P(alcanza)={c['p_optimum_overall']*100:.0f}% ({c['n_seeds']} seeds)")

    print("\nANOVA de dos factores B×L (caso b1, semillas = repeticiones)...")
    anova_ablation = build_anova_ablation(CSV_B3, "epsilon", "learning_rate", "E", "L", "SparseMLP")
    if anova_ablation:
        anova_ablation["case_id"] = "b3"
        print(f"    N={anova_ablation['N']}  n/celda={anova_ablation['n_per_cell']}  "
              f"equilibrado={anova_ablation['balanced']}  p via {anova_ablation['p_method']}")
        print(f"    {'Fuente':<10}{'gl':>6}{'SS':>12}{'F':>12}{'p':>10}{'η²':>8}")
        for row in anova_ablation["table"]:
            F = row.get("F"); p = row.get("p_disp", "—")
            print(f"    {row['source']:<10}{row['df']:>6}{row['SS']:>12.2f}"
                  f"{('%.2f' % F) if F is not None else '—':>12}{p:>10}{row['eta2']*100:>7.2f}%")

    print("\nDescomposición de varianza init×orden (η² observado · descriptiva, sin F/p)...")
    anova_variance = build_anova_variance()
    if anova_variance:
        print(f"    {anova_variance['note']}")
        for m in anova_variance["models"]:
            print(f"    {m}:")
            print(f"      {'Fuente':<16}{'gl':>6}{'η² (%var)':>11}")
            for row in anova_variance["anova"][m]["table"]:
                print(f"      {row['source']:<16}{row['df']:>6}{row['pct']*100:>10.1f}%")

    print("\nParseando tablas P(swap) vs N (MonteCarlo + Bootstrap)...")
    switch = parse_switch_tables()
    print(f"  {len(switch)} parejas con curva P(swap):")
    for p in sorted(switch):
        print(f"    {p}")

    # secuencia determinista para el slider auto-animado (reproducible)
    rng = random.Random(42)
    max_len = max((len(v) for v in samples.values()), default=0)
    draw_order = list(range(max_len)); rng.shuffle(draw_order)
    draw_order = draw_order[:300]  # suficientes frames

    data = {
        "meta": {
            "source": source,
            "note": "Datos pre-horneados para la presentacion AEMCCO. Sin computo en vivo.",
            "metric": "mse_x100 (menor es mejor)",
        },
        "models": models,
        "overlap_pair": overlap_pair,
        "overlap_candidates": [{"pair": [a, b], "overlap": oc}
                               for oc, a, b in pairs_overlap[:5]],
        "grids": grids,
        "ablation_tree": tree,
        "anova_ablation": anova_ablation,
        "anova_variance": anova_variance,
        "baldominos": build_baldominos(),
        "switch_prob": switch,
        "draw_order": draw_order,
    }
    with open(OUT_FILE, "w") as fh:
        json.dump(data, fh, ensure_ascii=False, separators=(",", ":"))
    kb = os.path.getsize(OUT_FILE) / 1024
    print(f"\n[OK] {OUT_FILE}  ({kb:.1f} KB)")
    print(f"     overlap_pair sugerido: {overlap_pair}")

if __name__ == "__main__":
    main()
