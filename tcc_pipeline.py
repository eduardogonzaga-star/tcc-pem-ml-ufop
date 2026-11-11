import os, sys, json, time, subprocess, warnings, zipfile
import numpy as np, pandas as pd, sklearn, scipy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

warnings.filterwarnings("ignore")
SEED, N_PERM, MLP_SEEDS, GPR_RESTARTS = 42, 30, [0, 1, 2, 3, 4], 2
DEGEN_R, DEGEN_M = 1e-4, 1e-4
OUT = os.environ.get("TCC_OUT_DIR", "resultados_tcc_final")
os.makedirs(OUT, exist_ok=True)
P0 = "Activation Test MEA "
PROTOCOLS = [P0 + s for s in ["Constant Current 0.25A", "Constant Voltage 0.6V", "Constant Voltage 0.6V-2",
                              "Constant Voltage 0.6V-3", "Constant Voltage 0.6V-4", "Cycling Potential",
                              "Standard Protocol", "Standard Protocol (Repeat)"]]
REF = {("Random Forest", "GroupKFold_curva"): .7218, ("Random Forest", "LOPO"): .7347,
       ("Gradient Boosting", "GroupKFold_curva"): .7546, ("Gradient Boosting", "LOPO"): .7681,
       ("Regressao Linear", "GroupKFold_curva"): .5013, ("Regressao Linear", "LOPO"): .5190,
       ("MLP", "GroupKFold_curva"): .5821, ("MLP", "LOPO"): -.7296}


def log(*a):
    print(*a, flush=True)


root = next((c for c in [os.environ.get("TCC_DATA_DIR"), "pem-dataset1", "/content/pem-dataset1",
                         "/home/claude/ecsim/pem-dataset1"] if c and os.path.isdir(os.path.join(c, PROTOCOLS[0]))), None)
if root is None:
    subprocess.run(["git", "clone", "-q", "https://github.com/ECSIM/pem-dataset1.git"], check=True)
    root = "pem-dataset1"

frames = []
for p in PROTOCOLS:
    for k in (4, 5):
        d = pd.read_csv(os.path.join(root, p, f"{k}.csv")).rename(
            columns={"current_density": "j", "cell_voltage": "V", "pressure": "P", "relative_humidity": "RH"})
        d["protocol"], d["file"] = p, k
        frames.append(d[["j", "V", "P", "RH", "protocol", "file"]])
data = pd.concat(frames, ignore_index=True)
data["curve_id"] = data.protocol + "|" + data.file.astype(str) + "|" + data.P.astype(str) + "|" + data.RH.astype(str)
assert len(data) == 2212 and data.curve_id.nunique() == 141 and data.protocol.nunique() == 8
assert set(data.P) == {5, 15, 25} and set(data.RH) == {30, 50, 100}
assert not data[["j", "V", "P", "RH"]].isna().any().any() and (data.j > 0).all()

ohe = OneHotEncoder(categories=[sorted(PROTOCOLS)], sparse_output=False)
Xp = ohe.fit_transform(data[["protocol"]])
X = np.hstack([data[["j", "P", "RH"]].to_numpy(float), Xp])
y = data.V.to_numpy(float)
NUM, ALL = [0, 1, 2], list(range(11))

def group_kfold(groups, n=5):
    u, inv = np.unique(groups, return_inverse=True)
    cnt = np.bincount(inv)
    w, g2f = np.zeros(n), np.zeros(len(u), int)
    for i in np.argsort(cnt)[::-1]:
        f = np.argmin(w)
        w[f] += cnt[i]
        g2f[i] = f
    fold = g2f[inv]
    return [(np.where(fold != f)[0], np.where(fold == f)[0]) for f in range(n)]


GROUPS = {"GroupKFold_curva": "curve_id", "LOPO": "protocol", "LO-Pressao": "P", "LO-UR": "RH"}
SPLITS = {}
for s, c in GROUPS.items():
    g = data[c].to_numpy()
    SPLITS[s] = group_kfold(g) if s == "GroupKFold_curva" else list(LeaveOneGroupOut().split(X, y, g))
    assert all(not set(g[a]) & set(g[b]) for a, b in SPLITS[s])


class Semi(BaseEstimator, RegressorMixin):
    def __init__(self, extended=False):
        self.extended = extended

    @staticmethod
    def f(j, E, b, R, m, n):
        return E - b * np.log(j) - R * j - m * np.exp(np.clip(n * j, None, 500))

    @staticmethod
    def fe(Z, E0, EP, ER, b, R0, RP, RR, m, n):
        j, P, RH = Z
        return E0 + EP * P + ER * RH - b * np.log(j) - (R0 + RP * P + RR * RH) * j - m * np.exp(np.clip(n * j, None, 500))

    def fit(self, A, t):
        A = np.asarray(A, float)
        if self.extended:
            self.p_, _ = curve_fit(self.fe, (A[:, 0], A[:, 1], A[:, 2]), t,
                                   p0=[1, 0, 0, .05, 1e-4, 0, 0, 1e-6, 1e-2],
                                   bounds=([.5, -.01, -.01, 0, -1e-3, -1e-3, 0, 0, 0],
                                           [1.5, .01, .01, 1, 1e-3, 1e-3, 1, 1, .05]), maxfev=40000)
        else:
            self.p_, _ = curve_fit(self.f, A[:, 0], t, p0=[1, .05, 1e-4, 1e-6, 1e-2],
                                   bounds=([.5, 0, 0, 0, 0], [1.5, 1, 1, 1, .05]), maxfev=40000)
        return self

    def predict(self, A):
        A = np.asarray(A, float)
        return self.fe((A[:, 0], A[:, 1], A[:, 2]), *self.p_) if self.extended else self.f(A[:, 0], *self.p_)


class LinLnJ(BaseEstimator, RegressorMixin):
    def fit(self, A, t):
        A = np.asarray(A, float)
        self.m_ = LinearRegression().fit(np.column_stack([np.log(A[:, 0]), A[:, 0]]), t)
        return self

    def predict(self, A):
        A = np.asarray(A, float)
        return self.m_.predict(np.column_stack([np.log(A[:, 0]), A[:, 0]]))


def mlp(hls=(64, 32), alpha=1e-3, seed=SEED):
    return MLPRegressor(hidden_layer_sizes=hls, alpha=alpha, max_iter=2000, early_stopping=True,
                        n_iter_no_change=20, random_state=seed)


SPECS = {
    "Baseline": (lambda: DummyRegressor(), NUM, True, {}),
    "Regressao Linear": (lambda: LinearRegression(), ALL, True, {}),
    "Random Forest": (lambda: RandomForestRegressor(n_estimators=300, max_depth=12, random_state=SEED, n_jobs=-1),
                      ALL, True, {"n_estimators": 300, "max_depth": 12}),
    "Gradient Boosting": (lambda: GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=.05,
                                                            random_state=SEED), ALL, True,
                          {"n_estimators": 300, "max_depth": 3, "learning_rate": .05}),
    "GPR (ARD)": (lambda: GaussianProcessRegressor(kernel=C(1.) * RBF([1., 1., 1.]) + WhiteKernel(1.),
                                                   n_restarts_optimizer=GPR_RESTARTS, normalize_y=True,
                                                   random_state=SEED), NUM, True,
                  {"kernel": "C*RBF(ARD,3)+White", "n_restarts_optimizer": GPR_RESTARTS}),
    "MLP": (lambda: mlp(), ALL, True, {"hidden": [64, 32], "alpha": 1e-3}),
    "MLP (sem protocolo)": (lambda: mlp(), NUM, True, {"hidden": [64, 32], "alpha": 1e-3}),
    "Semiempirico global (j)": (lambda: Semi(False), NUM, False, {}),
    "Semiempirico estendido (j,P,UR)": (lambda: Semi(True), NUM, False, {}),
    "Regressao (ln j, j)": (lambda: LinLnJ(), NUM, False, {}),
}
MAIN = ["Baseline", "Regressao Linear", "Random Forest", "Gradient Boosting", "GPR (ARD)", "MLP"]


def build(name):
    mk, _, sc, _ = SPECS[name]
    return Pipeline([("s", MinMaxScaler()), ("m", mk())]) if sc else mk()


def mape(a, b):
    return float(np.mean(np.abs((a - b) / a)) * 100)


def run(name, scheme):
    cols = SPECS[name][1]
    oof, rows = np.full(len(y), np.nan), []
    g = data[GROUPS[scheme]].to_numpy()
    for k, (tr, te) in enumerate(SPLITS[scheme], 1):
        try:
            est = build(name).fit(X[np.ix_(tr, cols)], y[tr])
            pr, st = est.predict(X[np.ix_(te, cols)]), "ok"
        except Exception as e:
            pr, st = np.full(len(te), np.nan), f"FALHA {type(e).__name__}: {e}"
            log("  !", name, scheme, k, st)
        oof[te] = pr
        ok = st == "ok"
        rows.append(dict(modelo=name, esquema=scheme, fold=k, grupo_teste=";".join(map(str, np.unique(g[te]))),
                         n_teste=len(te), MAE=mean_absolute_error(y[te], pr) if ok else np.nan,
                         RMSE=np.sqrt(mean_squared_error(y[te], pr)) if ok else np.nan,
                         R2=r2_score(y[te], pr) if ok else np.nan, MAPE=mape(y[te], pr) if ok else np.nan, status=st))
    return rows, oof


t0 = time.time()
rows_all, OOF = [], {}
for s in SPLITS:
    for n in SPECS:
        log(f"{s:17s} {n:34s} {time.time() - t0:6.0f}s")
        r, o = run(n, s)
        rows_all += r
        OOF[(n, s)] = o
folds = pd.DataFrame(rows_all)
folds.to_csv(f"{OUT}/resultados_por_fold.csv", index=False)
agg = folds.groupby(["modelo", "esquema"], sort=False).agg(
    n_folds=("R2", "size"), R2_medio=("R2", "mean"), R2_dp=("R2", lambda v: v.std(ddof=1)),
    MAE_medio=("MAE", "mean"), MAE_dp=("MAE", lambda v: v.std(ddof=1)), RMSE_medio=("RMSE", "mean"),
    MAPE_medio=("MAPE", "mean")).reset_index()
agg["R2_agregado_oof"] = [r2_score(y, OOF[(m, s)]) if not np.isnan(OOF[(m, s)]).any() else np.nan
                          for m, s in zip(agg.modelo, agg.esquema)]
agg.to_csv(f"{OUT}/tabela_modelos_esquemas.csv", index=False)
agg[agg.esquema.isin(["GroupKFold_curva", "LOPO"]) & agg.modelo.isin(MAIN)].to_csv(f"{OUT}/tabela3_principal.csv", index=False)
agg[agg.esquema.eq("LO-Pressao") & agg.modelo.isin(MAIN)].to_csv(f"{OUT}/tabela_3_1_leave_one_pressure_out.csv", index=False)
agg[agg.esquema.eq("LO-UR") & agg.modelo.isin(MAIN)].to_csv(f"{OUT}/tabela_3_1_leave_one_rh_out.csv", index=False)
agg[agg.modelo.isin(["Semiempirico global (j)", "Regressao (ln j, j)"]) & agg.esquema.isin(["GroupKFold_curva", "LOPO"])
    ].to_csv(f"{OUT}/tabela_4_3_1_semiempirico.csv", index=False)
agg[agg.modelo.eq("Semiempirico estendido (j,P,UR)")].to_csv(f"{OUT}/tabela_4_3_1_estendido.csv", index=False)
pd.concat([pd.DataFrame({"idx": np.arange(len(y)), "modelo": n, "esquema": s, "y": y, "pred": p})
           for (n, s), p in OOF.items()]).to_csv(f"{OUT}/predicoes_oof.csv", index=False)

R = lambda m, s: float(agg[(agg.modelo == m) & (agg.esquema == s)].R2_medio.iloc[0])
pd.DataFrame([{"Config": "com protocolo", "GroupKFold_R2": R("MLP", "GroupKFold_curva"), "LOPO_R2": R("MLP", "LOPO")},
              {"Config": "sem protocolo", "GroupKFold_R2": R("MLP (sem protocolo)", "GroupKFold_curva"),
               "LOPO_R2": R("MLP (sem protocolo)", "LOPO")}]).to_csv(f"{OUT}/mlp_ablacao_protocolo.csv", index=False)


def mlp_r2(scheme, hls, alpha, seed):
    out = []
    for tr, te in SPLITS[scheme]:
        e = Pipeline([("s", MinMaxScaler()), ("m", mlp(hls, alpha, seed))]).fit(X[np.ix_(tr, NUM)], y[tr])
        out.append(r2_score(y[te], e.predict(X[np.ix_(te, NUM)])))
    return np.mean(out)


hp = []
for hls, a in [((64,), 1e-3), ((64, 32), 1e-3), ((32,), 1e-3), ((64, 32), 1e-2), ((128,), 1e-3)]:
    gk = [mlp_r2("GroupKFold_curva", hls, a, s) for s in MLP_SEEDS]
    lo = [mlp_r2("LOPO", hls, a, s) for s in MLP_SEEDS]
    hp.append(dict(hidden=str(hls), alpha=a, GK_R2_medio=np.mean(gk), GK_R2_dp=np.std(gk, ddof=1),
                   LOPO_R2_medio=np.mean(lo), LOPO_R2_dp=np.std(lo, ddof=1)))
    log(hp[-1])
pd.DataFrame(hp).to_csv(f"{OUT}/mlp_busca_hiperparametros.csv", index=False)

rng = np.random.RandomState(SEED)
lab = data.protocol.to_numpy()
drops = {k: [] for k in ["Densidade de corrente", "Pressao", "Umidade relativa", "Protocolo (bloco)"]}
for tr, te in SPLITS["GroupKFold_curva"]:
    est = build("Random Forest").fit(X[tr], y[tr])
    Xt = X[te]
    base = r2_score(y[te], est.predict(Xt))
    for nm, c in zip(list(drops)[:3], [0, 1, 2]):
        d = []
        for _ in range(N_PERM):
            Z = Xt.copy()
            Z[:, c] = Z[rng.permutation(len(Z)), c]
            d.append(base - r2_score(y[te], est.predict(Z)))
        drops[nm].append(np.mean(d))
    d = []
    for _ in range(N_PERM):
        Z = Xt.copy()
        Z[:, 3:] = ohe.transform(pd.DataFrame({"protocol": lab[te][rng.permutation(len(te))]}))
        d.append(base - r2_score(y[te], est.predict(Z)))
    drops["Protocolo (bloco)"].append(np.mean(d))
perm = pd.DataFrame([dict(variavel=k, queda_R2_media=np.mean(v), dp_entre_folds=np.std(v, ddof=1)) for k, v in drops.items()])
perm["razao_vs_corrente"] = perm.queda_R2_media.iloc[0] / perm.queda_R2_media
perm.to_csv(f"{OUT}/tabela_5_1_permutacao.csv", index=False)

pc = []
for cid, g in data.groupby("curve_id"):
    j, v = g.j.to_numpy(float), g.V.to_numpy(float)
    rec = dict(curve_id=cid, protocol=g.protocol.iloc[0], P=g.P.iloc[0], RH=g.RH.iloc[0])
    try:
        po, _ = curve_fit(Semi.f, j, v, p0=[1, .05, 1e-4, 1e-6, 1e-2], bounds=([.5, 0, 0, 0, 0], [1.5, 1, 1, 1, .05]), maxfev=40000)
        rec.update(dict(zip(["Erev", "b", "R", "m", "n"], po)), R2=r2_score(v, Semi.f(j, *po)),
                   ohm_degenerado=bool(po[2] < DEGEN_R), conc_degenerado=bool(po[3] < DEGEN_M), status="ok")
    except Exception as e:
        rec["status"] = f"FALHA {type(e).__name__}"
    pc.append(rec)
pc = pd.DataFrame(pc)
pc.to_csv(f"{OUT}/ajustes_individuais_por_curva.csv", index=False)
ok = pc[pc.status == "ok"]
curva = dict(n_curvas=len(pc), n_falhas=int((pc.status != "ok").sum()), R2_mediano=float(ok.R2.median()),
             n_R2_ge_099=int((ok.R2 >= .99).sum()), ohm_degenerado=int(ok.ohm_degenerado.sum()),
             conc_degenerado=int(ok.conc_degenerado.sum()), limiar_R=DEGEN_R, limiar_m=DEGEN_M,
             R2_medio_por_UR={str(k): float(v) for k, v in ok.groupby("RH").R2.mean().items()},
             R2_medio_por_P={str(k): float(v) for k, v in ok.groupby("P").R2.mean().items()})

diag = {}
for nm in ["Random Forest", "Gradient Boosting", "GPR (ARD)"]:
    res = y - OOF[(nm, "GroupKFold_curva")]
    sw = stats.shapiro(res[np.random.RandomState(SEED).choice(len(res), 500, replace=False)])
    r, p = stats.pearsonr(res, data.j)
    diag[nm] = dict(residuo_medio=float(res.mean()), residuo_dp=float(res.std()), shapiro_p=float(sw.pvalue),
                    pearson_res_j=float(r), pearson_p=float(p))
rf = build("Random Forest").fit(X, y)
diag["RF treino completo"] = dict(R2=float(r2_score(y, rf.predict(X))),
                                  pearson_res_j=float(stats.pearsonr(y - rf.predict(X), data.j)[0]))
pd.DataFrame([dict(modelo=k, **v) for k, v in diag.items()]).to_csv(f"{OUT}/correlacao_residuo.csv", index=False)

fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
for nm in ["Random Forest", "Gradient Boosting", "GPR (ARD)"]:
    o = OOF[(nm, "GroupKFold_curva")]
    ax[0].scatter(y, o, s=5, alpha=.3, label=f"{nm} (R2={r2_score(y, o):.3f})")
ax[0].plot([0, 1.05], [0, 1.05], "k--", lw=1)
ax[0].set(xlabel="Tensao real (V)", ylabel="Tensao prevista, fold de teste (V)", title="Previsto x real")
ax[0].legend(markerscale=3, fontsize=8)
res = y - OOF[("Gradient Boosting", "GroupKFold_curva")]
ax[1].hist(res, bins=40, color="steelblue", edgecolor="white")
ax[1].set(xlabel="Residuo fora da amostra (V)", title="Residuos (Gradient Boosting)")
ax[2].scatter(data.j, res, s=5, alpha=.3)
ax[2].axhline(0, color="r", lw=1)
ax[2].set(xlabel="Densidade de corrente (mA/cm2)", ylabel="Residuo (V)", title="Residuo x j")
plt.tight_layout()
plt.savefig(f"{OUT}/fig_previsto_residuos.png", dpi=150)
plt.close()

fig, axs = plt.subplots(1, 3, figsize=(15, 4.4), sharey=True)
for a, rh in zip(axs, [30, 50, 100]):
    for Pp, col in zip([5, 15, 25], ["tab:blue", "tab:orange", "tab:green"]):
        m = (data.protocol == P0 + "Standard Protocol") & (data.file == 4) & (data.RH == rh) & (data.P == Pp)
        idx = np.where(m)[0]
        idx = idx[np.argsort(data.j.to_numpy()[idx])]
        a.plot(data.j.iloc[idx], y[idx], "o", color=col, ms=4, label=f"{Pp} psig")
        a.plot(data.j.iloc[idx], OOF[("Gradient Boosting", "GroupKFold_curva")][idx], "-", color=col)
    a.set(title=f"UR = {rh}%", xlabel="j (mA/cm2)")
axs[0].set_ylabel("Tensao (V)")
axs[0].legend()
plt.tight_layout()
plt.savefig(f"{OUT}/fig_curvas_polarizacao.png", dpi=150)
plt.close()

reg = pd.DataFrame([dict(modelo=m, esquema=s, referencia=v, obtido=R(m, s), dif=R(m, s) - v,
                         status="OK" if abs(R(m, s) - v) <= .01 else "DIVERGE") for (m, s), v in REF.items()])
reg.to_csv(f"{OUT}/checagem_regressao.csv", index=False)
cfg = dict(seed=SEED, n_perm=N_PERM, mlp_sementes=MLP_SEEDS, limiar_degenerado=dict(R=DEGEN_R, m=DEGEN_M),
           modelos={k: dict(colunas="3 numericas" if v[1] == NUM else "3 numericas + 8 protocolo",
                            minmax=v[2], params=v[3]) for k, v in SPECS.items()},
           versoes=dict(python=sys.version.split()[0], numpy=np.__version__, pandas=pd.__version__,
                        sklearn=sklearn.__version__, scipy=scipy.__version__),
           resumo_por_curva=curva, diagnostico_residuos=diag)
json.dump(cfg, open(f"{OUT}/config.json", "w"), indent=2, ensure_ascii=False, default=float)

pd.set_option("display.width", 250, "display.max_columns", 30)
log(agg.round(4).to_string(index=False))
log(perm.round(4).to_string(index=False))
log(pd.DataFrame(hp).round(4).to_string(index=False))
log(json.dumps(curva, indent=1))
log(json.dumps(diag, indent=1))
log(reg.round(4).to_string(index=False))
log(f"folds com falha: {int((folds.status != 'ok').sum())} | divergencias: {int((reg.status == 'DIVERGE').sum())} | {time.time() - t0:.0f}s")

zp = f"{OUT}.zip"
with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
    for f in sorted(os.listdir(OUT)):
        z.write(os.path.join(OUT, f), arcname=f)
try:
    from google.colab import files
    files.download(zp)
except Exception:
    log("Baixe:", zp)
