"""Pruebas estadísticas de la Tarea 1.

* Asociación: V de Cramér, prueba MCAR de Little.
* Comparación de clasificadores: DeLong rápido (Sun y Xu, 2014), DeLong directo O(mn) para
  validar, McNemar y bootstrap pareado con métricas ponderadas O(n) por réplica.

La implementación rápida de DeLong sigue el algoritmo de Sun y Xu (2014), en la versión de
referencia de Yandex Data School (https://github.com/yandexdataschool/roc_comparison, licencia MIT),
adaptada y comentada.
"""
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests

Z975 = stats.norm.ppf(0.975)


# ----------------------------------------------------------------------------------------------
# Asociación entre variables
# ----------------------------------------------------------------------------------------------
def cramer_v(x: pd.Series, y: pd.Series) -> tuple:
    """V de Cramér con la corrección de sesgo de Bergsma (2013). Devuelve (V, chi2, p, gl)."""
    tabla = pd.crosstab(x, y)
    chi2, p, gl, _ = stats.chi2_contingency(tabla, correction=False)
    n = tabla.to_numpy().sum()
    r, k = tabla.shape
    phi2 = chi2 / n
    phi2c = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
    rc, kc = r - (r - 1) ** 2 / (n - 1), k - (k - 1) ** 2 / (n - 1)
    v = np.sqrt(phi2c / max(min(kc - 1, rc - 1), 1e-12))
    return v, chi2, p, gl


def little_mcar(datos: pd.DataFrame, max_iter=200, tol=1e-6) -> dict:
    """Prueba MCAR de Little (1988) para variables numéricas con faltantes.

    Estima media y covarianza por máxima verosimilitud con el algoritmo EM bajo normalidad y
    compara la media observada de cada patrón de faltantes con la media global estimada:
    d2 = sum_j n_j (ybar_j - mu_j)' S_j^{-1} (ybar_j - mu_j) ~ chi2(sum_j p_j - p).
    """
    X = datos.to_numpy(dtype=float)
    n, p = X.shape
    M = np.isnan(X)
    patrones, inv = np.unique(M, axis=0, return_inverse=True)
    inv = inv.ravel()
    mu = np.nanmean(X, axis=0)
    S = np.diag(np.nanvar(X, axis=0))
    for _ in range(max_iter):
        suma_x = np.zeros(p)
        suma_xx = np.zeros((p, p))
        for j, pat in enumerate(patrones):
            filas = X[inv == j]
            o, m = ~pat, pat
            Xo = filas[:, o]
            if m.any():
                Soo_inv = np.linalg.pinv(S[np.ix_(o, o)])
                B = S[np.ix_(m, o)] @ Soo_inv
                Xm = mu[m] + (Xo - mu[o]) @ B.T
                completo = np.empty((len(filas), p))
                completo[:, o], completo[:, m] = Xo, Xm
                C = np.zeros((p, p))
                C[np.ix_(m, m)] = S[np.ix_(m, m)] - B @ S[np.ix_(o, m)]
                suma_xx += completo.T @ completo + len(filas) * C
            else:
                completo = Xo
                suma_xx += completo.T @ completo
            suma_x += completo.sum(axis=0)
        mu_nuevo = suma_x / n
        S_nuevo = suma_xx / n - np.outer(mu_nuevo, mu_nuevo)
        cambio = max(np.abs(mu_nuevo - mu).max(), np.abs(S_nuevo - S).max())
        mu, S = mu_nuevo, S_nuevo
        if cambio < tol * (1 + np.abs(S).max()):
            break
    d2, gl = 0.0, 0
    for j, pat in enumerate(patrones):
        o = ~pat
        if not o.any():
            continue
        filas = X[inv == j][:, o]
        dif = filas.mean(axis=0) - mu[o]
        d2 += len(filas) * dif @ np.linalg.pinv(S[np.ix_(o, o)]) @ dif
        gl += o.sum()
    gl -= p
    return {"d2": d2, "gl": int(gl), "p_valor": stats.chi2.sf(d2, gl), "patrones": len(patrones)}


# ----------------------------------------------------------------------------------------------
# DeLong
# ----------------------------------------------------------------------------------------------
def _rangos_medios(x: np.ndarray) -> np.ndarray:
    """Rangos medios (1..n) con empates promediados, en O(n log n)."""
    return stats.rankdata(x, method="average")


def delong_rapido(puntajes: np.ndarray, m: int):
    """Algoritmo de Sun y Xu (2014).

    puntajes: matriz k x (m + n) con los positivos en las primeras m columnas.
    Devuelve (aucs, matriz de covarianza k x k de los AUC).
    """
    n = puntajes.shape[1] - m
    k = puntajes.shape[0]
    pos, neg = puntajes[:, :m], puntajes[:, m:]
    tx = np.empty((k, m))
    ty = np.empty((k, n))
    tz = np.empty((k, m + n))
    for r in range(k):
        tx[r] = _rangos_medios(pos[r])
        ty[r] = _rangos_medios(neg[r])
        tz[r] = _rangos_medios(puntajes[r])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n          # componentes estructurales de los positivos (V_i)
    v10 = 1.0 - (tz[:, m:] - ty) / m    # componentes estructurales de los negativos (W_j)
    sx = np.atleast_2d(np.cov(v01))
    sy = np.atleast_2d(np.cov(v10))
    return aucs, sx / m + sy / n


def delong_directo(y: np.ndarray, s1: np.ndarray, s2: np.ndarray):
    """Versión directa O(m n) de DeLong et al. (1988); solo para validar con pocos datos."""
    y = np.asarray(y).astype(bool)
    aucs, V, W = [], [], []
    for s in (s1, s2):
        P, N = s[y], s[~y]
        phi = (P[:, None] > N[None, :]).astype(float) + 0.5 * (P[:, None] == N[None, :])
        aucs.append(phi.mean())
        V.append(phi.mean(axis=1))
        W.append(phi.mean(axis=0))
    m, n = y.sum(), (~y).sum()
    cov = np.cov(np.vstack(V)) / m + np.cov(np.vstack(W)) / n
    return np.array(aucs), cov


def _ordenar(y, *scores):
    y = np.asarray(y).astype(int)
    orden = np.r_[np.flatnonzero(y == 1), np.flatnonzero(y == 0)]
    return int(y.sum()), np.vstack([np.asarray(s, dtype=float)[orden] for s in scores])


def delong_auc(y, s) -> dict:
    """AUC con su IC del 95 % según la varianza de DeLong."""
    m, P = _ordenar(y, s)
    auc, cov = delong_rapido(P, m)
    se = np.sqrt(cov[0, 0])
    return {"auc": auc[0], "se": se, "ic_inf": auc[0] - Z975 * se, "ic_sup": auc[0] + Z975 * se}


def delong_prueba(y, s1, s2) -> dict:
    """Prueba de DeLong para H0: AUC1 = AUC2 sobre las mismas observaciones."""
    m, P = _ordenar(y, s1, s2)
    auc, cov = delong_rapido(P, m)
    delta = auc[0] - auc[1]
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    se = np.sqrt(max(var, 1e-300))
    z = delta / se
    se1, se2 = np.sqrt(cov[0, 0]), np.sqrt(cov[1, 1])
    return {"auc1": auc[0], "auc1_inf": auc[0] - Z975 * se1, "auc1_sup": auc[0] + Z975 * se1,
            "auc2": auc[1], "auc2_inf": auc[1] - Z975 * se2, "auc2_sup": auc[1] + Z975 * se2,
            "delta": delta, "delta_inf": delta - Z975 * se, "delta_sup": delta + Z975 * se,
            "z": z, "p": 2 * stats.norm.sf(abs(z)),
            "correlacion": cov[0, 1] / (se1 * se2)}


def holm(p) -> np.ndarray:
    return multipletests(np.asarray(p), alpha=0.05, method="holm")[1]


# ----------------------------------------------------------------------------------------------
# McNemar
# ----------------------------------------------------------------------------------------------
def mcnemar_prueba(y, pred1, pred2) -> dict:
    """McNemar con corrección de continuidad; exacta (binomial) si b + c < 25."""
    y = np.asarray(y)
    ok1, ok2 = np.asarray(pred1) == y, np.asarray(pred2) == y
    a = int(np.sum(ok1 & ok2))
    b = int(np.sum(ok1 & ~ok2))   # acierta el modelo 1, falla el 2
    c = int(np.sum(~ok1 & ok2))   # falla el 1, acierta el 2
    d = int(np.sum(~ok1 & ~ok2))
    exacta = (b + c) < 25
    r = mcnemar([[a, b], [c, d]], exact=exacta, correction=True)
    return {"a": a, "b": b, "c": c, "d": d, "chi2": np.nan if exacta else r.statistic,
            "p": r.pvalue, "exacta": exacta, "error1": 1 - ok1.mean(), "error2": 1 - ok2.mean()}


# ----------------------------------------------------------------------------------------------
# Bootstrap pareado
# ----------------------------------------------------------------------------------------------
class _Puntajes:
    """Precalcula el orden de un vector de puntajes para evaluar métricas ponderadas en O(n).

    Remuestrear con reemplazo equivale a ponderar cada observación por el número de veces que sale
    en la réplica. Con los valores únicos ordenados una sola vez, cada réplica solo necesita
    `bincount` y sumas acumuladas.
    """

    def __init__(self, s, y, umbral):
        self.unicos, self.inv = np.unique(np.asarray(s, dtype=float), return_inverse=True)
        self.inv = self.inv.ravel()
        self.y = np.asarray(y).astype(float)
        self.pred = (np.asarray(s) >= umbral).astype(float)

    def metricas(self, w):
        k = len(self.unicos)
        wp = np.bincount(self.inv, weights=w * self.y, minlength=k)
        wn = np.bincount(self.inv, weights=w * (1 - self.y), minlength=k)
        P, N = wp.sum(), wn.sum()
        # AUC: P(s+ > s-) + 0,5 P(s+ = s-)
        neg_debajo = np.cumsum(wn) - wn
        auc = (wp * (neg_debajo + 0.5 * wn)).sum() / (P * N)
        # AUC-PR como precisión promedio (misma definición que average_precision_score)
        tp = np.cumsum(wp[::-1])
        fp = np.cumsum(wn[::-1])
        prec = tp / np.maximum(tp + fp, 1e-300)
        ap = (np.diff(np.r_[0.0, tp]) / P * prec).sum()
        # F1 en el umbral fijo
        tpu = (w * self.y * self.pred).sum()
        fpu = (w * (1 - self.y) * self.pred).sum()
        fnu = (w * self.y * (1 - self.pred)).sum()
        f1 = 2 * tpu / max(2 * tpu + fpu + fnu, 1e-300)
        return np.array([auc, ap, f1])


def bootstrap_pareado(y, s1, s2, umbral1, umbral2, B=2000, semilla=42) -> pd.DataFrame:
    """Diferencias (modelo 1 - modelo 2) de AUC, AUC-PR y F1 en B réplicas con los mismos índices."""
    y = np.asarray(y)
    n = len(y)
    m1, m2 = _Puntajes(s1, y, umbral1), _Puntajes(s2, y, umbral2)
    obs = m1.metricas(np.ones(n)) - m2.metricas(np.ones(n))
    rng = np.random.default_rng(semilla)
    difs = np.empty((B, 3))
    for b in range(B):
        w = np.bincount(rng.integers(0, n, n), minlength=n).astype(float)
        difs[b] = m1.metricas(w) - m2.metricas(w)
    lo, hi = np.percentile(difs, [2.5, 97.5], axis=0)
    # p bootstrap bilateral: proporción de réplicas al otro lado del cero, duplicada
    p = np.minimum(1, 2 * np.minimum((difs <= 0).mean(axis=0), (difs >= 0).mean(axis=0)))
    return pd.DataFrame({"observada": obs, "media": difs.mean(axis=0), "ic_inf": lo, "ic_sup": hi,
                         "p_boot": np.maximum(p, 1 / B), "excluye_0": (lo > 0) | (hi < 0)},
                        index=["ΔAUC", "ΔAUC-PR", "ΔF1"])


def metricas_unicas(y, s, umbral) -> np.ndarray:
    """AUC, AUC-PR y F1 con pesos unitarios (para validar contra scikit-learn)."""
    return _Puntajes(s, y, umbral).metricas(np.ones(len(y)))
