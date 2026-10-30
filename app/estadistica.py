"""Funciones estadísticas compartidas por la aplicación (mismos cálculos del notebook, redondeo a 4 decimales)."""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mutual_info_score, normalized_mutual_info_score


def r4(v):
    return round(float(v), 4)


def medidas(s: pd.Series) -> pd.DataFrame:
    """Tendencia central, dispersión, posición y forma de una variable cuantitativa."""
    s = s.dropna()
    q = s.quantile([.10, .25, .50, .75, .90])
    g1 = stats.skew(s, bias=False)
    ek = stats.kurtosis(s, fisher=True, bias=False)
    filas = {
        'n': len(s), 'Media': s.mean(), 'Mediana': s.median(), 'Moda': s.mode().iloc[0],
        'Mínimo': s.min(), 'Máximo': s.max(), 'Rango': s.max() - s.min(),
        'Varianza': s.var(ddof=1), 'Desviación estándar': s.std(ddof=1), 'Coef. de variación (%)': s.std(ddof=1) / s.mean() * 100,
        'P10': q[.10], 'Q1': q[.25], 'Q2 (mediana)': q[.50], 'Q3': q[.75], 'P90': q[.90], 'IQR': q[.75] - q[.25],
        'Asimetría (g1)': g1, 'Curtosis': ek + 3, 'Exceso de curtosis': ek,
    }
    return pd.DataFrame({'Valor': {k: r4(v) for k, v in filas.items()}})


def forma_texto(s: pd.Series) -> str:
    s = s.dropna()
    g1 = stats.skew(s, bias=False); ek = stats.kurtosis(s, fisher=True, bias=False)
    asim = 'aproximadamente simétrica' if abs(g1) < .5 else ('asimétrica positiva (cola a la derecha)' if g1 > 0 else 'asimétrica negativa (cola a la izquierda)')
    curt = 'mesocúrtica' if abs(ek) < .5 else ('leptocúrtica (más apuntada que la normal)' if ek > 0 else 'platicúrtica (más aplanada que la normal)')
    return f'Distribución {asim} y {curt}.'


def atipicos_iqr(s: pd.Series) -> dict:
    s = s.dropna(); q1, q3 = s.quantile([.25, .75]); iqr = q3 - q1
    li, ls = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    out = (s < li) | (s > ls); s2 = s[~out]
    return {'Límite inferior': r4(li), 'Límite superior': r4(ls), 'N° atípicos': int(out.sum()), '% atípicos': r4(out.mean() * 100),
            'Media con atípicos': r4(s.mean()), 'Media sin atípicos': r4(s2.mean()),
            'Desv. est. con atípicos': r4(s.std()), 'Desv. est. sin atípicos': r4(s2.std())}


def correlaciones(x: pd.Series, y: pd.Series) -> pd.DataFrame:
    t = pd.concat([x, y], axis=1).dropna()
    a, b = t.iloc[:, 0], t.iloc[:, 1]
    pr = stats.pearsonr(a, b); sr = stats.spearmanr(a, b); kt = stats.kendalltau(a, b)
    return pd.DataFrame({
        'Coeficiente': [r4(np.cov(a, b, ddof=1)[0, 1]), r4(pr[0]), r4(sr[0]), r4(kt[0])],
        'p-valor': [None, r4(pr[1]), r4(sr[1]), r4(kt[1])],
    }, index=['Covarianza', 'Pearson r', 'Spearman ρ', 'Kendall τ'])


def independencia(a: pd.Series, b: pd.Series):
    obs = pd.crosstab(a, b)
    chi2, p, gl, esp = stats.chi2_contingency(obs)
    esp = pd.DataFrame(esp, index=obs.index, columns=obs.columns)
    n = obs.values.sum(); r, k = obs.shape
    mi = mutual_info_score(a.astype(str), b.astype(str))
    g = 2 * n * mi
    res = {'χ²': r4(chi2), 'Grados de libertad': int(gl), 'p-valor χ²': r4(p),
           'V de Cramér': r4(np.sqrt(chi2 / (n * (min(r, k) - 1)))),
           'Información mutua (nats)': r4(mi), 'Información mutua (bits)': r4(mi / np.log(2)),
           'IM normalizada': r4(normalized_mutual_info_score(a.astype(str), b.astype(str))),
           'G = 2·N·IM': r4(g), 'p-valor G': r4(stats.chi2.sf(g, gl)),
           '% celdas con esperado < 5': r4((esp.values < 5).mean() * 100)}
    return obs, esp.round(4), res


def comparar_grupos(df: pd.DataFrame, grupo: str, valor: str, orden=None):
    t = df[[grupo, valor]].dropna()
    niveles = [g for g in (orden or sorted(t[grupo].unique())) if g in set(t[grupo])]
    grupos = {g: t.loc[t[grupo] == g, valor].values for g in niveles}
    grupos = {g: v for g, v in grupos.items() if len(v) >= 3}
    filas = []
    for g, v in grupos.items():
        w, p = stats.shapiro(v) if len(v) <= 5000 else (np.nan, np.nan)
        filas.append({'Grupo': g, 'n': len(v), 'Media': r4(v.mean()), 'Mediana': r4(np.median(v)), 'Desv. est.': r4(v.std(ddof=1)),
                      'Shapiro-Wilk W': r4(w), 'p-valor': r4(p), '¿Normal?': 'Sí' if p > .05 else 'No'})
    vals = list(grupos.values())
    lev = stats.levene(*vals, center='median'); an = stats.f_oneway(*vals); kw = stats.kruskal(*vals)
    allv = np.concatenate(vals); N, k = len(allv), len(vals)
    eta2 = sum(len(v) * (v.mean() - allv.mean()) ** 2 for v in vals) / ((allv - allv.mean()) ** 2).sum()
    pruebas = {'Levene (mediana) p': r4(lev.pvalue), 'ANOVA F': r4(an.statistic), 'ANOVA p': r4(an.pvalue), 'η²': r4(eta2),
               'Kruskal-Wallis H': r4(kw.statistic), 'Kruskal-Wallis p': r4(kw.pvalue), 'ε²': r4((kw.statistic - k + 1) / (N - k))}
    return pd.DataFrame(filas).set_index('Grupo'), pruebas
