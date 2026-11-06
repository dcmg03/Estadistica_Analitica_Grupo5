"""Pestañas de modelación del Laboratorio 3: regresión lineal y regresión logística.

Los modelos los entrena y guarda el notebook 03_Lab3_Modelacion.ipynb en models/; la base de modelación (con la columna
`conjunto` = entrenamiento / prueba) está en data/processed/base_lab3.csv.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sklearn
import streamlit as st
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, mean_absolute_error, mean_squared_error, precision_score,
                             r2_score, recall_score, roc_auc_score, roc_curve)

RAIZ = Path(__file__).resolve().parent.parent
MODELOS, DATA = RAIZ / 'models', RAIZ / 'data' / 'processed'
AZUL, GRIS, NARANJA = '#2a78d6', '#a3a29b', '#eb6834'

ETIQUETAS = {
    'ipm': 'IPM municipal (%)', 'n_servicios_caidos': 'N° de servicios caídos o con afectaciones (0-7)',
    'n_infra_afectada': 'N° de elementos de infraestructura afectados', 'n_sistemas_afectados': 'N° de sistemas afectados',
    'indice_vulnerabilidad': 'Índice de vulnerabilidad (0-5)', 'tipo_entidad_g': 'Tipo de entidad', 'departamento': 'Departamento',
    'estado_conectividad': 'Estado de la conectividad', 'energia_g': 'Suministro de energía', 'personal_afectado': '¿Personal TIC afectado?',
    'estado_sistemas': 'Estado de los sistemas de información', 'riesgo_continuidad': '¿Riesgo para la continuidad de la operación?',
    'sede_contingencia': '¿Requiere sede de contingencia?', 'zonas_incomunicadas': '¿Hay zonas incomunicadas?',
    'estado_infra': 'Estado de la infraestructura física'}


@st.cache_resource
def cargar_modelos():
    meta = json.loads((MODELOS / 'metadatos.json').read_text(encoding='utf-8'))
    reg = joblib.load(MODELOS / 'regresion_lineal.joblib')
    clf = joblib.load(MODELOS / 'regresion_logistica.joblib')
    return reg, clf, meta


@st.cache_resource
def cargar_sin_risaralda():
    ruta = MODELOS / 'regresion_logistica_sin_risaralda.joblib'
    return joblib.load(ruta) if ruta.exists() else None


VERSIONES = ['Entrenado con toda la base (302 entidades)', 'Reentrenado sin Risaralda (183 entidades)']


def elegir_clasificador(clf, meta, clave):
    """Selector del modelo de clasificación. Devuelve (modelo, sin_risaralda, hiperparámetros, n de entrenamiento)."""
    alt = cargar_sin_risaralda()
    if alt is None:
        return clf, False, meta['clasificacion']['hiperparametros'], 302
    v = st.radio('Modelo de clasificación', VERSIONES, horizontal=True, key=clave,
                 help='El reentrenado aprendió solo de Caldas, Chocó, Quindío y Valle. Con él, la evaluación y las entidades excluyen '
                      'a Risaralda automáticamente.')
    if v == VERSIONES[1]:
        st.caption('Con este modelo se evalúa solo fuera de Risaralda (54 entidades de prueba). Con umbral 0,5 detecta 10 de 12 entidades en '
                   'afectación alta, lo mismo que el modelo original con umbral 0,3.')
        return alt, True, meta['clasificacion'].get('hiperparametros_sin_risaralda', {}), 183
    return clf, False, meta['clasificacion']['hiperparametros'], 302


@st.cache_data
def cargar_base():
    b = pd.read_csv(DATA / 'base_lab3.csv', dtype={'cod_mpio': str})
    b['mpio'] = (b['municipio'].str.title().str.replace(' De ', ' de ').str.replace(' Del ', ' del ')
                 + ' (' + b['departamento'].str.title().str.replace(' Del ', ' del ') + ')')
    return b


def disponible():
    """Devuelve (reg, clf, meta, base) o None si faltan archivos."""
    try:
        reg, clf, meta = cargar_modelos()
        base = cargar_base()
    except FileNotFoundError:
        st.error('No se encontraron los modelos. Ejecute el notebook `notebooks/03_Lab3_Modelacion.ipynb` (Run All): guarda '
                 '`models/*.joblib`, `models/metadatos.json` y `data/processed/base_lab3.csv`.')
        return None
    if meta.get('sklearn') != sklearn.__version__:
        st.caption(f'Nota: los modelos se entrenaron con scikit-learn {meta.get("sklearn")} y la app usa {sklearn.__version__}. '
                   'Si ve resultados extraños, vuelva a ejecutar el notebook 03.')
    return reg, clf, meta, base


def selector_municipio(base, clave):
    """Elige el municipio y devuelve departamento e IPM (dato del DANE)."""
    mun = base.drop_duplicates('cod_mpio').sort_values('mpio')
    m = st.selectbox('Municipio de la entidad', mun['mpio'], key=clave,
                     index=int(np.flatnonzero(mun['mpio'].str.startswith('Quibdó'))[0]) if mun['mpio'].str.startswith('Quibdó').any() else 0)
    fila = mun[mun['mpio'] == m].iloc[0]
    st.caption(f'IPM del municipio (DANE, 2018): **{fila["ipm"]:.1f} %** · Departamento: {fila["departamento"].title()}')
    return fila['departamento'], float(fila['ipm'])


def opciones(base, c):
    return sorted(base[c].dropna().unique())


def tabla4(d):
    return d.style.format(precision=4, na_rep='—')


# =============================== REGRESIÓN LINEAL ===============================
def _plantilla(nombre, datos):
    html = (Path(__file__).parent / nombre).read_text(encoding='utf-8')
    return html.replace('__DATOS__', json.dumps(datos, ensure_ascii=False))


def _municipios(base):
    m = base.drop_duplicates('cod_mpio').sort_values('mpio')
    return [{'n': r.mpio, 'ipm': float(r.ipm), 'dep': r.departamento} for r in m.itertuples() if pd.notna(r.ipm)]


def pestana_regresion(base, reg, meta, filtro):
    X = meta['regresion']['numericas'] + meta['regresion']['cualitativas']; y = meta['regresion']['respuesta']
    pts = filtro(base).dropna(subset=X + [y])
    rng = np.random.default_rng(7)
    munis = _municipios(base)
    ini = next((m for m in munis if m['n'].startswith('Quibdó')), munis[0])
    datos = {'coef': params_lineal(reg), 'munis': munis, 'tipos': sorted(base['tipo_entidad_g'].dropna().unique()),
             'init': {'mun': ini['n'], 'dep': ini['dep'], 'ipm': ini['ipm'], 'tipo': 'Alcaldía'},
             'puntos': [{'ipm': float(r.ipm), 'y': int(r.indice_vulnerabilidad), 'j': float(rng.uniform(-.18, .18)), 'dep': r.departamento,
                         'tipo': r.tipo_entidad_g, 'nom': str(r.razon_social).strip()[:60], 'mun': r.mpio} for r in pts.itertuples()]}
    datos['depnom'] = {d: d.title().replace(' Del ', ' del ') for d in base['departamento'].dropna().unique()}
    _iframe(_plantilla('lineal.html', datos), 610)

    st.subheader('¿Qué tan cerca queda el modelo? (entidades de prueba)')
    te = filtro(base[base['conjunto'] == 'prueba']).dropna(subset=X + [y])
    if len(te) < 5:
        st.warning('Hay muy pocas entidades de prueba con los filtros actuales.'); return
    pr = reg.predict(te[X])
    datos = {'test': [{'nom': str(r.razon_social).strip()[:60], 'real': int(r.indice_vulnerabilidad), 'pred': float(p)}
                      for r, p in zip(te.itertuples(), pr)]}
    _iframe(_plantilla('errores.html', datos), 425)


# =============================== REGRESIÓN LOGÍSTICA ===============================
def pestana_clasificacion(base, clf, meta, filtro):
    clf, sin_ris, hp, _ = elegir_clasificador(clf, meta, 'clf_version')
    num, cat = meta['clasificacion']['numericas'], meta['clasificacion']['cualitativas']; y = meta['clasificacion']['respuesta']
    munis = _municipios(base)
    ini_m = next((m for m in munis if m['n'].startswith('Quibdó')), munis[0])
    inicial = {c: str(base[c].mode()[0]) for c in cat}
    inicial['tipo_entidad_g'] = 'Alcaldía'
    for c in num:
        inicial[c] = int(base[c].median()) if c != 'ipm' else ini_m['ipm']
    datos = {'params': params_logistica(clf), 'munis': munis, 'mun_inicial': ini_m['n'], 'inicial': inicial, 'umbral': 0.5}
    _iframe(_plantilla('simulador.html', datos), 690)

    st.subheader('Mueva el umbral: ¿a quién detecta el modelo? (entidades de prueba)')
    te = filtro(base[base['conjunto'] == 'prueba']).dropna(subset=num + cat)
    if sin_ris:
        te = te[te['departamento'] != 'RISARALDA']
    if len(te) < 5 or te[y].nunique() < 2:
        st.warning('Hay muy pocas entidades de prueba (o una sola clase) con los filtros actuales.'); return
    p = clf.predict_proba(te[num + cat])[:, 1]
    datos = {'umbral': 0.5, 'solo_fuera': bool(sin_ris), 'sin_ris_inicial': False,
             'ents': [{'i': i, 'nom': str(r.razon_social).strip()[:60], 'mun': r.mpio, 'p': float(q), 'y': int(getattr(r, y)),
                       'ris': r.departamento == 'RISARALDA', 'nivel': str(r.nivel_afectacion_cat)}
                      for i, (r, q) in enumerate(zip(te.itertuples(), p))]}
    _iframe(_plantilla('umbral.html', datos), 470)

    with st.expander('Razones de odds del modelo'):
        pre, mod = clf.named_steps['preprocesamiento'], clf.named_steps['modelo']
        co = pd.DataFrame({'Variable': [_nombre_columna(n, cat) for n in pre.get_feature_names_out()], 'Coeficiente': mod.coef_[0]})
        co['Razón de odds'] = np.exp(co['Coeficiente'])
        co = co.reindex(co['Coeficiente'].abs().sort_values(ascending=False).index).head(14).sort_values('Coeficiente')
        fig = px.bar(co, x='Razón de odds', y='Variable', orientation='h', log_x=True, height=480,
                     color=np.where(co['Coeficiente'] > 0, 'Sube el riesgo', 'Baja el riesgo'),
                     color_discrete_map={'Sube el riesgo': AZUL, 'Baja el riesgo': GRIS})
        fig.add_vline(x=1, line_color='black')
        fig.update_layout(yaxis_title=None, legend_title=None, margin=dict(t=10))
        st.plotly_chart(fig, width='stretch')


# =============================== PIPELINE (presentación) ===============================
CORTO = {'n_servicios_caidos': 'Servicios caídos', 'n_infra_afectada': 'Infraestructura afectada', 'n_sistemas_afectados': 'Sistemas afectados',
         'ipm': 'Pobreza del municipio (IPM, %)', 'indice_vulnerabilidad': 'Índice de vulnerabilidad', 'tipo_entidad_g': 'Tipo de entidad',
         'estado_conectividad': 'Conectividad', 'energia_g': 'Energía', 'personal_afectado': 'Personal TIC afectado',
         'estado_sistemas': 'Sistemas de información', 'riesgo_continuidad': 'Riesgo para la operación', 'sede_contingencia': 'Requiere sede alterna',
         'zonas_incomunicadas': 'Zonas incomunicadas', 'estado_infra': 'Infraestructura física', 'departamento': 'Departamento'}


def _nombre_columna(col, cat):
    """'cualitativas__personal_afectado_Sí' -> 'Personal TIC afectado = Sí'."""
    bloque, resto = col.split('__', 1)
    if bloque != 'cualitativas':
        return CORTO.get(resto, resto)
    for v in sorted(cat, key=len, reverse=True):
        if resto.startswith(v + '_'):
            return f'{CORTO.get(v, v)} = {resto[len(v) + 1:]}'
    return resto


def _datos_linea(clf, fila, num, cat):
    """Datos de una entidad en cada estación del pipeline, para la línea de ensamble (HTML)."""
    pre, mod = clf.named_steps['preprocesamiento'], clf.named_steps['modelo']
    ent = fila[num + cat].to_frame().T.infer_objects()
    for c in num:
        ent[c] = pd.to_numeric(ent[c], errors='coerce')
    bloques = {n: (t, cols) for n, t, cols in pre.transformers_}
    f = lambda v: None if pd.isna(v) else float(v)
    numericas, conteos, escalado, cualitativas, onehot = [], [], [], [], []
    for nombre in ('conteos', 'numericas'):
        tr, cols = bloques[nombre]
        x0 = ent[cols].astype(float)
        imp = tr.named_steps['imputar'].transform(x0)
        x = imp
        if 'log1p' in tr.named_steps:
            x = tr.named_steps['log1p'].transform(x)
        z = tr.named_steps['escalar'].transform(x)
        for j, c in enumerate(cols):
            dec = 1 if c == 'ipm' else 0
            numericas.append({'lab': CORTO[c], 'orig': f(x0.iloc[0, j]), 'faltaba': bool(pd.isna(x0.iloc[0, j])), 'imputado': f(imp[0, j]), 'dec': dec})
            if nombre == 'conteos':
                conteos.append({'lab': CORTO[c], 'antes': f(imp[0, j]), 'despues': f(x[0, j])})
            escalado.append({'lab': CORTO[c] + (' (después del log)' if nombre == 'conteos' else ''), 'entrada': f(x[0, j]), 'z': f(z[0, j]),
                             'dec': 2 if nombre == 'conteos' else dec})
    tr, cols = bloques['cualitativas']
    imp = tr.named_steps['imputar'].transform(ent[cols])
    oh = tr.named_steps['onehot']
    for j, c in enumerate(cols):
        o = ent[c].iloc[0]
        cualitativas.append({'lab': CORTO[c], 'orig': None if pd.isna(o) else str(o), 'faltaba': bool(pd.isna(o)), 'imputado': str(imp[0, j])})
        onehot.append({'lab': CORTO[c], 'opciones': [{'nombre': str(k), 'on': str(k) == str(imp[0, j]), 'ref': i == 0}
                                                     for i, k in enumerate(oh.categories_[j])]})
    zt = pre.transform(ent)[0]
    nombres = pre.get_feature_names_out()
    ap = pd.DataFrame({'lab': [_nombre_columna(n, cat) for n in nombres], 'v': zt * mod.coef_[0]})
    ap = ap[ap['v'].abs() > 1e-9]
    top = ap.reindex(ap['v'].abs().sort_values(ascending=False).index).head(8)
    logit = float(mod.intercept_[0] + ap['v'].sum())
    esc = bloques['numericas'][0].named_steps['escalar']
    ipm = ent['ipm'].iloc[0]
    n_ref = sum(1 for v in onehot if any(o['on'] and o['ref'] for o in v['opciones']))
    return {'nombre': str(fila['razon_social']).strip(), 'lugar': f"{fila['mpio']} · {str(fila['tipo_entidad']).strip()}",
            'nivel': str(fila['nivel_afectacion_cat']), 'real': int(fila['alto_critico']),
            'numericas': numericas, 'cualitativas': cualitativas, 'n_faltantes': sum(v['faltaba'] for v in numericas + cualitativas),
            'conteos': conteos, 'escalado': escalado,
            'ejemplo_ipm': None if pd.isna(ipm) else {'x': float(ipm), 'media': float(esc.mean_[0]), 'sd': float(esc.scale_[0]),
                                                      'z': float((ipm - esc.mean_[0]) / esc.scale_[0])},
            'onehot': onehot, 'n_onehot': int(sum(len(v['opciones']) - 1 for v in onehot)), 'n_ref': n_ref,
            'aportes': [{'lab': r.lab, 'v': float(r.v)} for r in top.itertuples()], 'resto': float(ap['v'].sum() - top['v'].sum()),
            'intercepto': float(mod.intercept_[0]), 'logit': logit, 'prob': float(1 / (1 + np.exp(-logit)))}


def _iframe(html, alto):
    if hasattr(st, 'iframe'):
        st.iframe(html, height=alto)
    else:
        import streamlit.components.v1 as components
        components.html(html, height=alto, scrolling=True)


def pestana_pipeline(base, reg, clf, meta, filtro):
    st.markdown('Antes de predecir, cada respuesta de la encuesta pasa por una **línea de ensamble** de 6 estaciones. Elija una entidad que el '
                'modelo **nunca vio** al entrenar y siga su recorrido: con **Reproducir** avanza sola, o haga clic en cualquier estación.')
    num, cat = meta['clasificacion']['numericas'], meta['clasificacion']['cualitativas']
    modelo_clf, sin_ris, _, n_tr = elegir_clasificador(clf, meta, 'pipe_version')
    te = filtro(base[base['conjunto'] == 'prueba'])
    if sin_ris:
        te = te[te['departamento'] != 'RISARALDA']
    if te.empty:
        st.warning('No hay entidades de prueba con los filtros actuales.'); return
    etiqueta = te['razon_social'].str.strip().str.slice(0, 60) + ' · ' + te['mpio']
    sugeridas = te.index[te['razon_social'].str.contains('Atrato' if sin_ris else 'BERNARDO ARIAS', case=False, na=False)]
    sel = st.selectbox('Entidad que recorre la línea (conjunto de prueba)', etiqueta.index, format_func=lambda i: etiqueta[i], key=f'pipe_ent_{sin_ris}',
                       index=int(list(etiqueta.index).index(sugeridas[0])) if len(sugeridas) else 0)
    datos = _datos_linea(modelo_clf, te.loc[sel], num, cat)
    datos['n_entrenamiento'] = n_tr
    plantilla = (Path(__file__).parent / 'linea_ensamble.html').read_text(encoding='utf-8')
    _iframe(plantilla.replace('__DATOS__', json.dumps(datos, ensure_ascii=False)), 690)

    with st.expander('Ver el pipeline en lenguaje técnico (diagrama de scikit-learn)'):
        cual = st.radio('Modelo', ['Regresión logística (clasificación)', 'Regresión lineal (regresión)'], horizontal=True, key='pipe_modelo')
        from sklearn.utils import estimator_html_repr
        _iframe(estimator_html_repr(clf if cual.startswith('Regresión logística') else reg), 470)
        st.markdown('''
| Estación | Paso técnico | Variables | Por qué |
|---|---|---|---|
| Rellenar casillas vacías | `SimpleImputer` (mediana / moda) | todas | El modelo no admite vacíos; los valores típicos salen solo del entrenamiento |
| Acortar las colas | `FunctionTransformer(log1p)` | 3 conteos | Muy asimétricos (g1 ≈ 1,8–2,5 en el Lab 2) |
| Misma regla para medir | `StandardScaler` | 5 numéricas | Media 0 y desviación 1: efectos comparables |
| Traducir palabras a números | `OneHotEncoder(drop='first')` | 9 cualitativas → 23 columnas | El modelo solo recibe números; se quita una categoría de referencia |
| Sumar puntos y decidir | `LogisticRegression(C=0.1, class_weight='balanced')` | 28 columnas | Probabilidad = 1 / (1 + e^(−suma)) |
''')

    # ---------- Escalamiento ----------
    st.subheader('¿Por qué StandardScaler? Compare los escaladores')
    from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
    ESC = {'StandardScaler': StandardScaler(), 'MinMaxScaler': MinMaxScaler(), 'RobustScaler': RobustScaler()}
    a, b, c = st.columns(3)
    var = a.selectbox('Variable', num, format_func=ETIQUETAS.get, key='esc_var')
    nom = b.radio('Escalador', list(ESC), horizontal=True, key='esc_nom')
    uselog = c.checkbox('Aplicar log(1 + x) antes', value=var.startswith('n_'), key='esc_log')
    tr = base[base['conjunto'] == 'entrenamiento'][var].dropna().astype(float)
    x = np.log1p(tr) if uselog else tr
    xs = ESC[nom].fit_transform(x.values.reshape(-1, 1)).ravel()
    c1, c2 = st.columns(2)
    fig = px.histogram(x=tr, nbins=25, color_discrete_sequence=[GRIS], title='Original', labels={'x': ETIQUETAS[var]})
    fig.update_layout(yaxis_title='Entidades', height=300); c1.plotly_chart(fig, width='stretch')
    fig = px.histogram(x=xs, nbins=25, color_discrete_sequence=[AZUL], title=f'Después de {"log(1 + x) y " if uselog else ""}{nom}',
                       labels={'x': 'Valor transformado'})
    fig.update_layout(yaxis_title='Entidades', height=300); c2.plotly_chart(fig, width='stretch')
    resumen = pd.DataFrame({'Original': [tr.mean(), tr.std(), tr.median(), tr.min(), tr.max()],
                            'Transformada': [xs.mean(), xs.std(ddof=1), np.median(xs), xs.min(), xs.max()]},
                           index=['Media', 'Desv. est.', 'Mediana', 'Mínimo', 'Máximo']).T
    st.dataframe(tabla4(resumen), width='stretch')
    st.caption({'StandardScaler': 'Resta la media y divide por la desviación estándar: media 0 y desviación 1. Sensible a atípicos.',
                'MinMaxScaler': 'Lleva el mínimo a 0 y el máximo a 1. Un solo atípico comprime a todas las demás entidades.',
                'RobustScaler': 'Resta la mediana y divide por el rango intercuartílico: resistente a atípicos (mediana 0).'}[nom]
               + ' La forma de la distribución no cambia con ningún escalador; solo cambia la escala. El logaritmo sí cambia la forma.')
    try:
        comp = pd.read_csv(MODELOS / 'comparacion_escaladores.csv', index_col=0)
        largo = comp.reset_index().melt(id_vars='Escalador', var_name='Métrica', value_name='Valor')
        fig = px.bar(largo, x='Escalador', y='Valor', color='Escalador', facet_col='Métrica', facet_col_spacing=.06, text_auto='.4f',
                     color_discrete_sequence=[GRIS, AZUL, '#9ec5f4', '#1c5cab'], title='Validación cruzada (5 pliegues) con cada escalador')
        fig.update_yaxes(matches=None, showticklabels=True); fig.update_xaxes(showticklabels=False, title=None)
        fig.for_each_annotation(lambda t: t.update(text=t.text.split('=')[-1]))
        st.plotly_chart(fig, width='stretch')
        st.markdown('En la **regresión lineal** el escalamiento no cambia nada (mínimos cuadrados da las mismas predicciones con cualquier escala). '
                    'En la **logística** sí importa un poco, porque la regularización L2 penaliza igual a todos los coeficientes: MinMaxScaler queda '
                    'último. Se eligió **StandardScaler**, que permite leer los coeficientes como el efecto de aumentar una desviación estándar.')
    except FileNotFoundError:
        st.caption('Ejecute el notebook 03 para ver la comparación de escaladores con validación cruzada.')


# =============================== PARÁMETROS PARA LAS VISUALIZACIONES INTERACTIVAS ===============================
def params_lineal(reg):
    """Coeficientes de la regresión lineal en unidades originales: índice = b0 + b_ipm·IPM + efecto depto + efecto tipo."""
    pre, mod = reg.named_steps['preprocesamiento'], reg.named_steps['modelo']
    esc = pre.named_transformers_['numericas'].named_steps['escalar']
    oh = pre.named_transformers_['cualitativas'].named_steps['onehot']
    cols = [c for n, _, cs in pre.transformers_ if n == 'cualitativas' for c in cs]
    coef = dict(zip(pre.get_feature_names_out(), mod.coef_))
    b_ipm = coef['numericas__ipm'] / esc.scale_[0]
    efectos = {}
    for c, cats in zip(cols, oh.categories_):
        efectos[c] = {str(k): (0.0 if i == 0 else float(coef[f'cualitativas__{c}_{k}'])) for i, k in enumerate(cats)}
    return {'b0': float(mod.intercept_ - b_ipm * esc.mean_[0]), 'ipm': float(b_ipm),
            'dep': efectos['departamento'], 'tipo': efectos['tipo_entidad_g']}


def params_logistica(clf):
    """Parámetros del pipeline logístico para reproducirlo en JavaScript (log1p, estandarización, one-hot y coeficientes)."""
    pre, mod = clf.named_steps['preprocesamiento'], clf.named_steps['modelo']
    coef = dict(zip(pre.get_feature_names_out(), mod.coef_[0]))
    num = []
    for nombre, tr, cols in pre.transformers_:
        if nombre in ('conteos', 'numericas'):
            esc = tr.named_steps['escalar']
            for j, c in enumerate(cols):
                num.append({'var': c, 'lab': CORTO[c], 'log': nombre == 'conteos', 'media': float(esc.mean_[j]), 'sd': float(esc.scale_[j]),
                            'coef': float(coef[f'{nombre}__{c}'])})
    tr, cols = [(t, cs) for n, t, cs in pre.transformers_ if n == 'cualitativas'][0]
    oh = tr.named_steps['onehot']
    cat = [{'var': c, 'lab': CORTO[c], 'opciones': [{'v': str(k), 'coef': 0.0 if i == 0 else float(coef[f'cualitativas__{c}_{k}'])}
                                                     for i, k in enumerate(cats)]} for c, cats in zip(cols, oh.categories_)]
    return {'b0': float(mod.intercept_[0]), 'num': num, 'cat': cat}
