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
def pestana_regresion(base, reg, meta, filtro):
    st.markdown('**Variable respuesta:** índice de vulnerabilidad digital (0 a 5), suma de 5 condiciones desfavorables: sin energía estable, '
                'sin fibra óptica, sin póliza de seguro, sin planes de contingencia y personal TIC insuficiente. '
                '**Predictores:** IPM del municipio, departamento y tipo de entidad (ninguno de los componentes del índice, para evitar fuga de '
                'información). **Modelo:** regresión lineal por mínimos cuadrados dentro de un pipeline (escalamiento + one-hot).')
    X = meta['regresion']['numericas'] + meta['regresion']['cualitativas']; y = meta['regresion']['respuesta']

    st.subheader('1. Hacer una predicción')
    a, b = st.columns([1, 2])
    with a:
        dep, ipm = selector_municipio(base, 'reg_mpio')
        tipo = st.selectbox('Tipo de entidad', opciones(base, 'tipo_entidad_g'), key='reg_tipo')
        ipm = st.slider('IPM a usar (puede simular otro valor)', 0.0, 100.0, ipm, 0.5, key='reg_ipm')
        entrada = pd.DataFrame([{'ipm': ipm, 'departamento': dep, 'tipo_entidad_g': tipo}])
        pred = float(reg.predict(entrada[X])[0])
        st.metric('Índice de vulnerabilidad esperado', f'{pred:.2f} de 5')
        nivel = 'bajo' if pred < 1.5 else ('medio' if pred < 2.5 else 'alto')
        st.markdown(f'Vulnerabilidad esperada **{nivel}**. Promedio de la base: {base[y].mean():.2f}.')
        st.caption('El modelo explica ~23 % de la variabilidad (validación cruzada): sirve para comparar grupos de entidades, '
                   'no para conocer el índice exacto de una entidad.')
    with b:
        rejilla = pd.DataFrame([{'ipm': v, 'departamento': dep, 'tipo_entidad_g': t} for t in opciones(base, 'tipo_entidad_g')
                                for v in np.linspace(5, 95, 37)])
        rejilla['pred'] = reg.predict(rejilla[X])
        fig = px.line(rejilla, x='ipm', y='pred', color='tipo_entidad_g', labels={'ipm': 'IPM municipal (%)', 'pred': 'Índice esperado',
                      'tipo_entidad_g': 'Tipo de entidad'}, title=f'Índice esperado según el IPM · {dep.title()}',
                      color_discrete_sequence=['#0d366b', AZUL, '#9ec5f4', GRIS])
        fig.add_trace(go.Scatter(x=[ipm], y=[pred], mode='markers', marker=dict(size=14, color=NARANJA, line=dict(color='white', width=2)),
                                 name='Su entidad'))
        fig.update_yaxes(range=[0, 5])
        st.plotly_chart(fig, width='stretch')

    st.subheader('2. Evaluación del modelo')
    te = filtro(base[base['conjunto'] == 'prueba'])
    if len(te) < 10:
        st.warning('Hay muy pocas entidades de prueba con los filtros actuales.'); return
    p = reg.predict(te[X]); yt = te[y].values; m = yt > 0
    met = pd.DataFrame([{'MAE': mean_absolute_error(yt, p), 'RMSE': np.sqrt(mean_squared_error(yt, p)),
                         'MAPE (%)': np.mean(np.abs((yt[m] - p[m]) / yt[m])) * 100 if m.any() else np.nan,
                         'R²': r2_score(yt, p) if len(te) > 1 else np.nan, 'Entidades de prueba': len(te)}])
    st.dataframe(tabla4(met), hide_index=True, width='stretch')
    st.caption('Métricas sobre las entidades de prueba que cumplen los filtros de la barra lateral. MAPE calculado solo con índice > 0 '
               '(el porcentaje no está definido cuando el valor real es 0). En la partición completa el R² es bajo (0,07): esa partición quedó '
               'en el percentil 4 de la validación cruzada repetida, cuyo R² medio es 0,23.')
    c1, c2 = st.columns(2)
    jit = np.random.default_rng(1).uniform(-.12, .12, len(te))
    fig = px.scatter(x=yt + jit, y=p, color=te['departamento'].str.title(), hover_name=te['mpio'],
                     labels={'x': 'Índice real', 'y': 'Índice predicho', 'color': 'Departamento'}, title='Real vs predicho (prueba)')
    fig.add_shape(type='line', x0=0, y0=0, x1=5, y1=5, line=dict(dash='dash', color=NARANJA))
    c1.plotly_chart(fig, width='stretch')
    fig = px.histogram(x=yt - p, nbins=20, color_discrete_sequence=[AZUL], labels={'x': 'Residuo (real − predicho)'},
                       title='Distribución de los residuos (prueba)')
    fig.update_layout(yaxis_title='N° de entidades')
    c2.plotly_chart(fig, width='stretch')

    st.subheader('3. Coeficientes')
    pre, mod = reg.named_steps['preprocesamiento'], reg.named_steps['modelo']
    coef = pd.DataFrame({'Variable': [n.split('__', 1)[1] for n in pre.get_feature_names_out()], 'Coeficiente': mod.coef_})
    sd = pre.named_transformers_['numericas'].named_steps['escalar'].scale_[0]
    coef.loc[coef['Variable'] == 'ipm', 'Variable'] = f'ipm (por cada desviación estándar = {sd:.1f} puntos)'
    fig = px.bar(coef.sort_values('Coeficiente'), x='Coeficiente', y='Variable', orientation='h',
                 color=np.where(coef.sort_values('Coeficiente')['Coeficiente'] > 0, 'Aumenta', 'Disminuye'),
                 color_discrete_map={'Aumenta': AZUL, 'Disminuye': GRIS}, title='Efecto sobre el índice (referencias: Caldas y Alcaldía)')
    fig.update_layout(yaxis_title=None, legend_title=None)
    st.plotly_chart(fig, width='stretch')
    st.markdown('Por cada punto de IPM el índice sube **0,0118** (p = 0,0131). Las **instituciones educativas** tienen 1,11 puntos más que las '
                'alcaldías y los **hospitales** 0,65 menos. Prueba F global: p < 0,0001. Breusch-Pagan detectó heterocedasticidad, por lo que '
                'en el notebook los p-valores se verificaron con errores robustos (HC3).')


# =============================== REGRESIÓN LOGÍSTICA ===============================
def pestana_clasificacion(base, clf, meta, filtro):
    st.markdown('**Variable respuesta:** `alto_critico` = 1 si la entidad declara afectación **Alta o Crítica**; 0 si declara Sin afectación, '
                'Baja o Moderada. **Modelo:** regresión logística con regularización L2 (C = '
                f'{meta["clasificacion"]["hiperparametros"]["C"]}, pesos de clase {meta["clasificacion"]["hiperparametros"]["class_weight"]}) '
                'dentro de un pipeline (log de los conteos, escalamiento y one-hot).')
    num, cat = meta['clasificacion']['numericas'], meta['clasificacion']['cualitativas']; y = meta['clasificacion']['respuesta']

    st.subheader('1. Hacer una predicción')
    st.caption('Responda como si fuera una entidad que diligencia la encuesta. Los valores iniciales corresponden a una entidad típica.')
    entrada = {}
    c1, c2, c3 = st.columns(3)
    with c1:
        _, entrada['ipm'] = selector_municipio(base, 'clf_mpio')
        entrada['tipo_entidad_g'] = st.selectbox(ETIQUETAS['tipo_entidad_g'], opciones(base, 'tipo_entidad_g'), key='clf_tipo')
        for c in ['estado_conectividad', 'energia_g']:
            ops = opciones(base, c); moda = base[c].mode()[0]
            entrada[c] = st.selectbox(ETIQUETAS[c], ops, index=ops.index(moda), key=f'clf_{c}')
    with c2:
        for c in ['personal_afectado', 'estado_sistemas', 'riesgo_continuidad', 'sede_contingencia']:
            ops = opciones(base, c); moda = base[c].mode()[0]
            entrada[c] = st.selectbox(ETIQUETAS[c], ops, index=ops.index(moda), key=f'clf_{c}')
    with c3:
        for c in ['zonas_incomunicadas', 'estado_infra']:
            ops = opciones(base, c); moda = base[c].mode()[0]
            entrada[c] = st.selectbox(ETIQUETAS[c], ops, index=ops.index(moda), key=f'clf_{c}')
        for c, mx in [('n_servicios_caidos', 7), ('n_infra_afectada', 14), ('n_sistemas_afectados', 8), ('indice_vulnerabilidad', 5)]:
            entrada[c] = st.slider(ETIQUETAS[c], 0, mx, int(base[c].median()), key=f'clf_{c}')
    X = pd.DataFrame([entrada])[num + cat]
    prob = float(clf.predict_proba(X)[0, 1])
    umbral = st.slider('Umbral de decisión (bajarlo aumenta el recall: se detectan más entidades críticas, con más falsas alarmas)',
                       0.1, 0.9, 0.5, 0.05, key='clf_umbral')
    a, b = st.columns([1, 2])
    with a:
        fig = go.Figure(go.Indicator(mode='gauge+number', value=prob * 100, number={'suffix': ' %', 'valueformat': '.1f'},
                                     title={'text': 'Probabilidad de afectación Alta o Crítica'},
                                     gauge={'axis': {'range': [0, 100]}, 'bar': {'color': AZUL},
                                            'threshold': {'line': {'color': NARANJA, 'width': 3}, 'value': umbral * 100}}))
        fig.update_layout(height=260, margin=dict(t=60, b=10, l=20, r=20))
        st.plotly_chart(fig, width='stretch')
        if prob >= umbral:
            st.error(f'**Clasificación: afectación ALTA o CRÍTICA** (probabilidad {prob:.1%} ≥ umbral {umbral:.0%}). Priorizar atención.')
        else:
            st.success(f'**Clasificación: afectación BAJA** (probabilidad {prob:.1%} < umbral {umbral:.0%}).')
    with b:
        # Contribución de cada variable al log-odds (coeficiente × valor transformado)
        pre, mod = clf.named_steps['preprocesamiento'], clf.named_steps['modelo']
        z = pre.transform(X)[0] * mod.coef_[0]
        nombres = [n.split('__', 1)[1] for n in pre.get_feature_names_out()]
        contrib = pd.DataFrame({'Variable': nombres, 'Aporte': z})
        contrib = contrib[contrib['Aporte'].abs() > 1e-6].sort_values('Aporte', key=abs, ascending=False).head(10).sort_values('Aporte')
        fig = px.bar(contrib, x='Aporte', y='Variable', orientation='h', title='¿Qué empuja esta predicción? (aporte al log-odds)',
                     color=np.where(contrib['Aporte'] > 0, 'Sube el riesgo', 'Baja el riesgo'),
                     color_discrete_map={'Sube el riesgo': AZUL, 'Baja el riesgo': GRIS})
        fig.update_layout(yaxis_title=None, legend_title=None, height=360)
        st.plotly_chart(fig, width='stretch')
        st.caption('Variables numéricas: aporte respecto a una entidad promedio. Cualitativas: aporte frente a la categoría de referencia.')

    st.subheader('2. Evaluación del modelo')
    te = filtro(base[base['conjunto'] == 'prueba'])
    if len(te) < 10 or te[y].nunique() < 2:
        st.warning('Hay muy pocas entidades de prueba (o una sola clase) con los filtros actuales.'); return
    pr = clf.predict_proba(te[num + cat])[:, 1]; pc = (pr >= umbral).astype(int); yt = te[y].values
    met = pd.DataFrame([{'Accuracy': accuracy_score(yt, pc), 'Precision': precision_score(yt, pc, zero_division=0),
                         'Recall': recall_score(yt, pc), 'F1-score': f1_score(yt, pc), 'AUC-ROC': roc_auc_score(yt, pr),
                         'Entidades de prueba': len(te)}])
    st.dataframe(tabla4(met), hide_index=True, width='stretch')
    st.caption(f'Métricas sobre las entidades de prueba que cumplen los filtros, con umbral {umbral:.2f}. Active "Excluir Risaralda" en la barra '
               'lateral: el recall cae mucho, porque parte del desempeño viene del bloque de colegios de Risaralda con respuestas idénticas.')
    c1, c2 = st.columns(2)
    cm = confusion_matrix(yt, pc, labels=[0, 1])
    fig = px.imshow(cm, text_auto=True, color_continuous_scale='Blues', x=['Baja', 'Alta/Crítica'], y=['Baja', 'Alta/Crítica'],
                    labels={'x': 'Predicho', 'y': 'Real', 'color': 'Entidades'}, title='Matriz de confusión (prueba)')
    fig.update_layout(coloraxis_showscale=False)
    c1.plotly_chart(fig, width='stretch')
    fpr, tpr, _ = roc_curve(yt, pr)
    fig = go.Figure([go.Scatter(x=fpr, y=tpr, mode='lines', line=dict(color=AZUL, width=3), name=f'Logística (AUC = {roc_auc_score(yt, pr):.4f})'),
                     go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(color=GRIS, dash='dash'), name='Azar')])
    fig.update_layout(title='Curva ROC (prueba)', xaxis_title='Tasa de falsos positivos', yaxis_title='Recall',
                      legend=dict(x=.45, y=.1))
    c2.plotly_chart(fig, width='stretch')

    st.subheader('3. Razones de odds')
    pre, mod = clf.named_steps['preprocesamiento'], clf.named_steps['modelo']
    co = pd.DataFrame({'Variable': [n.split('__', 1)[1] for n in pre.get_feature_names_out()], 'Coeficiente': mod.coef_[0]})
    co['Razón de odds'] = np.exp(co['Coeficiente'])
    co = co.sort_values('Coeficiente')
    fig = px.bar(co, x='Razón de odds', y='Variable', orientation='h', log_x=True, height=650,
                 color=np.where(co['Coeficiente'] > 0, 'Aumenta la probabilidad', 'Disminuye la probabilidad'),
                 color_discrete_map={'Aumenta la probabilidad': AZUL, 'Disminuye la probabilidad': GRIS},
                 title='Razones de odds (escala logarítmica; 1 = sin efecto)')
    fig.add_vline(x=1, line_color='black')
    fig.update_layout(yaxis_title=None, legend_title=None)
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
    te = filtro(base[base['conjunto'] == 'prueba'])
    if te.empty:
        st.warning('No hay entidades de prueba con los filtros actuales.'); return
    etiqueta = te['razon_social'].str.strip().str.slice(0, 60) + ' · ' + te['mpio']
    sugeridas = te.index[te['razon_social'].str.contains('BERNARDO ARIAS', case=False, na=False)]
    sel = st.selectbox('Entidad que recorre la línea (conjunto de prueba)', etiqueta.index, format_func=lambda i: etiqueta[i], key='pipe_ent',
                       index=int(list(etiqueta.index).index(sugeridas[0])) if len(sugeridas) else 0)
    datos = _datos_linea(clf, te.loc[sel], num, cat)
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
