"""Páginas ejecutivas del Laboratorio 3: estilo común, resumen ejecutivo y priorización con conclusiones."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

AZUL, AZUL_OSC, GRIS, NARANJA, ROJO = '#2a78d6', '#0d366b', '#a3a29b', '#eb6834', '#c0392b'

CSS = """
<style>
  .block-container { padding-top: 2.2rem; max-width: 1280px; }
  .hero { background: linear-gradient(120deg, #0d366b 0%, #1c5cab 60%, #2a78d6 100%); color: #fff; border-radius: 16px;
          padding: 26px 30px 22px; margin-bottom: 18px; }
  .hero .kicker { text-transform: uppercase; letter-spacing: .08em; font-size: 12.5px; opacity: .85; font-weight: 600; }
  .hero h1 { color: #fff; font-size: 30px; line-height: 1.2; margin: 6px 0 8px; padding: 0; }
  .hero p { font-size: 16.5px; margin: 0; opacity: .95; max-width: 980px; }
  .tarjeta { border: 1px solid #e3e8ef; border-radius: 14px; padding: 16px 18px; background: #fff; height: 100%; }
  .tarjeta .n { display: inline-block; background: #eef4fc; color: #1c5cab; font-weight: 700; border-radius: 8px; padding: 2px 9px;
                font-size: 12.5px; margin-bottom: 8px; }
  .tarjeta h4 { margin: 0 0 6px; font-size: 17px; line-height: 1.25; }
  .tarjeta p { margin: 0; font-size: 14.5px; color: #3d3c38; line-height: 1.45; }
  .tarjeta.alerta { border-color: #f3c3ad; background: #fff7f2; }
  .tarjeta.alerta .n { background: #fde3d6; color: #b4471c; }
  .pasos { display: flex; gap: 8px; align-items: stretch; margin: 6px 0 4px; }
  .pasos div { flex: 1; background: #f6f8fb; border-radius: 10px; padding: 10px 12px; font-size: 13.5px; line-height: 1.35; }
  .pasos div b { display: block; color: #1c5cab; font-size: 12.5px; text-transform: uppercase; letter-spacing: .05em; }
  .pasos div.hoy { background: #0d366b; color: #fff; }
  .pasos div.hoy b { color: #9ec5f4; }
  .kpi { border: 1px solid #e3e8ef; border-radius: 14px; padding: 14px 18px; background: #fff; }
  .kpi .l { font-size: 13.5px; color: #5c5b56; }
  .kpi .v { font-size: 34px; font-weight: 700; line-height: 1.15; margin: 2px 0; color: #1a1a1a; }
  .kpi .s { font-size: 13px; color: #5c5b56; }
  .kpi .s.mal { color: #c0392b; font-weight: 600; }
  div[data-testid="stMetric"] { background: #fff; }
  div[data-testid="stMetricValue"] { font-size: 30px; }
</style>
"""


def estilo():
    st.markdown(CSS, unsafe_allow_html=True)


def encabezado(kicker, titulo, bajada):
    st.markdown(f'<div class="hero"><div class="kicker">{kicker}</div><h1>{titulo}</h1><p>{bajada}</p></div>', unsafe_allow_html=True)


def kpi(col, etiqueta, valor, sub, mal=False):
    col.markdown(f'<div class="kpi"><div class="l">{etiqueta}</div><div class="v">{valor}</div><div class="s{" mal" if mal else ""}">{sub}</div></div>',
                 unsafe_allow_html=True)


def tarjeta(col, n, titulo, texto, alerta=False):
    col.markdown(f'<div class="tarjeta{" alerta" if alerta else ""}"><div class="n">{n}</div><h4>{titulo}</h4><p>{texto}</p></div>',
                 unsafe_allow_html=True)


# Métricas de validación cruzada repetida (5 × 10) del notebook 03, sección 8.2
CV_CLF = pd.DataFrame({'Toda la base (403)': [0.8792, 0.9382, 0.8207, 0.8746, 0.9436],
                       'Sin Risaralda (237)': [0.7459, 0.4917, 0.6665, 0.5603, 0.8329]},
                      index=['Accuracy', 'Precision', 'Recall', 'F1-score', 'AUC-ROC'])


# =============================== RESUMEN EJECUTIVO ===============================
def pagina_resumen():
    encabezado('Laboratorio 3 · Proyecto final de modelación · Grupo 5',
               '¿Qué entidades públicas necesitan atención digital primero tras el terremoto?',
               'Dos modelos responden con datos de 403 entidades: una <b>regresión logística</b> detecta la afectación Alta o Crítica (F1 de 0,90) '
               'y una <b>regresión lineal</b> muestra que la pobreza del municipio aumenta la fragilidad digital. La clasificación depende del bloque '
               'de colegios de Risaralda, y sin él su desempeño baja.')

    c = st.columns(4)
    kpi(c[0], 'Entidades modeladas', '403', '5 departamentos · 107 municipios')
    kpi(c[1], 'Clasificación · F1 en prueba', '0,90', '▼ 0,56 sin Risaralda (CV)', mal=True)
    kpi(c[2], 'Regresión · R² (validación cruzada)', '0,23', 'explica ~23 % de la fragilidad')
    kpi(c[3], 'Regresión · error medio (MAE)', '0,71', 'puntos en una escala de 0 a 5')

    st.markdown('#### Tres hallazgos')
    c = st.columns(3)
    tarjeta(c[0], 'Clasificación', 'El personal y la conectividad definen la afectación',
            'Tener el <b>personal TIC afectado</b> duplica las chances de afectación alta (OR 2,10). Le siguen la <b>operación detenida</b> (1,96) '
            'y estar <b>sin conectividad</b> (1,79).')
    tarjeta(c[1], 'Regresión', 'La pobreza explica la fragilidad, no la afectación declarada',
            'Cada punto de IPM suma <b>0,0118</b> al índice de vulnerabilidad (p = 0,013). Los <b>colegios</b> tienen 1,11 puntos más que las alcaldías. '
            'El índice no se asocia con el nivel de afectación que la entidad declara.')
    tarjeta(c[2], 'Alerta de calidad', 'El bloque de Risaralda infla el clasificador',
            '145 de los 160 casos "Crítico" son colegios de Risaralda con respuestas casi idénticas. Sin ellos, el F1 baja de <b>0,87 a 0,56</b> '
            '(validación cruzada). Hay que validarlo con MinTIC.', alerta=True)

    st.markdown('')
    izq, der = st.columns([3, 2])
    with izq, st.container(border=True):
        largo = CV_CLF.reset_index(names='Métrica').melt(id_vars='Métrica', var_name='Base', value_name='Valor')
        fig = px.bar(largo, x='Métrica', y='Valor', color='Base', barmode='group', text_auto='.2f',
                     color_discrete_map={'Toda la base (403)': AZUL, 'Sin Risaralda (237)': GRIS},
                     title='Regresión logística: desempeño con y sin el bloque de Risaralda')
        fig.update_layout(height=360, yaxis_range=[0, 1.05], yaxis_title=None, xaxis_title=None, legend_title=None,
                          legend=dict(orientation='h', y=1.02, x=1, xanchor='right', yanchor='bottom'), margin=dict(t=70, b=10))
        st.plotly_chart(fig, width='stretch')
        st.caption('Validación cruzada repetida (5 pliegues × 10 repeticiones). El AUC se mantiene alto: el modelo ordena bien el riesgo, pero '
                   'con el umbral de 0,5 se le escapan entidades críticas fuera de Risaralda.')
    with der, st.container(border=True):
        st.markdown('**El proyecto en una mirada**')
        st.markdown('''
| | Clasificación | Regresión |
|---|---|---|
| **Pregunta** | ¿Afectación Alta o Crítica? | ¿Cuánta fragilidad se espera? |
| **Respuesta** | `alto_critico` (1/0) | Índice de vulnerabilidad (0-5) |
| **Modelo** | Regresión logística | Regresión lineal |
| **Predictores** | 14 (operación, conectividad, personal, IPM…) | 3 (IPM, departamento, tipo) |
| **Métricas** | Accuracy, Precision, Recall, F1 | MAE, RMSE, MAPE, R² |
''')
        st.caption('Fuentes: MinTIC, encuesta a entidades tras el sismo del 10-ago-2026 (datos.gov.co) · DANE, IPM municipal 2018.')

    st.markdown('#### Recorrido del proyecto')
    st.markdown('<div class="pasos">'
                '<div><b>Lab 1</b>Limpieza, EDA y cruce con la pobreza del DANE</div>'
                '<div><b>Lab 2</b>Estadística descriptiva y pruebas de asociación</div>'
                '<div class="hoy"><b>Lab 3 · hoy</b>Hipótesis, pipelines, modelos, evaluación y app</div>'
                '</div>', unsafe_allow_html=True)
    st.caption('Use el menú de la izquierda: **Pipeline** muestra cómo se preparan los datos; las páginas de cada modelo permiten hacer '
               'predicciones; **Priorización** cruza los dos modelos. Los filtros de la barra lateral aplican a todas las páginas. '
               'La exploración del Lab 2 sigue disponible en el **Anexo**.')


# =============================== PRIORIZACIÓN Y CONCLUSIONES ===============================
@st.cache_data
def _puntajes(_clf, _reg, base, num_c, cat_c, num_r, cat_r):
    t = base.copy()
    t['prob'] = _clf.predict_proba(t[num_c + cat_c])[:, 1]
    t['indice_esperado'] = _reg.predict(t[num_r + cat_r])
    return t


def pagina_priorizacion(base, reg, clf, meta, filtro):
    encabezado('Laboratorio 3 · Conclusiones', 'Priorización: dónde actuar primero',
               'Cruzamos los dos modelos en una matriz: la <b>probabilidad de afectación alta</b> (logística) y la <b>fragilidad esperada</b> (lineal). '
               'Las entidades del cuadrante superior derecho combinan crisis actual y fragilidad estructural.')
    c, r = meta['clasificacion'], meta['regresion']
    t = filtro(_puntajes(clf, reg, base, c['numericas'], c['cualitativas'], r['numericas'], r['cualitativas']))
    if len(t) < 5:
        st.warning('Muy pocas entidades con los filtros actuales.'); return

    a, b = st.columns([1, 3])
    umbral = a.slider('Umbral de probabilidad', 0.1, 0.9, 0.5, 0.05, key='prio_umbral',
                      help='0,5 es el estándar; fuera de Risaralda recomendamos 0,3 para no dejar entidades críticas sin atención.')
    corte = a.slider('Fragilidad alta desde (índice esperado)', 1.0, 3.5, 2.5, 0.1, key='prio_corte')
    t = t.assign(Cuadrante=np.select(
        [(t['prob'] >= umbral) & (t['indice_esperado'] >= corte), t['prob'] >= umbral, t['indice_esperado'] >= corte],
        ['1. Atender ya', '2. Crisis actual', '3. Prevenir'], '4. Seguimiento'))
    colores = {'1. Atender ya': ROJO, '2. Crisis actual': NARANJA, '3. Prevenir': AZUL, '4. Seguimiento': GRIS}
    conteo = t['Cuadrante'].value_counts()
    for k, nombre in enumerate(colores):
        a.metric(nombre, int(conteo.get(nombre, 0)), border=True)
    with b, st.container(border=True):
        fig = px.scatter(t, x='prob', y='indice_esperado', color='Cuadrante', color_discrete_map=colores,
                         category_orders={'Cuadrante': list(colores)}, hover_name='razon_social',
                         hover_data={'mpio': True, 'prob': ':.1%', 'indice_esperado': ':.2f', 'Cuadrante': False},
                         labels={'prob': 'Probabilidad de afectación Alta o Crítica (logística)', 'indice_esperado': 'Fragilidad esperada (lineal, 0-5)',
                                 'mpio': 'Municipio'})
        fig.update_traces(marker=dict(size=9, opacity=.8, line=dict(width=.5, color='white')))
        fig.add_vline(x=umbral, line_dash='dash', line_color='#5c5b56')
        fig.add_hline(y=corte, line_dash='dash', line_color='#5c5b56')
        for x, y, txt in [(.98, 3.95, 'ATENDER YA'), (.02, 3.95, 'PREVENIR'), (.98, .1, 'CRISIS ACTUAL'), (.02, .1, 'SEGUIMIENTO')]:
            fig.add_annotation(x=x, y=y, text=txt, showarrow=False, font=dict(size=11, color='#5c5b56'),
                               xanchor='right' if x > .5 else 'left')
        fig.update_layout(height=470, xaxis=dict(range=[0, 1], tickformat='.0%'), yaxis_range=[0, 4.1], legend_title=None,
                          legend=dict(orientation='h', y=1.02, x=1, xanchor='right', yanchor='bottom'), margin=dict(t=40, b=10))
        st.plotly_chart(fig, width='stretch')
        st.caption('Cada punto es una entidad (filtros de la barra lateral). Pase el mouse para ver su nombre. Las entidades de entrenamiento '
                   'también se puntúan: aquí la matriz se usa para priorizar, no para evaluar los modelos.')

    with st.container(border=True):
        st.markdown('**Entidades para atender primero** (cuadrante "Atender ya", ordenadas por probabilidad)')
        top = (t[t['Cuadrante'] == '1. Atender ya'].sort_values(['prob', 'indice_esperado'], ascending=False)
               [['razon_social', 'mpio', 'tipo_entidad_g', 'nivel_afectacion_cat', 'prob', 'indice_esperado']]
               .rename(columns={'razon_social': 'Entidad', 'mpio': 'Municipio', 'tipo_entidad_g': 'Tipo', 'nivel_afectacion_cat': 'Nivel declarado',
                                'prob': 'Probabilidad', 'indice_esperado': 'Fragilidad esperada'}))
        st.dataframe(top.head(15).style.format({'Probabilidad': '{:.1%}', 'Fragilidad esperada': '{:.2f}'}), hide_index=True, width='stretch')
        st.download_button('Descargar la lista completa (CSV)', top.to_csv(index=False).encode('utf-8-sig'), 'priorizacion_entidades.csv',
                           'text/csv')

    st.markdown('#### Conclusiones y recomendaciones para MinTIC')
    c = st.columns(3)
    tarjeta(c[0], 'Recomendación 1', 'Priorizar con los dos modelos, no solo con la autoevaluación',
            'La afectación declarada no refleja la pobreza ni la fragilidad. Las entidades del <b>Chocó</b> declaran niveles similares al resto, '
            'pero son más frágiles: la matriz las hace visibles.')
    tarjeta(c[1], 'Recomendación 2', 'Bajar el umbral fuera de Risaralda',
            'Con umbral 0,3 el recall fuera de Risaralda sube de <b>0,42 a 0,83</b>. Para MinTIC es más grave no atender una entidad crítica '
            'que revisar una que no lo era.')
    tarjeta(c[2], 'Recomendación 3', 'Validar el bloque de Risaralda y reentrenar', 'Confirmar con MinTIC si las 145 respuestas casi '
            'idénticas son reales. Después, reentrenar sin ese bloque y probar modelos de Poisson u ordinales para el índice.', alerta=True)
    st.markdown('')
    with st.container(border=True):
        st.markdown('''
**Lo que demuestra el proyecto**
1. **Pruebas de hipótesis:** 8 de 9 variables cualitativas se asocian con la afectación (χ² con corrección de Holm, V de Cramér 0,47–0,71); IPM, departamento y tipo de entidad se asocian con la fragilidad (Spearman y Kruskal-Wallis).
2. **Pipelines:** imputación, logaritmo, escalamiento y one-hot dentro del modelo; todo se aprende solo con el entrenamiento, sin fuga de información.
3. **Evaluación honesta:** partición de prueba + validación cruzada repetida, análisis sin Risaralda, supuestos del modelo lineal y errores robustos HC3.
4. **Producto usable:** esta app predice para cualquier entidad y entrega una lista priorizada descargable.
''')
