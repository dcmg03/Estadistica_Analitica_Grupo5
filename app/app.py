"""Dashboard del Laboratorio 2 — Afectación digital de entidades públicas por el terremoto del 10 de agosto de 2026.

Ejecutar desde la carpeta del proyecto:  streamlit run app/app.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from estadistica import medidas, forma_texto, atipicos_iqr, correlaciones, independencia, comparar_grupos

st.set_page_config(page_title='Afectación digital · Terremoto 2026', page_icon='📊', layout='wide')


def f4(tabla):
    """Muestra los números con 4 decimales (requisito del laboratorio)."""
    return tabla.style.format(precision=4, na_rep='—', thousands='')


DATA = Path(__file__).resolve().parent.parent / 'data' / 'processed'
AZUL, GRIS, AMARILLO = '#2a78d6', '#a3a29b', '#FFC000'
ORD_OBJ = ['Sin afectación TIC', 'Bajo', 'Moderado', 'Alto', 'Crítico']
SEQ = ['#9ec5f4', '#6da7ec', '#3987e5', '#1c5cab', '#0d366b']
CONECT = ['Normal.', 'Degradada o inestable.', 'Disponible únicamente mediante mecanismos alternativos.', 'Sin conectividad.']
ORDENES = {'nivel_afectacion_cat': ORD_OBJ, 'estado_conectividad': CONECT,
           'energia_estable': ['Sí, normal.', 'Sí, pero intermitente.', 'No.'],
           'ipm_grupo': ['IPM bajo', 'IPM medio', 'IPM alto']}


def ordenar(tabla, filas, columnas):
    """Ordena filas y columnas de una tabla de contingencia según el orden lógico de cada variable."""
    if filas in ORDENES:
        tabla = tabla.reindex([o for o in ORDENES[filas] if o in tabla.index] + [o for o in tabla.index if o not in ORDENES[filas]])
    if columnas in ORDENES:
        tabla = tabla.reindex(columns=[o for o in ORDENES[columnas] if o in tabla.columns] + [o for o in tabla.columns if o not in ORDENES[columnas]])
    return tabla

NUM = {'ipm': 'IPM municipal (%)', 'ipm_rural': 'IPM rural (%)', 'ipm_cabecera': 'IPM cabecera (%)',
       'duracion_min': 'Duración del formulario (min)', 'n_infra_afectada': 'N° de elementos de infraestructura afectados',
       'n_servicios_caidos': 'N° de servicios caídos o con afectaciones', 'n_sistemas_afectados': 'N° de sistemas afectados',
       'indice_vulnerabilidad': 'Índice de vulnerabilidad digital (0-5)', 'dias_post_sismo': 'Días después del sismo'}
CAT = {'nivel_afectacion_cat': 'Nivel de afectación (objetivo)', 'departamento': 'Departamento', 'tipo_entidad': 'Tipo de entidad',
       'estado_conectividad': 'Estado de la conectividad', 'energia_estable': 'Suministro de energía', 'personal_suficiente': 'Personal TIC',
       'riesgo_continuidad': 'Riesgo para la continuidad', 'personal_afectado': 'Afectación del personal TIC',
       'estado_sistemas': 'Estado de los sistemas', 'ipm_grupo': 'Tercil de IPM'}
MUN = {'ipm': 'IPM municipal (%)', 'pct_satelite': '% entidades que dependen del satélite',
       'pct_sin_conectividad': '% entidades sin conectividad', 'pct_alto_critico': '% entidades en Alto o Crítico',
       'vulnerabilidad_media': 'Índice de vulnerabilidad medio', 'sin_agua_mejorada': '% hogares sin fuente de agua mejorada',
       'Trabajo informal': '% trabajo informal', 'Analfabetismo': '% analfabetismo'}


@st.cache_data
def cargar():
    base = pd.read_csv(DATA / 'base_lab2.csv', dtype={'cod_mpio': str})
    mun = pd.read_csv(DATA / 'municipios_lab2.csv', dtype={'cod_mpio': str})
    base['nivel_afectacion_cat'] = pd.Categorical(base['nivel_afectacion_cat'], ORD_OBJ, ordered=True)
    for c in base.select_dtypes(include=['object', 'string']):
        base[c] = base[c].str.strip()
    return base, mun


try:
    base, mun = cargar()
except FileNotFoundError:
    st.error('No se encontraron los datos en `data/processed/`. Ejecute primero los notebooks 01 (Lab 1) y 02 (Lab 2) con Run All para generar '
             '`base_lab2.csv` y `municipios_lab2.csv`.')
    st.stop()

# ---------------- barra lateral ----------------
st.sidebar.header('Filtros')
deps = st.sidebar.multiselect('Departamento', sorted(base['departamento'].unique()), default=sorted(base['departamento'].unique()))
tipos = st.sidebar.multiselect('Tipo de entidad', sorted(base['tipo_entidad'].unique()), default=sorted(base['tipo_entidad'].unique()))
sin_ris = st.sidebar.checkbox('Excluir Risaralda (bloque de respuestas idénticas)', value=False,
                              help='145 de las 160 entidades en "Crítico" son colegios de Risaralda con respuestas casi idénticas.')
df = base[base['departamento'].isin(deps) & base['tipo_entidad'].isin(tipos)]
if sin_ris:
    df = df[df['departamento'] != 'RISARALDA']
st.sidebar.markdown(f'**Entidades seleccionadas:** {len(df)} de {len(base)}')
st.sidebar.caption('Fuentes: MinTIC, *Resultados encuesta entidades Terremoto* (datos.gov.co, 2026) · '
                   'DANE, IPM municipal de fuente censal 2018.')
st.sidebar.caption('Edwart Stevens González Ríos · Karen Julieth Hernández Chaparro · Deisy Carolina Monroy Gutiérrez — '
                   'Especialización en Analítica Estratégica de Datos, UPTC')

if len(df) < 10:
    st.warning('Hay muy pocas entidades con los filtros actuales. Amplíe la selección.')
    st.stop()

st.title('Afectación digital de entidades públicas tras el terremoto del 10 de agosto de 2026')
st.caption('Laboratorio 2 · Exploración de datos e hipótesis de asociación · Resultados redondeados a 4 decimales · '
           'Base: 408 entidades con IPM del DANE (Nuevo Belén de Bajirá no tiene dato censal)')

t1, t2, t3, t4, t5, t6 = st.tabs(['Resumen', 'Univariado', 'Correlación', 'Independencia (χ²)', 'ANOVA vs Kruskal-Wallis', 'Foco Chocó'])

# ---------------- Resumen ----------------
with t1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Entidades', f'{len(df)}')
    c2.metric('En nivel Crítico', f'{(df["nivel_afectacion_cat"] == "Crítico").mean() * 100:.1f} %')
    c3.metric('Sin conectividad', f'{(df["estado_conectividad"] == "Sin conectividad.").mean() * 100:.1f} %')
    c4.metric('IPM mediano del municipio', f'{df["ipm"].median():.1f} %')
    a, b = st.columns(2)
    vc = df['nivel_afectacion_cat'].value_counts().reindex(ORD_OBJ).fillna(0).reset_index()
    vc.columns = ['Nivel', 'Entidades']
    fig = px.bar(vc, x='Nivel', y='Entidades', color='Nivel', color_discrete_sequence=SEQ, text='Entidades',
                 title='Variable objetivo: nivel de afectación')
    fig.update_layout(showlegend=False, xaxis_title=None)
    a.plotly_chart(fig, width='stretch')
    ct = (pd.crosstab(df['departamento'], df['nivel_afectacion_cat'], normalize='index') * 100).reindex(columns=ORD_OBJ).fillna(0)
    fig = px.bar(ct.reset_index().melt(id_vars='departamento', var_name='Nivel', value_name='%'), y='departamento', x='%', color='Nivel',
                 orientation='h', color_discrete_sequence=SEQ, category_orders={'Nivel': ORD_OBJ},
                 title='Nivel de afectación por departamento (%)')
    fig.update_layout(yaxis_title=None, legend_title=None)
    b.plotly_chart(fig, width='stretch')
    st.info('**Hallazgo clave:** la clase "Crítico" se concentra en colegios de Risaralda con respuestas casi idénticas. '
            'Active el filtro de la barra lateral para ver cómo cambia la distribución sin ese bloque.')

# ---------------- Univariado ----------------
with t2:
    v = st.selectbox('Variable cuantitativa', list(NUM), format_func=NUM.get, key='uni')
    s = df[v].dropna()
    a, b = st.columns([1, 2])
    a.dataframe(f4(medidas(s)), width='stretch', height=700)
    with b:
        log = st.checkbox('Escala logarítmica en el eje x', value=(v == 'duracion_min'))
        fig = px.histogram(df, x=v, nbins=40, marginal='box', color_discrete_sequence=[AZUL], labels={v: NUM[v]},
                           title=f'Histograma y diagrama de caja · {NUM[v]}', log_x=log)
        fig.add_vline(x=s.mean(), line_color='#eb6834', row=1, col=1)
        fig.add_vline(x=s.median(), line_dash='dash', line_color='#0b0b0b', row=1, col=1)
        st.plotly_chart(fig, width='stretch')
        st.caption('Línea naranja: media · línea punteada: mediana')
        st.markdown(f'**Forma:** {forma_texto(s)}')
        st.markdown('**Atípicos por el criterio IQR (1,5 × IQR):**')
        st.dataframe(f4(pd.DataFrame([atipicos_iqr(s)])), width='stretch', hide_index=True)
    st.subheader('Variables cualitativas')
    vc_ = st.selectbox('Variable cualitativa', list(CAT), format_func=CAT.get, key='unicat')
    f = df[vc_].astype(str).value_counts().reset_index(); f.columns = ['Categoría', 'Frecuencia']
    f['%'] = (f['Frecuencia'] / f['Frecuencia'].sum() * 100).round(4)
    fig = px.bar(f.sort_values('Frecuencia'), x='Frecuencia', y='Categoría', orientation='h', text='Frecuencia',
                 color_discrete_sequence=[AZUL], title=CAT[vc_])
    st.plotly_chart(fig, width='stretch')

# ---------------- Correlación ----------------
with t3:
    nivel = st.radio('Nivel de análisis', ['Municipal (recomendado para variables del territorio)', 'Entidad'], horizontal=True)
    if nivel.startswith('Municipal'):
        mm = mun[mun['departamento'].isin(deps)]
        if sin_ris:
            mm = mm[mm['departamento'] != 'RISARALDA']
        opciones, datos, nombres = list(MUN), mm, MUN
        st.caption(f'{len(mm)} municipios. Las proporciones se calculan con las entidades de cada municipio que respondieron la encuesta.')
    else:
        opciones, datos, nombres = list(NUM), df, NUM
    a, b = st.columns(2)
    x = a.selectbox('Variable X', opciones, format_func=nombres.get, index=0)
    y = b.selectbox('Variable Y', [o for o in opciones if o != x], format_func=nombres.get, index=0)
    c1, c2 = st.columns([2, 1])
    datos = datos.assign(Grupo=np.where(datos['departamento'] == 'CHOCÓ', 'Chocó', 'Otros departamentos'))
    fig = px.scatter(datos, x=x, y=y, color='Grupo', color_discrete_map={'Chocó': AZUL, 'Otros departamentos': GRIS},
                     labels={x: nombres[x], y: nombres[y]}, title=f'{nombres[y]} vs {nombres[x]}', opacity=.8)
    c1.plotly_chart(fig, width='stretch')
    with c2:
        st.dataframe(f4(correlaciones(datos[x], datos[y])), width='stretch')
        st.markdown('**Cómo leerlo:** Pearson mide relación **lineal** y es sensible a atípicos; Spearman y Kendall usan **rangos** y '
                    'detectan relaciones monótonas aunque no sean lineales. Si Pearson y Spearman difieren mucho, revise atípicos o no linealidad.')

# ---------------- Independencia ----------------
with t4:
    a, b = st.columns(2)
    v1 = a.selectbox('Variable 1', list(CAT), format_func=CAT.get, index=list(CAT).index('estado_conectividad'))
    v2 = b.selectbox('Variable 2', [c for c in CAT if c != v1], format_func=CAT.get, index=0)
    t = df[[v1, v2]].dropna()
    obs, esp, res = independencia(t[v1].astype(str), t[v2].astype(str))
    obs, esp = ordenar(obs, v1, v2), ordenar(esp, v1, v2)
    st.markdown('**Frecuencias observadas**'); st.dataframe(obs.assign(Total=obs.sum(axis=1)), width='stretch')
    st.markdown('**Frecuencias esperadas bajo independencia**'); st.dataframe(f4(esp), width='stretch')
    st.dataframe(f4(pd.DataFrame([res])), width='stretch', hide_index=True)
    concl = 'se **rechaza** la independencia: hay dependencia estadísticamente significativa' if res['p-valor χ²'] < .05 \
        else '**no se rechaza** la independencia'
    st.markdown(f'Con α = 0,05, {concl} (p = {res["p-valor χ²"]:.4f}). La información mutua mide la dependencia sin suponer una forma de relación; '
                f'su prueba G = 2·N·IM lleva a la misma conclusión que χ² (p = {res["p-valor G"]:.4f}).')
    if res['% celdas con esperado < 5'] > 20:
        st.warning(f'{res["% celdas con esperado < 5"]} % de las celdas tiene frecuencia esperada menor a 5: la aproximación χ² es menos confiable.')
    fig = px.imshow((obs.div(obs.sum(axis=1), axis=0) * 100).round(1), text_auto=True, color_continuous_scale='Blues', aspect='auto',
                    title=f'% por fila: {CAT[v1]} × {CAT[v2]}')
    st.plotly_chart(fig, width='stretch')

# ---------------- ANOVA vs Kruskal ----------------
with t5:
    a, b = st.columns(2)
    g = a.selectbox('Variable cualitativa (grupos)', list(CAT), format_func=CAT.get, index=0, key='grp')
    v = b.selectbox('Variable cuantitativa', list(NUM), format_func=NUM.get, index=0, key='val')
    orden = ORDENES.get(g)
    tab, pr = comparar_grupos(df.assign(**{g: df[g].astype(str)}), g, v, orden)
    st.markdown('**Normalidad por grupo (Shapiro-Wilk)**')
    st.dataframe(f4(tab), width='stretch')
    st.dataframe(f4(pd.DataFrame([pr])), width='stretch', hide_index=True)
    normal = (tab['¿Normal?'] == 'Sí').all(); homo = pr['Levene (mediana) p'] > .05
    rec = 'ANOVA' if (normal and homo) else 'Kruskal-Wallis'
    st.markdown(f'- Normalidad en todos los grupos: **{"sí" if normal else "no"}** · Homocedasticidad (Levene): **{"sí" if homo else "no"}**\n'
                f'- ANOVA: p = {pr["ANOVA p"]:.4f} · Kruskal-Wallis: p = {pr["Kruskal-Wallis p"]:.4f}\n'
                f'- **Prueba recomendada: {rec}**' + ('' if rec == 'ANOVA' else ', porque se viola al menos un supuesto del ANOVA.'))
    fig = px.box(df.assign(**{g: df[g].astype(str)}), x=g, y=v, color_discrete_sequence=[AZUL], points='outliers',
                 category_orders={g: orden} if orden else None, labels={g: CAT[g], v: NUM[v]}, title=f'{NUM[v]} por {CAT[g]}')
    st.plotly_chart(fig, width='stretch')

# ---------------- Foco Chocó ----------------
with t6:
    st.markdown('Comparación de **alcaldías** del Chocó frente a las de los demás departamentos (para no mezclar tipos de entidad).')
    al = base[base['tipo_entidad'] == 'Alcaldía'].assign(Grupo=lambda x: np.where(x['departamento'] == 'CHOCÓ', 'Chocó', 'Otros 4 departamentos'))
    ind = {'Energía intermitente o sin energía': al['sin_energia_estable'] == 1,
           'Depende de internet satelital': al['depende_satelite'] == 1,
           'Con fibra óptica': al['sin_fibra_optica'] == 0,
           'Con póliza de seguro': al['poliza_seguro'] == 'Sí.',
           'Se declara Alto o Crítico': al['nivel_afectacion_cat'].isin(['Alto', 'Crítico'])}
    comp = pd.DataFrame({k: v.groupby(al['Grupo']).mean() * 100 for k, v in ind.items()}).T.round(1)
    fig = px.bar(comp.reset_index().melt(id_vars='index', var_name='Grupo', value_name='%'), y='index', x='%', color='Grupo', barmode='group',
                 orientation='h', color_discrete_map={'Chocó': AZUL, 'Otros 4 departamentos': GRIS}, text='%',
                 title=f'Alcaldías: Chocó (n = {(al.Grupo == "Chocó").sum()}) frente a otros departamentos (n = {(al.Grupo != "Chocó").sum()})')
    fig.update_layout(yaxis_title=None)
    st.plotly_chart(fig, width='stretch')
    st.success('El Chocó **declara un nivel de afectación similar** al resto, pero su infraestructura es **mucho más frágil**: '
               'más dependencia del satélite, energía inestable y casi sin seguros. Priorizar solo por autoevaluación puede relegarlo.')
    st.caption('Siguiente fase propuesta: contrastar la pobreza de 2025 (línea base) con la de 2026 cuando el DANE la publique, '
               'por departamento y en los municipios PDET.')
