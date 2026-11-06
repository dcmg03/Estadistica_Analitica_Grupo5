"""Dashboard de los Laboratorios 2 y 3 — Afectación digital de entidades públicas por el terremoto del 10 de agosto de 2026.

Ejecutar desde la carpeta del proyecto:  streamlit run app/app.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from estadistica import medidas, forma_texto, atipicos_iqr, correlaciones, independencia, comparar_grupos
import modelos
import ejecutivo

st.set_page_config(page_title='Afectación digital · Terremoto 2026', layout='wide', initial_sidebar_state='expanded')


def f4(tabla):
    """Muestra los números con 4 decimales (requisito del laboratorio)."""
    return tabla.style.format(precision=4, na_rep='—', thousands='')


DATA = Path(__file__).resolve().parent.parent / 'data' / 'processed'
AZUL, GRIS = '#2a78d6', '#a3a29b'
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


def nombre_mpio(s):
    """PUEBLO RICO -> Pueblo Rico; BELÉN DE UMBRÍA -> Belén de Umbría."""
    return s.str.title().str.replace(' De ', ' de ').str.replace(' Del ', ' del ')


def nombre_propio(txt):
    """VALLE DEL CAUCA -> Valle del Cauca (una sola cadena de texto, no una Series)."""
    t = str(txt).title()
    for a, b in [(' De ', ' de '), (' Del ', ' del '), (' La ', ' la '), (' Las ', ' las '), (' Los ', ' los ')]:
        t = t.replace(a, b)
    return t


def histograma(df, v, log=False, nbins=40):
    """Histograma con diagrama de caja; al pasar el mouse muestra los municipios de cada barra y de cada atípico."""
    s = df[v].dropna()
    titulo = f'Histograma y diagrama de caja · {NUM[v]}'
    if log:   # escala logarítmica (duración): histograma estándar
        fig = px.histogram(df, x=v, nbins=nbins, marginal='box', color_discrete_sequence=[AZUL], labels={v: NUM[v]},
                           title=titulo, log_x=True, hover_data=['mpio'])
        fig.update_traces(hovertemplate=f'{NUM[v]}: %{{x}}<br>N° de entidades: %{{y}}<extra></extra>', selector=dict(type='histogram'))
        fig.update_layout(yaxis_title='N° de entidades')
        fila = 1
    else:
        t = df.loc[df[v].notna(), [v, 'mpio']]
        bordes = np.histogram_bin_edges(t[v], bins=nbins)
        t = t.assign(barra=pd.cut(t[v], bordes, include_lowest=True, labels=False))
        filas = []
        for i in range(len(bordes) - 1):
            top = t.loc[t['barra'] == i, 'mpio'].value_counts()
            txt = '<br>'.join(f'{m}: {n}' for m, n in top.head(6).items())
            if len(top) > 6:
                txt += f'<br>… y {len(top) - 6} municipios más'
            filas.append((bordes[i], bordes[i + 1], int(top.sum()), txt or '—'))
        h = pd.DataFrame(filas, columns=['desde', 'hasta', 'n', 'mpios'])
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[.18, .82], vertical_spacing=.03)
        fig.add_trace(go.Box(x=t[v], name='', boxpoints='outliers', notched=True, marker_color=AZUL, text=t['mpio'],
                             hoverinfo='x+text'), row=1, col=1)
        fig.add_trace(go.Bar(x=(h['desde'] + h['hasta']) / 2, y=h['n'], width=h['hasta'] - h['desde'], marker_color=AZUL,
                             marker_line=dict(color='white', width=.5), customdata=h[['desde', 'hasta', 'mpios']],
                             hovertemplate=f'{NUM[v]}: %{{customdata[0]:.1f}} a %{{customdata[1]:.1f}}<br>'
                                           '<b>N° de entidades: %{y}</b><br>%{customdata[2]}<extra></extra>'), row=2, col=1)
        fig.update_layout(title=titulo, showlegend=False, bargap=0)
        fig.update_yaxes(showticklabels=False, row=1, col=1)
        fig.update_yaxes(title_text='N° de entidades', row=2, col=1)
        fig.update_xaxes(title_text=NUM[v], row=2, col=1)
        fila = 2
    fig.add_vline(x=s.mean(), line_color='#eb6834', row=fila, col=1)
    fig.add_vline(x=s.median(), line_dash='dash', line_color='#0b0b0b', row=fila, col=1)
    return fig


@st.cache_data
def cargar():
    base = pd.read_csv(DATA / 'base_lab2.csv', dtype={'cod_mpio': str})
    mun = pd.read_csv(DATA / 'municipios_lab2.csv', dtype={'cod_mpio': str})
    base['nivel_afectacion_cat'] = pd.Categorical(base['nivel_afectacion_cat'], ORD_OBJ, ordered=True)
    for c in base.select_dtypes(include=['object', 'string']):
        base[c] = base[c].str.strip()
    base['mpio'] = nombre_mpio(base['municipio']) + ' (' + base['departamento'].str.title() + ')'
    mun = mun.merge(base[['cod_mpio', 'mpio']].drop_duplicates('cod_mpio'), on='cod_mpio', how='left')
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

def filtro_modelos(t):
    """Aplica los filtros de la barra lateral a la base de modelación."""
    t = t[t['departamento'].isin(deps) & t['tipo_entidad'].str.strip().isin(tipos)]
    return t[t['departamento'] != 'RISARALDA'] if sin_ris else t



def pagina_anexo():
    ejecutivo.encabezado('Anexo · Laboratorio 2', 'Exploración de datos e hipótesis de asociación',
                         'Lo que ya socializamos en el Lab 2: descriptivas, correlación, independencia, ANOVA frente a Kruskal-Wallis y el foco en el Chocó. '
                         'Base de 408 entidades con IPM del DANE; resultados redondeados a 4 decimales.')
    a1, a2, a3, a4, a5, a6 = st.tabs(['Resumen', 'Univariado', 'Correlación', 'Independencia (χ²)', 'ANOVA vs Kruskal-Wallis', 'Foco Chocó'])
    # ---------------- Resumen ----------------
    with a1:
        c1, c2, c3, c4 = st.columns(4)
        pct_critico = (df['nivel_afectacion_cat'] == 'Crítico').mean() * 100
        pct_sinconec = (df['estado_conectividad'] == 'Sin conectividad.').mean() * 100
        c1.metric('Entidades', f'{len(df)}')
        c2.metric('En nivel Crítico', f'{pct_critico:.1f} %')
        c3.metric('Sin conectividad', f'{pct_sinconec:.1f} %')
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
    with a2:
        v = st.selectbox('Variable cuantitativa', list(NUM), format_func=NUM.get, key='uni')
        s = df[v].dropna()
        a, b = st.columns([1, 2])
        a.dataframe(f4(medidas(s)), width='stretch', height=700)
        with b:
            log = st.checkbox('Escala logarítmica en el eje x', value=(v == 'duracion_min'))
            fig = histograma(df, v, log)
            st.plotly_chart(fig, width='stretch')
            st.caption('Línea naranja: media · línea punteada: mediana · Pase el mouse por una barra o un punto para ver de qué municipios son esas entidades.')
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
    with a3:
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
                         labels={x: nombres[x], y: nombres[y]}, title=f'{nombres[y]} vs {nombres[x]}', opacity=.8,
                         hover_name='mpio')
        c1.plotly_chart(fig, width='stretch')
        with c2:
            st.dataframe(f4(correlaciones(datos[x], datos[y])), width='stretch')
            st.markdown('**Cómo leerlo:** Pearson mide relación **lineal** y es sensible a atípicos; Spearman y Kendall usan **rangos** y '
                        'detectan relaciones monótonas aunque no sean lineales. Si Pearson y Spearman difieren mucho, revise atípicos o no linealidad.')

    # ---------------- Independencia ----------------
    with a4:
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
    with a5:
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
    with a6:
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



# ---------------- Páginas del Lab 3 ----------------
_m = modelos.disponible()


def pagina_pipeline():
    ejecutivo.encabezado('Laboratorio 3 · Construcción de pipelines', 'Cómo se preparan los datos antes de predecir',
                         'El modelo no lee la encuesta tal cual: una línea de ensamble rellena vacíos, acorta colas, pone todo en la misma escala y '
                         'traduce las palabras a números. Todo se aprende solo con las entidades de entrenamiento.')
    if _m:
        modelos.pestana_pipeline(_m[3], _m[0], _m[1], _m[2], filtro_modelos)


def pagina_lineal():
    ejecutivo.encabezado('Laboratorio 3 · Modelo de regresión', 'Regresión lineal: fragilidad digital esperada',
                         'El índice de vulnerabilidad (0 a 5) sube 0,0118 por cada punto de pobreza del municipio y 1,11 en los colegios. '
                         'El modelo explica ~23 % de la variabilidad (R² en validación cruzada) con un error medio de 0,71 puntos.')
    if _m:
        modelos.pestana_regresion(_m[3], _m[0], _m[2], filtro_modelos)


def pagina_logistica():
    ejecutivo.encabezado('Laboratorio 3 · Modelo de clasificación', 'Regresión logística: ¿afectación Alta o Crítica?',
                         'En prueba acierta el 90 % (F1 0,90; AUC 0,97). Active "Excluir Risaralda" en la barra lateral para ver la limitación: '
                         'fuera de ese bloque conviene bajar el umbral.')
    if _m:
        modelos.pestana_clasificacion(_m[3], _m[1], _m[2], filtro_modelos)


def pagina_priorizacion():
    if _m:
        ejecutivo.pagina_priorizacion(_m[3], _m[0], _m[1], _m[2], filtro_modelos)


ejecutivo.estilo()
nav = st.navigation({
    'Laboratorio 3 · Modelación': [
        st.Page(ejecutivo.pagina_resumen, title='Resumen ejecutivo', icon=':material/dashboard:', url_path='resumen', default=True),
        st.Page(pagina_pipeline, title='Pipeline', icon=':material/conveyor_belt:', url_path='pipeline'),
        st.Page(pagina_lineal, title='Regresión lineal', icon=':material/trending_up:', url_path='regresion-lineal'),
        st.Page(pagina_logistica, title='Regresión logística', icon=':material/rule:', url_path='regresion-logistica'),
        st.Page(pagina_priorizacion, title='Priorización y conclusiones', icon=':material/flag:', url_path='priorizacion'),
    ],
    'Anexo': [st.Page(pagina_anexo, title='Exploración (Lab 2)', icon=':material/query_stats:', url_path='anexo-lab2')],
})
nav.run()
