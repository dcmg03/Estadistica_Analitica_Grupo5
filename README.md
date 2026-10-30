# Afectación digital de entidades públicas tras el terremoto del 10 de agosto de 2026

**Estadística para Analítica de Datos: laboratorios del curso · Grupo 5**
Especialización en Analítica Estratégica de Datos · Universidad Pedagógica y Tecnológica de Colombia (UPTC)
Profesor: Duván Cataño

Equipo: Edwart Stevens González Ríos · Karen Julieth Hernández Chaparro · Deisy Carolina Monroy Gutiérrez

Analizamos la encuesta de MinTIC a 409 entidades públicas afectadas por el sismo, cruzada con el índice de pobreza
multidimensional municipal del DANE, con foco en el Chocó. El objetivo de modelación es clasificar el nivel de afectación
digital de cada entidad.

## Estructura del proyecto

```
Estadistica_Analitica_Grupo5/
├── notebooks/
│   ├── 01_Lab1_EDA_Encuesta_Terremoto.ipynb      ← Lab 1: exploración, calidad de datos, objetivo, cruce con el DANE
│   └── 02_Lab2_Estadistica_Asociacion.ipynb      ← Lab 2: descriptivas, forma, correlación, χ², ANOVA/Kruskal-Wallis
├── app/
│   ├── app.py                ← dashboard en Streamlit (entregable del Lab 2)
│   └── estadistica.py        ← funciones estadísticas del dashboard
├── data/
│   ├── raw/                  ← datos originales: NO se editan
│   │   ├── Resultados_encuesta_entidades_Terremoto_20261022.csv   (MinTIC, datos.gov.co)
│   │   └── anexo-censal-pobreza-municipal-2018.xlsx               (DANE, IPM municipal 2018)
│   └── processed/            ← generados por los notebooks (se pueden borrar y regenerar)
│       ├── base_lab1_preparada.csv, base_lab1_con_ipm.csv, diccionario_variables.csv   (salen del Lab 1)
│       └── base_lab2.csv, municipios_lab2.csv                                           (salen del Lab 2)
├── docs/
│   └── lab1/                 ← entregables del Lab 1: presentación, propuesta APA, guion, diccionario
├── .vscode/                  ← extensiones recomendadas y lanzador del dashboard
├── .streamlit/config.toml    ← tema del dashboard
└── requirements.txt
```

Flujo de los datos: `data/raw` → **notebook 01** → `data/processed` (bases del Lab 1) → **notebook 02** → `data/processed`
(bases del Lab 2) → **app**. El Lab 2 no repite la limpieza del Lab 1: lee lo que este exporta.

## Cómo ejecutarlo en Visual Studio Code

1. Clonar el repositorio (o abrir la carpeta) en VS Code y aceptar las extensiones recomendadas (Python y Jupyter).
2. Crear y activar el entorno virtual en la terminal de VS Code:
   - Windows: `python -m venv .venv` y luego `.venv\Scripts\activate`
   - macOS / Linux: `python3 -m venv .venv` y luego `source .venv/bin/activate`
3. Instalar dependencias: `pip install -r requirements.txt`
4. Abrir `notebooks/01_Lab1_EDA_Encuesta_Terremoto.ipynb`, elegir el kernel `.venv` y ejecutar **Run All**.
5. Abrir `notebooks/02_Lab2_Estadistica_Asociacion.ipynb` y ejecutar **Run All**.
6. Dashboard: desde la carpeta del proyecto, `streamlit run app/app.py`
   (o *Run and Debug → "Streamlit: dashboard del laboratorio"*). Abre en `http://localhost:8501`.

## Cómo trabajamos en equipo con GitHub

1. **Antes de empezar:** `git pull` para traer lo último de tus compañeros.
2. **Una rama por tarea:** `git checkout -b karen/correlaciones` (nombre/tema).
3. **Un notebook, una persona a la vez.** Los `.ipynb` son archivos JSON grandes y los conflictos son difíciles de
   resolver. Avisen en el grupo antes de editar un notebook.
4. **Antes de hacer commit de un notebook:** *Restart* del kernel y *Run All*, para que las salidas correspondan al código.
5. Guardar cambios: `git add .` → `git commit -m "Lab 2: agrega Kendall a la sección L2-4"` → `git push`.
6. Abrir un *Pull Request* en GitHub para que otra persona lo revise antes de unirlo a `main`.
7. **No modificar `data/raw/`.** Todo lo derivado va a `data/processed/` y se regenera con los notebooks.

## Fuentes

- Ministerio de Tecnologías de la Información y las Comunicaciones. (2026). *Resultados encuesta entidades Terremoto*
  [Conjunto de datos]. Portal Nacional de Datos Abiertos. Licencia CC BY-SA 4.0.
  https://www.datos.gov.co/Ciencia-Tecnolog-a-e-Innovaci-n/Resultados-encuesta-entidades-Terremoto/v3tn-7h59/about_data
- Departamento Administrativo Nacional de Estadística. (2020). *Medida de pobreza multidimensional municipal de fuente
  censal 2018: Anexos* [Conjunto de datos].
  https://www.dane.gov.co/files/investigaciones/condiciones_vida/pobreza/2018/informacion-censal/anexo-censal-pobreza-municipal-2018.xlsx
