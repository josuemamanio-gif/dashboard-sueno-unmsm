"""
app.py - Dashboard de sueño y espacios (11 dimensiones)
Ejecutar con: streamlit run app.py
"""
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import statsmodels.api as sm

st.set_page_config(page_title="Sueño UNMSM - CCSS", layout="wide")

INDICES = [
    "D1_duracion", "D2_calidad", "D3_latencia", "D4_disfuncion",
    "D5_somnolencia", "D6_higiene", "D7_academico",
    "D8_disponibilidad", "D9_confort", "D10_uso_demanda", "D11_bienestar_estres",
]
NICE = {
    "D1_duracion": "D1. Duración (h)",
    "D2_calidad": "D2. Calidad",
    "D3_latencia": "D3. Latencia",
    "D4_disfuncion": "D4. Disfunción diurna",
    "D5_somnolencia": "D5. Somnolencia",
    "D6_higiene": "D6. Higiene",
    "D7_academico": "D7. Académico",
    "D8_disponibilidad": "D8. Disponibilidad",
    "D9_confort": "D9. Confort",
    "D10_uso_demanda": "D10. Uso/demanda",
    "D11_bienestar_estres": "D11. Bienestar/estrés",
}

@st.cache_data
def load_data():
    return pd.read_excel("datos_limpios.xlsx")

@st.cache_data
def cronbach_alpha(items):
    items = items.dropna()
    k = items.shape[1]
    if k < 2 or len(items) < 3: return float("nan")
    var_items = items.var(axis=0, ddof=1).sum()
    var_total = items.sum(axis=1).var(ddof=1)
    if var_total == 0: return float("nan")
    return (k / (k - 1)) * (1 - var_items / var_total)

@st.cache_data
@st.cache_data
def construir_items(df):
    def to_num(serie, mapping=None):
        s = serie.astype(str).str.strip().str.lower()
        num = pd.to_numeric(s, errors="coerce")
        if num.isna().any():
            extracted = s.str.extract(r'^\s*(\d+(?:\.\d+)?)\s*=', expand=False)
            num = num.fillna(pd.to_numeric(extracted, errors="coerce"))
        if mapping is not None and num.isna().any():
            num = num.fillna(s.map(mapping))
        return num

    def block(start, end, mapping=None):
        cols = [to_num(df.iloc[:, i], mapping) for i in range(start, end)]
        return pd.concat(cols, axis=1, ignore_index=True)

    MAP_05  = {"nunca": 0, "rara vez": 1, "a veces": 2, "casi siempre": 3, "siempre": 4}
    MAP_EPW = {"nunca": 0, "ligera": 1, "moderada": 2, "alta": 3}
    MAP_DIF = {"ninguna": 0, "leve": 1, "moderada": 2, "severa": 3}
    MAP_15  = {
        "nunca": 1, "casi nunca": 2, "a veces": 3, "casi siempre": 4, "siempre": 5,
        "totalmente en desacuerdo": 1, "en desacuerdo": 2,
        "ni de acuerdo ni en desacuerdo": 3, "de acuerdo": 4,
        "totalmente de acuerdo": 5,
    }
    MAP_CAL = {"muy buena": 1, "buena": 2, "regular": 3, "mala": 4, "muy mala": 5}

    d = {}
    d["D2 Calidad"]           = block(6, 7, MAP_CAL)
    d["D3 Latencia"]          = block(7, 9)
    d["D4 Disfunción"]        = block(9, 11, MAP_DIF)
    d["D5 Somnolencia"]       = block(11, 19, MAP_EPW)
    d["D6 Higiene"]           = block(19, 24, MAP_05)
    d["D7 Académico"]         = block(24, 29, MAP_05)
    d["D8 Disponibilidad"]    = 6 - block(29, 34, MAP_15)
    d["D9 Confort"]           = 6 - block(34, 39, MAP_15)
    d["D10 Uso/demanda"]      = block(39, 44, MAP_15)
    d11 = block(44, 49, MAP_15)
    d11.iloc[:, 3] = 6 - d11.iloc[:, 3]
    d["D11 Bienestar/estrés"] = d11
    return d

try:
    df = load_data()
except FileNotFoundError:
    st.error("No existe datos_limpios.xlsx. Ejecuta primero: python corregir_excel.py")
    st.stop()

if "h_ac_corr" not in df.columns:
    st.error("datos_limpios.xlsx es de una versión vieja. Vuelve a ejecutar corregir_excel.py.")
    st.stop()

h_franja = df["h_ac_corr"].copy()
h_franja = h_franja.where(h_franja >= 6, h_franja + 24)  # 0.5 → 24.5; 2.0 → 26.0
df["franja"] = pd.cut(h_franja,
                     bins=[-1, 23, 24, 27, 30],
                     labels=["Antes 23h", "23-24h", "00-03h", "Después 03h"])

st.sidebar.title("Filtros")
franjas = st.sidebar.multiselect(
    "Franja horaria de acostarse",
    options=df["franja"].dropna().unique().tolist(),
    default=df["franja"].dropna().unique().tolist(),
)
excl_out = st.sidebar.checkbox("Excluir outliers", value=False)

dff = df[df["franja"].isin(franjas)].copy()
if excl_out and "outlier_flag" in dff.columns:
    dff = dff[~dff["outlier_flag"].fillna(False)]

if dff.empty:
    st.sidebar.error("Los filtros no dejan ningún caso. Marca al menos una franja.")
    st.stop()

st.sidebar.markdown(f"**Muestra:** {len(dff)} de {len(df)} casos")
st.sidebar.caption("n pequeño: interpretar con cautela.")

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
    ["Resumen", "Descriptivos", "Fiabilidad", "Correlaciones", "Modelo", "Datos", "Árboles Lógicos"]
)

with tab1:
    st.header("Resumen ejecutivo")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Casos", len(dff))
    c2.metric("Horas sueño", f"{dff['D1_duracion'].mean():.1f} h")
    c3.metric("% sueño <7 h", f"{(dff['D1_duracion'] < 7).mean()*100:.0f}%")
    c4.metric("Epworth mediana", f"{dff['D5_somnolencia'].median():.1f}/24")

    fig = px.histogram(dff.dropna(subset=["D1_duracion"]),
                       x="D1_duracion", nbins=15,
                       labels={"D1_duracion": "Horas de sueño entre semana"})
    fig.add_vline(x=7, line_dash="dash", line_color="red", annotation_text="7 h")
    st.plotly_chart(fig, use_container_width=True)

    fig2 = px.box(dff.dropna(subset=["D1_duracion", "franja"]),
                  x="franja", y="D1_duracion",
                  labels={"franja": "Franja", "D1_duracion": "Horas"})
    st.plotly_chart(fig2, use_container_width=True)

    fig3 = px.histogram(dff.dropna(subset=["D5_somnolencia"]),
                        x="D5_somnolencia", nbins=24,
                        labels={"D5_somnolencia": "Epworth (0-24)"})
    fig3.add_vline(x=10, line_dash="dash", line_color="red",
                   annotation_text=">10 excesiva")
    st.plotly_chart(fig3, use_container_width=True)

with tab2:
    st.header("Descriptivos de las 11 dimensiones")
    cols_present = [c for c in INDICES if c in dff.columns]
    desc = dff[cols_present].describe().T
    desc["mediana"] = dff[cols_present].median()
    desc = desc[["mean", "std", "mediana", "min", "max"]]
    desc.columns = ["Media", "DE", "Mediana", "Mín", "Máx"]
    desc.index = [NICE.get(c, c) for c in desc.index]
    st.dataframe(desc.round(2), use_container_width=True)

    if "eficiencia_sueno" in dff.columns:
        st.subheader("Indicador derivado: eficiencia del sueño")
        eff = dff["eficiencia_sueno"].describe()
        st.write(f"Eficiencia media: {eff['mean']:.1f} % (mediana {eff['50%']:.1f} %, "
                 f"rango {eff['min']:.1f} – {eff['max']:.1f} %)")
        st.caption("Eficiencia = horas dormidas / (horas dormidas + horas en cama sin dormir) × 100")

    long = dff[cols_present].melt(var_name="Dimensión", value_name="Valor")
    long["Dimensión"] = long["Dimensión"].map(NICE)
    fig = px.box(long.dropna(), x="Dimensión", y="Valor", color="Dimensión")
    fig.update_layout(showlegend=False, xaxis_tickangle=-40, height=500)
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.header("Fiabilidad (alfa de Cronbach)")
    items = construir_items(df)
    filas = []
    for dim, mat in items.items():
        a = cronbach_alpha(mat)
        interp = "-"
        if not np.isnan(a):
            interp = ("Excelente" if a >= 0.90 else "Buena" if a >= 0.80 else
                      "Aceptable" if a >= 0.70 else "Cuestionable" if a >= 0.60 else "Baja")
        filas.append({"Dimensión": dim, "Ítems": mat.shape[1],
                      "Alfa": round(a, 3) if not np.isnan(a) else None,
                      "Interpretación": interp})
    st.dataframe(pd.DataFrame(filas), use_container_width=True)
    st.caption("D1 (Duración) y la Eficiencia no se evalúan con alfa (son derivadas).")

with tab4:
    st.header("Correlaciones de Spearman")
    corr = dff[cols_present].corr(method="spearman")
    corr.index   = [NICE.get(c, c) for c in corr.index]
    corr.columns = [NICE.get(c, c) for c in corr.columns]
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                    zmin=-1, zmax=1, aspect="auto")
    fig.update_layout(height=650)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("En D8/D9 un puntaje alto significa menor cobertura (más déficit).")

with tab5:
    st.header("Regresión lineal (exploratorio)")
    y_col = st.selectbox("Variable dependiente", cols_present,
                         index=cols_present.index("D11_bienestar_estres") if "D11_bienestar_estres" in cols_present else 0,
                         format_func=lambda x: NICE.get(x, x))
    xs = [c for c in cols_present if c != y_col]
    x_cols = st.multiselect("Predictores (máx 3)", xs, default=xs[:3],
                            max_selections=3, format_func=lambda x: NICE.get(x, x))

    if x_cols:
        datos = dff[[y_col] + x_cols].dropna()
        n_ok = len(datos)
        if n_ok <= len(x_cols) + 2:
            st.warning(f"Solo hay {n_ok} casos completos. Se necesitan al menos "
                       f"{len(x_cols) + 3}. Prueba otros predictores.")
        else:
            X = sm.add_constant(datos[x_cols])
            modelo = sm.OLS(datos[y_col], X).fit()
            coefs = pd.DataFrame({"Coef.": modelo.params,
                                  "EE": modelo.bse,
                                  "p": modelo.pvalues}).round(4)
            st.dataframe(coefs, use_container_width=True)
            c1, c2, c3 = st.columns(3)
            c1.metric("R²", f"{modelo.rsquared:.3f}")
            c2.metric("R² ajustado", f"{modelo.rsquared_adj:.3f}")
            c3.metric("n", n_ok)
    else:
        st.info("Elige al menos un predictor.")

with tab6:
    st.header("Datos limpios")
    st.dataframe(dff, use_container_width=True, height=500)
    st.download_button("Descargar CSV",
                       dff.to_csv(index=False).encode("utf-8"),
                       "datos_filtrados.csv", "text/csv")

with tab7:
    st.header("Metodología de Marco Lógico (MML)")
    st.write("Representación gráfica del diagnóstico y objetivos de la investigación, basados en los datos de la encuesta.")

    # --- ÁRBOL DE PROBLEMAS ---
    st.subheader("1. Árbol de Problemas")
    st.write("Refleja la situación negativa actual: el problema central, sus causas (raíces) y sus efectos (ramas).")
    
    arbol_problemas = """
    digraph Problemas {
        rankdir=BT; // De abajo (causas) hacia arriba (efectos)
        bgcolor="transparent";
        node [shape=box, style="filled,rounded", color="gray", fontname="Arial", fontsize=12];
        edge [color="gray40"];
        
        // Problema Central
        P [label="Deficiente calidad del sueño y altos niveles\\nde estrés académico en estudiantes CCSS", fillcolor="#ffb3b3"];
        
        // Causas (Parte de abajo)
        C1 [label="Inadecuada disponibilidad y\\nconfort de espacios (D8, D9)", fillcolor="#f0f0f0"];
        C2 [label="Prácticas deficientes de\\nhigiene del sueño (D6)", fillcolor="#f0f0f0"];
        C3 [label="Alta carga académica y\\nhorarios desfavorables (D7)", fillcolor="#f0f0f0"];
        
        // Efectos (Parte de arriba)
        E1 [label="Somnolencia diurna excesiva\\ny disfunción (D4, D5)", fillcolor="#d9e6f2"];
        E2 [label="Fatiga, agotamiento y bajo\\nrendimiento (D11)", fillcolor="#d9e6f2"];
        E3 [label="Reducción de horas de sueño\\ny latencia prolongada (D1, D3)", fillcolor="#d9e6f2"];
        
        // Relaciones
        C1 -> P;
        C2 -> P;
        C3 -> P;
        P -> E1;
        P -> E2;
        P -> E3;
    }
    """
    st.graphviz_chart(arbol_problemas)

    st.divider()

    # --- ÁRBOL DE OBJETIVOS ---
    st.subheader("2. Árbol de Objetivos")
    st.write("Refleja la situación futura deseada (el 'espejo' positivo del árbol de problemas).")
    
    arbol_objetivos = """
    digraph Objetivos {
        rankdir=BT; 
        bgcolor="transparent";
        node [shape=box, style="filled,rounded", color="gray", fontname="Arial", fontsize=12];
        edge [color="gray40"];
        
        // Objetivo Central (Propósito)
        O [label="Mejorar la calidad del sueño y reducir\\nel estrés académico en estudiantes CCSS", fillcolor="#b3ffb3"];
        
        // Medios (Parte de abajo)
        M1 [label="Adecuada disponibilidad y\\nconfort de espacios (D8, D9)", fillcolor="#f0f0f0"];
        M2 [label="Promoción de una buena\\nhigiene del sueño (D6)", fillcolor="#f0f0f0"];
        M3 [label="Gestión eficiente de la carga\\nacadémica (D7)", fillcolor="#f0f0f0"];
        
        // Fines (Parte de arriba)
        F1 [label="Niveles óptimos de alerta\\n(disminución D4, D5)", fillcolor="#d9e6f2"];
        F2 [label="Mejor rendimiento y reducción\\ndel estrés (D11)", fillcolor="#d9e6f2"];
        F3 [label="Horas de sueño efectivas\\ny menor latencia (D1, D3)", fillcolor="#d9e6f2"];
        
        // Relaciones
        M1 -> O;
        M2 -> O;
        M3 -> O;
        O -> F1;
        O -> F2;
        O -> F3;
    }
    """
    st.graphviz_chart(arbol_objetivos)