import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.figure_factory as ff

st.set_page_config(page_title="Fraude CB – EDA", layout="wide")

@st.cache_data
def load_df(path: str, sample: int | None):
    df = pd.read_csv(path)
    keep = [c for c in df.columns if c in ['Time','Amount','Class'] or c.startswith('V')]
    df = df[keep]
    if sample and sample < len(df):
        df = df.sample(sample, random_state=42).reset_index(drop=True)
    return df

st.title("Analyse des transactions")

default_path = "creditcard.csv"
uploaded = st.file_uploader("Choisir un fichier CSV", type=["csv"])
path = default_path if (uploaded is None and os.path.exists(default_path)) else None

if uploaded is not None:
    df = load_df(uploaded, None)
elif path:
    st.caption(f"Lecture : `{default_path}`")
    sample_n = st.sidebar.number_input("Taille d’échantillon", 1000, 200000, 30000, step=1000)
    df = load_df(default_path, sample_n)
else:
    st.warning("Aucun fichier trouvé.")
    st.stop()

# métriques
c0, c1, c2 = st.columns(3)
c0.metric("Transactions", f"{len(df):,}")
fraud = df['Class'].sum()
perc = 100 * fraud / len(df)
c1.metric("Fraudes", f"{int(fraud):,}", f"{perc:.3f}%")
c2.metric("Montant moyen (€)", f"{df['Amount'].mean():,.2f}")

st.divider()

# settings
with st.sidebar:
    st.header("Contrôles")
    bins = st.slider("Nombre de bacs", 20, 150, 50)
    log_y = st.checkbox("Échelle Y logarithmique", value=False)
    log_amount = st.checkbox("Log(Amount+1)", value=False) #réduit l'effet des valeurs extrêmes et rend la visualisation plus facile
    show_kde = st.checkbox("Courbe KDE", value=True)

df_plot = df.copy()
if log_amount:
    df_plot['Amount_log1p'] = np.log1p(df_plot['Amount'])

# 1. Répartition des classes
st.subheader("1- Répartition des classes")
classes = df_plot['Class'].value_counts().rename_axis("Classe").reset_index(name="Nombre")
fig1 = px.bar(classes, x="Classe", y="Nombre", text="Nombre",
              color="Classe", color_discrete_map={0: "skyblue", 1: "crimson"},
              title="0 = normale | 1 = fraude")
if log_y:
    fig1.update_layout(yaxis_type="log")
fig1.update_traces(textposition="outside")
st.plotly_chart(fig1, use_container_width=True)

# 2. Distribution du montant
st.subheader("2- Distribution du montant des transactions")
amt_col = "Amount_log1p" if log_amount else "Amount"
fig2 = px.histogram(df_plot, x=amt_col, color=df_plot['Class'],
                    nbins=bins, barmode="overlay",
                    color_discrete_map={0: "skyblue", 1: "crimson"},
                    labels={amt_col: "Montant (log1p)" if log_amount else "Montant (€)"},
                    title="Montants par classe")
if show_kde:
    try:
        groups = [df_plot[df_plot['Class'] == k][amt_col] for k in sorted(df_plot['Class'].unique())]
        names = [f"Classe {int(k)}" for k in sorted(df_plot['Class'].unique())]
        kde = ff.create_distplot(groups, names, show_hist=False)
        for trace in kde.data:
            fig2.add_trace(trace)
    except:
        pass
st.plotly_chart(fig2, use_container_width=True)

# 3. Distribution temporelle
st.subheader("3- Distribution temporelle")
fig3 = px.histogram(df_plot, x="Time", nbins=bins,
                    title="Transactions au fil du temps",
                    color=df_plot['Class'],
                    color_discrete_map={0: "skyblue", 1: "crimson"})
st.plotly_chart(fig3, use_container_width=True)
