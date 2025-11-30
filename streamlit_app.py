# streamlit_app.py
# Streamlit app: MMR Offshore Decision Support (AHP + SAW + MonteCarlo + FPSO compatibility)
# Author: Jullyano Lino
# Run: streamlit run streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from scipy import linalg
import io
import base64

st.set_page_config(layout="wide", page_title="MMR Offshore Selector", initial_sidebar_state="expanded")

# ---------------------------
# Constants / Defaults
# ---------------------------
MMRS = ['Oklo Aurora', 'U-Battery', 'eVinci', 'StarCore', 'HolosGen', 'USNC MMR', 'Kronos (PWR)', 'XAMR (HTGR)']
DEFAULT_CRITERIA = ['Segurança', 'Custo-benefício', 'Volume compacto', 'Potência', 'Regulação', 'Massa total', 'CG sensitivity', 'Shield mass', 'Heat rejection', 'LCOE']

# ---------------------------
# Utils
# ---------------------------
def read_dataset(path="mmr_data.csv"):
    """Read reactor dataset. Expect columns:
    name, mmr, CAPEX, OPEX, P_rated_MW, CF, mass_kg, footprint_m2, shield_mass_kg, cg_m_from_ref, inertia_kgm2,
    heat_rejection_kW, fuel_type, ref_source (URL), last_update (YYYY-MM-DD)

    If the file is not present, return a template DataFrame with one row per MMRS.
    """
    import os
    cols = ["name","mmr","CAPEX","OPEX","P_rated_MW","CF","mass_kg","footprint_m2","shield_mass_kg",
            "cg_m_from_ref","inertia_kgm2","heat_rejection_kW","fuel_type","ref_source","last_update"]

    if os.path.isfile(path):
        try:
            df = pd.read_csv(path)
            # ensure all expected columns exist
            for c in cols:
                if c not in df.columns:
                    df[c] = np.nan
            return df[cols]
        except Exception as e:
            st.warning(f"Erro ao ler '{path}': {e}. Gerando template vazio.")
    # create template skeleton
    rows = []
    for m in MMRS:
        rows.append({
            "name": m,
            "mmr": m,
            "CAPEX": np.nan,
            "OPEX": np.nan,
            "P_rated_MW": np.nan,
            "CF": np.nan,
            "mass_kg": np.nan,
            "footprint_m2": np.nan,
            "shield_mass_kg": np.nan,
            "cg_m_from_ref": np.nan,
            "inertia_kgm2": np.nan,
            "heat_rejection_kW": np.nan,
            "fuel_type": "",
            "ref_source": "",
            "last_update": ""
        })
    df = pd.DataFrame(rows, columns=cols)
    return df



def save_df_to_csv(df, filename="mmr_data_out.csv"):
    df.to_csv(filename, index=False)
    return filename

def normalize_series(s, method="minmax", benefit=True):
    # benefit=True means higher is better; if false, invert after normalization
    if method == "minmax":
        mn, mx = np.nanmin(s), np.nanmax(s)
        if np.isclose(mx, mn):
            res = np.zeros_like(s, dtype=float)
        else:
            res = (s - mn) / (mx - mn)
    elif method == "zscore":
        mu = np.nanmean(s)
        sigma = np.nanstd(s)
        if np.isclose(sigma, 0):
            res = np.zeros_like(s, dtype=float)
        else:
            res = (s - mu) / sigma
            # put into 0..1 range
            res = (res - np.nanmin(res)) / (np.nanmax(res) - np.nanmin(res) + 1e-12)
    else:
        # identity
        res = (s - np.nanmin(s)) / (np.nanmax(s) - np.nanmin(s) + 1e-12)
    if not benefit:
        res = 1.0 - res
    return res

def compute_lcoe(capex, opex, prated_mw, cf, r=0.07, n=20):
    # CAPEX [$], OPEX [$/yr], prated_mw [MW], CF [0-1]
    if np.isnan(capex) or np.isnan(prated_mw) or np.isnan(cf) or prated_mw <=0 or cf<=0:
        return np.nan
    crf = (r*(1+r)**n)/((1+r)**n - 1)
    annual_energy = prated_mw * 1e3 * cf * 8760.0  # MWh -> kWh? keep as kWh; we'll output $/MWh later
    lcoe = (capex * crf + opex) / (annual_energy/1000.0)  # $/MWh
    return lcoe

def ahp_weights_from_pairwise(A):
    # A: numpy array (n,n) pairwise comparison matrix
    vals, vecs = np.linalg.eig(A)
    max_idx = np.argmax(vals.real)
    w = vecs[:, max_idx].real
    w = np.abs(w)
    w = w / np.sum(w)
    return w

def create_pairwise_matrix(n, ui_values=None):
    # ui_values: upper triangular entries from user in matrix form or None
    # default identity
    A = np.ones((n,n))
    if ui_values is not None:
        # ui_values a dict with (i,j): val for i<j
        for (i,j),v in ui_values.items():
            A[i,j] = v
            A[j,i] = 1.0/v if v !=0 else 1.0
    return A

def montecarlo_scores(scores_matrix, weights, n_runs=2000, noise_scale=0.10):
    # scores_matrix: reactors x criteria scaled 0..1
    n_reactors, n_criteria = scores_matrix.shape
    results = np.zeros((n_runs, n_reactors))
    for k in range(n_runs):
        noise = np.random.normal(1.0, noise_scale, scores_matrix.shape)
        noisy = np.clip(scores_matrix * noise, 0.0, 1.0)
        results[k,:] = noisy.dot(weights)
    return results

def tornado_sensitivity(base_weights, scores_matrix, delta=0.10):
    # For each criterion vary weight +/- delta (redistribute remaining proportionally) and compute winner change
    n = len(base_weights)
    base_scores = scores_matrix.dot(base_weights)
    base_rank = np.argsort(-base_scores)
    impacts = []
    for i in range(n):
        w_up = base_weights.copy()
        w_down = base_weights.copy()
        # add delta proportion of total to criterion i and remove proportionally from others
        add = delta * base_weights.sum()
        # simpler: scale i by (1+delta) then renormalize
        w_up[i] *= (1+delta)
        w_up = w_up / w_up.sum()
        w_down[i] *= (1-delta)
        w_down = w_down / w_down.sum()
        score_up = scores_matrix.dot(w_up)
        score_down = scores_matrix.dot(w_down)
        impacts.append({
            "criterion_index": i,
            "score_change_up": (score_up - base_scores).mean(),
            "score_change_down": (score_down - base_scores).mean(),
            "max_delta": max(np.abs(score_up - base_scores).max(), np.abs(score_down - base_scores).max())
        })
    return impacts

# ---------------------------
# Load Data
# ---------------------------
st.sidebar.title("MMR Offshore Selector")
st.sidebar.markdown("Carregue um CSV com dados dos MMRs (ex.: `mmr_data.csv`) com colunas técnicas. Se não, um template será criado automaticamente.")

uploaded = st.sidebar.file_uploader("Upload CSV de MMRs (opcional)", type=["csv"])
if uploaded:
    df_mmr = pd.read_csv(uploaded)
else:
    df_mmr = read_dataset()  # placeholder skeleton

st.sidebar.markdown("### Escolha método de decisão")
method = st.sidebar.selectbox("Método", ["SAW (Weighted Sum)", "AHP (Pairwise)"])

# criteria management
st.sidebar.markdown("### Critérios (revise/adicione)")
criteria = st.sidebar.multiselect("Selecione critérios", options=list(df_mmr.columns)+DEFAULT_CRITERIA, default=DEFAULT_CRITERIA)
if not criteria:
    st.sidebar.error("Selecione ao menos um critério.")
    st.stop()

# allow user to provide numeric vectors for criteria or use computed (e.g., LCOE)
# compute derived fields (LCOE)
if 'CAPEX' in df_mmr.columns and 'OPEX' in df_mmr.columns and 'P_rated_MW' in df_mmr.columns and 'CF' in df_mmr.columns:
    df_mmr['LCOE_$perMWh'] = df_mmr.apply(lambda r: compute_lcoe(r.get('CAPEX', np.nan), r.get('OPEX', np.nan), r.get('P_rated_MW', np.nan), r.get('CF', np.nan)), axis=1)
else:
    df_mmr['LCOE_$perMWh'] = np.nan

# Display dataset and allow editing of key numeric fields inline
st.header("1) Dados dos MMRs (fonte e último update obrigatório)")
st.markdown("Cada reator deve ter `ref_source` (URL/PDF) e `last_update` para garantir rastreabilidade das notas.")

# -------------------------
# Editable dataset UI (compatível com várias versões do Streamlit)
# -------------------------
with st.expander("Ver / editar dados de reatores"):
    # Prefer st.data_editor (API estável nas versões recentes)
    try:
        if hasattr(st, "data_editor"):
            # st.data_editor returns a DataFrame when editable by user
            edited = st.data_editor(df_mmr, num_rows="dynamic")
            df_mmr = pd.DataFrame(edited)
        elif hasattr(st, "experimental_data_editor"):
            # fallback para versões onde a API é experimental
            edited = st.experimental_data_editor(df_mmr, num_rows="dynamic")
            df_mmr = pd.DataFrame(edited)
        else:
            # Fallback manual: criar inputs por célula (menos elegante, mas compatível)
            st.warning("Seu Streamlit não possui data_editor integrado — usando editor manual de células. Recomendo atualizar o Streamlit para a versão mais recente.")
            rows = []
            for i, row in df_mmr.iterrows():
                st.markdown(f"**Reator {i+1}: {row.get('name') or row.get('mmr','(sem nome)')}**")
                new_row = {}
                for col in df_mmr.columns:
                    val = row[col]
                    if pd.api.types.is_numeric_dtype(df_mmr[col]) or (isinstance(val, (int, float, np.floating, np.integer))):
                        new_val = st.number_input(f"{col} (linha {i+1})", value=float(val) if not pd.isna(val) else 0.0, key=f"manual_{i}_{col}")
                    else:
                        new_val = st.text_input(f"{col} (linha {i+1})", value=str(val) if not pd.isna(val) else "", key=f"manual_{i}_{col}")
                    new_row[col] = new_val
                rows.append(new_row)
            # reconstruir df_mmr
            try:
                df_mmr = pd.DataFrame(rows, columns=df_mmr.columns)
            except Exception:
                # se algo falhar ao reconstruir, manter o original
                st.error("Erro ao reconstruir DataFrame a partir do editor manual; mantendo dados originais.")
    except Exception as e:
        st.error(f"Erro ao abrir editor de dados: {e}")
        st.write("Mostrando DataFrame apenas para visualização (não editável).")
        st.dataframe(df_mmr)


# Build the score matrix (reactors x criteria)

# ---------------------------
# Build the score matrix (reactors x criteria)  -- REPLACEMENT ROBUSTO
# ---------------------------

# sanitize obvious string placeholders that came from editors
df_mmr = df_mmr.replace({"None": np.nan, "none": np.nan, "NULL": np.nan, "NoneType": np.nan})

# ensure name column exists
if 'name' not in df_mmr.columns:
    df_mmr['name'] = df_mmr['mmr'].fillna('Unknown')

reactor_names = df_mmr['name'].fillna(df_mmr['mmr']).tolist()
n_reactors = len(reactor_names)
n_criteria = len(criteria)

# prepare numeric matrix; if column exists numeric, use it; else ask user to input subjective scores
scores_matrix = np.zeros((n_reactors, n_criteria))
metric_info = []

st.header("2) Defina notas / métricas por critério (versão robusta)")
cols_input = st.columns(2)

for j, c in enumerate(criteria):
    with cols_input[j % 2]:
        st.subheader(c)

        # If the criterion is exactly 'LCOE' but our df uses 'LCOE_$perMWh', map it:
        df_colname = c
        if c.lower() in ['lcoe', 'lcoe_$permwh', 'lcoe_$permwh']:
            if 'LCOE_$perMWh' in df_mmr.columns:
                df_colname = 'LCOE_$perMWh'
            elif 'LCOE' in df_mmr.columns:
                df_colname = 'LCOE'

        # If column exists in df_mmr, try to coerce to numeric and check
        if df_colname in df_mmr.columns:
            # coerce to numeric (safe), convert 'inf' strings, etc.
            series_raw = df_mmr[df_colname].replace([None, ""], np.nan)
            # try numeric coercion
            series_num = pd.to_numeric(series_raw, errors='coerce').astype(float).values

            # if the entire series is NaN -> ask user for subjective scores instead of leaving NaNs
            if np.all(np.isnan(series_num)):
                st.warning(f"A coluna '{df_colname}' existe mas está vazia. Insira notas subjetivas 0..10 para '{c}'.")
                subj = []
                for i, rname in enumerate(reactor_names):
                    default = 5.0
                    val = st.number_input(f"{rname} — {c} (col '{df_colname}' vazio)", min_value=0.0, max_value=10.0, value=float(default), step=0.5, key=f"subj_{j}_{i}")
                    subj.append(val)
                subj = np.array(subj, dtype=float)
                normed = normalize_series(subj, method="minmax", benefit=True)
                scores_matrix[:, j] = normed
                metric_info.append((c, False, f"subjective_from_empty_column_{df_colname}"))
            else:
                # we have some numeric data -> normalize it
                # ask user whether it's benefit or cost
                benefit = st.checkbox(f"'{c}' é benefício (maior é melhor)?", value=True, key=f"benefit_{j}")
                norm_method = st.selectbox(f"Normalização '{c}'", options=["minmax","zscore"], index=0, key=f"norm_{j}")
                normed = normalize_series(series_num, method=norm_method, benefit=benefit)
                # fill any remaining NaNs in normalized with column mean
                if np.isnan(normed).any():
                    col_mean = np.nanmean(normed)
                    normed = np.where(np.isfinite(normed), normed, col_mean if np.isfinite(col_mean) else 0.0)
                scores_matrix[:, j] = normed
                metric_info.append((c, True, df_colname))
        else:
            # column not present in dataframe: ask for subjective scores 0..10
            st.write(f"Critério '{c}' não encontrado no dataset — insira notas subjetivas (0..10)." )
            subj = []
            for i, rname in enumerate(reactor_names):
                default = 5.0
                val = st.number_input(f"{rname} — {c}", min_value=0.0, max_value=10.0, value=float(default), step=0.5, key=f"{j}_{i}")
                subj.append(val)
            subj = np.array(subj, dtype=float)
            normed = normalize_series(subj, method="minmax", benefit=True)
            scores_matrix[:, j] = normed
            metric_info.append((c, False, "subjective_0_10"))

# Special handling for LCOE if present as a raw column name
if 'LCOE_$perMWh' in df_mmr.columns and 'LCOE_$perMWh' in criteria:
    idx = criteria.index('LCOE_$perMWh')
    series = pd.to_numeric(df_mmr['LCOE_$perMWh'], errors='coerce').astype(float).values
    if np.all(np.isnan(series)):
        st.warning("LCOE presente mas sem valores numéricos; verifique CAPEX/OPEX/P_rated/CF ou preencha manualmente.")
    else:
        scores_matrix[:, idx] = normalize_series(series, method="minmax", benefit=False)
        metric_info[idx] = ('LCOE_$perMWh', True, 'minmax_cost')

# Final sanitation: ensure finite and in 0..1
if not np.isfinite(scores_matrix).all():
    st.warning("Algumas entradas na matriz de notas não eram finitas e foram ajustadas (NaN -> 0.5 onde apropriado)." )
    scores_matrix = np.where(np.isfinite(scores_matrix), scores_matrix, 0.5)

# If matrix is all zeros (happens se tudo faltou), set to 0.5 neutral to avoid blank tornado/zero ranking
if np.allclose(scores_matrix, 0.0):
    st.warning("Todas as notas normalizadas são zero — preenchendo com 0.5 (neutro) para permitir análises exploratórias. Recomendo preencher dados reais.")
    scores_matrix[:] = 0.5

# Show normalized matrix preview
st.subheader("Matriz normalizada (0..1) — prévia")
df_norm_preview = pd.DataFrame(scores_matrix, index=reactor_names, columns=criteria)
st.dataframe(df_norm_preview.style.format("{:.3f}"))

# ---------------------------
# Deterministic ranking (safe)
# ---------------------------
# Ensure weights variable exists (if not yet elicited, make uniform placeholder)
try:
    _ = weights
except NameError:
    weights = np.ones(n_criteria) / float(n_criteria)
    st.info("Pesos ainda não definidos: usando pesos uniformes temporariamente (sugestão: configure SAW ou AHP na seção de pesos)." )

# guard against NaN in weights (e.g., AHP failed)
if not np.isfinite(weights).all() or weights.sum() == 0:
    st.warning("Pesos inválidos detectados; usando pesos uniformes como fallback.")
    weights = np.ones(n_criteria) / float(n_criteria)

# normalize weights strictly
weights = np.array(weights, dtype=float)
if weights.sum() <= 0:
    weights = np.ones_like(weights) / len(weights)
else:
    weights = weights / weights.sum()

scores = scores_matrix.dot(weights)
# guard: if scores contain NaN/inf, replace with -inf for ranking
scores = np.where(np.isfinite(scores), scores, -1e9)

df_scores = pd.DataFrame({
    "Reactor": reactor_names,
    "Score": scores
}).sort_values("Score", ascending=False)

st.subheader("Ranqueamento (determinístico)")
st.dataframe(df_scores.style.format({"Score":"{:.4f}"}))
# Weight elicitation (AHP or SAW)
# ---------------------------
st.header("3) Elicitando pesos (AHP ou SAW)")
if method == "SAW (Weighted Sum)":
    st.markdown("Use sliders para definir pesos e pressione 'Normalizar pesos' para obter soma = 1.")
    weights = []
    w_cols = st.columns(len(criteria))
    for j, c in enumerate(criteria):
        with w_cols[j]:
            w = st.slider(f"Peso — {c}", min_value=0.0, max_value=5.0, value=1.0, step=0.1, key=f"w_{j}")
            weights.append(w)
    weights = np.array(weights, dtype=float)
    if weights.sum() == 0:
        st.warning("A soma dos pesos é zero; pesos iguais serão usados.")
        weights = np.ones_like(weights)
    weights = weights / weights.sum()
    st.write("Pesos normalizados:", {c: float(weights[i]) for i,c in enumerate(criteria)})

else:
    st.markdown("AHP — insira as comparações pareadas (i vs j). Valor >1 indica i > j.")
    n = len(criteria)
    st.write("Forneça valores de comparação para pares (i,j). Ex.: 1,3,5,7,9 escala de Saaty ou frações.")
    # we present a compressed UI: a table upper-triangular inputs
    pair_vals = {}
    for i in range(n):
        for j in range(i+1, n):
            default = 1.0
            val = st.number_input(f"{criteria[i]}  ÷  {criteria[j]}", min_value=1e-6, max_value=1e6, value=float(default), step=0.1, key=f"pair_{i}_{j}")
            pair_vals[(i,j)] = float(val)
    A = create_pairwise_matrix(n, ui_values=pair_vals)
    try:
        w = ahp_weights_from_pairwise(A)
        weights = w
        st.write("Pesos (autovetor) normalizados (AHP):")
        st.write({criteria[i]: float(weights[i]) for i in range(len(criteria))})
        # consistency check (CR) basic: compute lambda_max
        vals, vecs = np.linalg.eig(A)
        lambda_max = max(vals.real)
        n = A.shape[0]
        ci = (lambda_max - n)/(n-1) if n>1 else 0.0
        # Saaty random index (approx for n<=10)
        RI_dict = {1:0.00,2:0.00,3:0.58,4:0.90,5:1.12,6:1.24,7:1.32,8:1.41,9:1.45,10:1.49}
        ri = RI_dict.get(n, 1.49)
        cr = ci/ri if ri>0 else 0.0
        st.write(f"Índice de Consistência (CR) ≈ {cr:.3f}. (CR < 0.10 geralmente aceitável)")
    except Exception as e:
        st.error("Erro ao calcular pesos AHP: " + str(e))
        st.stop()

# ---------------------------
# Deterministic ranking
# ---------------------------
st.header("4) Ranqueamento determinístico e probabilístico")

scores = scores_matrix.dot(weights)
df_scores = pd.DataFrame({
    "Reactor": reactor_names,
    "Score": scores
}).sort_values("Score", ascending=False)
st.subheader("Ranqueamento (determinístico)")
st.dataframe(df_scores.style.format({"Score":"{:.4f}"}))

# Monte Carlo
st.subheader("Análise probabilística (Monte Carlo)")
mc_runs = st.number_input("Monte Carlo runs", min_value=100, max_value=20000, value=2000, step=100)
noise_scale = st.slider("Incerteza nas notas (%)", min_value=0.0, max_value=0.5, value=0.10, step=0.01)
if st.button("Rodar Monte Carlo"):
    with st.spinner("Executando Monte Carlo..."):
        mc_results = montecarlo_scores(scores_matrix, weights, n_runs=mc_runs, noise_scale=noise_scale)
        winners = np.argmax(mc_results, axis=1)
        probs = [(winners==i).mean() for i in range(n_reactors)]
        df_prob = pd.DataFrame({"Reactor":reactor_names, "Prob_top1":probs}).sort_values("Prob_top1", ascending=False)
        st.write("Probabilidade de ser o melhor (Top-1) considerando incerteza nas notas")
        st.dataframe(df_prob.style.format({"Prob_top1":"{:.3%}"}))
        # violin / histogram for top candidate
        top_idx = int(np.argmax(probs))
        st.markdown(f"Distribuição de pontuação do vencedor provável: **{reactor_names[top_idx]}**")
        plt.figure(figsize=(6,3))
        plt.hist(mc_results[:, top_idx], bins=40)
        plt.title(f"Histograma de scores (MC) — {reactor_names[top_idx]}")
        st.pyplot(plt.gcf())
        plt.clf()

# ---------------------------
# Sensitivity (tornado)
# ---------------------------
st.header("5) Análise de sensibilidade (tornado / elasticidade)")
delta_pct = st.slider("Variação para sensibilidade (%)", min_value=1, max_value=50, value=10, step=1)
impacts = tornado_sensitivity(weights, scores_matrix, delta=delta_pct/100.0)
# prepare tornado dataframe
tornado_df = []
for it in impacts:
    idx = it['criterion_index']
    tornado_df.append({
        "criterion": criteria[idx],
        "max_delta": it['max_delta'],
        "mean_change_up": it['score_change_up'],
        "mean_change_down": it['score_change_down']
    })
tornado_df = pd.DataFrame(tornado_df).sort_values("max_delta", ascending=False)
st.subheader("Tornado — critérios mais sensíveis")
st.dataframe(tornado_df.style.format({"max_delta":"{:.6f}", "mean_change_up":"{:.6f}", "mean_change_down":"{:.6f}"}))

# plot tornado bars
fig, ax = plt.subplots(figsize=(8, max(3, len(criteria)*0.4)))
y = tornado_df['criterion']
x = tornado_df['max_delta']
ax.barh(y, x)
ax.set_xlabel("Impacto máximo médio no score")
ax.set_title(f"Tornado (variação ±{delta_pct}%)")
st.pyplot(fig)

# Elasticidade LCOE: if LCOE present
if 'LCOE_$perMWh' in criteria:
    idx_lcoe = criteria.index('LCOE_$perMWh')
    orig = scores_matrix[:, idx_lcoe].copy()
    # increase LCOE by +10% (i.e., worsen)
    # since LCOE already normalized as cost (higher worse), increasing raw LCOE reduces normalized score; but for elasticities we approximate by perturbing the normalized column
    pert = orig * 1.10
    pert = np.clip(pert, 0, 1)
    scores_pert = scores_matrix.copy()
    scores_pert[:, idx_lcoe] = pert
    new_scores = scores_pert.dot(weights)
    elasticity = (new_scores - scores).mean() / (0.10)  # per 10% change
    st.subheader("Elasticidade média da LCOE (variação +10%)")
    st.write(f"Média de mudança no score por +10% LCOE (normalizado): {elasticity:.6f}")

# ---------------------------
# FPSO compatibility module (engineering outputs)
# ---------------------------
st.header("6) Módulo de compatibilidade FPSO (massa, CG, GM, deck capacity checks)")
st.markdown("Forneça propriedades do topside / deck do FPSO para verificação.")

fpsodeck = {}
fpsodeck['deck_capacity_kg'] = st.number_input("Deck allowable distributed mass (kg)", value=2_000_000.0, step=10000.0)
fpsodeck['deck_point_capacity_kg'] = st.number_input("Deck allowable point load (kg)", value=200_000.0, step=1000.0)
fpsodeck['crane_capacity_kg'] = st.number_input("Crane capacity (kg)", value=50_000.0, step=1000.0)
fpsodeck['allowable_cg_shift_m'] = st.number_input("Allowable CG shift (m)", value=0.5, step=0.01)
fpsodeck['metacentric_height_m'] = st.number_input("GM metacentric height baseline (m)", value=1.0, step=0.01)
fpsodeck['deck_area_m2'] = st.number_input("Deck area (m^2)", value=500.0, step=1.0)

# compute compatibility: mass total, including shield
out_rows = []
for i, r in df_mmr.iterrows():
    name = r.get('name') or r.get('mmr')
    mass = float(r.get('mass_kg') if not pd.isna(r.get('mass_kg')) else 0.0)
    shield = float(r.get('shield_mass_kg') if not pd.isna(r.get('shield_mass_kg')) else 0.0)
    total_mass = mass + shield
    cg = float(r.get('cg_m_from_ref') if not pd.isna(r.get('cg_m_from_ref')) else 0.0)
    inertia = float(r.get('inertia_kgm2') if not pd.isna(r.get('inertia_kgm2')) else 0.0)
    # approximate CG shift: assume installation changes CG proportionally to (total_mass / deck_capacity) * cg
    cg_shift = cg * (total_mass / max(1.0, fpsodeck['deck_capacity_kg']))
    # approximate GM new: GM_new = GM_old - delta where delta ~ cg_shift * (total_mass / (1e6)) (rough linear heuristic)
    gm_new = fpsodeck['metacentric_height_m'] - cg_shift * (total_mass / 1e6)
    deck_ok = total_mass <= fpsodeck['deck_capacity_kg']
    point_ok = total_mass <= fpsodeck['deck_point_capacity_kg']
    crane_ok = total_mass <= fpsodeck['crane_capacity_kg']
    out_rows.append({
        "Reactor": name,
        "mass_kg": total_mass,
        "cg_shift_m": cg_shift,
        "GM_new_m_est": gm_new,
        "deck_OK": deck_ok,
        "point_load_OK": point_ok,
        "crane_OK": crane_ok,
        "heat_rejection_kW": float(r.get('heat_rejection_kW') if not pd.isna(r.get('heat_rejection_kW')) else 0.0),
        "notes_sources": r.get('ref_source', "")
    })

df_fspo = pd.DataFrame(out_rows)
st.dataframe(df_fspo.style.format({"mass_kg":"{:.0f}", "cg_shift_m":"{:.4f}", "GM_new_m_est":"{:.4f}"}))

# Flagging critical issues
st.subheader("Flags de incompatibilidade física (automáticas)")
flags = []
for i, row in df_fspo.iterrows():
    issues = []
    if not row['deck_OK']:
        issues.append("Deck capacity exceeded")
    if not row['point_load_OK']:
        issues.append("Point load exceeded")
    if not row['crane_OK']:
        issues.append("Crane cannot lift")
    if row['GM_new_m_est'] < 0.2:
        issues.append("Estimated GM too low -> estabilidade insuficiente")
    flags.append({"Reactor": row['Reactor'], "Issues": "; ".join(issues) if issues else "OK"})
st.dataframe(pd.DataFrame(flags))

# Export engineering CSV for FEA/CAD integration
if st.button("Exportar relatório técnico (CSV) para engenheiros navais"):
    outname = "mmr_fspo_compatibility_report.csv"
    df_fspo.to_csv(outname, index=False)
    st.success(f"Arquivo gerado: {outname}. Faça download a partir do diretório do app.")

# ---------------------------
# Regulatory checklist & licensing scoring
# ---------------------------
st.header("7) Checklist regulatório por jurisdição (CNEN / ANTAQ / IMO / Port Authority)")
st.markdown("Referências oficiais carregadas automaticamente (quando possível). Consulte links de referência no README.")
# Simple scoring model: for each reactor compute 'licenciability_score' based on: known fuel constraints, transport complexity, size, shielding, and documented pre-licensing progress
def compute_licensing_score(row):
    score = 0.0
    # fuel: HALEU increases complexity (penalty)
    fuel = str(row.get('fuel_type','')).lower()
    if 'haleu' in fuel:
        score -= 0.2
    if 'triso' in fuel or 'htgr' in fuel:
        score += 0.1
    # if ref_source present add trust
    if row.get('ref_source'):
        score += 0.2
    # if mass small -> easier
    mass = row.get('mass_kg',0.0)
    if mass < 50000:
        score += 0.2
    elif mass < 200000:
        score += 0.1
    # if shielding massive, penalty
    if row.get('shield_mass_kg',0.0) > 100000:
        score -= 0.2
    # clamp between 0 and 1
    s = min(1.0, max(0.0, 0.5 + score))
    return s

licscores = []
for i, r in df_mmr.iterrows():
    lscore = compute_licensing_score(r)
    # rough time estimate months: lower score -> longer
    eta_months = int((1.0 - lscore) * 36 + 6)  # between 6 and ~42 months
    lproba = lscore
    lrefs = r.get('ref_source','')
    lrow = {"Reactor": r.get('name', r.get('mmr')), "lic_score": lscore, "est_months": eta_months, "prob_success": lproba, "refs": lrefs}
    licscores.append(lrow)

df_lic = pd.DataFrame(licscores).sort_values("lic_score", ascending=False)
st.dataframe(df_lic.style.format({"lic_score":"{:.3f}", "prob_success":"{:.3%}"}))

# ---------------------------
# Risks and mitigations
# ---------------------------
st.header("8) Riscos específicos e recomendações mitigatórias (automático)")
st.markdown("O sistema sumariza riscos chave e recomendações a partir das entradas; revise e acrescente ações locais.")

# Generate automatic recommendations based on flags and fuel types
recs = []
for i, r in df_mmr.iterrows():
    name = r.get('name', r.get('mmr'))
    issues = []
    if r.get('heat_rejection_kW',0.0) > 5000:
        issues.append("Alto requisito de rejeição térmica -> implementar loop intermediário e heat exchangers de alta eficiência (evitar seawater primário).")
    fuel = str(r.get('fuel_type','')).lower()
    if 'haleu' in fuel:
        issues.append("HALEU: negociar logística de fornecimento e transporte com autoridades e verificar capacidade de armazenagem segura onshore.")
    if r.get('shield_mass_kg',0.0) > 100000:
        issues.append("Shield mass elevado: avaliar reforço de deck e possibilidade de reduzir shield com materiais avançados ou aumentar afastamento/layers.")
    if pd.isna(r.get('ref_source')) or r.get('ref_source','')=='':
        issues.append("Sem fontes/datasheets: requer coleta de dados e verificação documental (não use para decisão final).")
    if not issues:
        issues.append("Nenhuma mitigação automática detectada; validar planos operacionais e P&Ps locais.")
    recs.append({"Reactor": name, "Mitigations": " ; ".join(issues)})
st.dataframe(pd.DataFrame(recs))

# ---------------------------
# Export full decision report (minimal HTML)
# ---------------------------
st.header("9) Exportar relatório resumido (HTML)")
def generate_html_report(df_scores, df_fspo, df_lic):
    html = "<html><head><meta charset='utf-8'><title>MMR Offshore Decision Report</title></head><body>"
    html += "<h1>MMR Offshore Decision Report</h1>"
    html += "<h2>Ranqueamento determinístico</h2>"
    html += df_scores.to_html(index=False)
    html += "<h2>FPSO compatibility</h2>"
    html += df_fspo.to_html(index=False)
    html += "<h2>Licensing scores</h2>"
    html += df_lic.to_html(index=False)
    html += "<p>Gerado por MMR Offshore Selector</p>"
    html += "</body></html>"
    return html

if st.button("Gerar relatório HTML"):
    html = generate_html_report(df_scores, df_fspo, df_lic)
    b = html.encode('utf-8')
    b64 = base64.b64encode(b).decode()
    href = f'<a href="data:text/html;base64,{b64}" download="mmr_offshore_report.html">Download relatório HTML</a>'
    st.markdown(href, unsafe_allow_html=True)

st.markdown("---")
st.markdown("Referências / fontes públicas (exemplos): CNEN normas e transporte; IAEA SMR siting guidance; IMO nuclear & SOLAS/IMDG; fabricantes Oklo, Westinghouse (eVinci), USNC, X-energy. Consulte README para links e instruções detalhadas.")
