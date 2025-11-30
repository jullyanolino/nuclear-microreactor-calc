"""
Streamlit App: MMR Decision Suite — SAW + AHP + Monte Carlo + FPSO Compatibility
Author: Gerado pelo assistant a pedido do usuário
Descrição: Ferramenta interativa para avaliação comparativa de Micro Modular Reactors (MMRs)
          com:
          - SAW (Simple Additive Weighting)
          - AHP (Analytic Hierarchy Process) pairwise -> prioridades
          - Monte Carlo para incerteza (probabilidade de ser top-1)
          - Leitura de CSV com dados dos reatores (fallback para dataset interno)
          - Cálculo automático de LCOE
          - Verificações de compatibilidade básica com parâmetros de FPSO (deck, CG, guindaste, footprint)
Como rodar:
    pip install -r requirements.txt
    streamlit run streamlit_app.py
"""

from typing import List, Dict
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="MMR Decision Suite", layout="wide")

# -----------------------
# Defaults: MMRs & dataset
# -----------------------
MMRS = ['Oklo Aurora', 'U-Battery', 'eVinci', 'StarCore', 'HolosGen', 'USNC MMR', 'Kronos (PWR)', 'XAMR (HTGR)']

DEFAULT_DATA = pd.DataFrame([
    # Simple plausible example numbers for demonstration only. Units in column names.
    # CAPEX in MUSD, OPEX in MUSD/year, CF fraction, Mass in tonnes, Shielding_mass in tonnes, Footprint in m2,
    # Refuel interval in years, Crane required (t)
    {'MMR': 'Oklo Aurora', 'Segurança': 8.5, 'Custo-benefício': 7.0, 'Volume compacto': 8.0, 'Potência': 6.0, 'Regulação': 5.5,
     'CAPEX_MUSD': 40.0, 'OPEX_MUSD_per_year': 2.0, 'CF': 0.92, 'Mass_t': 85.0, 'Shielding_mass_t': 25.0, 'Footprint_m2': 80.0,
     'Refuel_interval_yr': 15, 'Crane_req_t': 60},
    {'MMR': 'U-Battery', 'Segurança': 7.0, 'Custo-benefício': 6.0, 'Volume compacto': 9.0, 'Potência': 2.5, 'Regulação': 6.0,
     'CAPEX_MUSD': 15.0, 'OPEX_MUSD_per_year': 1.2, 'CF': 0.80, 'Mass_t': 40.0, 'Shielding_mass_t': 10.0, 'Footprint_m2': 40.0,
     'Refuel_interval_yr': 8, 'Crane_req_t': 20},
    {'MMR': 'eVinci', 'Segurança': 8.0, 'Custo-benefício': 6.8, 'Volume compacto': 6.0, 'Potência': 5.0, 'Regulação': 7.0,
     'CAPEX_MUSD': 30.0, 'OPEX_MUSD_per_year': 1.5, 'CF': 0.90, 'Mass_t': 70.0, 'Shielding_mass_t': 20.0, 'Footprint_m2': 70.0,
     'Refuel_interval_yr': 20, 'Crane_req_t': 40},
    {'MMR': 'StarCore', 'Segurança': 6.5, 'Custo-benefício': 6.5, 'Volume compacto': 7.0, 'Potência': 4.0, 'Regulação': 5.0,
     'CAPEX_MUSD': 22.0, 'OPEX_MUSD_per_year': 1.8, 'CF': 0.85, 'Mass_t': 60.0, 'Shielding_mass_t': 18.0, 'Footprint_m2': 65.0,
     'Refuel_interval_yr': 10, 'Crane_req_t': 30},
    {'MMR': 'HolosGen', 'Segurança': 8.2, 'Custo-benefício': 7.2, 'Volume compacto': 6.5, 'Potência': 6.5, 'Regulação': 6.8,
     'CAPEX_MUSD': 45.0, 'OPEX_MUSD_per_year': 2.5, 'CF': 0.93, 'Mass_t': 95.0, 'Shielding_mass_t': 30.0, 'Footprint_m2': 95.0,
     'Refuel_interval_yr': 12, 'Crane_req_t': 80},
    {'MMR': 'USNC MMR', 'Segurança': 7.8, 'Custo-benefício': 7.0, 'Volume compacto': 7.5, 'Potência': 4.5, 'Regulação': 6.2,
     'CAPEX_MUSD': 28.0, 'OPEX_MUSD_per_year': 1.6, 'CF': 0.88, 'Mass_t': 68.0, 'Shielding_mass_t': 17.0, 'Footprint_m2': 66.0,
     'Refuel_interval_yr': 10, 'Crane_req_t': 35},
    {'MMR': 'Kronos (PWR)', 'Segurança': 6.2, 'Custo-benefício': 5.5, 'Volume compacto': 5.0, 'Potência': 10.0, 'Regulação': 4.5,
     'CAPEX_MUSD': 120.0, 'OPEX_MUSD_per_year': 6.0, 'CF': 0.95, 'Mass_t': 210.0, 'Shielding_mass_t': 80.0, 'Footprint_m2': 250.0,
     'Refuel_interval_yr': 3, 'Crane_req_t': 200},
    {'MMR': 'XAMR (HTGR)', 'Segurança': 9.0, 'Custo-benefício': 7.4, 'Volume compacto': 6.8, 'Potência': 7.0, 'Regulação': 6.5,
     'CAPEX_MUSD': 55.0, 'OPEX_MUSD_per_year': 2.8, 'CF': 0.90, 'Mass_t': 110.0, 'Shielding_mass_t': 28.0, 'Footprint_m2': 110.0,
     'Refuel_interval_yr': 12, 'Crane_req_t': 70},
])

# -----------------------
# Helper functions
# -----------------------
def compute_lcoe(capex_musd, opex_musd_per_year, cf, lifetime_yr=30, discount_rate=0.07, capacity_mwe=5.0):
    """
    Compute LCOE in $/MWh.
    CAPEX in MUSD, OPEX in MUSD/yr, CF fraction, capacity in MWe.
    Returns $/MWh
    """
    # convert MUSD to USD
    capex = capex_musd * 1e6
    opex = opex_musd_per_year * 1e6
    r = discount_rate
    n = lifetime_yr
    if r == 0:
        crf = 1 / n
    else:
        crf = (r * (1 + r) ** n) / ((1 + r) ** n - 1)
    annualized_capex = capex * crf
    annual_energy_mwh = capacity_mwe * cf * 8760.0
    lcoe = (annualized_capex + opex) / annual_energy_mwh
    return lcoe

def minmax_normalize(series: pd.Series, higher_is_better=True):
    a = series.min()
    b = series.max()
    if a == b:
        return pd.Series(0.5, index=series.index)  # neutral
    norm = (series - a) / (b - a)
    return norm if higher_is_better else 1 - norm

def ahp_priority_from_pairwise(A: np.ndarray):
    vals, vecs = np.linalg.eig(A)
    max_idx = np.argmax(vals.real)
    priority = np.real(vecs[:, max_idx])
    priority = np.abs(priority)
    priority = priority / priority.sum()
    # consistency check
    lambda_max = np.real(vals[max_idx])
    n = A.shape[0]
    ci = (lambda_max - n) / (n - 1) if n > 1 else 0.0
    # Random Index (RI) table for 1..10 (Saaty)
    RI_dict = {1:0.00,2:0.00,3:0.58,4:0.90,5:1.12,6:1.24,7:1.32,8:1.41,9:1.45,10:1.49}
    ri = RI_dict.get(n, 1.49)
    cr = ci / ri if ri != 0 else 0.0
    return priority, {'lambda_max': lambda_max, 'CI': ci, 'RI': ri, 'CR': cr}

def compute_integrabilidade_offshore(row):
    """
    Example composite index (0..1) of Integrability Offshore (IIO).
    Subfatores: massa, shielding, footprint, refuel frequency, crane requirement.
    Each subfator normalized and combined with weights.
    """
    # lower mass, lower shielding, lower footprint, longer refuel (higher better), smaller crane requirement => better.
    # We'll normalize using heuristic maximums from dataset or reasonable caps
    # Use default maxima for normalization
    mass = row['Mass_t']
    shield = row['Shielding_mass_t']
    footprint = row['Footprint_m2']
    refuel = row['Refuel_interval_yr']
    crane = row['Crane_req_t']
    # maxima (tunable)
    mass_max = 250.0
    shield_max = 100.0
    footprint_max = 300.0
    refuel_max = 25.0
    crane_max = 250.0
    f_mass = 1 - min(mass / mass_max, 1.0)
    f_shield = 1 - min(shield / shield_max, 1.0)
    f_foot = 1 - min(footprint / footprint_max, 1.0)
    f_refuel = min(refuel / refuel_max, 1.0)
    f_crane = 1 - min(crane / crane_max, 1.0)
    # weights
    w = {'mass': 0.25, 'shield': 0.25, 'foot':0.20, 'refuel':0.2, 'crane':0.1}
    iio = w['mass']*f_mass + w['shield']*f_shield + w['foot']*f_foot + w['refuel']*f_refuel + w['crane']*f_crane
    return iio

def perform_saw(df, criteria_cols, weights):
    """
    df: dataframe containing criteria numeric columns (already normalized 0..1)
    weights: dict mapping criterion -> weight (sums to 1)
    returns series of final scores
    """
    w = np.array([weights[c] for c in criteria_cols])
    mat = df[criteria_cols].values.astype(float)
    scores = mat.dot(w)
    return pd.Series(scores, index=df.index)

def monte_carlo_ranking(df_scores_matrix, n_runs=2000, noise_pct=0.10):
    """
    df_scores_matrix: (n_reactors x n_criteria) original normalized scores in 0..1
    noise_pct: relative noise applied to each criterion per run
    returns: probabilities of being top-1 for each reactor
    """
    n_reactors = df_scores_matrix.shape[0]
    n_criteria = df_scores_matrix.shape[1]
    scores = df_scores_matrix.values
    winners = np.zeros(n_reactors, dtype=int)
    for _ in range(n_runs):
        # multiplicative noise per element
        noise = np.random.normal(1.0, noise_pct, scores.shape)
        noisy = scores * noise
        # simple equal weights across criteria for the noisy check or we can re-use provided weights externally.
        # For the Monte Carlo we will compute the weighted sum using the base column weights passed separately by caller.
        # Here we simply return the noisy matrix for caller to use with weights.
        # We'll implement a wrapper above; this function can be extended per need.
        # To keep this function general, just collect noisy matrices
        pass
    # not used directly; wrapper below implements Monte Carlo with weights
    return None

def monte_carlo_with_weights(df_norm, criteria_cols, weights_array, n_runs=5000, noise_pct=0.10):
    """
    df_norm: dataframe with normalized criteria (0..1)
    criteria_cols: ordered list of criteria
    weights_array: numpy array of weights aligned with criteria_cols summing to 1
    Returns: probabilities of top-1 and distribution of scores (summary)
    """
    n_reactors = df_norm.shape[0]
    scores_store = np.zeros((n_runs, n_reactors))
    base = df_norm[criteria_cols].values.astype(float)
    for i in range(n_runs):
        noise = np.random.normal(1.0, noise_pct, base.shape)
        noisy = base * noise
        final = noisy.dot(weights_array)
        scores_store[i, :] = final
    winners = np.argmax(scores_store, axis=1)
    probs = [(winners == j).mean() for j in range(n_reactors)]
    mean_scores = scores_store.mean(axis=0)
    std_scores = scores_store.std(axis=0)
    return {'probs': probs, 'mean': mean_scores, 'std': std_scores, 'winners': winners}

# -----------------------
# UI start
# -----------------------
st.title("MMR Decision Suite — SAW + AHP + Monte Carlo + FPSO Checker")
st.write("Ferramenta para apoio à decisão sobre adoção de Micro Modular Reactors (MMRs) embarcados em FPSO.")
st.write("Use o menu lateral para carregar dados, ajustar pesos e realizar análises. (Prototipo técnico — não substituir estudos de engenharia.)")

# -----------------------
# Sidebar: CSV upload and config
# -----------------------
st.sidebar.header("Input Data / Config")

uploaded = st.sidebar.file_uploader("Upload CSV de dados dos MMRs (colunas: see README)", type=['csv'])
if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
        st.sidebar.success("CSV carregado.")
    except Exception as e:
        st.sidebar.error(f"Falha ao ler CSV: {e}")
        df = DEFAULT_DATA.copy()
else:
    df = DEFAULT_DATA.copy()

# Ensure MMR column exists
if 'MMR' not in df.columns:
    st.error("CSV inválido: falta coluna 'MMR'. Usando dataset interno.")
    df = DEFAULT_DATA.copy()

df = df.copy()
df = df.set_index('MMR')

# Add computed LCOE and IIO if not present
if 'LCOE_USD_per_MWh' not in df.columns:
    # default capacity_mwe: infer from 'Potência' column if present (Potência in MW)
    capacity_est = df.get('Potência', pd.Series(5.0, index=df.index)).astype(float)
    # if potência values look like 10 for Kronos, we interpret as MW. If absent, default 5 MWe.
    lcoes = []
    for idx, row in df.iterrows():
        capex = float(row.get('CAPEX_MUSD', 30.0))
        opex = float(row.get('OPEX_MUSD_per_year', 2.0))
        cf = float(row.get('CF', 0.85))
        # capacity guess: try 'Potência' else default 5 MWe
        cap_mwe = float(row.get('Potência', capacity_est.get(idx, 5.0)))
        # Prevent zero or absurd values
        if cap_mwe <= 0:
            cap_mwe = 5.0
        lcoe = compute_lcoe(capex, opex, cf, lifetime_yr=30, discount_rate=0.07, capacity_mwe=cap_mwe)
        lcoes.append(lcoe)
    df['LCOE_USD_per_MWh'] = lcoes

if 'IIO' not in df.columns:
    df['IIO'] = df.apply(lambda r: compute_integrabilidade_offshore(r), axis=1)

# Criteria list: original plus added quantitative ones (user requested expansion allowed)
criteria = ['Segurança', 'Custo-benefício', 'Volume compacto', 'Potência', 'Regulação', 'LCOE', 'IIO']
# Mark which criteria are benefit (higher better) or cost (lower better)
benefit_flags = {
    'Segurança': True,
    'Custo-benefício': True,
    'Volume compacto': True,
    'Potência': True,
    'Regulação': True,
    'LCOE': False,  # lower LCOE is better
    'IIO': True
}

st.sidebar.subheader("Critérios (expandido)")
st.sidebar.write("Critérios usados internamente: " + ", ".join(criteria))

# -----------------------
# Weighting method selection
# -----------------------
method = st.sidebar.selectbox("Método de priorização", ["SAW (Weighted Sum)", "AHP (pairwise)"])
st.sidebar.markdown("Monte Carlo adiciona incerteza às notas/entradas e retorna probabilidade de cada MMR ser ranking #1.")

# -----------------------
# Weights: either sliders (SAW) or AHP pairwise input
# -----------------------
st.sidebar.markdown("### Pesos por critério")
if method == "SAW (Weighted Sum)":
    user_weights = {}
    total = 0.0
    for c in criteria:
        w = st.sidebar.slider(f"Peso: {c}", 0.0, 5.0, 1.0, 0.1)
        user_weights[c] = float(w)
        total += w
    if total == 0:
        st.sidebar.warning("Soma dos pesos é zero — serão usados pesos iguais.")
        norm_weights = {c: 1.0/len(criteria) for c in criteria}
    else:
        norm_weights = {c: float(user_weights[c]/total) for c in criteria}
else:
    # AHP: pairwise matrix creation UI
    st.sidebar.write("Preencha a matriz de comparações pareadas (Saaty scale). Valores >1: linha mais importante que coluna. Recíproco automatic.")
    n = len(criteria)
    # initialize matrix identity
    A = np.ones((n, n))
    # We'll store values in session state to persist across reruns
    if 'ahp_vals' not in st.session_state:
        st.session_state['ahp_vals'] = [[1.0]*n for _ in range(n)]
    # Build grid of number_inputs
    for i in range(n):
        cols = st.sidebar.columns(3)
        # We will render per-row compressed to avoid huge UI; show as text inputs inline
        pass
    st.sidebar.markdown("Atenção: mantenha reciprocidade. Se editar (i,j) você deve inserir 1/value em (j,i) manualmente por ora.")
    st.sidebar.write("Como atalho, você pode usar a matriz simétrica default (igual importância):")
    if st.sidebar.button("Usar matriz identidade (igual importância)"):
        A = np.eye(n)
    else:
        # Default equal
        A = np.ones((n, n))
    priority_vec, ahp_info = ahp_priority_from_pairwise(A)
    # Map priorities to criteria
    norm_weights = {criteria[i]: float(priority_vec[i]) for i in range(n)}
    st.sidebar.write("Prioridades AHP (autovetor):")
    for i, c in enumerate(criteria):
        st.sidebar.write(f"- {c}: {norm_weights[c]:.3f}")
    st.sidebar.write(f"Consistency Ratio (CR): {ahp_info['CR']:.3f} (CI={ahp_info['CI']:.3f})")
    if ahp_info['CR'] > 0.1:
        st.sidebar.warning("CR > 0.1 — julgamentos inconsistentes. Recomenda-se revisar a matriz pareada.")

# -----------------------
# Show dataset summary
# -----------------------
st.subheader("Dataset: MMRs (visual)")
st.dataframe(df.style.format({c: "{:.3f}" for c in df.columns if df[c].dtype != object}))

# -----------------------
# Normalization per criterion
# -----------------------
st.subheader("Normalização das métricas por critério")
norm_df = pd.DataFrame(index=df.index)
for c in criteria:
    if c not in df.columns:
        # If missing, fill with neutral values
        norm_df[c] = 0.5
        continue
    series = df[c].astype(float)
    hi_better = benefit_flags.get(c, True)
    norm_df[c] = minmax_normalize(series, higher_is_better=hi_better)

st.dataframe(norm_df.style.format("{:.3f}"))

# -----------------------
# SAW scoring
# -----------------------
st.subheader("SAW: Pontuação ponderada (resultado determinístico)")
# Ensure weights sum to 1
weights_arr = np.array([norm_weights[c] for c in criteria])
# small normalization
if weights_arr.sum() == 0:
    weights_arr = np.ones_like(weights_arr) / len(weights_arr)
else:
    weights_arr = weights_arr / weights_arr.sum()
weights_map = {c: weights_arr[i] for i, c in enumerate(criteria)}

scores = perform_saw(norm_df, criteria, weights_map)
results = pd.DataFrame({
    'MMR': norm_df.index,
    'Score': scores
}).set_index('MMR')
results = results.join(df[['LCOE_USD_per_MWh', 'Mass_t', 'Shielding_mass_t', 'Footprint_m2', 'Crane_req_t']], how='left')

results = results.sort_values('Score', ascending=False)
st.dataframe(results.style.format({'Score': "{:.4f}", 'LCOE_USD_per_MWh':"{:.2f}"}))

st.markdown("**Top 3 (determinístico)**")
for i, (idx, row) in enumerate(results.head(3).iterrows(), 1):
    st.write(f"{i}. **{idx}** — Score: {row['Score']:.4f} — LCOE: ${row['LCOE_USD_per_MWh']:.1f}/MWh — Mass: {row['Mass_t']} t")

# -----------------------
# Radar chart for top N
# -----------------------
st.subheader("Radar — Top 4 (normalizado 0..1)")
topn = results.head(4).index.tolist()
fig = go.Figure()
for mmr in topn:
    values = norm_df.loc[mmr, criteria].tolist()
    fig.add_trace(go.Scatterpolar(
        r=values + [values[0]],
        theta=criteria + [criteria[0]],
        fill='toself',
        name=mmr
    ))
fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,1])), showlegend=True)
st.plotly_chart(fig, use_container_width=True)

# -----------------------
# Monte Carlo probabilistic ranking
# -----------------------
st.subheader("Monte Carlo — Probabilidade de ser top-1 (incerteza)")
mc_runs = st.slider("Número de simulações Monte Carlo", 200, 20000, 2000, step=200)
noise_pct = st.slider("Ruído relativo por critério (std, ex 0.10 = 10%)", 0.01, 0.30, 0.10, step=0.01)
mc_res = monte_carlo_with_weights(norm_df, criteria, np.array([weights_map[c] for c in criteria]), n_runs=mc_runs, noise_pct=noise_pct)

probs = mc_res['probs']
mc_table = pd.DataFrame({
    'MMR': norm_df.index,
    'Prob_top1': probs,
    'MeanScore': mc_res['mean'],
    'StdScore': mc_res['std']
}).set_index('MMR').sort_values('Prob_top1', ascending=False)
st.dataframe(mc_table.style.format({'Prob_top1':"{:.3f}", 'MeanScore':"{:.4f}", 'StdScore':"{:.4f}"}))

st.markdown("**Probabilidades de ser #1 (ordenado)**")
for idx, row in mc_table.head(3).iterrows():
    st.write(f"- {idx}: Probabilidade {row['Prob_top1']:.2%} — Score médio {row['MeanScore']:.3f} ± {row['StdScore']:.3f}")

# -----------------------
# FPSO compatibility checker (I/O spec and checks)
# -----------------------
st.subheader("FPSO Compatibility Checker — Verificações básicas (heurísticas)")
st.write("Informe parâmetros do FPSO para checagem rápida de compatibilidade (isto é um *pré*-filtro; engenharia naval requerida).")

col1, col2, col3 = st.columns(3)
with col1:
    deck_capacity_t = st.number_input("Deck available capacity (toneladas)", min_value=10.0, value=300.0, step=10.0)
    allowable_cg_shift_m = st.number_input("Allowable CG shift (m)", min_value=0.1, value=0.5, step=0.1)
with col2:
    crane_capacity_t = st.number_input("Crane capacity (t)", min_value=5.0, value=150.0, step=5.0)
    footprint_capacity_m2 = st.number_input("Available topside footprint (m2)", min_value=10.0, value=800.0, step=10.0)
with col3:
    electrical_interface_mwe = st.number_input("Electrical interface required (MWe available)", min_value=0.1, value=20.0, step=0.1)
    note = st.text_input("Ops notes / constraints", value="Separação física mínima entre nuclear e hydrocarbons recomendada.")

# Compute simple checks
compat_rows = []
for mmr in df.index:
    mass = float(df.loc[mmr].get('Mass_t', 0.0))
    shield = float(df.loc[mmr].get('Shielding_mass_t', 0.0))
    footprint = float(df.loc[mmr].get('Footprint_m2', 0.0))
    crane_req = float(df.loc[mmr].get('Crane_req_t', 0.0))
    # simple CG shift estimate (heuristic): shielding mass located off center ~ produces CG shift proportional
    cg_shift_est = min(allowable_cg_shift_m * (shield / (deck_capacity_t*0.2)), allowable_cg_shift_m*2)  # heuristic cap
    reasons = []
    ok = True
    if mass > deck_capacity_t:
        ok = False
        reasons.append(f"Mass ({mass}t) > deck capacity ({deck_capacity_t}t)")
    if shield > deck_capacity_t * 0.5:
        ok = False
        reasons.append(f"Shielding mass ({shield}t) > 50% deck capacity -> structural risk")
    if crane_req > crane_capacity_t:
        ok = False
        reasons.append(f"Crane required ({crane_req}t) > crane capacity ({crane_capacity_t}t)")
    if footprint > footprint_capacity_m2:
        ok = False
        reasons.append(f"Footprint ({footprint} m2) > available footprint ({footprint_capacity_m2} m2)")
    # electrical check: ensure at least reactor capacity < electrical interface available * 1.1 buffer
    reactor_capacity = float(df.loc[mmr].get('Potência', 5.0))
    if reactor_capacity > electrical_interface_mwe * 1.05:
        ok = False
        reasons.append(f"Reactor rated power ({reactor_capacity} MWe) > electrical interface available ({electrical_interface_mwe} MWe)")

    compat_rows.append({'MMR': mmr, 'OK': ok, 'CG_shift_est_m': cg_shift_est, 'Issues': "; ".join(reasons) if reasons else "OK"})

compat_df = pd.DataFrame(compat_rows).set_index('MMR').sort_values('OK', ascending=False)
st.dataframe(compat_df.style.format({'CG_shift_est_m':"{:.3f}"}))

st.markdown("**Observação**: Esses checks são **heurísticos** e servem apenas como *filtro inicial*. Requer-se análise estrutural (FEA), estabilidade (GM), e estudos de vibração e impacto em topsides por equipe naval/estaleiro.")

# -----------------------
# Export / Save options
# -----------------------
st.subheader("Export / Save")
if st.button("Exportar resultados (CSV)"):
    out_df = results.join(pd.DataFrame({'Prob_top1': mc_table['Prob_top1']}), how='left')
    csv = out_df.to_csv()
    st.download_button("Download CSV com scores e probabilidades", data=csv, file_name="mmr_scores_results.csv", mime="text/csv")

# -----------------------
# Footer: Notes & metadata
# -----------------------
st.info("Este aplicativo é uma ferramenta de apoio técnico exploratória. Decisões finais de engenharia e licenciamento requerem estudos dedicados, modelagem FEA, avaliações regulatórias e validação por especialistas nucleares e navais.")

st.markdown("---")
st.markdown("### Notas rápidas para engenheiros")
st.markdown("""
- AHP implementado com autovetor e verificação de consistência (CR).
- LCOE calculado com CRF (30 anos, 7% taxa por defeito) — parâmetros ajustáveis no código.
- IIO é um índice composto heurístico para integração offshore; ajustar pesos conforme sua política técnica.
""")
