# mmr_selector.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

st.set_page_config(page_title="Seleção de MMR Offshore", layout="wide")

st.title("⚛️ Seletor Interativo de Micro Reatores Modulares (MMRs) Offshore")

st.markdown("""
Calculadora de comparação de MMRs para embarque em plataformas marítimas usando **Multicritérios Ponderados (Simple Additive Weighting- SAW)**.
Altere os pesos dos critérios abaixo e veja o novo ranking dos reatores!
""")

# Dados dos MMRs
mmrs = ['Oklo Aurora', 'U-Battery', 'eVinci', 'StarCore', 'HolosGen', 'USNC MMR', 'Kronos (PWR)', 'XAMR (HTGR)']
criteria = ['Segurança', 'Custo-benefício', 'Volume compacto', 'Potência', 'Regulação']

# Notas normalizadas (0-10)
scores = np.array([
    [8, 10, 9, 5, 7],  # Oklo
    [10, 8, 8, 6, 8],  # U-Battery
    [10, 7, 7, 6, 6],  # eVinci
    [9, 6, 6, 7, 5],   # StarCore
    [9, 8, 10, 6, 4],  # HolosGen
    [10, 8, 8, 6, 7],  # USNC MMR
    [8, 7, 6, 6, 6],   # Kronos (PWR)
    [9, 8, 8, 7, 6],   # XAMR (HTGR)
])

st.sidebar.header("🎛️ Ajuste os Pesos dos Critérios")
peso_seguranca = st.sidebar.slider('Peso - Segurança', 0.0, 1.0, 0.30, step=0.05)
peso_custo = st.sidebar.slider('Peso - Custo-benefício', 0.0, 1.0, 0.25, step=0.05)
peso_volume = st.sidebar.slider('Peso - Volume compacto', 0.0, 1.0, 0.20, step=0.05)
peso_potencia = st.sidebar.slider('Peso - Potência', 0.0, 1.0, 0.15, step=0.05)
peso_regulacao = st.sidebar.slider('Peso - Regulação', 0.0, 1.0, 0.10, step=0.05)

pesos = np.array([peso_seguranca, peso_custo, peso_volume, peso_potencia, peso_regulacao])

# CORREÇÃO: Verificar se a soma dos pesos é zero para evitar divisão por zero
if pesos.sum() == 0:
    st.sidebar.warning("⚠️ Todos os pesos estão em zero! Usando pesos iguais temporariamente.")
    pesos = np.ones(5) / 5  # Pesos iguais quando todos são zero
else:
    pesos = pesos / pesos.sum()

# Mostrar pesos normalizados
st.sidebar.markdown("### Pesos Normalizados:")
for i, criterio in enumerate(criteria):
    st.sidebar.write(f"**{criterio}**: {pesos[i]:.3f}")

weighted_scores = scores * pesos
final_scores = weighted_scores.sum(axis=1)

df_resultado = pd.DataFrame({
    'MMR': mmrs,
    'Score Final': final_scores
}).sort_values(by='Score Final', ascending=False).reset_index(drop=True)

# Adicionar ranking
df_resultado['Ranking'] = range(1, len(df_resultado) + 1)
df_resultado = df_resultado[['Ranking', 'MMR', 'Score Final']].round(3)

st.subheader("🏆 Ranking dos MMRs Offshore (com base nos pesos ajustados)")

# CORREÇÃO 1: Destacar toda a linha do vencedor, não só o score máximo
def highlight_winner(row):
    if row['Ranking'] == 1:
        return ['background-color: lightgreen'] * len(row)
    else:
        return [''] * len(row)

st.dataframe(
    df_resultado.style.apply(highlight_winner, axis=1), 
    height=350,
    use_container_width=True
)

with st.expander("🔬 Ver Detalhamento Técnico dos MMRs"):
    st.markdown("""
| MMR           | Potência (MWe) | Combustível | Resfriamento     | Vida útil (anos) | Volume estimado |
|---------------|----------------|-------------|------------------|------------------|------------------|
| Oklo Aurora   | 1.5            | HALEU       | Metálico         | 20               | Muito compacto   |
| U-Battery     | 4              | TRISO       | Hélio            | 5                | Compacto         |
| eVinci        | 5              | TRISO       | Metálico         | 8-10             | Muito compacto   |
| StarCore      | 14             | TRISO       | Gás hélio        | 10               | Médio            |
| HolosGen      | 3-10           | HALEU       | Ar/Gás           | 8-10             | ISO container    |
| USNC MMR      | 5              | TRISO/FCM   | Gás hélio        | 20               | Containerizado   |
| Kronos (PWR)  | 5              | LEU         | Água pressurizada| 10               | Médio            |
| XAMR (HTGR)   | 6              | TRISO       | Gás hélio        | 10-15            | Compacto         |
""")

# CORREÇÃO 2: Gráfico radar dinâmico baseado nos scores ponderados
st.subheader("📊 Gráfico Radar Interativo (baseado nos pesos atuais)")

# Criar dados dinâmicos baseados nos pesos atuais
num_vars = len(criteria)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(12, 8), subplot_kw=dict(polar=True))

# Mostrar apenas os top 4 MMRs para melhor visualização
top_mmrs = 4
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

for i in range(min(top_mmrs, len(df_resultado))):
    mmr_name = df_resultado.iloc[i]['MMR']
    mmr_index = mmrs.index(mmr_name)
    
    # Usar scores ponderados para o gráfico
    weighted_values = weighted_scores[mmr_index].tolist()
    stats = weighted_values + weighted_values[:1]  # Fechar o polígono
    
    ax.plot(angles, stats, 'o-', linewidth=2, label=f"{i+1}º {mmr_name}", color=colors[i])
    ax.fill(angles, stats, alpha=0.15, color=colors[i])

ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.set_thetagrids(np.degrees(angles[:-1]), criteria)
# CORREÇÃO: Garantir que os limites do eixo sejam válidos
y_max = weighted_scores.max() if not np.isnan(weighted_scores.max()) else 1.0
ax.set_ylim(0, max(y_max * 1.1, 1.0))
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
ax.set_title("Top 4 MMRs - Scores Ponderados", pad=20)
plt.tight_layout()
st.pyplot(fig)

# Mostrar análise de sensibilidade
st.subheader("🔍 Análise de Sensibilidade")
col1, col2 = st.columns(2)

with col1:
    st.markdown("**MMR Vencedor atual:**")
    if not np.isnan(df_resultado.iloc[0]['Score Final']):
        winner = df_resultado.iloc[0]
        st.success(f"🥇 **{winner['MMR']}** com score {winner['Score Final']:.3f}")
    else:
        st.warning("⚠️ Defina pelo menos um peso para ver o vencedor!")

with col2:
    st.markdown("**Critério mais influente:**")
    if pesos.sum() > 0:
        criterio_dominante = criteria[np.argmax(pesos)]
        st.info(f"📈 **{criterio_dominante}** (peso: {pesos[np.argmax(pesos)]:.3f})")
    else:
        st.info("📈 **Todos iguais** (pesos zerados)")

st.markdown("💡 Desenvolvido por Equipe Nautilus®, 2025")
