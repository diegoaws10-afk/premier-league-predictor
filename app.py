import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
from datetime import datetime

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="Tony Bloom AI 🤖",
    page_icon="⚽",
    layout="centered"
)

# --- 1. MOTOR DE DADOS (CACHEADO) ---
@st.cache_data(ttl=3600) # Atualiza o cache a cada 1 hora
def carregar_dados():
    URL = "https://www.football-data.co.uk/mmz4281/2526/E0.csv"
    try:
        df = pd.read_csv(URL)
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
        # Filtrar colunas necessárias para Gols, Cantos e Chutes
        cols = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'HC', 'AC', 'HST', 'AST']
        df = df[cols].dropna()
        return df
    except Exception as e:
        st.error(f"Erro ao baixar dados: {e}")
        return pd.DataFrame()

def calcular_forcas(df, alpha):
    # Decaimento Temporal
    data_hoje = pd.to_datetime('today')
    df['days_ago'] = (data_hoje - df['Date']).dt.days
    df['weight'] = np.exp(-alpha * df['days_ago'])

    def weighted_mean(values, weights):
        return np.average(values, weights=weights)

    # Médias da Liga (Ponderadas)
    medias = {
        'gols_home': np.average(df['FTHG'], weights=df['weight']),
        'gols_away': np.average(df['FTAG'], weights=df['weight']),
        'cantos_home': np.average(df['HC'], weights=df['weight']),
        'cantos_away': np.average(df['AC'], weights=df['weight']),
        'chutes_home': np.average(df['HST'], weights=df['weight']),
        'chutes_away': np.average(df['AST'], weights=df['weight']),
    }

    # Função auxiliar para criar tabela de força
    def get_strength(metric_home, metric_away, league_avg_h, league_avg_a):
        att_h = df.groupby('HomeTeam').apply(lambda x: weighted_mean(x[metric_home], x['weight']), include_groups=False) / league_avg_h
        att_a = df.groupby('AwayTeam').apply(lambda x: weighted_mean(x[metric_away], x['weight']), include_groups=False) / league_avg_a
        def_h = df.groupby('HomeTeam').apply(lambda x: weighted_mean(x[metric_away], x['weight']), include_groups=False) / league_avg_a
        def_a = df.groupby('AwayTeam').apply(lambda x: weighted_mean(x[metric_home], x['weight']), include_groups=False) / league_avg_h
        return att_h, att_a, def_h, def_a

    # Processar Forças
    g_att_h, g_att_a, g_def_h, g_def_a = get_strength('FTHG', 'FTAG', medias['gols_home'], medias['gols_away'])
    c_att_h, c_att_a, c_def_h, c_def_a = get_strength('HC', 'AC', medias['cantos_home'], medias['cantos_away'])
    s_att_h, s_att_a, s_def_h, s_def_a = get_strength('HST', 'AST', medias['chutes_home'], medias['chutes_away'])

    # Dataframe Único de Stats
    stats = pd.DataFrame({
        # GOLS
        'g_att_h': g_att_h, 'g_att_a': g_att_a, 'g_def_h': g_def_h, 'g_def_a': g_def_a,
        # CANTOS
        'c_att_h': c_att_h, 'c_att_a': c_att_a, 'c_def_h': c_def_h, 'c_def_a': c_def_a,
        # CHUTES
        's_att_h': s_att_h, 's_att_a': s_att_a, 's_def_h': s_def_h, 's_def_a': s_def_a,
    })
    
    return stats, medias

# --- 2. INTERFACE SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Configurações")
    banca = st.number_input("Sua Banca (R$)", value=30.0, step=10.0)
    kelly_frac = st.slider("Gestão Kelly (%)", 0.05, 0.25, 0.10)
    alpha = st.slider("Peso Temporal (Alpha)", 0.001, 0.020, 0.006, format="%.3f")
    st.markdown("---")
    st.markdown("Developed with **Tony Bloom AI** Logic")

# --- 3. CARREGAMENTO DOS DADOS ---
df = carregar_dados()
if not df.empty:
    stats, medias = calcular_forcas(df, alpha)
    lista_times = sorted(stats.index.tolist())
else:
    st.stop()

# --- 4. INTERFACE PRINCIPAL ---
st.title("Premier League 25/26 - Smart Picks 🧠")

# Criar Abas
tab1, tab2 = st.tabs(["⚽ Vencedor (Match Odds)", "🚩 Escanteios & Chutes"])

# === ABA 1: MATCH ODDS ===
with tab1:
    col1, col2 = st.columns(2)
    with col1:
        home = st.selectbox("Time Casa", lista_times, key='h1')
    with col2:
        away = st.selectbox("Time Visitante", lista_times, key='a1')
    
    col3, col4, col5 = st.columns(3)
    with col3:
        odd_h = st.number_input("Odd Casa (1)", value=0.0, step=0.01)
    with col4:
        odd_d = st.number_input("Odd Empate (X)", value=0.0, step=0.01)
    with col5:
        odd_a = st.number_input("Odd Fora (2)", value=0.0, step=0.01)

    if st.button("Analisar Jogo", type="primary"):
        if home == away:
            st.error("Times iguais!")
        else:
            # Poisson Gols
            lh = stats.loc[home, 'g_att_h'] * stats.loc[away, 'g_def_a'] * medias['gols_home']
            la = stats.loc[away, 'g_att_a'] * stats.loc[home, 'g_def_h'] * medias['gols_away']
            
            prob_h, prob_a, prob_d = 0, 0, 0
            for x in range(7):
                for y in range(7):
                    p = poisson.pmf(x, lh) * poisson.pmf(y, la)
                    if x > y: prob_h += p
                    elif x < y: prob_a += p
                    else: prob_d += p
            
            # Função Kelly
            def show_result(label, prob, odd_book):
                odd_fair = 1/prob
                st.markdown(f"**{label}**")
                cols = st.columns(3)
                cols[0].metric("Prob. Real", f"{prob*100:.1f}%")
                cols[1].metric("Odd Justa", f"{odd_fair:.2f}")
                
                if odd_book > odd_fair:
                    b = odd_book - 1
                    q = 1 - prob
                    f = (b * prob - q) / b
                    stake = banca * (f * kelly_frac)
                    cols[2].metric("Odd Bookie", f"{odd_book:.2f}", delta="VALOR! (+EV)")
                    st.success(f"✅ **APOSTAR: R$ {stake:.2f}**")
                else:
                    cols[2].metric("Odd Bookie", f"{odd_book:.2f}", delta="-EV", delta_color="inverse")
                    st.caption("Sem valor matemático.")
                st.divider()

            show_result(f"Vitória {home}", prob_h, odd_h)
            show_result("Empate", prob_d, odd_d)
            show_result(f"Vitória {away}", prob_a, odd_a)

# === ABA 2: PROPS ===
with tab2:
    col1, col2 = st.columns(2)
    with col1:
        home_p = st.selectbox("Time Casa", lista_times, key='h2')
    with col2:
        away_p = st.selectbox("Time Visitante", lista_times, key='a2')
    
    tipo_aposta = st.radio("Mercado", ["Escanteios", "Chutes no Gol (On Target)"], horizontal=True)
    
    col3, col4 = st.columns(2)
    with col3:
        linha = st.number_input("Linha (Over)", value=9.5, step=0.5)
    with col4:
        odd_prop = st.number_input("Odd (Bet365)", value=1.90, step=0.01)

    if st.button("Analisar Prop", type="primary"):
        if home_p == away_p:
            st.error("Times iguais!")
        else:
            if tipo_aposta == "Escanteios":
                lh = stats.loc[home_p, 'c_att_h'] * stats.loc[away_p, 'c_def_a'] * medias['cantos_home']
                la = stats.loc[away_p, 'c_att_a'] * stats.loc[home_p, 'c_def_h'] * medias['cantos_away']
            else:
                lh = stats.loc[home_p, 's_att_h'] * stats.loc[away_p, 's_def_a'] * medias['chutes_home']
                la = stats.loc[away_p, 's_att_a'] * stats.loc[home_p, 's_def_h'] * medias['chutes_away']

            lambda_total = lh + la
            # Probabilidade OVER = 1 - CDF da parte inteira da linha
            # Ex: Over 10.5 -> Prob de ser > 10.5 (ou seja, 11, 12...)
            prob_over = 1 - poisson.cdf(int(linha), lambda_total)
            odd_fair = 1/prob_over if prob_over > 0 else 0

            st.info(f"Expectativa do Modelo: **{lambda_total:.2f}** no total.")
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Prob. Over", f"{prob_over*100:.1f}%")
            c2.metric("Odd Justa", f"{odd_fair:.2f}")
            
            if odd_prop > odd_fair:
                b = odd_prop - 1
                q = 1 - prob_over
                f = (b * prob_over - q) / b
                stake = banca * (f * kelly_frac)
                c3.metric("Odd Bookie", f"{odd_prop:.2f}", delta="VALOR! (+EV)")
                st.success(f"✅ **APOSTAR: R$ {stake:.2f}**")
            else:
                c3.metric("Odd Bookie", f"{odd_prop:.2f}", delta="-EV", delta_color="inverse")
                st.warning("Sem valor. A linha é muito alta ou a odd muito baixa.")