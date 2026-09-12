import os
import logging
import json
import urllib.request
from datetime import datetime
import numpy as np
import pandas as pd
import yfinance as yf
from hmmlearn import hmm

# Configuración del registro de eventos (logging)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ==========================================
# CONFIGURACIÓN DE NOTIFICACIONES
# ==========================================
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "TU_BOT_TOKEN_AQUI")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "TU_CHAT_ID_AQUI")

def send_telegram_alert(regime, probabilities, strategies, top_candidates, date_str):
    """Envia un resumen estructurado al canal de Telegram."""
    if TELEGRAM_BOT_TOKEN == "TU_BOT_TOKEN_AQUI" or TELEGRAM_CHAT_ID == "TU_CHAT_ID_AQUI":
        logging.warning("Telegram no configurado. Omite el envío de alerta.")
        return

    top_stocks = ", ".join(top_candidates.index[:3])
    max_prob = max(probabilities.values())
    
    message = (
        f"🚨 *ALERTA DE RÉGIMEN DE MERCADO* 🚨\n\n"
        f"📅 *Fecha:* {date_str}\n"
        f"📊 *Régimen Activo:* `{regime}` (Confianza: {max_prob:.1%})\n\n"
        f"✅ *Estrategias ACTIVAS:* {', '.join(strategies['Active'])}\n"
        f"⏸️ *Estrategias STANDBY:* {', '.join(strategies['Standby'])}\n\n"
        f"🏆 *Top Candidates:* `{top_stocks}`\n"
    )

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown"
    }

    try:
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req) as response:
            if response.status == 200:
                logging.info("Notificación enviada a Telegram con éxito.")
    except Exception as e:
        logging.error(f"Error al enviar la notificación a Telegram: {e}")

# ==========================================
# 1. INGESTA DE DATOS Y CARACTERÍSTICAS MACRO
# ==========================================
def fetch_macro_data(period="5y"):
    tickers = ["^GSPC", "^VIX", "^TNX", "^IRX"]
    data = yf.download(tickers, period=period, progress=False)["Close"]
    
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(1)
        
    df = pd.DataFrame(index=data.index)
    df["Returns"] = data["^GSPC"].pct_change()
    df["Vol_21d"] = df["Returns"].rolling(21).std() * np.sqrt(252)
    df["Yield_Spread"] = data["^TNX"] - data["^IRX"]
    df["VIX_Level"] = data["^VIX"] / 100.0
    
    df = df.dropna()
    X = df[["Returns", "Vol_21d", "Yield_Spread", "VIX_Level"]].values
    return df, X

# ==========================================
# 2. CLASIFICADOR HMM DE RÉGIMEN
# ==========================================
def fit_hmm_regimes(df, X, n_components=3):
    model = hmm.GaussianHMM(
        n_components=n_components,
        covariance_type="full",
        n_iter=1000,
        random_state=42
    )
    model.fit(X)
    
    hidden_states = model.predict(X)
    probs = model.predict_proba(X)[-1]
    current_state_idx = hidden_states[-1]
    
    state_vols = [model.means_[i][1] for i in range(n_components)]
    sorted_vol_indices = np.argsort(state_vols)
    
    labels_map = {
        sorted_vol_indices[0]: "LOW_VOLATILITY / TRENDING",
        sorted_vol_indices[1]: "NEUTRAL / MEAN_REVERTING",
        sorted_vol_indices[2]: "HIGH_VOLATILITY / CRISIS"
    }
    
    current_regime = labels_map[current_state_idx]
    state_probabilities = {labels_map[i]: float(probs[i]) for i in range(n_components)}
    latest_date = df.index[-1].strftime("%Y-%m-%d")
    
    return current_regime, state_probabilities, latest_date

# ==========================================
# 3. SELECCIÓN HÍBRIDA DE ACCIONES
# ==========================================
def rank_portfolio_candidates(tickers, current_regime):
    scores = {}
    hist_data = yf.download(tickers, period="6mo", progress=False)["Close"]
    if isinstance(hist_data.columns, pd.MultiIndex):
        hist_data.columns = hist_data.columns.get_level_values(1)

    for ticker in tickers:
        try:
            tk = yf.Ticker(ticker)
            info = tk.info
            
            pe_ratio = info.get("forwardPE", None)
            profit_margins = info.get("profitMargins", 0)
            debt_to_equity = info.get("debtToEquity", 100)
            free_cashflow = info.get("freeCashflow", 0)
            
            f_score = 0
            if profit_margins and profit_margins > 0.15:
                f_score += 35
            if debt_to_equity and debt_to_equity < 80:
                f_score += 35
            if free_cashflow and free_cashflow > 0:
                f_score += 30
                
            hist = hist_data[ticker].dropna()
            returns = hist.pct_change().dropna()
            
            t_score = 0
            if "TRENDING" in current_regime:
                ret_3m = (hist.iloc[-1] / hist.iloc[-60]) - 1 if len(hist) >= 60 else 0
                vol = returns.std() * np.sqrt(252)
                sharpe_proxy = ret_3m / (vol + 1e-6)
                t_score = np.clip(sharpe_proxy * 50, 0, 100)
            elif "MEAN_REVERTING" in current_regime:
                sma_50 = hist.rolling(50).mean().iloc[-1]
                std_50 = hist.rolling(50).std().iloc[-1]
                z_score = (hist.iloc[-1] - sma_50) / (std_50 + 1e-6)
                t_score = np.clip(-z_score * 35 + 50, 0, 100)
            elif "CRISIS" in current_regime:
                vol = returns.std() * np.sqrt(252)
                t_score = np.clip((0.40 - vol) * 200, 0, 100)
                
            total_score = 0.5 * f_score + 0.5 * t_score
            scores[ticker] = {
                "Total_Score": round(total_score, 1),
                "F_Score": f_score,
                "T_Score": round(t_score, 1),
                "P_E": round(pe_ratio, 1) if pe_ratio else "N/A"
            }
        except Exception as e:
            logging.warning(f"Error procesando {ticker}: {e}")
            continue
            
    if not scores:
        logging.error("No se pudieron obtener datos para ningún ticker.")
        return pd.DataFrame(columns=["Total_Score", "F_Score", "T_Score", "P_E"])

    df_scores = pd.DataFrame.from_dict(scores, orient="index")
    return df_scores.sort_values(by="Total_Score", ascending=False)

# ==========================================
# 4. MAPEO DE ESTRATEGIAS
# ==========================================
def map_strategies(regime):
    mapping = {
        "LOW_VOLATILITY / TRENDING": {
            "Active": ["Momentum", "Breakout", "Trend Following"],
            "Standby": ["Mean Reversion", "Tail Risk Hedging"]
        },
        "NEUTRAL / MEAN_REVERTING": {
            "Active": ["Statistical Arbitrage", "Pairs Trading", "Grid Trading"],
            "Standby": ["Aggressive Momentum"]
        },
        "HIGH_VOLATILITY / CRISIS": {
            "Active": ["Tail Risk Hedging", "Short Volatility Spreads", "Cash"],
            "Standby": ["Long-Only Momentum", "Unhedged Exposure"]
        }
    }
    return mapping.get(regime, {"Active": [], "Standby": []})

# ==========================================
# 5. ACTUALIZACIÓN DE ARCHIVOS DE CONTEXTO (.md)
# ==========================================
def update_context_files(regime, probabilities, ranked_stocks, strategies, date_str):
    prob_lines = [f"- **{k}:** {v:.1%}" for k, v in probabilities.items()]
    prob_str = "\n".join(prob_lines)
    top_candidates = ranked_stocks.head(5)
    
    table_str = "| Ticker | Combined Score | Fundamental Score | Technical Score | Forward P/E |\n| :--- | :--- | :--- | :--- | :--- |\n"
    for ticker, row in top_candidates.iterrows():
        table_str += f"| **{ticker}** | {row['Total_Score']} | {row['F_Score']}/100 | {row['T_Score']}/100 | {row['P_E']} |\n"

    current_content = f"""# Current Market Regime & Portfolio Diagnosis

- **Date:** {date_str}
- **Last Run:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Active Regime:** {regime}

## HMM State Probabilities
{prob_str}

## Active Strategy Status
- **ACTIVE:** {', '.join(strategies['Active'])}
- **STANDBY:** {', '.join(strategies['Standby'])}

## Recommended Stock Portfolio (Hybrid Scoring)
{table_str}

## Manual Execution Plan
1. Review the top candidates listed above.
2. Check position sizing manually before placing broker orders.
"""
    
    with open("current-regime.md", "w", encoding="utf-8") as f:
        f.write(current_content)
        
    history_file = "regime-history.md"
    if not os.path.exists(history_file):
        with open(history_file, "w", encoding="utf-8") as f:
            f.write("# Market Regime History Log\n\n| Date | Detected Regime | Max Probability | Active Strategies |\n| :--- | :--- | :--- | :--- |\n")
            
    max_prob = max(probabilities.values())
    history_line = f"| {date_str} | **{regime}** | {max_prob:.1%} | {', '.join(strategies['Active'])} |\n"
    with open(history_file, "a", encoding="utf-8") as f:
        f.write(history_line)

# ==========================================
# PIPELINE PRINCIPAL
# ==========================================
def get_expanded_universe():
    return [
       
    # Tecnología / Magnificent 7 + AI
    "NVDA", "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "META", "TSLA",
    "AVGO", "AMD", "ORCL", "CRM", "ADBE", "INTC", "QCOM", "TXN",
    "MU", "AMAT", "LRCX", "KLAC", "SNPS", "CDNS", "NOW", "INTU",
    "PANW", "CRWD", "SNOW", "PLTR", "ANET",

    # Semiconductores / Hardware adicional
    "TSM", "ASML", "DELL", "HPE", "HPQ",

    # Financieras
    "JPM", "BAC", "WFC", "C", "GS", "MS", "BLK", "SCHW",
    "V", "MA", "AXP", "PYPL", "BRK-B", "SPGI", "MCO",

    # Salud / Farmacéuticas / Biotech
    "LLY", "UNH", "JNJ", "ABBV", "MRK", "PFE", "AMGN", "GILD",
    "BMY", "TMO", "DHR", "ISRG", "SYK", "BSX", "MDT", "CI",
    "ELV", "CVS", "REGN", "VRTX",

    # Consumo Discretionary
    "HD", "LOW", "NKE", "SBUX", "MCD", "BKNG", "CMG", "TJX",
    "ROST", "ORLY", "AZO", "ULTA", "MAR", "HLT",

    # Consumo Staples
    "WMT", "COST", "PG", "KO", "PEP", "PM", "MO", "CL",
    "KMB", "GIS", "KHC", "MDLZ",

    # Energía
    "XOM", "CVX", "COP", "EOG", "SLB", "OXY", "MPC", "VLO",
    "PSX",

    # Industriales
    "CAT", "DE", "GE", "HON", "UNP", "UPS", "FDX", "BA",
    "LMT", "RTX", "GD", "NOC", "EMR", "ETN", "ITW", "PH",

    # Comunicaciones / Media
    "NFLX", "DIS", "CMCSA", "T", "VZ", "TMUS", "CHTR",

    # Utilities / Real Estate / Otros de calidad
    "NEE", "DUK", "SO", "D", "AEP", "SRE",
    "AMT", "PLD", "EQIX", "CCI",
    "IBM", "CSCO", "ACN", "ADP", "PAYX", "FIS", "FISV"
]
    

# Eliminamos duplicados por si acaso al concatenar listas
expanded_universe = list(set(get_expanded_universe()))

def run_pipeline():
    universe = list(set(get_expanded_universe()))
    
    logging.info(f"1. Fetching macro data and scoring {len(universe)} stocks...")
    df, X = fetch_macro_data(period="5y")
    
    logging.info("2. Classifying regime via HMM...")
    regime, probabilities, date_str = fit_hmm_regimes(df, X, n_components=3)
    
    logging.info("3. Scoring candidates...")
    ranked_stocks = rank_portfolio_candidates(universe, regime)
    strategies = map_strategies(regime)
    
    logging.info("4. Writing markdown context files...")
    update_context_files(regime, probabilities, ranked_stocks, strategies, date_str)
    
    logging.info("5. Sending Telegram notification...")
    send_telegram_alert(regime, probabilities, strategies, ranked_stocks, date_str)
    
    logging.info(f"Pipeline finished [{date_str}]. Active regime: {regime}")
 

if __name__ == "__main__":
    run_pipeline()