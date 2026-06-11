"""
app.py — FIFA World Cup 2026 Match Predictor
Deploy: Hugging Face Spaces (SDK: Gradio)

Before deploying, upload these files to your Space:
  - wc2026_model.pkl
  - wc2026_scaler.pkl
  - wc2026_features.pkl
  - wc2026.db  (optional — will be created fresh if missing)
"""

import gradio as gr
import pandas as pd
import numpy as np
import sqlite3
import pickle
import os

# ── Load model ─────────────────────────────────────────────────────────────────
with open("wc2026_model.pkl",    "rb") as f: model    = pickle.load(f)
with open("wc2026_scaler.pkl",   "rb") as f: scaler   = pickle.load(f)
with open("wc2026_features.pkl", "rb") as f: FEATURES = pickle.load(f)

MODEL_NAME = type(model).__name__

# ── Elo ratings (pre-tournament snapshot June 2026) ───────────────────────────
ELO_2026 = {
    "Spain":2155,"Argentina":2113,"France":2062,"England":2020,"Brazil":1988,
    "Portugal":1984,"Colombia":1977,"Netherlands":1944,"Germany":1925,"Belgium":1900,
    "Morocco":1880,"Italy":1870,"Uruguay":1860,"Croatia":1855,"Japan":1850,
    "Mexico":1845,"United States":1840,"Ecuador":1820,"Senegal":1815,"Australia":1810,
    "Switzerland":1805,"Denmark":1800,"Serbia":1795,"Poland":1790,"South Korea":1785,
    "Tunisia":1775,"Canada":1770,"Costa Rica":1755,"Cameroon":1750,"Ghana":1745,
    "Iran":1740,"Saudi Arabia":1715,"Qatar":1700,"Panama":1695,"Venezuela":1690,
    "Paraguay":1685,"Peru":1680,"Chile":1675,"Algeria":1670,"Egypt":1665,
    "Nigeria":1660,"South Africa":1650,"Kenya":1600,"New Zealand":1580,
    "Indonesia":1560,"Honduras":1555,"Jamaica":1550,"Haiti":1530,"Angola":1525,"Ukraine":1810,
}

WC2026_TEAMS = sorted(ELO_2026.keys())

# ── Matches ────────────────────────────────────────────────────────────────────
WC2026_MATCHES = [
    (1,"United States","Serbia",   "2026-06-12","Group A"),
    (2,"Panama",       "Mexico",   "2026-06-12","Group A"),
    (3,"Argentina",    "Morocco",  "2026-06-13","Group B"),
    (4,"Angola",       "Ukraine",  "2026-06-13","Group B"),
    (5,"France",       "Japan",    "2026-06-13","Group C"),
    (6,"Paraguay",     "Saudi Arabia","2026-06-13","Group C"),
    (7,"Brazil",       "Germany",  "2026-06-14","Group D"),
    (8,"Switzerland",  "Chile",    "2026-06-14","Group D"),
    (9,"Spain",        "Netherlands","2026-06-14","Group E"),
    (10,"England",     "Nigeria",  "2026-06-14","Group E"),
    (11,"Portugal",    "Colombia", "2026-06-15","Group F"),
    (12,"Belgium",     "Uruguay",  "2026-06-15","Group F"),
    (13,"United States","Panama",  "2026-06-18","Group A"),
    (14,"Mexico",      "Serbia",   "2026-06-18","Group A"),
    (15,"Argentina",   "Angola",   "2026-06-19","Group B"),
    (16,"Ukraine",     "Morocco",  "2026-06-19","Group B"),
    (17,"France",      "Paraguay", "2026-06-19","Group C"),
    (18,"Saudi Arabia","Japan",    "2026-06-19","Group C"),
    (19,"Brazil",      "Switzerland","2026-06-20","Group D"),
    (20,"Chile",       "Germany",  "2026-06-20","Group D"),
    (21,"Spain",       "England",  "2026-06-20","Group E"),
    (22,"Nigeria",     "Netherlands","2026-06-20","Group E"),
    (23,"Portugal",    "Belgium",  "2026-06-21","Group F"),
    (24,"Uruguay",     "Colombia", "2026-06-21","Group F"),
]

# ── Database setup ─────────────────────────────────────────────────────────────
DB_PATH = "wc2026.db"

def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            total_points INTEGER DEFAULT 0
        )""")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS matches (
            id INTEGER PRIMARY KEY,
            home_team TEXT, away_team TEXT,
            match_date TEXT, stage TEXT,
            actual_result TEXT, home_score INTEGER, away_score INTEGER,
            ai_prediction TEXT, ai_confidence REAL
        )""")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER, match_id INTEGER,
            prediction TEXT,
            pred_home_score INTEGER, pred_away_score INTEGER,
            points_earned INTEGER DEFAULT 0,
            UNIQUE(user_id, match_id)
        )""")
    conn.commit()
    for mid, h, a, dt, stage in WC2026_MATCHES:
        conn.execute(
            "INSERT OR IGNORE INTO matches (id,home_team,away_team,match_date,stage) VALUES (?,?,?,?,?)",
            (mid, h, a, dt, stage)
        )
    conn.commit()
    return conn

conn = get_conn()

# ── Prediction engine ──────────────────────────────────────────────────────────
def predict_match(home_team, away_team):
    h_elo    = ELO_2026.get(home_team, 1600)
    a_elo    = ELO_2026.get(away_team, 1600)
    elo_diff = h_elo - a_elo

    fv = np.array([[1.3, 1.1, 0.40, 1.1, 1.3, 0.35,
                    elo_diff, h_elo, a_elo, 0]], dtype=float)

    if MODEL_NAME == "LogisticRegression":
        fv = scaler.transform(fv)

    proba = model.predict_proba(fv)[0]
    pred  = model.predict(fv)[0]

    label_map  = {0:"Home Win", 1:"Draw", 2:"Away Win"}
    proba_dict = {
        f"{home_team} Win": round(proba[0]*100, 1),
        "Draw":              round(proba[1]*100, 1),
        f"{away_team} Win": round(proba[2]*100, 1),
    }
    return label_map[pred], proba_dict, elo_diff, h_elo, a_elo


def ai_analysis(home_team, away_team):
    pred, proba, elo_diff, h_elo, a_elo = predict_match(home_team, away_team)
    try:
        import anthropic
        client = anthropic.Anthropic()
        msg = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=180,
            messages=[{"role":"user","content":(
                f"Analiza el partido Copa del Mundo 2026: {home_team} vs {away_team}. "
                f"Elo {home_team}={h_elo}, Elo {away_team}={a_elo}, diferencia={elo_diff:+d}. "
                f"Predicción: {pred} ({proba}). "
                f"Genera 2 oraciones en español, estilo periodismo deportivo."
            )}]
        )
        return msg.content[0].text
    except Exception:
        stronger = home_team if elo_diff > 0 else away_team
        max_k    = max(proba, key=proba.get)
        return (
            f"{stronger} llega con ventaja Elo de {abs(elo_diff)} puntos "
            f"({h_elo} vs {a_elo}). "
            f"El modelo proyecta {max_k} con {proba[max_k]}% de probabilidad."
        )


# ── App functions ──────────────────────────────────────────────────────────────
def app_predict(home_team, away_team):
    if home_team == away_team:
        return "❌ Selecciona dos equipos diferentes.", "", ""
    pred, proba, elo_diff, h_elo, a_elo = predict_match(home_team, away_team)
    emoji   = {"Home Win":"🏠✅","Draw":"🤝","Away Win":"✈️✅"}
    res_str = f"{emoji.get(pred,'')} **Predicción: {pred}**"
    prob_md = "| Resultado | Probabilidad |\n|-----------|:------------:|\n"
    for k, v in proba.items():
        prob_md += f"| {k} | {'█'*int(v/5)} {v}% |\n"
    elo_str = (
        f"**Elo {home_team}**: {h_elo}  \n"
        f"**Elo {away_team}**: {a_elo}  \n"
        f"**Diferencia**: {elo_diff:+d}"
    )
    analysis = ai_analysis(home_team, away_team)
    return res_str, prob_md + "\n" + elo_str, f"🤖 {analysis}"


def app_matches():
    rows = conn.execute(
        "SELECT id,home_team,away_team,match_date,stage,ai_prediction,ai_confidence "
        "FROM matches ORDER BY match_date LIMIT 24"
    ).fetchall()
    md = "| # | Partido | Fecha | Fase | IA | Confianza |\n"
    md += "|---|---------|-------|------|:--:|:---------:|\n"
    for mid, h, a, dt, s, aip, aic in rows:
        conf = f"{aic:.1f}%" if aic else "—"
        md  += f"| {mid} | {h} vs {a} | {dt} | {s} | {aip or '—'} | {conf} |\n"
    return md


def app_submit(username, match_id, prediction, home_score, away_score):
    if not username.strip():
        return "❌ Ingresa un nombre de usuario."
    try:
        conn.execute("INSERT OR IGNORE INTO users (username) VALUES (?)", (username.strip(),))
        conn.commit()
        uid = conn.execute("SELECT id FROM users WHERE username=?", (username.strip(),)).fetchone()[0]
        try:
            hs  = int(home_score) if str(home_score).strip() not in ("","None") else None
            as_ = int(away_score) if str(away_score).strip() not in ("","None") else None
        except (ValueError, TypeError):
            hs = as_ = None
        conn.execute(
            "INSERT OR IGNORE INTO predictions (user_id,match_id,prediction,pred_home_score,pred_away_score) VALUES (?,?,?,?,?)",
            (uid, int(match_id), prediction, hs, as_)
        )
        conn.commit()
        return f"✅ **{username}** predijo Partido #{int(match_id)}: {prediction}"
    except Exception as e:
        return f"❌ Error: {e}"


def app_leaderboard():
    rows = conn.execute("""
        SELECT u.username, u.total_points, COUNT(p.id),
               SUM(CASE WHEN p.points_earned >= 3 THEN 1 ELSE 0 END)
        FROM users u
        LEFT JOIN predictions p ON u.id = p.user_id
        GROUP BY u.id ORDER BY u.total_points DESC
    """).fetchall()
    if not rows:
        return "Sin predicciones aún. ¡Sé el primero!"
    md  = "| Pos | Usuario | Puntos | Predicciones | Correctas |\n"
    md += "|:---:|---------|:------:|:------------:|:---------:|\n"
    medals = ["🥇","🥈","🥉"]
    for i, (u, p, pr, c) in enumerate(rows, 1):
        medal = medals[i-1] if i <= 3 else str(i)
        md   += f"| {medal} | {u} | {p} | {pr} | {c or 0} |\n"
    return md


# ── Gradio UI ──────────────────────────────────────────────────────────────────
with gr.Blocks(
    title="⚽ WC 2026 Predictor",
    theme=gr.themes.Soft(primary_hue="blue")
) as demo:

    gr.Markdown(
        "# ⚽ FIFA World Cup 2026 — Match Predictor\n"
        "### Powered by Machine Learning + Claude AI\n"
        "*Predictions based on 49,000+ historical matches and Elo ratings.*"
    )

    with gr.Tabs():

        with gr.TabItem("🤖 Predicción IA"):
            with gr.Row():
                h_dd = gr.Dropdown(WC2026_TEAMS, label="🏠 Equipo Local",    value="Argentina")
                a_dd = gr.Dropdown(WC2026_TEAMS, label="✈️ Equipo Visitante", value="France")
            btn    = gr.Button("🔮 Predecir resultado", variant="primary")
            r_out  = gr.Markdown(label="Resultado")
            p_out  = gr.Markdown(label="Probabilidades")
            an_out = gr.Markdown(label="Análisis IA")
            btn.click(fn=app_predict, inputs=[h_dd, a_dd], outputs=[r_out, p_out, an_out])

        with gr.TabItem("📅 Próximos Partidos"):
            m_out = gr.Markdown()
            gr.Button("🔄 Actualizar", variant="secondary").click(fn=app_matches, outputs=m_out)
            demo.load(fn=app_matches, outputs=m_out)

        with gr.TabItem("📝 Mi Predicción"):
            u_in   = gr.Textbox(label="👤 Usuario", placeholder="ej: futbolero99")
            mid_in = gr.Number(label="🆔 ID Partido (ver tabla Próximos Partidos)", value=1, precision=0)
            p_in   = gr.Radio(["Home Win","Draw","Away Win"], label="🎯 Tu predicción", value="Home Win")
            with gr.Row():
                hs_in = gr.Number(label="⚽ Goles local (opcional, +2 pts)", value=None, precision=0)
                as_in = gr.Number(label="⚽ Goles visitante (opcional)",     value=None, precision=0)
            s_out = gr.Markdown()
            gr.Button("📤 Enviar predicción", variant="primary").click(
                fn=app_submit, inputs=[u_in, mid_in, p_in, hs_in, as_in], outputs=s_out
            )

        with gr.TabItem("🏆 Leaderboard"):
            lb_out = gr.Markdown()
            gr.Button("🔄 Actualizar", variant="secondary").click(fn=app_leaderboard, outputs=lb_out)
            demo.load(fn=app_leaderboard, outputs=lb_out)

    gr.Markdown(
        "---\n"
        "**Scoring**: +3 resultado correcto · +2 marcador exacto · 0 fallo  \n"
        "*Data: martj42/Kaggle (CC BY-SA) · Elo: eloratings.net*"
    )

if __name__ == "__main__":
    demo.launch()
