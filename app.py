"""Streamlit front-end: pick an algorithm, predict the WC2026 final + third-place match.

Run: streamlit run app.py   (needs models/ -- run train_model.py first)
"""
import streamlit as st

from train_model import ALGORITHMS, DISPLAY_NAMES, load_artifacts, load_tables, predict_matchup

DEFAULT_ALGO = "ensemble"

CODE = {"Argentina": "ARG", "Spain": "ESP", "England": "ENG", "France": "FRA"}
FIXTURES = [("Final", "Argentina", "Spain"), ("Third Place", "England", "France")]

st.set_page_config(page_title="World Cup 2026 Predictor", page_icon=":soccer:", layout="centered")
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;600;700&family=Inter:wght@400;500;600&display=swap');

  .stApp {
    background:
      radial-gradient(900px 500px at 50% -10%, rgba(214,168,74,.18), transparent 70%),
      linear-gradient(170deg, #04120a 0%, #0a2f19 45%, #04120a 100%);
  }
  /* pitch stripes */
  .stApp::before {
    content: ""; position: fixed; inset: 0; pointer-events: none; opacity: .05;
    background: repeating-linear-gradient(90deg, #fff 0 60px, transparent 60px 120px);
  }
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  .hero { text-align: center; padding: 1.6rem 0 .4rem; }
  .hero .kicker {
    font-family: 'Barlow Condensed', sans-serif; letter-spacing: .38em;
    font-size: .78rem; font-weight: 600; color: #d6a84a; text-transform: uppercase;
  }
  .hero h1 {
    font-family: 'Barlow Condensed', sans-serif; font-size: 3.4rem; line-height: 1;
    font-weight: 700; letter-spacing: .02em; margin: .3rem 0 .5rem;
    color: #f6f8f5; text-transform: uppercase;
  }
  .hero .rule {
    width: 120px; height: 3px; margin: 0 auto 1rem;
    background: linear-gradient(90deg, transparent, #d6a84a, transparent);
  }
  .hero .sub { color: #9fc4ae; font-size: .88rem; }

  .card {
    background: linear-gradient(180deg, rgba(255,255,255,.07), rgba(255,255,255,.03));
    border: 1px solid rgba(214,168,74,.28); border-left: 3px solid #d6a84a;
    border-radius: 4px; padding: 1.2rem 1.4rem 1rem; margin: 1.2rem 0;
    box-shadow: 0 18px 40px rgba(0,0,0,.35);
  }
  .card .stage {
    font-family: 'Barlow Condensed', sans-serif; text-transform: uppercase;
    letter-spacing: .3em; font-size: .72rem; color: #d6a84a; font-weight: 600;
  }
  .card .fixture {
    font-family: 'Barlow Condensed', sans-serif; font-size: 1.9rem; font-weight: 700;
    color: #f6f8f5; text-transform: uppercase; letter-spacing: .01em; margin-top: .1rem;
  }
  .card .fixture .vs { color: #6f9781; margin: 0 .5rem; font-size: 1.2rem; }

  .row { display: flex; align-items: baseline; justify-content: space-between; margin-top: .9rem; }
  .row .name { font-weight: 600; color: #f6f8f5; font-size: 1rem; }
  .row .code {
    font-family: 'Barlow Condensed', sans-serif; font-weight: 700; color: #d6a84a;
    letter-spacing: .12em; margin-right: .55rem;
  }
  .row .pct { font-family: 'Barlow Condensed', sans-serif; font-size: 1.35rem;
              font-weight: 700; color: #f6f8f5; }

  .bar { height: 7px; border-radius: 99px; background: rgba(255,255,255,.09); overflow: hidden; margin-top: .35rem; }
  .bar > i { display: block; height: 100%; border-radius: 99px;
             background: linear-gradient(90deg, #1f7a45, #d6a84a); }
  .bar.dim > i { background: linear-gradient(90deg, #2a4d3a, #6f9781); }

  .snap { color: #82a891; font-size: .74rem; letter-spacing: .02em; margin-top: .35rem; display: block; }

  .verdict {
    margin-top: 1.1rem; padding-top: .8rem; border-top: 1px solid rgba(214,168,74,.2);
    font-family: 'Barlow Condensed', sans-serif; text-transform: uppercase;
    letter-spacing: .1em; font-size: 1.05rem; color: #f6f8f5;
  }
  .verdict b { color: #d6a84a; }

  .stButton > button {
    font-family: 'Barlow Condensed', sans-serif; text-transform: uppercase;
    letter-spacing: .22em; font-weight: 700; font-size: 1rem;
    background: #d6a84a; color: #04120a; border: 0; border-radius: 3px; padding: .7rem 0;
  }
  .stButton > button:hover { background: #e8bd63; color: #04120a; }
</style>""", unsafe_allow_html=True)


@st.cache_resource
def _artifacts():
    return load_artifacts()


@st.cache_resource
def _tables():
    return load_tables()


@st.cache_data
def _predict(team_a, team_b, algorithm):
    return predict_matchup(team_a, team_b, algorithm, neutral=True,
                           artifacts=_artifacts(), tables=_tables())


def snapshot(ctx):
    star = ctx.get("squad_star_power")
    star = f"{star:.2f} G+A/app" if star == star and star is not None else "n/a"
    return f"Elo {ctx['elo']:.0f} · form(5) {ctx['form_pts_5']:.2f} · star power {star}"


def team_row(team, p, ctx, favored):
    pct = min(max(p, 0.0), 1.0)
    return f"""
      <div class='row'>
        <span class='name'><span class='code'>{CODE[team]}</span>{team}</span>
        <span class='pct'>{p:.1%}</span>
      </div>
      <div class='bar {"" if favored else "dim"}'><i style='width:{pct:.1%}'></i></div>
      <span class='snap'>{snapshot(ctx)}</span>"""


def match_card(label, team_a, team_b, algorithm):
    r = _predict(team_a, team_b, algorithm)
    fav, p_fav = (team_a, r["p_a"]) if r["p_a"] >= r["p_b"] else (team_b, r["p_b"])
    draw = f" &middot; {r['p_draw']:.0%} draw in 90' before the knockout split" if r["p_draw"] else ""
    st.markdown(f"""
    <div class='card'>
      <div class='stage'>{label}</div>
      <div class='fixture'>{team_a}<span class='vs'>vs</span>{team_b}</div>
      {team_row(team_a, r["p_a"], r["ctx_a"], fav == team_a)}
      {team_row(team_b, r["p_b"], r["ctx_b"], fav == team_b)}
      <div class='verdict'><b>{fav}</b> favored &middot; {p_fav:.0%}</div>
      <span class='snap'>{DISPLAY_NAMES[algorithm]} &middot; neutral venue &middot; draw mass split 50/50{draw}</span>
    </div>""", unsafe_allow_html=True)

    with st.expander("Compare all algorithms"):
        for alg in ALGORITHMS:
            o = _predict(team_a, team_b, alg)
            st.write(f"{DISPLAY_NAMES[alg]}: {team_a} {o['p_a']:.1%} — {team_b} {o['p_b']:.1%}")


st.markdown("""
<div class='hero'>
  <div class='kicker'>FIFA World Cup 2026</div>
  <h1>Match Predictor</h1>
  <div class='rule'></div>
  <div class='sub'>Trained through the semifinals &middot; Elo, form, tournament profile and squad rates</div>
</div>""", unsafe_allow_html=True)

choice = st.selectbox("Algorithm", ALGORITHMS, index=ALGORITHMS.index(DEFAULT_ALGO),
                      format_func=lambda a: DISPLAY_NAMES[a])

if st.button("Start Prediction", type="primary", use_container_width=True):
    with st.spinner("Crunching the numbers..."):
        for label, a, b in FIXTURES:
            match_card(label, a, b, choice)
