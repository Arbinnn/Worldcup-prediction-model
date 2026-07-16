"""Streamlit front-end: pick an algorithm, predict the WC2026 final + third-place match.

Run: streamlit run app.py   (needs models/ -- run train_model.py first)
"""
import streamlit as st

from train_model import ALGORITHMS, DISPLAY_NAMES, load_artifacts, load_tables, predict_matchup

DEFAULT_ALGO = "ensemble"

FLAG = {"Argentina": "🇦🇷", "Spain": "🇪🇸", "England": "🏴󠁧󠁢󠁥󠁮󠁧󠁿", "France": "🇫🇷"}
FIXTURES = [("🏆 Final", "Argentina", "Spain"), ("🥉 Third Place", "England", "France")]

st.set_page_config(page_title="World Cup 2026 Predictor", page_icon="⚽", layout="centered")
st.markdown("""
<style>
  .stApp { background: linear-gradient(160deg, #0b3d20 0%, #14512c 60%, #0b3d20 100%); }
  .card { background: rgba(255,255,255,.06); border: 1px solid rgba(255,255,255,.15);
          border-radius: 14px; padding: 1.1rem 1.3rem; margin-bottom: 1rem; }
  .team { font-size: 1.15rem; font-weight: 600; }
  .snap { color: #b7d8c2; font-size: .8rem; }
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


def match_card(label, team_a, team_b, algorithm):
    r = _predict(team_a, team_b, algorithm)
    st.markdown(f"<div class='card'><h3>{label}: {FLAG[team_a]} {team_a} vs {FLAG[team_b]} {team_b}</h3>",
                unsafe_allow_html=True)
    for team, p, ctx in [(team_a, r["p_a"], r["ctx_a"]), (team_b, r["p_b"], r["ctx_b"])]:
        st.markdown(f"<span class='team'>{FLAG[team]} {team} — {p:.1%}</span>", unsafe_allow_html=True)
        st.progress(min(max(p, 0.0), 1.0))
        st.markdown(f"<span class='snap'>{snapshot(ctx)}</span>", unsafe_allow_html=True)
    fav, p_fav = (team_a, r["p_a"]) if r["p_a"] >= r["p_b"] else (team_b, r["p_b"])
    st.markdown(f"**{fav} favored, {p_fav:.0%}** ({DISPLAY_NAMES[algorithm]})")
    draw = f" · {r['p_draw']:.0%} draw in 90' before the knockout split" if r["p_draw"] else ""
    st.caption(f"Model: {DISPLAY_NAMES[algorithm]} · neutral venue · draw mass split 50/50{draw}")

    with st.expander("Compare all algorithms"):
        for alg in ALGORITHMS:
            o = _predict(team_a, team_b, alg)
            st.write(f"{DISPLAY_NAMES[alg]}: {team_a} {o['p_a']:.1%} — {team_b} {o['p_b']:.1%}")
    st.markdown("</div>", unsafe_allow_html=True)


st.title("⚽ World Cup 2026 — Match Predictor")
st.caption("Trained through the semifinals · Elo, form, tournament profile and squad rates")

choice = st.selectbox("Algorithm", ALGORITHMS, index=ALGORITHMS.index(DEFAULT_ALGO),
                      format_func=lambda a: DISPLAY_NAMES[a])

if st.button("⚽ Start Prediction", type="primary", use_container_width=True):
    with st.spinner("Crunching the numbers..."):
        for label, a, b in FIXTURES:
            match_card(label, a, b, choice)
