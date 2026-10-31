import json
import struct
import time
from datetime import datetime, timezone

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pyodbc
import requests
import streamlit as st
import streamlit.components.v1 as components
from azure.identity import InteractiveBrowserCredential
from plotly.subplots import make_subplots
from streamlit_autorefresh import st_autorefresh

SERVER = "oe3xc2subvwevo757s5gzbossa-nby64cvvc6benanjye2rzyyu34.datawarehouse.fabric.microsoft.com"
DATABASE = "Retail_Clickstream_Lakehouse"
USER = "2401526@rst.edu.eg"
TRIGGER_SECONDS = 10
WORKSPACE_ID = "0aee7168-17b5-4682-81a9-c1351ce314df"

C = {
    "blue": "#5b8cff", "cyan": "#22d3ee", "green": "#34d399", "amber": "#fbbf24",
    "red": "#fb7185", "purple": "#a78bfa", "pink": "#f472b6", "gray": "#94a3b8",
}
PALETTE = [C["blue"], C["purple"], C["cyan"], C["amber"], C["pink"], C["green"], C["red"]]
WINDOWS = {"Last 15 min": 15, "Last 1 hour": 60, "Last 6 hours": 360, "All data": None}

st.set_page_config(page_title="Clickstream Anomaly Monitor", page_icon="🛰️", layout="wide",
                   initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&family=JetBrains+Mono:wght@600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp {
    background:
        radial-gradient(1200px 600px at 10% -10%, rgba(91,140,255,.18), transparent 60%),
        radial-gradient(900px 500px at 100% 0%, rgba(167,139,250,.16), transparent 60%),
        radial-gradient(800px 600px at 50% 120%, rgba(34,211,238,.10), transparent 60%),
        #070a14;
}
header[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer, [data-testid="stToolbar"] { visibility: hidden; }
.block-container { padding-top: 1rem; padding-bottom: 3rem; max-width: 1560px; }
iframe { border-radius: 20px; }
.section { display:flex; align-items:center; gap:10px; margin:4px 0 8px; font-size:1.02rem; font-weight:700; color:#e2e8f0; }
.section::before { content:""; width:6px; height:20px; border-radius:4px; background:linear-gradient(180deg,#5b8cff,#a78bfa); }
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: linear-gradient(145deg, rgba(255,255,255,.045), rgba(255,255,255,.012));
    border: 1px solid rgba(255,255,255,.07) !important; border-radius: 20px !important;
    box-shadow: 0 20px 50px -30px rgba(0,0,0,.8);
}
.kpi { position: relative; overflow: hidden; border-radius: 16px; padding: 14px 16px; margin-bottom: 12px; min-height: 112px;
  background: linear-gradient(145deg, rgba(255,255,255,.06), rgba(255,255,255,.02));
  border: 1px solid rgba(255,255,255,.08); transition: transform .25s, box-shadow .25s; animation: rise .6s ease both; }
.kpi::before { content: ""; position: absolute; inset: 0 auto 0 0; width: 4px; background: linear-gradient(180deg, var(--c), transparent); }
.kpi::after { content: ""; position: absolute; width: 140px; height: 140px; right: -55px; top: -65px;
  background: radial-gradient(circle, color-mix(in srgb, var(--c) 38%, transparent), transparent 70%); opacity: .45; }
.kpi:hover { transform: translateY(-5px); box-shadow: 0 16px 34px -14px color-mix(in srgb, var(--c) 65%, transparent); }
@keyframes rise { from { opacity: 0; transform: translateY(14px); } to { opacity: 1; transform: none; } }
.kpi .label { color: #94a3b8; font-size: .66rem; font-weight: 700; text-transform: uppercase; letter-spacing: .08em; position: relative; z-index: 1; }
.kpi .value { font-family: 'JetBrains Mono', monospace; font-size: 1.5rem; font-weight: 700; color: #f8fafc; margin-top: 8px;
  text-shadow: 0 0 18px color-mix(in srgb, var(--c) 40%, transparent); position: relative; z-index: 1; }
.kpi .sub { color: #94a3b8; font-size: .74rem; margin-top: 4px; position: relative; z-index: 1; }
.stTabs [data-baseweb="tab-list"] { gap: 8px; padding: 4px 0 12px; }
.stTabs [data-baseweb="tab"] {
    height: 44px; padding: 0 22px; border-radius: 999px !important; color: #94a3b8; font-weight: 600;
    background: rgba(255,255,255,.04); border: 1px solid rgba(255,255,255,.07); transition: all .2s ease;
}
.stTabs [data-baseweb="tab"] p { font-size: .92rem; }
.stTabs [data-baseweb="tab"]:hover { color: #e2e8f0; border-color: rgba(91,140,255,.5); transform: translateY(-2px); }
.stTabs [aria-selected="true"] {
    color: #fff !important; border-color: transparent !important;
    background: linear-gradient(90deg, #5b8cff, #a78bfa) !important; box-shadow: 0 8px 24px -8px rgba(124,108,255,.8);
}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] {
    display: none !important; background: transparent !important; height: 0 !important;
}
[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }
[data-baseweb="tag"] { background: linear-gradient(90deg, #5b8cff, #a78bfa) !important; border-radius: 999px !important; }
[data-baseweb="tag"] span { color: #fff !important; font-weight: 600; }
[data-baseweb="select"] > div { background: rgba(255,255,255,.04) !important; border-color: rgba(255,255,255,.1) !important; border-radius: 12px !important; }
div[role="radiogroup"] { flex-wrap: nowrap !important; gap: 14px; }
div[role="radiogroup"] label { white-space: nowrap; }
</style>
""", unsafe_allow_html=True)

HERO_TEMPLATE = r"""
<!doctype html><html><head><meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap" rel="stylesheet">
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: Inter, sans-serif; color: #e2e8f0; background: transparent; overflow: hidden; }
.wrap { padding: 4px 4px 0; }
.top { display: flex; justify-content: space-between; align-items: flex-start; gap: 16px; }
.title { font-size: 2.2rem; font-weight: 800; letter-spacing: -.02em;
  background: linear-gradient(90deg,#5b8cff,#a78bfa,#22d3ee,#f472b6,#5b8cff); background-size: 400% 100%;
  -webkit-background-clip: text; background-clip: text; color: transparent; animation: shine 10s linear infinite; }
@keyframes shine { to { background-position: 400% 0; } }
.sub { color: #94a3b8; font-size: .84rem; margin-top: 6px; }
.sub b { color: #cbd5e1; }
.win { display: inline-block; margin-top: 8px; padding: 4px 12px; border-radius: 999px; font-size: .78rem; font-weight: 700;
  color: #c4b5fd; background: rgba(167,139,250,.12); border: 1px solid rgba(167,139,250,.35); }
.status { display: inline-flex; align-items: center; gap: 10px; padding: 10px 20px; border-radius: 999px;
  font-weight: 800; letter-spacing: .14em; color: var(--sc); border: 1px solid var(--sc);
  background: color-mix(in srgb, var(--sc) 12%, transparent); animation: glow 2.2s ease-in-out infinite; }
.status .dot { width: 10px; height: 10px; border-radius: 50%; background: var(--sc); animation: blink 1.1s infinite; }
@keyframes glow { 0%,100% { box-shadow: 0 0 0 0 transparent; } 50% { box-shadow: 0 0 28px 3px color-mix(in srgb, var(--sc) 50%, transparent); } }
@keyframes blink { 50% { opacity: .2; transform: scale(.6); } }
.smeta { color: #94a3b8; font-size: .78rem; margin-top: 8px; text-align: right; line-height: 1.5; }
.smeta .up { color: #34d399; font-family: 'JetBrains Mono', monospace; font-weight: 700; }
.pipe { margin-top: 14px; padding: 14px 16px; border-radius: 20px;
  background: linear-gradient(145deg, rgba(255,255,255,.05), rgba(255,255,255,.015));
  border: 1px solid rgba(255,255,255,.08); display: flex; align-items: center; }
.node { flex: 0 0 auto; width: 140px; text-align: center; position: relative; }
.node .bubble { width: 50px; height: 50px; margin: 0 auto; border-radius: 15px; display: grid; place-items: center;
  font-size: 1.4rem; background: rgba(91,140,255,.12); border: 1px solid rgba(91,140,255,.35);
  box-shadow: 0 0 24px -6px rgba(91,140,255,.6); animation: float 4s ease-in-out infinite; animation-delay: var(--d); }
@keyframes float { 50% { transform: translateY(-4px); } }
.node .name { font-weight: 700; font-size: .82rem; margin-top: 8px; }
.node .desc { color: #94a3b8; font-size: .68rem; margin-top: 2px; }
.node .num { font-family: 'JetBrains Mono', monospace; font-weight: 700; font-size: .92rem; margin-top: 6px; color: #f8fafc; }
.node .badge { position: absolute; top: -6px; right: 14px; background: #fb7185; color: #0b0f1c; font-weight: 800;
  font-size: .62rem; padding: 2px 7px; border-radius: 999px; box-shadow: 0 0 14px rgba(251,113,133,.7); }
.link { flex: 1 1 auto; height: 4px; margin: 0 4px 46px; border-radius: 4px; position: relative; overflow: hidden;
  background: rgba(255,255,255,.07); }
.link::before { content: ""; position: absolute; inset: 0; border-radius: 4px;
  background: linear-gradient(90deg, transparent, var(--sc), transparent); background-size: 50% 100%;
  background-repeat: no-repeat; animation: flow 1.4s linear infinite; }
.link::after { content: ""; position: absolute; top: -3px; width: 10px; height: 10px; border-radius: 50%;
  background: var(--sc); box-shadow: 0 0 12px var(--sc); animation: dot 1.4s linear infinite; }
@keyframes flow { from { background-position: -50% 0; } to { background-position: 150% 0; } }
@keyframes dot { from { left: -10px; } to { left: 100%; } }
.stalled .link::before, .stalled .link::after { animation-play-state: paused; opacity: .35; }
.grid { display: grid; grid-template-columns: repeat(6, minmax(0, 1fr)); gap: 12px; margin-top: 14px; }
.lbl { grid-column: 1 / -1; display: flex; align-items: center; gap: 10px; font-weight: 700; font-size: .93rem; margin-top: 2px; }
.lbl::before { content: ""; width: 6px; height: 18px; border-radius: 4px; background: linear-gradient(180deg,#5b8cff,#a78bfa); }
.card { position: relative; overflow: hidden; border-radius: 18px; padding: 13px 14px 10px; min-height: 142px;
  background: linear-gradient(145deg, rgba(255,255,255,.065), rgba(255,255,255,.02));
  border: 1px solid rgba(255,255,255,.08); transition: transform .25s, box-shadow .25s, border-color .25s;
  animation: rise .7s ease both; animation-delay: var(--d); cursor: default; }
.card::before { content: ""; position: absolute; inset: 0 auto 0 0; width: 4px; background: linear-gradient(180deg, var(--c), transparent); }
.card::after { content: ""; position: absolute; width: 160px; height: 160px; right: -60px; top: -70px;
  background: radial-gradient(circle, color-mix(in srgb, var(--c) 40%, transparent), transparent 70%); opacity: .45; transition: opacity .25s; }
.card:hover { transform: translateY(-6px) scale(1.015); border-color: color-mix(in srgb, var(--c) 65%, transparent);
  box-shadow: 0 18px 40px -14px color-mix(in srgb, var(--c) 65%, transparent); }
.card:hover::after { opacity: 1; }
@keyframes rise { from { opacity: 0; transform: translateY(18px) scale(.98); } to { opacity: 1; transform: none; } }
.head { display: flex; align-items: center; gap: 9px; position: relative; z-index: 1; }
.icon { flex: 0 0 auto; width: 32px; height: 32px; border-radius: 10px; display: grid; place-items: center; font-size: 1rem;
  background: color-mix(in srgb, var(--c) 18%, transparent); border: 1px solid color-mix(in srgb, var(--c) 40%, transparent); }
.clabel { color: #94a3b8; font-size: .64rem; font-weight: 700; text-transform: uppercase; letter-spacing: .07em;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.value { font-family: 'JetBrains Mono', monospace; font-size: 1.55rem; font-weight: 700; margin-top: 10px; color: #f8fafc;
  text-shadow: 0 0 20px color-mix(in srgb, var(--c) 40%, transparent); position: relative; z-index: 1;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.csub { color: #94a3b8; font-size: .72rem; margin-top: 3px; position: relative; z-index: 1;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.spark { position: absolute; left: 0; right: 0; bottom: 0; height: 38px; opacity: .9; }
@media (max-width: 1350px) {
  .value { font-size: 1.25rem; } .clabel { font-size: .58rem; letter-spacing: .04em; }
  .card { padding: 11px 11px 9px; min-height: 134px; } .node { width: 112px; } .icon { width: 28px; height: 28px; }
  .title { font-size: 1.9rem; }
}
</style></head><body><div class="wrap" id="root"></div>
<script>
const D = __DATA__;
const root = document.getElementById('root');
const fmtNum = (v, k) => k === 'pct' ? v.toFixed(2) + '%' : k === 'money' ? '$' + Math.round(v).toLocaleString() : Math.round(v).toLocaleString();
root.innerHTML = `
 <div class="top">
   <div><div class="title">Clickstream Anomaly Monitor</div>
     <div class="sub">Real-time anomaly detection · refreshed <b>${D.now} UTC</b> · every <b>${D.refresh}s</b> · refresh <b>#${D.tick}</b></div>
     <div class="win">🕒 ${D.window}</div></div>
   <div><div style="text-align:right"><span class="status" style="--sc:${D.scolor}"><span class="dot"></span>${D.status}</span></div>
     <div class="smeta">last batch <b>${D.lag}</b> · ${D.last5.toLocaleString()} rows in last 5 min<br>
       <span class="up">+${D.delta.toLocaleString()}</span> rows since last refresh</div></div>
 </div>
 <div class="pipe ${D.status === 'STALLED' ? 'stalled' : ''}" style="--sc:${D.scolor}" id="pipe"></div>
 <div class="grid" id="grid"></div>`;
const pipe = document.getElementById('pipe');
D.pipeline.forEach((p, i) => {
  if (i > 0) { const l = document.createElement('div'); l.className = 'link'; pipe.appendChild(l); }
  const n = document.createElement('div'); n.className = 'node';
  n.innerHTML = `${p.badge ? `<div class="badge">${p.badge}</div>` : ''}
    <div class="bubble" style="--d:${i * 0.4}s">${p.icon}</div>
    <div class="name">${p.name}</div><div class="desc">${p.desc}</div>
    <div class="num">${p.value || '&nbsp;'}</div>`;
  pipe.appendChild(n);
});
const grid = document.getElementById('grid');
let idx = 0;
D.groups.forEach(g => {
  const l = document.createElement('div'); l.className = 'lbl'; l.textContent = g.title; grid.appendChild(l);
  g.cards.forEach(c => {
    const el = document.createElement('div'); el.className = 'card';
    el.style.setProperty('--c', c.color); el.style.setProperty('--d', (idx++ * 0.06) + 's');
    let spark = '';
    if (c.spark && c.spark.length > 1) {
      const w = 200, h = 38, mx = Math.max(...c.spark, 1);
      const pts = c.spark.map((v, j) => `${(j / (c.spark.length - 1)) * w},${h - 4 - (v / mx) * (h - 8)}`).join(' ');
      spark = `<svg class="spark" viewBox="0 0 ${w} ${h}" preserveAspectRatio="none">
        <defs><linearGradient id="g${c.id}" x1="0" x2="0" y1="0" y2="1">
          <stop offset="0%" stop-color="${c.color}" stop-opacity=".45"/><stop offset="100%" stop-color="${c.color}" stop-opacity="0"/></linearGradient></defs>
        <polygon points="0,${h} ${pts} ${w},${h}" fill="url(#g${c.id})"/>
        <polyline points="${pts}" fill="none" stroke="${c.color}" stroke-width="2" vector-effect="non-scaling-stroke"/></svg>`;
    }
    el.innerHTML = `<div class="head"><div class="icon">${c.icon}</div><div class="clabel">${c.label}</div></div>
      <div class="value" id="v_${c.id}">0</div><div class="csub">${c.sub}</div>${spark}`;
    grid.appendChild(el);
    const target = document.getElementById('v_' + c.id);
    let from = 0;
    try { from = parseFloat(localStorage.getItem('kpi_' + c.id) || '0'); } catch (e) {}
    const start = performance.now(), dur = 1300;
    const tickFn = now => {
      const t = Math.min((now - start) / dur, 1), e = 1 - Math.pow(1 - t, 3);
      target.textContent = fmtNum(from + (c.value - from) * e, c.kind);
      if (t < 1) requestAnimationFrame(tickFn);
    };
    requestAnimationFrame(tickFn);
    try { localStorage.setItem('kpi_' + c.id, c.value); } catch (e) {}
  });
});
</script></body></html>
"""


@st.cache_resource
def get_credential():
    return InteractiveBrowserCredential(login_hint=USER)


def get_connection():
    token = get_credential().get_token("https://database.windows.net/.default").token
    token_bytes = token.encode("utf-16-le")
    token_struct = struct.pack(f"<I{len(token_bytes)}s", len(token_bytes), token_bytes)
    last_error = None
    for attempt in range(3):
        try:
            return pyodbc.connect(
                "Driver={ODBC Driver 18 for SQL Server};"
                f"Server={SERVER},1433;"
                f"Database={DATABASE};"
                "Encrypt=yes;TrustServerCertificate=no;",
                attrs_before={1256: token_struct},
                timeout=60,
            )
        except pyodbc.Error as e:
            last_error = e
            time.sleep(3 * (attempt + 1))
    raise last_error


def _query(sql):
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(sql)
        columns = [c[0] for c in cur.description]
        return pd.DataFrame([tuple(r) for r in cur.fetchall()], columns=columns)
    finally:
        conn.close()


@st.cache_data(ttl=10, show_spinner=False)
def run_query(sql):
    return _query(sql)


@st.cache_data(ttl=60, show_spinner=False)
def run_heavy(sql):
    return _query(sql)


def safe(fn, sql):
    try:
        return fn(sql)
    except Exception as e:
        st.warning(f"Query failed: {e}")
        return pd.DataFrame()


def section(title):
    st.markdown(f'<div class="section">{title}</div>', unsafe_allow_html=True)


def kpi(col, label, value, sub="", color=C["blue"]):
    col.markdown(f'<div class="kpi" style="--c:{color}"><div class="label">{label}</div>'
                 f'<div class="value">{value}</div><div class="sub">{sub}</div></div>', unsafe_allow_html=True)


def n(x):
    return int(x) if x is not None and pd.notna(x) else 0


def fl(x):
    return float(x) if x is not None and pd.notna(x) else 0.0


def ratio(a, b):
    return a / b * 100 if b else 0.0


def pct(a, b):
    return f"{a / b * 100:.2f}%" if b else "–"


def short(x):
    x = n(x)
    if x >= 1_000_000:
        return f"{x / 1_000_000:.2f}M"
    if x >= 1_000:
        return f"{x / 1_000:.1f}K"
    return f"{x:,}"


def style(fig, height=380):
    fig.update_layout(
        template="plotly_dark", height=height, margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#cbd5e1"), colorway=PALETTE,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title_text="", bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor="#111827", bordercolor="#334155", font_family="Inter"),
        hovermode="x unified",
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,.06)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(255,255,255,.06)", zeroline=False)
    return fig


def in_list(column, values):
    quoted = ", ".join("'" + str(v).replace("'", "''") + "'" for v in values)
    return f"{column} IN ({quoted})"


def human(seconds):
    if seconds is None:
        return "no data"
    seconds = int(seconds)
    if seconds < 120:
        return f"{seconds} s ago"
    if seconds < 7200:
        return f"{seconds // 60} min ago"
    if seconds < 172800:
        return f"{seconds // 3600} h ago"
    return f"{seconds // 86400} days ago"

def dur_s(d):
    if not isinstance(d, dict) or d.get("value") is None:
        return None
    unit = str(d.get("timeUnit", "Seconds")).lower()
    mult = {"milliseconds": 0.001, "seconds": 1, "minutes": 60, "hours": 3600, "days": 86400}.get(unit, 1)
    return float(d["value"]) * mult


def to_naive(x):
    t = pd.to_datetime(x, utc=True, errors="coerce")
    return t.tz_convert(None) if t is not None and pd.notna(t) else pd.NaT


@st.cache_data(ttl=120, show_spinner=False)
def spark_sessions():
    token = get_credential().get_token("https://api.fabric.microsoft.com/.default").token
    url = f"https://api.fabric.microsoft.com/v1/workspaces/{WORKSPACE_ID}/spark/livySessions"
    raw = []
    for _ in range(20):
        r = requests.get(url, headers={"Authorization": f"Bearer {token}"}, timeout=30)
        r.raise_for_status()
        body = r.json()
        raw.extend(body.get("value", []))
        url = body.get("continuationUri")
        if not url:
            break
    now_s = datetime.now(timezone.utc).replace(tzinfo=None)
    rows = []
    for x in raw:
        start = to_naive(x.get("startDateTime") or x.get("submittedDateTime"))
        end = to_naive(x.get("endDateTime"))
        running = dur_s(x.get("runningDuration"))
        if running is None and pd.notna(start):
            running = ((end if pd.notna(end) else now_s) - start).total_seconds()
        item = x.get("itemName") or "unknown"
        itype = x.get("itemType") or ""
        path = "stream" if item.lower().startswith("nb_stream") else "batch" if itype in ("Notebook", "SparkJobDefinition") else "other"
        rows.append({
            "item": item, "item_type": itype, "path": path, "state": x.get("state") or "Unknown",
            "start": start, "end": end if pd.notna(end) else now_s,
            "running_s": running or 0.0, "queued_s": dur_s(x.get("queuedDuration")) or 0.0,
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df[df["start"].notna()]
        df["hours"] = df["running_s"] / 3600
    return df


def load_sessions():
    try:
        return spark_sessions()
    except Exception as e:
        st.warning(f"Could not read Spark sessions from the Fabric API: {e}")
        return pd.DataFrame()


STATE_COLORS = {"Succeeded": C["green"], "Failed": C["red"], "Cancelled": C["gray"],
                "InProgress": C["blue"], "NotStarted": C["amber"], "Unknown": C["purple"]}

with st.container(border=True):
    f1, f2, f3, f4 = st.columns([1.7, 0.8, 1.1, 2.4])
    window = f1.radio("🕒 Time window (event time)", list(WINDOWS), index=3, horizontal=True)
    refresh = f2.selectbox("🔄 Auto-refresh", [15, 30, 60, 120], index=1, format_func=lambda s: f"every {s}s")
    options = safe(run_heavy, "SELECT DISTINCT platform, marketing_channel FROM Silver.stream_events")
    platforms = sorted(options["platform"].dropna().unique()) if not options.empty else []
    channels = sorted(options["marketing_channel"].dropna().unique()) if not options.empty else []
    sel_platforms = f3.multiselect("📱 Platform", platforms, default=platforms)
    sel_channels = f4.multiselect("📣 Marketing channel", channels, default=channels)

tick = st_autorefresh(interval=refresh * 1000, key="auto_refresh")

if not sel_platforms or not sel_channels:
    st.warning("Select at least one platform and one channel.")
    st.stop()

WHERE = f"{in_list('platform', sel_platforms)} AND {in_list('marketing_channel', sel_channels)}"
WHERE_P = f"{in_list('p.platform', sel_platforms)} AND {in_list('p.marketing_channel', sel_channels)}"

max_et_df = safe(run_query, f"SELECT MAX(event_time) AS max_et FROM Silver.stream_events WHERE {WHERE}")
max_et = pd.to_datetime(max_et_df["max_et"].iloc[0]) if not max_et_df.empty else None
mins_back = WINDOWS[window]
if mins_back and max_et is not None and pd.notna(max_et):
    cutoff = max_et - pd.Timedelta(minutes=mins_back)
    cut = cutoff.strftime("%Y-%m-%d %H:%M:%S")
    T, T_P, T_CART, T_MIN = (f"event_time >= '{cut}'", f"p.event_time >= '{cut}'",
                             f"cart_time >= '{cut}'", f"minute_start >= '{cut}'")
    window_label = f"{window} · {cutoff:%Y-%m-%d %H:%M} → {max_et:%H:%M} event time"
else:
    cutoff = None
    T = T_P = T_CART = T_MIN = "1=1"
    window_label = f"All data · up to {max_et:%Y-%m-%d %H:%M} event time" if max_et is not None and pd.notna(max_et) else "All data"

status_df = safe(run_query, """
    SELECT SUM(CASE WHEN processed_time >= DATEADD(minute, -5, SYSUTCDATETIME()) THEN 1 ELSE 0 END) AS last_5_min,
           MAX(processed_time) AS last_processed, COUNT(*) AS all_rows
    FROM Silver.stream_events
""")
st_row = status_df.iloc[0] if not status_df.empty else pd.Series(dtype=object)

summary = safe(run_query, f"""
    SELECT
        COUNT(*) AS total_events,
        SUM(CAST(is_duplicate_event     AS INT)) AS duplicate_events,
        SUM(CAST(is_bot_like            AS INT)) AS bot_like_views,
        COUNT(DISTINCT CASE WHEN is_bot_session = 1 THEN user_session END) AS bot_sessions,
        SUM(CAST(is_impossible_sequence AS INT)) AS impossible_sequences,
        SUM(CAST(exclude_from_kpis      AS INT)) AS excluded_from_kpis,
        SUM(CASE WHEN is_duplicate_event = 1 AND is_bot_like = 1 THEN 1 ELSE 0 END) AS overlap,
        SUM(CASE WHEN platform = 'unknown' OR marketing_channel = 'unknown' THEN 1 ELSE 0 END) AS unknown_rows
    FROM Silver.stream_events
    WHERE {WHERE} AND {T}
""")
s = summary.iloc[0] if not summary.empty else pd.Series(dtype=object)

metrics = safe(run_query, """
    SELECT stage, metric, SUM(value) AS total, COUNT(DISTINCT run_id) AS runs
    FROM Silver.pipeline_metrics WHERE stage IN ('stream_silver', 'stream_replay')
    GROUP BY stage, metric
""")
m = metrics.pivot(index="metric", columns="stage", values="total").fillna(0) if not metrics.empty else pd.DataFrame()


def metric_sum(name):
    return n(m.loc[name].sum()) if not m.empty and name in m.index else 0


batch_count = n(metrics.loc[metrics["metric"] == "rows_in", "runs"].sum()) if not metrics.empty else 0

no_view = safe(run_heavy, f"""
    SELECT COUNT(*) AS purchases_without_view
    FROM Silver.stream_events p
    WHERE p.event_type = 'purchase' AND p.exclude_from_kpis = 0 AND {WHERE_P} AND {T_P}
      AND NOT EXISTS (
          SELECT 1 FROM Silver.stream_events v
          WHERE v.user_session = p.user_session AND v.product_id = p.product_id
            AND v.event_type IN ('view', 'cart') AND v.event_time <= p.event_time)
""")

abandon = safe(run_heavy, f"""
    WITH c AS (
        SELECT user_session, product_id, platform, marketing_channel,
               MIN(event_time) AS cart_time, MAX(price) AS cart_price
        FROM Silver.stream_events
        WHERE event_type = 'cart' AND exclude_from_kpis = 0 AND {WHERE}
        GROUP BY user_session, product_id, platform, marketing_channel
    ),
    p AS (
        SELECT DISTINCT user_session, product_id
        FROM Silver.stream_events
        WHERE event_type = 'purchase'
    ),
    a AS (
        SELECT c.platform, c.marketing_channel, c.cart_time, c.cart_price,
               CASE WHEN p.user_session IS NULL THEN 1 ELSE 0 END AS is_abandoned
        FROM c LEFT JOIN p ON p.user_session = c.user_session AND p.product_id = c.product_id
    )
    SELECT platform, marketing_channel,
           COUNT(*) AS carts, SUM(is_abandoned) AS abandoned,
           SUM(CASE WHEN is_abandoned = 1 THEN cart_price ELSE 0 END) AS lost_revenue
    FROM a
    WHERE {T_CART}
    GROUP BY platform, marketing_channel
""")

dlq = safe(run_query, """
    SELECT dlq_reason, status, COUNT(*) AS rows_rejected
    FROM Silver.dlq_events WHERE source = 'stream'
    GROUP BY dlq_reason, status
""")

rt_all = safe(run_query, f"""
    SELECT minute_start, SUM(events) AS events, SUM(views) AS views, SUM(carts) AS carts,
           SUM(purchases) AS purchases, SUM(revenue) AS revenue, SUM(active_sessions) AS active_sessions,
           AVG(avg_latency_s) AS latency_s
    FROM Silver.stream_minute_kpis
    WHERE {WHERE}
    GROUP BY minute_start
    ORDER BY minute_start
""")

if not rt_all.empty:
    rt_all["minute_start"] = pd.to_datetime(rt_all["minute_start"])
    p = rt_all["purchases"].astype(float)
    rt_all["baseline"] = p.rolling(60, min_periods=15).mean().shift(1)
    rt_all["std"] = p.rolling(60, min_periods=15).std().shift(1)
    rt_all["z"] = (p - rt_all["baseline"]) / rt_all["std"]
    rt_all["is_surge"] = rt_all["z"] > 3
    rt_all["cart_to_purchase"] = rt_all["purchases"] / rt_all["carts"].where(rt_all["carts"] > 0)
rt = rt_all[rt_all["minute_start"] >= cutoff] if (cutoff is not None and not rt_all.empty) else rt_all
surge_minutes = int(rt["is_surge"].sum()) if not rt.empty else 0
surge_spark = rt["purchases"].tail(60).astype(float).tolist() if not rt.empty else []

trend = safe(run_query, f"""
    WITH b AS (
        SELECT DATEADD(minute, (DATEDIFF(minute, CAST('2019-10-01' AS datetime2), event_time) / 10) * 10,
                       CAST('2019-10-01' AS datetime2)) AS bucket,
               is_duplicate_event, is_bot_like, is_impossible_sequence, is_bot_session, exclude_from_kpis
        FROM Silver.stream_events WHERE {WHERE} AND {T}
    )
    SELECT bucket,
           SUM(CAST(is_duplicate_event     AS INT)) AS [Duplicate events],
           SUM(CAST(is_bot_like            AS INT)) AS [Bot-like views],
           SUM(CAST(is_bot_session         AS INT)) AS [Bot-session events],
           SUM(CAST(is_impossible_sequence AS INT)) AS [Impossible sequences],
           SUM(CAST(exclude_from_kpis      AS INT)) AS [Excluded]
    FROM b GROUP BY bucket ORDER BY bucket
""")


def spark(col, k=48):
    return trend[col].tail(k).astype(float).tolist() if not trend.empty and col in trend else []


total = n(s.get("total_events"))
all_rows = n(st_row.get("all_rows"))
prev_total = st.session_state.get("prev_total")
delta_rows = max(all_rows - prev_total, 0) if prev_total is not None else 0
st.session_state["prev_total"] = all_rows

last_processed = pd.to_datetime(st_row.get("last_processed")) if not status_df.empty else None
now_utc = datetime.now(timezone.utc).replace(tzinfo=None)
lag_s = (now_utc - last_processed).total_seconds() if last_processed is not None and pd.notna(last_processed) else None
if lag_s is not None and lag_s <= 120:
    status, scolor = "LIVE", C["green"]
elif lag_s is not None and lag_s <= 600:
    status, scolor = "DELAYED", C["amber"]
else:
    status, scolor = "STALLED", C["red"]

carts = n(abandon["carts"].sum()) if not abandon.empty else 0
abandoned = n(abandon["abandoned"].sum()) if not abandon.empty else 0
lost = fl(abandon["lost_revenue"].sum()) if not abandon.empty else 0.0
dlq_total = n(dlq["rows_rejected"].sum()) if not dlq.empty else 0
dlq_open = n(dlq.loc[dlq["status"] == "new", "rows_rejected"].sum()) if not dlq.empty else 0
dups_removed = metric_sum("duplicates_removed")
ignored_ops = metric_sum("ignored_ops")
messages = all_rows + dlq_open + dups_removed + ignored_ops
kpi_minutes = int(rt_all["minute_start"].nunique()) if not rt_all.empty else 0
no_view_count = n(no_view["purchases_without_view"].iloc[0]) if not no_view.empty else 0

hero = {
    "now": f"{now_utc:%H:%M:%S}", "refresh": refresh, "tick": tick, "window": window_label,
    "status": status, "scolor": scolor, "lag": human(lag_s),
    "last5": n(st_row.get("last_5_min")), "delta": delta_rows,
    "pipeline": [
        {"icon": "🐘", "name": "PostgreSQL", "desc": "Debezium CDC", "value": ""},
        {"icon": "📨", "name": "Event Hubs", "desc": "Kafka endpoint", "value": short(messages)},
        {"icon": "⚙️", "name": "Spark Stream", "desc": f"{batch_count} micro-batches",
         "value": short(messages - dlq_open), "badge": f"DLQ {short(dlq_open)}" if dlq_open else ""},
        {"icon": "🥈", "name": "Silver", "desc": "stream_events", "value": short(all_rows)},
        {"icon": "⏱️", "name": "Minute KPIs", "desc": "watermark 2 min", "value": f"{kpi_minutes:,} min"},
        {"icon": "📊", "name": "Dashboard", "desc": f"every {refresh}s", "value": "live view"},
    ],
    "groups": [
        {"title": "Event anomalies", "cards": [
            {"id": "dup", "icon": "🔁", "label": "Duplicates (1 s)", "value": n(s.get("duplicate_events")), "kind": "int",
             "sub": f"{ratio(n(s.get('duplicate_events')), total):.2f}% of events", "color": C["amber"], "spark": spark("Duplicate events")},
            {"id": "exact", "icon": "🧹", "label": "Exact dups removed", "value": dups_removed, "kind": "int",
             "sub": "all time · dedup key", "color": C["amber"]},
            {"id": "bot", "icon": "🤖", "label": "Bot-like views", "value": n(s.get("bot_like_views")), "kind": "int",
             "sub": "repeat view in 1–3 s", "color": C["purple"], "spark": spark("Bot-like views")},
            {"id": "bots", "icon": "🕷️", "label": "Bot sessions", "value": n(s.get("bot_sessions")), "kind": "int",
             "sub": ">100 views / 60 s, no cart", "color": C["purple"], "spark": spark("Bot-session events")},
            {"id": "imp", "icon": "⛔", "label": "Impossible seq.", "value": n(s.get("impossible_sequences")), "kind": "int",
             "sub": "cart 1 s after purchase", "color": C["red"], "spark": spark("Impossible sequences")},
            {"id": "nov", "icon": "👻", "label": "Purchase w/o view", "value": no_view_count, "kind": "int",
             "sub": "no earlier view/cart", "color": C["red"]},
        ]},
        {"title": "Business & data-quality anomalies", "cards": [
            {"id": "aband", "icon": "🛒", "label": "Cart abandonment", "value": ratio(abandoned, carts), "kind": "pct",
             "sub": f"{abandoned:,} of {carts:,} carts", "color": C["cyan"]},
            {"id": "lost", "icon": "💸", "label": "Lost revenue", "value": lost, "kind": "money",
             "sub": "abandoned cart value", "color": C["cyan"]},
            {"id": "surge", "icon": "⚡", "label": "Surge minutes", "value": surge_minutes, "kind": "int",
             "sub": "purchases z-score > 3", "color": C["blue"], "spark": surge_spark},
            {"id": "dlq", "icon": "🧪", "label": "DLQ open", "value": dlq_open, "kind": "int",
             "sub": f"all time · {dlq_total:,} total", "color": C["pink"]},
            {"id": "excl", "icon": "🚫", "label": "Excluded from KPIs", "value": ratio(n(s.get("excluded_from_kpis")), total), "kind": "pct",
             "sub": f"{n(s.get('excluded_from_kpis')):,} rows", "color": C["gray"], "spark": spark("Excluded")},
            {"id": "ovl", "icon": "🔀", "label": "Dup + bot-like", "value": n(s.get("overlap")), "kind": "int",
             "sub": "rows with both flags", "color": C["gray"]},
        ]},
    ],
}

components.html(HERO_TEMPLATE.replace("__DATA__", json.dumps(hero)), height=820, scrolling=False)

tab_c, tab_a, tab_h = st.tabs(["🚨  Anomalies", "⚡  Real-time", "🩺  Pipeline Health"])

with tab_c:
    c1, c2 = st.columns([1, 1.4])
    with c1:
        with st.container(border=True):
            section("C8 · Funnel conversion & drop-off (sessions)")
            fn = safe(run_query, f"""
                WITH s AS (
                    SELECT user_session,
                           MAX(CASE WHEN event_type = 'view'     THEN 1 ELSE 0 END) AS v,
                           MAX(CASE WHEN event_type = 'cart'     THEN 1 ELSE 0 END) AS c,
                           MAX(CASE WHEN event_type = 'purchase' THEN 1 ELSE 0 END) AS p
                    FROM Silver.stream_events
                    WHERE exclude_from_kpis = 0 AND {WHERE} AND {T}
                    GROUP BY user_session)
                SELECT SUM(v) AS v, SUM(c) AS c, SUM(p) AS p FROM s
            """)
            if not fn.empty:
                v, c, pp = n(fn["v"].iloc[0]), n(fn["c"].iloc[0]), n(fn["p"].iloc[0])
                fig = go.Figure(go.Funnel(y=["View", "Cart", "Purchase"], x=[v, c, pp],
                                          textinfo="value+percent previous",
                                          marker=dict(color=[C["blue"], C["purple"], C["green"]])))
                fig.update_layout(hovermode="closest")
                st.plotly_chart(style(fig, 290), use_container_width=True)
                k = st.columns(3)
                kpi(k[0], "View → Cart", pct(c, v), f"drop-off {pct(v - c, v)}", C["blue"])
                kpi(k[1], "Cart → Purchase", pct(pp, c), f"drop-off {pct(c - pp, c)}", C["purple"])
                kpi(k[2], "Overall", pct(pp, v), "view → purchase", C["green"])
    with c2:
        with st.container(border=True):
            section("C2 & C3 · Near duplicates and bot-like views over time (10-min buckets)")
            if not trend.empty:
                long = trend[["bucket", "Duplicate events", "Bot-like views"]].melt(
                    id_vars="bucket", var_name="anomaly", value_name="rows")
                fig = px.line(long, x="bucket", y="rows", color="anomaly", line_shape="spline",
                              color_discrete_sequence=[C["amber"], C["purple"]], labels={"bucket": "", "rows": "rows"})
                fig.update_traces(line=dict(width=2.2))
                st.plotly_chart(style(fig, 440), use_container_width=True)

    c3, c4 = st.columns([1.4, 1])
    with c3:
        with st.container(border=True):
            section("C3 · Top 10 sessions by bot-like views")
            top = safe(run_query, f"""
                SELECT TOP 10 user_session, MAX(platform) AS platform, MAX(marketing_channel) AS channel,
                       SUM(CAST(is_bot_like AS INT)) AS bot_like_views, COUNT(*) AS events
                FROM Silver.stream_events WHERE {WHERE} AND {T}
                GROUP BY user_session
                HAVING SUM(CAST(is_bot_like AS INT)) > 0
                ORDER BY bot_like_views DESC, events DESC
            """)
            st.dataframe(top, use_container_width=True, hide_index=True, height=380,
                         column_config={"bot_like_views": st.column_config.ProgressColumn(
                             "bot-like views", min_value=0,
                             max_value=int(top["bot_like_views"].max()) if not top.empty else 1, format="%d")})
    with c4:
        with st.container(border=True):
            section("C6 · Cart abandonment by channel")
            if not abandon.empty:
                a = abandon.groupby("marketing_channel", as_index=False)[["carts", "abandoned", "lost_revenue"]].sum()
                a["rate"] = a["abandoned"] / a["carts"]
                a = a.sort_values("rate")
                fig = px.bar(a, x="rate", y="marketing_channel", orientation="h",
                             text=a["rate"].map("{:.1%}".format), hover_data={"lost_revenue": ":,.0f"},
                             color="rate", color_continuous_scale=["#1e3a8a", C["cyan"]],
                             labels={"rate": "", "marketing_channel": ""})
                fig.update_xaxes(tickformat=".0%")
                fig.update_layout(coloraxis_showscale=False, hovermode="closest")
                st.plotly_chart(style(fig, 380), use_container_width=True)

    with st.container(border=True):
        section("C7 · Purchase surge · purchases per minute vs 60-minute baseline (± 3σ band)")
        if rt.empty:
            st.warning("No minute KPIs in this window yet. Check that the KPI stream (cell 9) is running.")
        else:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=rt["minute_start"], y=rt["baseline"] + 3 * rt["std"],
                                     line=dict(width=0), showlegend=False, hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=rt["minute_start"], y=rt["baseline"], name="baseline ± 3σ",
                                     fill="tonexty", fillcolor="rgba(148,163,184,.10)",
                                     line=dict(color=C["gray"], dash="dash", width=1)))
            fig.add_trace(go.Scatter(x=rt["minute_start"], y=rt["purchases"], name="purchases / minute",
                                     line=dict(color=C["blue"], width=2, shape="spline")))
            spikes = rt[rt["is_surge"]]
            fig.add_trace(go.Scatter(x=spikes["minute_start"], y=spikes["purchases"], mode="markers",
                                     name=f"surge ({len(spikes)} minutes)",
                                     marker=dict(color=C["red"], size=13, symbol="diamond", line=dict(color="#fff", width=1))))
            fig.update_layout(xaxis_title="", yaxis_title="purchases")
            st.plotly_chart(style(fig, 440), use_container_width=True)

with tab_a:
    if rt.empty:
        st.warning("No minute KPIs in this window yet. Check that the KPI stream (cell 9) is running.")
    else:
        tot_p, tot_c, tot_r = fl(rt["purchases"].sum()), fl(rt["carts"].sum()), fl(rt["revenue"].sum())
        latest = rt.iloc[-1]
        r = st.columns(4)
        kpi(r[0], "A3 · AOV", f"${tot_r / tot_p:,.2f}" if tot_p else "–", "revenue ÷ purchases", C["green"])
        kpi(r[1], "A4 · Cart → Purchase", pct(tot_p, tot_c), "closed minutes in window", C["purple"])
        kpi(r[2], "A2 · Revenue", f"${tot_r:,.0f}", f"{n(tot_p):,} purchases", C["blue"])
        kpi(r[3], "Latest closed minute", f"{latest['minute_start']:%H:%M}",
            f"{n(latest['events']):,} events", C["cyan"])

        c1, c2 = st.columns(2)
        with c1:
            with st.container(border=True):
                section("A1 · Events per minute by type")
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                fig.add_trace(go.Scatter(x=rt["minute_start"], y=rt["views"], name="views",
                                         line=dict(color=C["blue"], width=1.6)), secondary_y=False)
                fig.add_trace(go.Scatter(x=rt["minute_start"], y=rt["carts"], name="carts",
                                         line=dict(color=C["purple"], width=1.6)), secondary_y=True)
                fig.add_trace(go.Scatter(x=rt["minute_start"], y=rt["purchases"], name="purchases",
                                         line=dict(color=C["green"], width=1.6)), secondary_y=True)
                fig.update_yaxes(title_text="views", secondary_y=False)
                fig.update_yaxes(title_text="carts / purchases", secondary_y=True, showgrid=False)
                st.plotly_chart(style(fig), use_container_width=True)
        with c2:
            with st.container(border=True):
                section("A2 · Purchases and revenue per minute")
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                fig.add_trace(go.Bar(x=rt["minute_start"], y=rt["purchases"], name="purchases",
                                     marker_color=C["green"], opacity=.55), secondary_y=False)
                fig.add_trace(go.Scatter(x=rt["minute_start"], y=rt["revenue"], name="revenue ($)",
                                         line=dict(color=C["amber"], width=1.8)), secondary_y=True)
                fig.update_yaxes(title_text="purchases", secondary_y=False)
                fig.update_yaxes(title_text="revenue ($)", secondary_y=True, showgrid=False)
                st.plotly_chart(style(fig), use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            with st.container(border=True):
                section("A4 · Cart → purchase ratio per minute")
                fig = px.line(rt, x="minute_start", y="cart_to_purchase", color_discrete_sequence=[C["purple"]],
                              labels={"minute_start": "", "cart_to_purchase": "purchases ÷ carts"})
                fig.update_yaxes(tickformat=".0%")
                st.plotly_chart(style(fig), use_container_width=True)
        with c4:
            with st.container(border=True):
                section("A5 · Active sessions per minute (approximate)")
                fig = px.area(rt, x="minute_start", y="active_sessions", color_discrete_sequence=[C["cyan"]],
                              labels={"minute_start": "", "active_sessions": "sessions"})
                st.plotly_chart(style(fig), use_container_width=True)

        def split(dim):
            d = safe(run_query, f"""
                SELECT {dim} AS dim, SUM(carts) AS carts, SUM(purchases) AS purchases, SUM(revenue) AS revenue
                FROM Silver.stream_minute_kpis WHERE {WHERE} AND {T_MIN} GROUP BY {dim}
            """)
            if not d.empty:
                d["cart_to_purchase"] = d["purchases"] / d["carts"].where(d["carts"] > 0)
                d = d.sort_values("revenue", ascending=False)
            return d

        c5, c6 = st.columns(2)
        for col, dim, title in [(c5, "platform", "A6 · By platform"), (c6, "marketing_channel", "A7 · By channel")]:
            with col:
                with st.container(border=True):
                    section(f"{title} · revenue (bars) and cart → purchase (line)")
                    d = split(dim)
                    if not d.empty:
                        fig = make_subplots(specs=[[{"secondary_y": True}]])
                        fig.add_trace(go.Bar(x=d["dim"], y=d["revenue"], name="revenue ($)",
                                             marker=dict(color=d["revenue"], colorscale=[[0, "#1e3a8a"], [1, C["blue"]]])),
                                      secondary_y=False)
                        fig.add_trace(go.Scatter(x=d["dim"], y=d["cart_to_purchase"], name="cart → purchase",
                                                 mode="lines+markers", line=dict(color=C["amber"], width=2),
                                                 marker=dict(size=9)), secondary_y=True)
                        fig.update_yaxes(tickformat=".0%", secondary_y=True, showgrid=False)
                        fig.update_layout(hovermode="closest")
                        st.plotly_chart(style(fig), use_container_width=True)

with tab_h:
    sess = load_sessions()
    now_h = datetime.now(timezone.utc).replace(tzinfo=None)
    day_ago, week_ago = now_h - pd.Timedelta(days=1), now_h - pd.Timedelta(days=7)

    ops = safe(run_query, """
        SELECT stage, metric, run_id, MAX(ts) AS ts, SUM(value) AS value
        FROM Silver.pipeline_metrics
        WHERE ts >= DATEADD(day, -7, SYSUTCDATETIME())
        GROUP BY stage, metric, run_id
    """)
    if not ops.empty:
        ops["ts"] = pd.to_datetime(ops["ts"])
        ops["path"] = ops["stage"].apply(lambda x: "stream" if str(x).startswith("stream") else "batch")

    s7 = sess[sess["start"] >= week_ago] if not sess.empty else sess
    s24 = sess[sess["end"] >= day_ago] if not sess.empty else sess

    section("🖥️ Compute overview · Spark session hours (what the capacity is spent on)")
    r = st.columns(4)
    stream_h24 = fl(s24.loc[s24["path"] == "stream", "hours"].sum()) if not s24.empty else 0.0
    batch_h24 = fl(s24.loc[s24["path"] == "batch", "hours"].sum()) if not s24.empty else 0.0
    failed7 = int((s7["state"] == "Failed").sum()) if not s7.empty else 0
    queue7 = fl(s7["queued_s"].mean()) if not s7.empty else 0.0
    kpi(r[0], "⚡ Stream compute · 24 h", f"{stream_h24:,.1f} h", "Spark session hours", C["purple"])
    kpi(r[1], "📦 Batch compute · 24 h", f"{batch_h24:,.1f} h", "Spark session hours", C["cyan"])
    kpi(r[2], "Failed runs · 7 d", f"{failed7:,}", "all notebooks", C["red"] if failed7 else C["green"])
    kpi(r[3], "Avg queue time · 7 d", f"{queue7:,.0f} s", "waiting for Spark capacity", C["amber"] if queue7 > 60 else C["green"])

    if not s7.empty:
        c1, c2 = st.columns([1.6, 1])
        with c1:
            with st.container(border=True):
                section("Spark hours per day · batch vs stream")
                daily = (s7[s7["path"] != "other"].assign(day=s7["start"].dt.date)
                         .groupby(["day", "path"], as_index=False)["hours"].sum())
                fig = px.bar(daily, x="day", y="hours", color="path", barmode="stack",
                             color_discrete_map={"stream": C["purple"], "batch": C["cyan"]},
                             labels={"day": "", "hours": "session hours"})
                fig.update_layout(hovermode="closest")
                st.plotly_chart(style(fig, 340), use_container_width=True)
        with c2:
            with st.container(border=True):
                section("Share of compute by notebook · 7 d")
                share = s7.groupby("item", as_index=False)["hours"].sum().sort_values("hours", ascending=False)
                fig = px.pie(share, names="item", values="hours", hole=0.62, color_discrete_sequence=PALETTE)
                fig.update_traces(textinfo="percent", marker=dict(line=dict(color="#070a14", width=3)))
                fig.update_layout(hovermode="closest")
                st.plotly_chart(style(fig, 340), use_container_width=True)

    st.write("")
    sub_stream, sub_batch = st.tabs(["⚡  Stream", "📦  Batch"])

    with sub_stream:
        so = ops[ops["stage"] == "stream_silver"] if not ops.empty else pd.DataFrame()
        per_batch = (so.pivot_table(index=["run_id", "ts"], columns="metric", values="value", aggfunc="sum")
                     .reset_index().sort_values("ts")) if not so.empty else pd.DataFrame()

        def col_sum(df, name, since=None):
            if df.empty or name not in df:
                return 0.0
            d = df if since is None else df[df["ts"] >= since]
            return fl(d[name].sum())

        rows_15 = col_sum(per_batch, "rows_in", now_h - pd.Timedelta(minutes=15))
        avg_dur = fl(per_batch["batch_duration_s"].dropna().tail(20).mean()) if "batch_duration_s" in per_batch else 0.0
        dlq_24, in_24 = col_sum(per_batch, "dlq_count", day_ago), col_sum(per_batch, "rows_in", day_ago)
        lat = safe(run_query, """
            SELECT AVG(CAST(DATEDIFF(second, ingestion_time, processed_time) AS FLOAT)) AS eh_to_silver,
                   AVG(pipeline_latency_s) AS pg_to_eh
            FROM Silver.stream_events
            WHERE processed_time >= DATEADD(minute, -15, SYSUTCDATETIME())
        """)
        eh_lat = fl(lat["eh_to_silver"].iloc[0]) if not lat.empty else 0.0
        live_m = m["stream_silver"] if not m.empty and "stream_silver" in m.columns else pd.Series(dtype=float)
        rin = n(live_m.get("rows_in")) if not live_m.empty else 0
        rout = sum(n(live_m.get(k)) for k in ["rows_inserted", "dlq_count", "duplicates_removed", "ignored_ops"]) if not live_m.empty else 0

        r = st.columns(6)
        kpi(r[0], "Status", status, f"last batch {human(lag_s)}", scolor)
        kpi(r[1], "Throughput · 15 min", f"{rows_15 / 900:,.1f}/s", f"{int(rows_15):,} messages", C["blue"])
        kpi(r[2], "Avg batch duration", f"{avg_dur:,.1f} s", f"last 20 · trigger {TRIGGER_SECONDS} s",
            C["green"] if 0 < avg_dur <= TRIGGER_SECONDS * 3 else C["amber"])
        kpi(r[3], "Event Hubs → Silver", f"{eh_lat:,.0f} s" if eh_lat else "–", "avg latency · 15 min", C["cyan"])
        kpi(r[4], "DLQ rate · 24 h", pct(dlq_24, in_24), f"{int(dlq_24):,} of {int(in_24):,}", C["pink"])
        kpi(r[5], "Reconciliation", "✅ balanced" if rin == rout else "❌ gap", f"in {rin:,} · out {rout:,}",
            C["green"] if rin == rout else C["red"])

        c1, c2 = st.columns(2)
        with c1:
            with st.container(border=True):
                section("Rows and duration per micro-batch")
                if not per_batch.empty:
                    b = per_batch.tail(60)
                    fig = make_subplots(specs=[[{"secondary_y": True}]])
                    fig.add_trace(go.Bar(x=b["ts"], y=b.get("rows_inserted"), name="inserted", marker_color=C["blue"]), secondary_y=False)
                    fig.add_trace(go.Bar(x=b["ts"], y=b.get("dlq_count"), name="DLQ", marker_color=C["pink"]), secondary_y=False)
                    if "batch_duration_s" in b and b["batch_duration_s"].notna().any():
                        fig.add_trace(go.Scatter(x=b["ts"], y=b["batch_duration_s"], name="duration (s)", mode="lines+markers",
                                                 line=dict(color=C["amber"], width=2)), secondary_y=True)
                        fig.add_hline(y=TRIGGER_SECONDS, line_dash="dot", line_color=C["red"], secondary_y=True,
                                      annotation_text=f"trigger {TRIGGER_SECONDS} s")
                    fig.update_layout(barmode="stack")
                    fig.update_yaxes(title_text="rows", secondary_y=False)
                    fig.update_yaxes(title_text="seconds", secondary_y=True, showgrid=False)
                    st.plotly_chart(style(fig), use_container_width=True)
        with c2:
            with st.container(border=True):
                section("Latency per minute · last 6 h of processing")
                lat_m = safe(run_query, """
                    WITH x AS (
                        SELECT DATEADD(minute, DATEDIFF(minute, CAST('2000-01-01' AS datetime2), processed_time),
                                       CAST('2000-01-01' AS datetime2)) AS minute,
                               DATEDIFF(second, ingestion_time, processed_time) AS eh_s, pipeline_latency_s AS pg_s
                        FROM Silver.stream_events
                        WHERE processed_time >= DATEADD(hour, -6, SYSUTCDATETIME()))
                    SELECT minute, AVG(CAST(eh_s AS FLOAT)) AS [Event Hubs → Silver],
                           AVG(pg_s) AS [Postgres → Event Hubs]
                    FROM x GROUP BY minute ORDER BY minute
                """)
                if not lat_m.empty:
                    long = lat_m.melt(id_vars="minute", var_name="hop", value_name="seconds")
                    fig = px.line(long, x="minute", y="seconds", color="hop", log_y=True,
                                  color_discrete_sequence=[C["cyan"], C["amber"]], labels={"minute": ""})
                    st.plotly_chart(style(fig), use_container_width=True)
                else:
                    st.info("No rows processed in the last 6 hours.")

        c3, c4 = st.columns(2)
        with c3:
            with st.container(border=True):
                section("Data quality per micro-batch · DLQ % and duplicates %")
                if not per_batch.empty and "rows_in" in per_batch:
                    q = per_batch.tail(60).copy()
                    base = q["rows_in"].where(q["rows_in"] > 0)
                    q["DLQ %"] = q.get("dlq_count", 0) / base
                    q["Duplicates %"] = q.get("duplicates_removed", 0) / base
                    long = q[["ts", "DLQ %", "Duplicates %"]].melt(id_vars="ts", var_name="kind", value_name="share")
                    fig = px.line(long, x="ts", y="share", color="kind", markers=True,
                                  color_discrete_sequence=[C["pink"], C["amber"]], labels={"ts": "", "share": ""})
                    fig.update_yaxes(tickformat=".2%")
                    st.plotly_chart(style(fig), use_container_width=True)
        with c4:
            with st.container(border=True):
                section("Stream Spark sessions · timeline (7 d)")
                st_s = s7[s7["path"] == "stream"] if not s7.empty else s7
                if not st_s.empty:
                    fig = px.timeline(st_s, x_start="start", x_end="end", y="item", color="state",
                                      color_discrete_map=STATE_COLORS, labels={"item": ""})
                    fig.update_layout(hovermode="closest")
                    st.plotly_chart(style(fig), use_container_width=True)
                else:
                    st.info("No stream sessions in the last 7 days.")

        with st.container(border=True):
            section("Stream DLQ by reason")
            if not dlq.empty:
                fig = px.bar(dlq, x="rows_rejected", y="dlq_reason", color="status", orientation="h", log_x=True,
                             color_discrete_map={"new": C["red"], "replayed": C["green"]},
                             labels={"rows_rejected": "rows (log scale)", "dlq_reason": ""})
                fig.update_layout(hovermode="closest")
                st.plotly_chart(style(fig, 280), use_container_width=True)
            else:
                st.success("No stream messages in the DLQ.")

    with sub_batch:
        bs = s7[s7["path"] == "batch"] if not s7.empty else s7
        runs = len(bs)
        ok = int((bs["state"] == "Succeeded").sum()) if runs else 0
        failed = int((bs["state"] == "Failed").sum()) if runs else 0
        avg_run = fl(bs["running_s"].mean()) / 60 if runs else 0.0
        hours7 = fl(bs["hours"].sum()) if runs else 0.0
        last_run = bs["end"].max() if runs else None
        last_age = (now_h - last_run).total_seconds() if last_run is not None and pd.notna(last_run) else None

        r = st.columns(6)
        kpi(r[0], "Runs · 7 d", f"{runs:,}", "batch notebook sessions", C["cyan"])
        kpi(r[1], "Success rate", pct(ok, runs), f"{ok:,} succeeded", C["green"] if runs and ok == runs else C["amber"])
        kpi(r[2], "Failed", f"{failed:,}", "last 7 days", C["red"] if failed else C["green"])
        kpi(r[3], "Avg run duration", f"{avg_run:,.1f} min", "running time", C["blue"])
        kpi(r[4], "Compute · 7 d", f"{hours7:,.1f} h", "Spark session hours", C["purple"])
        kpi(r[5], "Last run", human(last_age), "any batch notebook",
            C["green"] if last_age is not None and last_age < 86400 else C["amber"])

        c1, c2 = st.columns(2)
        with c1:
            with st.container(border=True):
                section("Batch runs · timeline by notebook (7 d)")
                if runs:
                    fig = px.timeline(bs, x_start="start", x_end="end", y="item", color="state",
                                      color_discrete_map=STATE_COLORS, labels={"item": ""})
                    fig.update_layout(hovermode="closest")
                    st.plotly_chart(style(fig), use_container_width=True)
                else:
                    st.info("No batch sessions in the last 7 days.")
        with c2:
            with st.container(border=True):
                section("Run duration per notebook")
                if runs:
                    d = bs.sort_values("start").assign(minutes=bs["running_s"] / 60)
                    fig = px.line(d, x="start", y="minutes", color="item", markers=True,
                                  color_discrete_sequence=PALETTE, labels={"start": "", "minutes": "minutes"})
                    fig.update_layout(hovermode="closest")
                    st.plotly_chart(style(fig), use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            with st.container(border=True):
                section("Spark hours per day by notebook")
                if runs:
                    dd = bs.assign(day=bs["start"].dt.date).groupby(["day", "item"], as_index=False)["hours"].sum()
                    fig = px.bar(dd, x="day", y="hours", color="item", barmode="stack",
                                 color_discrete_sequence=PALETTE, labels={"day": "", "hours": "session hours"})
                    fig.update_layout(hovermode="closest")
                    st.plotly_chart(style(fig), use_container_width=True)
        with c4:
            with st.container(border=True):
                section("Batch pipeline metrics per run")
                bo = ops[ops["path"] == "batch"] if not ops.empty else pd.DataFrame()
                if bo.empty:
                    st.info("No batch metrics in Silver.pipeline_metrics in the last 7 days.")
                else:
                    metric_options = sorted(bo["metric"].unique())
                    pick = st.selectbox("Metric", metric_options, key="batch_metric",
                                        index=metric_options.index("rows_in") if "rows_in" in metric_options else 0)
                    d = bo[bo["metric"] == pick].sort_values("ts")
                    fig = px.line(d, x="ts", y="value", color="stage", markers=True,
                                  color_discrete_sequence=PALETTE, labels={"ts": "", "value": pick})
                    fig.update_layout(hovermode="closest")
                    st.plotly_chart(style(fig, 320), use_container_width=True)

        with st.container(border=True):
            section("Batch DLQ by reason")
            bdlq = safe(run_query, """
                SELECT dlq_reason, COUNT(*) AS rows_rejected FROM Silver.dlq_events
                WHERE source = 'batch' GROUP BY dlq_reason
            """)
            if not bdlq.empty:
                fig = px.bar(bdlq.sort_values("rows_rejected"), x="rows_rejected", y="dlq_reason", orientation="h",
                             log_x=True, text="rows_rejected", color_discrete_sequence=[C["cyan"]],
                             labels={"rows_rejected": "rows (log scale)", "dlq_reason": ""})
                fig.update_layout(hovermode="closest")
                st.plotly_chart(style(fig, 300), use_container_width=True)
            else:
                st.success("No batch rows in the DLQ.")