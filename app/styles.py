"""Theme tokens and custom CSS for the Streamlit dashboard.

Streamlit owns the outer page shell; the dashboard layers a compact design
system on top of it so the demo looks like a product rather than a notebook.
Both palettes share the same layout tokens, which keeps the HTML components in
``app.components`` theme-agnostic.
"""

from __future__ import annotations

_DARK = {
    "BG": "#070b16",
    "GLOW1": "rgba(76, 110, 245, 0.22)",
    "GLOW2": "rgba(45, 212, 191, 0.12)",
    "PANEL": "rgba(17, 24, 43, 0.88)",
    "PANEL2": "rgba(25, 35, 60, 0.92)",
    "PANEL3": "rgba(255, 255, 255, 0.035)",
    "SIDEBAR": "rgba(9, 14, 27, 0.92)",
    "INPUT": "rgba(8, 13, 26, 0.72)",
    "BORDER": "rgba(148, 163, 184, 0.16)",
    "BORDER_STRONG": "rgba(148, 163, 184, 0.30)",
    "TRACK": "rgba(148, 163, 184, 0.14)",
    "TEXT": "#e9eefb",
    "TEXT_SOFT": "#c3cde1",
    "MUTED": "#8b99b8",
    "ACCENT": "#7c9cff",
    "ACCENT2": "#3ddbc0",
    "ACCENT_TEXT": "#08101f",
    "TOP_BORDER": "rgba(124, 156, 255, 0.55)",
    "RANK_BG": "rgba(124, 156, 255, 0.16)",
    "HOVER": "rgba(124, 156, 255, 0.10)",
    "SHADOW": "0 22px 55px rgba(0, 0, 0, 0.38)",
    "CARD_SHADOW": "0 10px 26px rgba(0, 0, 0, 0.24)",
    "LOGO_SHADOW": "rgba(61, 219, 192, 0.22)",
    "DOT_GLOW": "rgba(61, 219, 192, 0.18)",
}

_LIGHT = {
    "BG": "#f3f6fc",
    "GLOW1": "rgba(89, 124, 255, 0.16)",
    "GLOW2": "rgba(13, 148, 136, 0.10)",
    "PANEL": "#ffffff",
    "PANEL2": "#f5f8ff",
    "PANEL3": "rgba(246, 249, 255, 0.85)",
    "SIDEBAR": "#ffffff",
    "INPUT": "#ffffff",
    "BORDER": "#e2e9f6",
    "BORDER_STRONG": "#cbd7ea",
    "TRACK": "rgba(86, 108, 145, 0.13)",
    "TEXT": "#101a33",
    "TEXT_SOFT": "#34415f",
    "MUTED": "#66738f",
    "ACCENT": "#3b5bdb",
    "ACCENT2": "#0d9488",
    "ACCENT_TEXT": "#ffffff",
    "TOP_BORDER": "rgba(59, 91, 219, 0.45)",
    "RANK_BG": "rgba(59, 91, 219, 0.10)",
    "HOVER": "rgba(59, 91, 219, 0.06)",
    "SHADOW": "0 22px 55px rgba(29, 49, 94, 0.12)",
    "CARD_SHADOW": "0 8px 22px rgba(29, 49, 94, 0.07)",
    "LOGO_SHADOW": "rgba(13, 148, 136, 0.18)",
    "DOT_GLOW": "rgba(13, 148, 136, 0.16)",
}

_TEMPLATE = """
<style>
:root {
    --sic-bg: __BG__;
    --sic-panel: __PANEL__;
    --sic-panel-2: __PANEL2__;
    --sic-panel-3: __PANEL3__;
    --sic-input: __INPUT__;
    --sic-border: __BORDER__;
    --sic-border-strong: __BORDER_STRONG__;
    --sic-track: __TRACK__;
    --sic-text: __TEXT__;
    --sic-text-soft: __TEXT_SOFT__;
    --sic-muted: __MUTED__;
    --sic-accent: __ACCENT__;
    --sic-accent-2: __ACCENT2__;
    --sic-accent-text: __ACCENT_TEXT__;
    --sic-top-border: __TOP_BORDER__;
    --sic-rank-bg: __RANK_BG__;
    --sic-hover: __HOVER__;
    --sic-shadow: __SHADOW__;
    --sic-card-shadow: __CARD_SHADOW__;
}

html, body, [class*="css"] {
    font-family: "Inter", "Segoe UI", "Helvetica Neue", system-ui, -apple-system,
        sans-serif;
}

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(1150px 620px at 6% -12%, __GLOW1__, transparent 62%),
        radial-gradient(900px 520px at 100% 0%, __GLOW2__, transparent 58%),
        __BG__ !important;
    color: var(--sic-text);
    min-height: 100vh;
}

[data-testid="stHeader"] {
    background: transparent !important;
}

[data-testid="stToolbar"] {
    right: 1rem;
}

[data-testid="stSidebar"] {
    background: __SIDEBAR__ !important;
    border-right: 1px solid var(--sic-border);
}

[data-testid="stSidebar"] > div:first-child {
    padding-top: 1.2rem;
}

[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div {
    color: var(--sic-text);
}

[data-testid="stSidebar"] .stCaption,
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
    color: var(--sic-muted) !important;
}

.block-container {
    max-width: 1320px;
    padding-top: 1.5rem;
    padding-bottom: 4rem;
}

h1, h2, h3, h4, h5, h6 {
    color: var(--sic-text);
    letter-spacing: -0.015em;
}

p, span, label, li {
    color: var(--sic-text-soft);
}

a {
    color: var(--sic-accent);
}

hr {
    border-color: var(--sic-border) !important;
}

/* ------------------------------------------------------------------ */
/* Streamlit widgets                                                   */
/*                                                                     */
/* Streamlit 1.66 renders widgets with react-aria, so the stable       */
/* data-testid hooks are used instead of generated emotion classes.    */
/* The BaseWeb rules remain as compatibility fallbacks.                */
/* ------------------------------------------------------------------ */
[data-baseweb="select"] > div,
[data-baseweb="input"] > div,
[data-baseweb="base-input"],
[data-testid="stSelectbox"] .react-aria-ComboBox > div[role="group"],
[data-testid="stTextInput"] [data-testid="stTextInputRootElement"],
[data-testid="stNumberInput"] [data-testid="stNumberInputContainer"] {
    background: var(--sic-input) !important;
    border: 1px solid var(--sic-border-strong) !important;
    border-radius: 10px !important;
    color: var(--sic-text) !important;
    box-shadow: none !important;
    transition: border-color 0.15s ease, box-shadow 0.15s ease;
}

[data-testid="stSelectbox"] .react-aria-ComboBox > div[role="group"][data-focus-within="true"],
[data-testid="stTextInput"] [data-testid="stTextInputRootElement"]:focus-within,
[data-testid="stNumberInput"] [data-testid="stNumberInputContainer"]:focus-within {
    border-color: var(--sic-accent) !important;
    box-shadow: 0 0 0 3px var(--sic-rank-bg) !important;
}

[data-testid="stSelectbox"] input,
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input {
    background: transparent !important;
    color: var(--sic-text) !important;
    -webkit-text-fill-color: var(--sic-text);
    caret-color: var(--sic-accent);
}

[data-testid="stSelectbox"] input::placeholder,
[data-testid="stTextInput"] input::placeholder,
[data-testid="stNumberInput"] input::placeholder {
    color: var(--sic-muted) !important;
    -webkit-text-fill-color: var(--sic-muted);
    opacity: 1;
}

[data-baseweb="select"] svg,
[data-testid="stSelectbox"] .react-aria-ComboBox svg {
    color: var(--sic-muted) !important;
    fill: var(--sic-muted) !important;
}

/* Dropdown menu is portalled to <body>, so it is themed by its own hook. */
[data-testid="stSelectboxVirtualDropdown"] {
    background: var(--sic-panel) !important;
    border: 1px solid var(--sic-border-strong) !important;
    border-radius: 12px !important;
    color: var(--sic-text) !important;
    box-shadow: var(--sic-shadow) !important;
    overflow: hidden;
}

[data-testid="stSelectboxVirtualDropdown"] [role="listbox"] {
    background: transparent !important;
    color: var(--sic-text) !important;
}

[data-testid="stSelectboxVirtualDropdown"] [role="option"] {
    background: transparent !important;
    color: var(--sic-text) !important;
    border-radius: 8px;
}

[data-testid="stSelectboxVirtualDropdown"] [role="option"]:hover,
[data-testid="stSelectboxVirtualDropdown"] [role="option"][data-focused="true"],
[data-testid="stSelectboxVirtualDropdown"] [role="option"][data-selected="true"] {
    background: var(--sic-hover) !important;
    color: var(--sic-text) !important;
}

div[data-baseweb="popover"] ul {
    background: var(--sic-panel) !important;
    border: 1px solid var(--sic-border) !important;
}

div[data-baseweb="popover"] li {
    color: var(--sic-text) !important;
}

div[data-baseweb="popover"] li:hover {
    background: var(--sic-hover) !important;
}

/* Radio and checkbox controls ship dark fills from the base theme. */
[data-testid="stRadioOption"] > div > div:first-child {
    background: var(--sic-input) !important;
    border-color: var(--sic-border-strong) !important;
    box-shadow: inset 0 0 0 1px var(--sic-border-strong) !important;
}

[data-testid="stRadioOption"]:has(input:checked) > div > div:first-child {
    background: var(--sic-accent) !important;
    border-color: var(--sic-accent) !important;
    box-shadow: none !important;
}

[data-testid="stCheckbox"] [data-testid="stCheckboxRootElement"] {
    background: var(--sic-input) !important;
    border-color: var(--sic-border-strong) !important;
}

/* Checkbox controls ship dark fills from the base theme. */
[data-testid="stCheckbox"] > div > div,
[data-testid="stCheckbox"] [data-testid="stCheckboxRootElement"] {
    background: var(--sic-input) !important;
    border-color: var(--sic-border-strong) !important;
}

[data-testid="stCheckbox"] input:checked + div,
[data-testid="stCheckbox"] [data-testid="stCheckboxRootElement"][data-checked="true"] {
    background: var(--sic-accent) !important;
    border-color: var(--sic-accent) !important;
}

.stButton > button,
.stFormSubmitButton > button {
    background: linear-gradient(135deg, var(--sic-accent), var(--sic-accent-2));
    color: var(--sic-accent-text) !important;
    border: none;
    border-radius: 10px;
    font-weight: 650;
    transition: transform 0.15s ease, box-shadow 0.15s ease, opacity 0.15s ease;
    box-shadow: 0 8px 20px var(--sic-rank-bg);
}

.stButton > button:hover,
.stFormSubmitButton > button:hover {
    transform: translateY(-1px);
    opacity: 0.94;
}

.stButton > button:active,
.stFormSubmitButton > button:active {
    transform: translateY(0);
}

[data-testid="stExpander"] details {
    background: var(--sic-panel);
    border: 1px solid var(--sic-border);
    border-radius: 14px;
}

[data-testid="stExpander"] summary {
    color: var(--sic-text) !important;
}

[data-testid="stDataFrame"] {
    border: 1px solid var(--sic-border);
    border-radius: 14px;
    overflow: hidden;
}

[data-testid="stAlert"] {
    background: var(--sic-panel-3) !important;
    border: 1px solid var(--sic-border) !important;
    border-radius: 12px;
}

[data-testid="stAlert"] p {
    color: var(--sic-text-soft) !important;
}

/* ------------------------------------------------------------------ */
/* Sidebar branding                                                    */
/* ------------------------------------------------------------------ */
.sic-side-brand {
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 0 0 14px;
}

.sic-logo {
    width: 44px;
    height: 44px;
    border-radius: 13px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 13px;
    font-weight: 800;
    letter-spacing: 0.04em;
    color: var(--sic-accent-text);
    background: linear-gradient(135deg, var(--sic-accent), var(--sic-accent-2));
    box-shadow: 0 10px 22px __LOGO_SHADOW__;
}

.sic-side-title {
    font-size: 16px;
    font-weight: 750;
    color: var(--sic-text);
    line-height: 1.15;
}

.sic-side-sub {
    font-size: 11px;
    color: var(--sic-muted);
    margin-top: 3px;
}

.sic-status {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    font-size: 12px;
    color: var(--sic-muted);
}

.sic-status__dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--sic-accent-2);
    box-shadow: 0 0 0 4px __DOT_GLOW__;
}

.sic-status--offline .sic-status__dot {
    background: #f87171;
    box-shadow: 0 0 0 4px rgba(248, 113, 113, 0.16);
}

/* ------------------------------------------------------------------ */
/* Hero                                                               */
/* ------------------------------------------------------------------ */
.sic-hero {
    position: relative;
    overflow: hidden;
    border: 1px solid var(--sic-border);
    border-radius: 24px;
    padding: 30px 32px 28px;
    background: linear-gradient(135deg, var(--sic-panel) 0%, var(--sic-panel-2) 100%);
    box-shadow: var(--sic-shadow);
    margin-bottom: 18px;
}

.sic-hero::after {
    content: "";
    position: absolute;
    width: 340px;
    height: 340px;
    right: -90px;
    top: -160px;
    border-radius: 50%;
    background: radial-gradient(circle, __GLOW1__, transparent 66%);
    pointer-events: none;
}

.sic-hero__eyebrow {
    display: flex;
    align-items: center;
    gap: 9px;
    font-size: 11px;
    letter-spacing: 0.13em;
    text-transform: uppercase;
    color: var(--sic-muted);
    font-weight: 650;
}

.sic-hero__title {
    font-size: clamp(26px, 3vw, 38px);
    line-height: 1.12;
    margin: 12px 0 9px;
    font-weight: 760;
    letter-spacing: -0.025em;
    color: var(--sic-text);
}

.sic-hero__subtitle {
    max-width: 760px;
    color: var(--sic-text-soft);
    font-size: 15px;
    line-height: 1.65;
    margin: 0;
}

.sic-badges {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 18px;
}

.sic-badge {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.02em;
    color: var(--sic-text-soft);
    background: var(--sic-panel-3);
    border: 1px solid var(--sic-border);
    border-radius: 999px;
    padding: 5px 10px;
}

.sic-badge--accent {
    color: var(--sic-accent);
    background: var(--sic-rank-bg);
    border-color: transparent;
}

/* ------------------------------------------------------------------ */
/* KPI grid                                                           */
/* ------------------------------------------------------------------ */
.sic-kpis {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 14px;
    margin-bottom: 24px;
}

.sic-kpi {
    position: relative;
    overflow: hidden;
    background: var(--sic-panel);
    border: 1px solid var(--sic-border);
    border-radius: 18px;
    padding: 18px 18px 16px;
    box-shadow: var(--sic-card-shadow);
}

.sic-kpi::before {
    content: "";
    position: absolute;
    inset: 0 0 auto 0;
    height: 2px;
    background: linear-gradient(90deg, var(--sic-accent), transparent 80%);
}

.sic-kpi__label {
    font-size: 11px;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    color: var(--sic-muted);
    font-weight: 650;
}

.sic-kpi__value {
    font-size: 27px;
    font-weight: 760;
    letter-spacing: -0.025em;
    color: var(--sic-text);
    margin-top: 7px;
    font-variant-numeric: tabular-nums;
}

.sic-kpi__meta {
    font-size: 11.5px;
    color: var(--sic-muted);
    margin-top: 4px;
}

/* ------------------------------------------------------------------ */
/* Section headings                                                    */
/* ------------------------------------------------------------------ */
.sic-section {
    margin: 6px 0 22px;
}

.sic-section__head {
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
    gap: 12px;
    flex-wrap: wrap;
    margin: 0 0 12px;
}

.sic-section__title {
    font-size: 19px;
    font-weight: 720;
    color: var(--sic-text);
    margin: 0;
    letter-spacing: -0.015em;
}

.sic-section__hint {
    font-size: 12px;
    color: var(--sic-muted);
    font-variant-numeric: tabular-nums;
}

/* ------------------------------------------------------------------ */
/* Profile                                                             */
/* ------------------------------------------------------------------ */
.sic-mini-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 10px;
    margin-bottom: 14px;
}

.sic-mini {
    background: var(--sic-panel);
    border: 1px solid var(--sic-border);
    border-radius: 14px;
    padding: 13px 14px;
    box-shadow: var(--sic-card-shadow);
}

.sic-mini__value {
    font-size: 21px;
    font-weight: 740;
    color: var(--sic-text);
    font-variant-numeric: tabular-nums;
    letter-spacing: -0.02em;
}

.sic-mini__label {
    font-size: 10.5px;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    color: var(--sic-muted);
    margin-top: 4px;
    font-weight: 600;
}

.sic-chip-row {
    display: flex;
    flex-wrap: wrap;
    gap: 7px;
    margin: 0 0 14px;
}

.sic-chip {
    font-size: 11px;
    padding: 4px 9px;
    border-radius: 999px;
    border: 1px solid var(--sic-border);
    color: var(--sic-text-soft);
    background: var(--sic-panel-3);
}

.sic-chip--accent {
    color: var(--sic-accent);
    border-color: var(--sic-top-border);
    background: var(--sic-rank-bg);
}

.sic-profile-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 14px;
}

.sic-panel {
    background: var(--sic-panel);
    border: 1px solid var(--sic-border);
    border-radius: 18px;
    padding: 17px 18px;
    box-shadow: var(--sic-card-shadow);
}

.sic-panel__title {
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--sic-muted);
    margin-bottom: 13px;
}

.sic-facet {
    margin-top: 12px;
}

.sic-facet:first-of-type {
    margin-top: 0;
}

.sic-facet__row {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    gap: 10px;
    font-size: 13px;
    margin-bottom: 6px;
}

.sic-facet__value {
    color: var(--sic-text);
    font-weight: 560;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    text-transform: capitalize;
}

.sic-facet__count {
    color: var(--sic-muted);
    font-variant-numeric: tabular-nums;
    flex: 0 0 auto;
    font-size: 11.5px;
}

.sic-facet__track {
    height: 6px;
    border-radius: 999px;
    background: var(--sic-track);
    overflow: hidden;
}

.sic-facet__fill {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, var(--sic-accent), var(--sic-accent-2));
}

.sic-muted {
    color: var(--sic-muted);
    font-size: 12px;
}

.sic-userkey {
    display: inline-block;
    font-family: ui-monospace, SFMono-Regular, Consolas, "Liberation Mono", monospace;
    font-size: 11px;
    color: var(--sic-text-soft);
    background: var(--sic-panel-3);
    border: 1px solid var(--sic-border);
    border-radius: 8px;
    padding: 3px 8px;
    margin-left: 6px;
}

/* ------------------------------------------------------------------ */
/* Recommendation cards                                               */
/* ------------------------------------------------------------------ */
.sic-rec-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(235px, 1fr));
    gap: 14px;
}

.sic-rec {
    position: relative;
    display: flex;
    flex-direction: column;
    gap: 7px;
    min-height: 172px;
    padding: 16px;
    border: 1px solid var(--sic-border);
    border-radius: 18px;
    background: var(--sic-panel);
    box-shadow: var(--sic-card-shadow);
    transition: transform 0.18s ease, border-color 0.18s ease,
        box-shadow 0.18s ease;
    animation: sicRise 0.35s ease both;
}

.sic-rec:hover {
    transform: translateY(-3px);
    border-color: var(--sic-accent);
    box-shadow: var(--sic-shadow);
}

.sic-rec--top {
    border-color: var(--sic-top-border);
}

.sic-rec__top {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 8px;
}

.sic-rec__cat {
    font-size: 10.5px;
    line-height: 1.45;
    letter-spacing: 0.07em;
    text-transform: uppercase;
    color: var(--sic-muted);
    padding-right: 6px;
}

.sic-rec__rank {
    flex: 0 0 auto;
    font-size: 11px;
    font-weight: 750;
    color: var(--sic-accent);
    background: var(--sic-rank-bg);
    border-radius: 999px;
    padding: 3px 8px;
}

.sic-rec__brand {
    font-size: 17px;
    font-weight: 720;
    letter-spacing: -0.015em;
    line-height: 1.25;
    color: var(--sic-text);
    text-transform: capitalize;
}

.sic-rec__id {
    font-family: ui-monospace, SFMono-Regular, Consolas, "Liberation Mono", monospace;
    font-size: 10.5px;
    color: var(--sic-muted);
}

.sic-rec__footer {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    gap: 8px;
    margin-top: auto;
    padding-top: 6px;
}

.sic-rec__price {
    font-size: 16px;
    font-weight: 730;
    color: var(--sic-text);
    font-variant-numeric: tabular-nums;
}

.sic-rec__match {
    font-size: 12px;
    font-weight: 720;
    color: var(--sic-accent-2);
}

.sic-rec__bar {
    height: 7px;
    border-radius: 999px;
    background: var(--sic-track);
    overflow: hidden;
}

.sic-rec__bar > span {
    display: block;
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, var(--sic-accent), var(--sic-accent-2));
}

.sic-rec__caption {
    display: flex;
    justify-content: space-between;
    gap: 8px;
    font-size: 10px;
    color: var(--sic-muted);
    font-variant-numeric: tabular-nums;
}

/* ------------------------------------------------------------------ */
/* Table view                                                          */
/* ------------------------------------------------------------------ */
.sic-table-wrap {
    width: 100%;
    overflow-x: auto;
    border: 1px solid var(--sic-border);
    border-radius: 14px;
    background: var(--sic-panel);
}

.sic-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 12.5px;
    font-variant-numeric: tabular-nums;
}

.sic-table thead th {
    position: sticky;
    top: 0;
    text-align: left;
    font-size: 10px;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    font-weight: 700;
    color: var(--sic-muted);
    background: var(--sic-panel-2);
    padding: 10px 12px;
    border-bottom: 1px solid var(--sic-border);
    white-space: nowrap;
}

.sic-table tbody td {
    padding: 9px 12px;
    border-bottom: 1px solid var(--sic-border);
    color: var(--sic-text-soft);
    vertical-align: middle;
}

.sic-table tbody tr:last-child td {
    border-bottom: none;
}

.sic-table tbody tr:hover td {
    background: var(--sic-hover);
}

.sic-table__rank {
    color: var(--sic-accent) !important;
    font-weight: 700;
}

.sic-table__product {
    font-family: ui-monospace, SFMono-Regular, Consolas, "Liberation Mono", monospace;
    font-size: 11px;
    color: var(--sic-text) !important;
}

.sic-table__brand {
    color: var(--sic-text) !important;
    font-weight: 600;
    text-transform: capitalize;
    white-space: nowrap;
}

.sic-table__num {
    text-align: right;
    white-space: nowrap;
}

.sic-table__match {
    display: flex;
    align-items: center;
    gap: 8px;
    min-width: 130px;
}

.sic-table__match-track {
    flex: 1 1 auto;
    height: 6px;
    border-radius: 999px;
    background: var(--sic-track);
    overflow: hidden;
}

.sic-table__match-fill {
    display: block;
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, var(--sic-accent), var(--sic-accent-2));
}

.sic-table__match-value {
    flex: 0 0 auto;
    font-size: 11px;
    font-weight: 650;
    color: var(--sic-accent-2) !important;
}

.sic-table__category {
    max-width: 260px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}

/* ------------------------------------------------------------------ */
/* Empty state and animation                                          */
/* ------------------------------------------------------------------ */
.sic-empty {
    border: 1px dashed var(--sic-border-strong);
    border-radius: 18px;
    padding: 30px 24px;
    text-align: center;
    color: var(--sic-muted);
    background: var(--sic-panel-3);
    font-size: 13px;
    line-height: 1.6;
}

.sic-empty strong {
    display: block;
    color: var(--sic-text);
    font-size: 15px;
    margin-bottom: 4px;
}

@keyframes sicRise {
    from {
        opacity: 0;
        transform: translateY(8px);
    }
    to {
        opacity: 1;
        transform: none;
    }
}

@media (max-width: 900px) {
    .sic-profile-grid {
        grid-template-columns: 1fr;
    }
    .sic-mini-grid {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }
    .sic-hero {
        padding: 24px 22px;
    }
}

@media (prefers-reduced-motion: reduce) {
    .sic-rec {
        animation: none;
        transition: none;
    }
    .sic-rec:hover {
        transform: none;
    }
}
</style>
"""


def build_css(dark: bool) -> str:
    """Render the dashboard stylesheet for the requested theme.

    Args:
        dark: Use the dark palette when true, the light palette otherwise.

    Returns:
        A ``<style>`` block ready for ``st.markdown``.
    """
    palette = _DARK if dark else _LIGHT
    css = _TEMPLATE
    for key, value in palette.items():
        css = css.replace(f"__{key}__", value)
    return css
