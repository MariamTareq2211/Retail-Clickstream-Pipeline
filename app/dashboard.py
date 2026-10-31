"""Streamlit dashboard for the content-based recommendation demo.

The dashboard talks to ``app.api`` over HTTP. It never imports the artefacts
directly, which keeps the UI process small and lets the API be deployed or
restarted independently.

Run the API first, then start this page from the project root::

    venv\\Scripts\\streamlit run app/dashboard.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

# Streamlit executes this file as a script; make the project root importable.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import streamlit as st

from app.api_client import ApiError, DemoApiClient
from app.components import (
    format_int,
    render_empty_state,
    render_filter_chips,
    render_hero,
    render_kpis,
    render_profile,
    render_recommendation_grid,
    render_recommendation_table,
    render_section_header,
    render_sidebar_brand,
    render_status,
    short_key,
)
from app.config import DemoSettings
from app.styles import build_css

st.set_page_config(
    page_title="SIC Recommender",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner=False)
def get_client(base_url: str, timeout: float) -> DemoApiClient:
    """Create one HTTP client per Streamlit process."""
    return DemoApiClient(base_url, timeout=timeout)


def reset_filters(default_top_k: int, price_range: tuple[float, float]) -> None:
    """Restore every sidebar filter to its catalogue default.

    The function runs as a widget callback, before the next script run, which
    is the only point where Streamlit allows widget state to be overwritten.
    """
    for key in list(st.session_state):
        if key.startswith("filter_level2::"):
            st.session_state.pop(key, None)
    st.session_state["filter_level2::all"] = "All categories"
    st.session_state["filter_level1"] = "All departments"
    st.session_state["filter_brand"] = "All brands"
    st.session_state["filter_price"] = price_range
    st.session_state["top_k"] = default_top_k


settings = DemoSettings()
client = get_client(settings.api_url, settings.request_timeout_seconds)

with st.sidebar:
    st.markdown(render_sidebar_brand(), unsafe_allow_html=True)
    dark = st.toggle("Dark appearance", value=True, key="dark_mode")
    st.markdown(build_css(dark), unsafe_allow_html=True)

    try:
        with st.spinner("Connecting to the recommendation API…"):
            health = client.health()
            metadata = client.metadata()
            user_payload = client.users(limit=settings.sample_users)
    except ApiError as error:
        st.markdown(
            render_status(False, settings.api_url), unsafe_allow_html=True
        )
        st.error(f"API unavailable: {error}")
        st.markdown("**Start the API service from the project root:**")
        st.code(
            "venv\\Scripts\\python.exe -m uvicorn app.api:app "
            f"--host {settings.api_host} --port {settings.api_port}",
            language="powershell",
        )
        st.caption("Then refresh this page.")
        st.stop()

    st.markdown(render_status(True, settings.api_url), unsafe_allow_html=True)
    st.divider()

    st.markdown("#### User")
    source = st.radio(
        "User source",
        ("Sample users", "Custom key"),
        horizontal=True,
        key="user_source",
        label_visibility="collapsed",
    )

    sample_users = user_payload["users"]
    sample_meta = {user["user_key"]: user for user in sample_users}
    sample_keys = [user["user_key"] for user in sample_users]
    active_key: str | None = None

    if source == "Sample users":
        selected = st.selectbox(
            "Sample user",
            options=sample_keys,
            format_func=lambda key: (
                f"{short_key(key)} · {sample_meta[key]['interactions']:,} interactions"
            ),
            key="sample_user",
            help=(
                "Curated from users with at least "
                f"{settings.sample_min_interactions} training interactions."
            ),
        )
        active_key = selected
    else:
        with st.form("custom_user_form", border=True):
            raw_key = st.text_input(
                "Custom user_key",
                value=st.session_state.get("custom_input", ""),
                placeholder="-9223190590261032003",
            )
            submitted = st.form_submit_button(
                "Load profile", type="primary", width="stretch"
            )
        if submitted:
            cleaned = raw_key.strip()
            st.session_state["custom_input"] = cleaned
            st.session_state["custom_active"] = cleaned
        active_key = st.session_state.get("custom_active") or None
        if active_key:
            try:
                int(active_key)
            except ValueError:
                st.error("A user_key must be a signed 64-bit integer.")
                active_key = None
        elif submitted:
            st.warning("Enter a user_key first.")
        else:
            st.caption(
                "Paste any user_key with training history. Cold-start users "
                "are not in the profile matrix and return a 404."
            )

    st.divider()
    st.markdown("#### Filters")
    top_k = st.slider(
        "Recommendations",
        min_value=settings.min_top_k,
        max_value=settings.max_top_k,
        value=settings.default_top_k,
        step=1,
        key="top_k",
        help="The API applies every filter before selecting the top-K items.",
    )

    level1_entries = metadata.get("categories", {}).get("level1", [])
    level2_entries = metadata.get("categories", {}).get("level2", [])
    level1_options = ["All departments"] + [
        entry["value"] for entry in level1_entries
    ]
    level1_label = st.selectbox(
        "Department", level1_options, key="filter_level1"
    )
    level1 = None if level1_label == "All departments" else level1_label

    level2_values = [
        entry["value"]
        for entry in level2_entries
        if not level1 or level1 in entry.get("parents", [])
    ]
    level2_label = st.selectbox(
        "Category",
        ["All categories"] + level2_values,
        key=f"filter_level2::{level1 or 'all'}",
    )
    level2 = None if level2_label == "All categories" else level2_label

    brand_entries = metadata.get("brands", [])
    brand_options = ["All brands"] + [entry["value"] for entry in brand_entries]
    brand_label = st.selectbox(
        "Brand",
        brand_options,
        key="filter_brand",
        help=(
            f"Showing the {len(brand_entries)} most common brands out of "
            f"{metadata.get('brand_count', len(brand_entries)):,}."
        ),
    )
    brand = None if brand_label == "All brands" else brand_label

    catalogue_low = math.floor(float(metadata["price"]["min"]))
    catalogue_high = math.ceil(float(metadata["price"]["max"]))
    price_low, price_high = st.slider(
        "Price range ($)",
        min_value=float(catalogue_low),
        max_value=float(catalogue_high),
        value=(float(catalogue_low), float(catalogue_high)),
        step=1.0,
        format="$%.0f",
        key="filter_price",
    )
    min_price = None if price_low <= float(metadata["price"]["min"]) else price_low
    max_price = None if price_high >= float(metadata["price"]["max"]) else price_high

    st.button(
        "Reset filters",
        width="stretch",
        on_click=reset_filters,
        args=(
            settings.default_top_k,
            (float(catalogue_low), float(catalogue_high)),
        ),
    )

    st.caption("Filters run inside the API before the top-K cut.")

st.markdown(render_hero(metadata, online=True), unsafe_allow_html=True)
st.markdown(render_kpis(metadata), unsafe_allow_html=True)

if active_key is None:
    st.markdown(
        render_empty_state(
            "Choose a user to start",
            "Select a curated sample in the sidebar or load a custom user_key. "
            "Every recommendation excludes products the user already interacted with.",
        ),
        unsafe_allow_html=True,
    )
else:
    try:
        result = client.recommend(
            user_key=active_key,
            top_k=top_k,
            category_level1=level1,
            category_level2=level2,
            category_level3=None,
            brand=brand,
            min_price=min_price,
            max_price=max_price,
        )
    except ApiError as error:
        if error.status_code == 404:
            st.markdown(
                render_empty_state(
                    "No profile for this user",
                    "The key is valid but has no training interactions, so it is "
                    "a cold-start user and cannot be scored. Pick a key from the "
                    "sample dropdown or check /users.",
                ),
                unsafe_allow_html=True,
            )
        elif error.status_code == 400:
            st.error(f"Invalid request: {error}")
        else:
            st.error(f"Recommendation request failed: {error}")
    else:
        profile = result["profile"]
        st.markdown(
            render_section_header(
                "Profile snapshot",
                f"User {short_key(active_key)} · "
                f"{format_int(profile['total_interactions'])} interactions",
            ),
            unsafe_allow_html=True,
        )
        st.markdown(render_profile(profile), unsafe_allow_html=True)
        st.caption(f"user_key · {active_key}")

        st.markdown(
            render_section_header(
                "Recommended products",
                f"{result['returned']} products · "
                f"{result['ranking_ms']:.0f} ms · "
                f"{format_int(result['candidates_considered'])} eligible",
            ),
            unsafe_allow_html=True,
        )
        st.markdown(
            render_filter_chips(result["filters"], top_k), unsafe_allow_html=True
        )

        recommendations = result["recommendations"]
        if recommendations:
            st.markdown(
                render_recommendation_grid(recommendations),
                unsafe_allow_html=True,
            )
            if result["returned"] < top_k:
                st.info(
                    f"Only {result['returned']} products match the current "
                    f"filters; widen the price range or clear a filter to see "
                    f"the full top {top_k}."
                )

            frame = pd.DataFrame(
                [
                    {
                        "Rank": item["rank"],
                        "Product ID": item["product_id"],
                        "Category": item["category_path"],
                        "Brand": item["brand"],
                        "Price": item["price"],
                        "Cosine": item["score"],
                        "Match %": item["match_percent"],
                    }
                    for item in recommendations
                ]
            )
            with st.expander("Table view and CSV export"):
                st.markdown(
                    render_recommendation_table(recommendations),
                    unsafe_allow_html=True,
                )
                st.download_button(
                    "Download CSV",
                    data=frame.to_csv(index=False).encode("utf-8"),
                    file_name=f"recommendations_{active_key.lstrip('-')}.csv",
                    mime="text/csv",
                    width="stretch",
                )
        else:
            st.markdown(
                render_empty_state(
                    "No products match these filters",
                    "Widen the price range or clear the department, category or "
                    "brand filter. The API only ranks unseen catalogue items.",
                ),
                unsafe_allow_html=True,
            )

        st.caption(
            "Cosine similarity reflects content and price affinity, not a "
            f"purchase probability. Model: {metadata['model']['label']}."
        )

with st.expander("How this demo works"):
    st.markdown(
        """
The API loads the Module 2 catalogue matrix and user profiles once at start-up.
Both matrices are L2-normalised, so a user's score for every product is a single
sparse dot product (`profile @ catalogue.T`).

For every request the service:

1. masks products the user already interacted with during training,
2. applies the department, category, brand and price filters,
3. sorts the remaining cosine scores and returns the top-K.

The highlighted offline metrics come from the Module 3 hidden-test evaluation.
The dashboard is deliberately stateless: the API is the only source of truth.
        """
    )

st.caption(
    "SIC graduation project · Modules 1–3 · FastAPI + Streamlit demo layer"
)
