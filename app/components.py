"""Reusable HTML fragments for the Streamlit dashboard.

All product and profile text is escaped before it is interpolated into markup.
The fragments return plain strings so they can be unit-tested without a running
Streamlit server.
"""

from __future__ import annotations

from html import escape
from typing import Any, Iterable


def format_int(value: int | float | None) -> str:
    """Format a number with thousands separators, or an em dash."""
    if value is None:
        return "—"
    return f"{int(value):,}"


def format_price(value: float | None) -> str:
    """Format a price in US dollars."""
    if value is None:
        return "—"
    return f"${value:,.2f}"


def short_key(user_key: str) -> str:
    """Shorten a long user key for display while preserving its suffix."""
    text = str(user_key)
    return f"…{text[-6:]}" if len(text) > 10 else text


def _display_label(value: str) -> str:
    """Map internal sentinels to reader-friendly labels."""
    return "Unknown" if value == "Unknown" else value


def render_sidebar_brand() -> str:
    """Return the sidebar brand block."""
    return (
        '<div class="sic-side-brand">'
        '<div class="sic-logo">SIC</div>'
        "<div>"
        '<div class="sic-side-title">Recommender</div>'
        '<div class="sic-side-sub">Content-based demo console</div>'
        "</div>"
        "</div>"
    )


def render_status(online: bool, base_url: str) -> str:
    """Return the API status line used in the sidebar."""
    state = "online" if online else "offline"
    klass = "sic-status" if online else "sic-status sic-status--offline"
    return (
        f'<div class="{klass}"><span class="sic-status__dot"></span>'
        f"API {state} · {escape(base_url)}</div>"
    )


def render_hero(meta: dict[str, Any], online: bool) -> str:
    """Return the page hero with model and dataset badges."""
    model = meta.get("model", {})
    label = escape(str(model.get("label") or "Content-based recommender"))
    status = "Live API" if online else "Offline mode"
    catalogue = format_int(meta.get("catalogue_items"))
    profiles = format_int(meta.get("profile_users"))
    return (
        '<section class="sic-hero">'
        '<div class="sic-hero__eyebrow">'
        '<span class="sic-status__dot"></span>'
        f"{escape(status)} · Model 3"
        "</div>"
        '<h1 class="sic-hero__title">Find the next product worth your attention.</h1>'
        '<p class="sic-hero__subtitle">'
        "A content-based recommender that blends product text and price into a "
        "single cosine profile. Recommendations are ranked from a time-decayed "
        "weighted centroid of each shopper's recent clicks, carts and purchases."
        "</p>"
        '<div class="sic-badges">'
        f'<span class="sic-badge sic-badge--accent">{label}</span>'
        f'<span class="sic-badge">{catalogue} products</span>'
        f'<span class="sic-badge">{profiles} profiled users</span>'
        '<span class="sic-badge">Seen items excluded</span>'
        "</div>"
        "</section>"
    )


def render_kpis(meta: dict[str, Any]) -> str:
    """Return the catalogue and offline-evaluation KPI cards."""
    evaluation = meta.get("evaluation", {})
    by_k = evaluation.get("by_k", {})
    at_ten = by_k.get("10", {}) if isinstance(by_k, dict) else {}
    scored = evaluation.get("scored_users")
    recall = at_ten.get("recall")
    hit_rate = at_ten.get("hit_rate")

    cards = [
        (
            "Catalogue",
            format_int(meta.get("catalogue_items")),
            "Products with content features",
        ),
        (
            "Profiled users",
            format_int(meta.get("profile_users")),
            "Users with training history",
        ),
        (
            "Recall@10",
            f"{recall * 100:.2f}%" if recall is not None else "—",
            f"Hidden test · {format_int(scored)} scored users",
        ),
        (
            "HitRate@10",
            f"{hit_rate * 100:.2f}%" if hit_rate is not None else "—",
            "Lists containing a relevant item",
        ),
    ]
    body = "".join(
        '<div class="sic-kpi">'
        f'<div class="sic-kpi__label">{escape(label)}</div>'
        f'<div class="sic-kpi__value">{escape(value)}</div>'
        f'<div class="sic-kpi__meta">{escape(meta_text)}</div>'
        "</div>"
        for label, value, meta_text in cards
    )
    return f'<section class="sic-kpis">{body}</section>'


def render_section_header(title: str, hint: str = "") -> str:
    """Return a section heading with an optional right-aligned hint."""
    hint_html = (
        f'<div class="sic-section__hint">{escape(hint)}</div>' if hint else ""
    )
    return (
        '<div class="sic-section__head">'
        f'<h2 class="sic-section__title">{escape(title)}</h2>'
        f"{hint_html}"
        "</div>"
    )


def _facet_block(title: str, facets: Iterable[dict[str, Any]]) -> str:
    """Render one panel of horizontal facet bars."""
    facets = list(facets)
    if facets:
        maximum = max(int(facet.get("count", 0)) for facet in facets) or 1
        rows = []
        for facet in facets:
            value = _display_label(str(facet.get("value", "")))
            count = int(facet.get("count", 0))
            width = min(100.0, 100.0 * count / maximum)
            rows.append(
                '<div class="sic-facet">'
                '<div class="sic-facet__row">'
                f'<span class="sic-facet__value" title="{escape(value)}">'
                f"{escape(value)}</span>"
                f'<span class="sic-facet__count">{format_int(count)}</span>'
                "</div>"
                '<div class="sic-facet__track">'
                f'<div class="sic-facet__fill" style="width:{width:.1f}%"></div>'
                "</div>"
                "</div>"
            )
        body = "".join(rows)
    else:
        body = '<div class="sic-muted">No interactions recorded.</div>'
    return (
        '<div class="sic-panel">'
        f'<div class="sic-panel__title">{escape(title)}</div>'
        f"{body}"
        "</div>"
    )


def render_profile(profile: dict[str, Any]) -> str:
    """Return the profile metric grid, event mix and facet panels."""
    events = profile.get("event_counts", {})
    view = int(events.get("view", 0))
    cart = int(events.get("cart", 0))
    purchase = int(events.get("purchase", 0))

    mini = [
        ("Interactions", format_int(profile.get("total_interactions"))),
        ("Products", format_int(profile.get("distinct_products"))),
        ("Departments", format_int(profile.get("categories_explored"))),
        ("Brands", format_int(profile.get("brands_explored"))),
    ]
    mini_html = "".join(
        '<div class="sic-mini">'
        f'<div class="sic-mini__value">{escape(value)}</div>'
        f'<div class="sic-mini__label">{escape(label)}</div>'
        "</div>"
        for label, value in mini
    )

    chips = [
        f'<span class="sic-chip sic-chip--accent">{format_int(view)} views</span>',
        f'<span class="sic-chip">{format_int(cart)} carts</span>',
        f'<span class="sic-chip">{format_int(purchase)} purchases</span>',
    ]
    departments = profile.get("top_departments", [])
    if departments:
        top = _display_label(str(departments[0].get("value", "")))
        chips.append(f'<span class="sic-chip">Top department · {escape(top)}</span>')

    categories = _facet_block("Top categories", profile.get("top_categories", []))
    brands = _facet_block("Top brands", profile.get("top_brands", []))
    return (
        f'<div class="sic-mini-grid">{mini_html}</div>'
        f'<div class="sic-chip-row">{"".join(chips)}</div>'
        f'<div class="sic-profile-grid">{categories}{brands}</div>'
    )


def render_filter_chips(filters: dict[str, Any], top_k: int) -> str:
    """Return a compact summary of the active filters."""
    chips = [
        f'<span class="sic-chip sic-chip--accent">Top {int(top_k)}</span>',
        '<span class="sic-chip">Seen items hidden</span>',
    ]
    level1 = filters.get("category_level1")
    level2 = filters.get("category_level2")
    level3 = filters.get("category_level3")
    category_path = " > ".join(
        _display_label(str(part)) for part in (level1, level2, level3) if part
    )
    if category_path:
        chips.append(f'<span class="sic-chip">{escape(category_path)}</span>')
    if filters.get("brand"):
        chips.append(f'<span class="sic-chip">{escape(str(filters["brand"]))}</span>')
    minimum = filters.get("min_price")
    maximum = filters.get("max_price")
    if minimum is not None or maximum is not None:
        low = format_price(minimum) if minimum is not None else "Any"
        high = format_price(maximum) if maximum is not None else "Any"
        chips.append(
            f'<span class="sic-chip">Price {escape(low)} – {escape(high)}</span>'
        )
    return f'<div class="sic-chip-row">{"".join(chips)}</div>'


def render_recommendation_grid(items: Iterable[dict[str, Any]]) -> str:
    """Render recommendation cards as a responsive CSS grid."""
    cards = []
    for index, item in enumerate(items):
        rank = int(item.get("rank", index + 1))
        category = _display_label(str(item.get("category_path", "")))
        brand = _display_label(str(item.get("brand", "")))
        if brand == "Unknown":
            brand = "Unbranded"
        product_id = int(item.get("product_id", 0))
        price = format_price(float(item.get("price", 0.0)))
        match = float(item.get("match_percent", 0.0))
        score = float(item.get("score", 0.0))
        width = min(max(match, 0.0), 100.0)
        klass = "sic-rec sic-rec--top" if rank <= 3 else "sic-rec"
        delay = min(index * 25, 200)
        cards.append(
            f'<article class="{klass}" style="animation-delay:{delay}ms">'
            '<div class="sic-rec__top">'
            f'<span class="sic-rec__cat" title="{escape(category)}">'
            f"{escape(category)}</span>"
            f'<span class="sic-rec__rank">#{rank}</span>'
            "</div>"
            f'<div class="sic-rec__brand">{escape(brand)}</div>'
            f'<div class="sic-rec__id">Product {product_id}</div>'
            '<div class="sic-rec__footer">'
            f'<span class="sic-rec__price">{escape(price)}</span>'
            f'<span class="sic-rec__match">{match:.1f}% match</span>'
            "</div>"
            '<div class="sic-rec__bar">'
            f'<span style="width:{width:.1f}%"></span>'
            "</div>"
            '<div class="sic-rec__caption">'
            f"<span>cosine {score:.3f}</span>"
            f"<span>{match:.0f}%</span>"
            "</div>"
            "</article>"
        )
    return f'<div class="sic-rec-grid">{"".join(cards)}</div>'


def render_recommendation_table(items: Iterable[dict[str, Any]]) -> str:
    """Render recommendations as a theme-aware HTML table.

    The native Streamlit data grid is canvas-based and always follows the
    configured base theme, which would clash with the light palette. A plain
    HTML table keeps both appearances consistent while preserving the match
    progress indicator.

    Args:
        items: Recommendation payloads returned by the API.

    Returns:
        A ``<table>`` wrapped for horizontal overflow.
    """
    rows = []
    for index, item in enumerate(items):
        rank = int(item.get("rank", index + 1))
        brand = _display_label(str(item.get("brand", "")))
        if brand == "Unknown":
            brand = "Unbranded"
        category = _display_label(str(item.get("category_path", "")))
        product_id = int(item.get("product_id", 0))
        price = format_price(float(item.get("price", 0.0)))
        match = min(max(float(item.get("match_percent", 0.0)), 0.0), 100.0)
        score = float(item.get("score", 0.0))
        rows.append(
            "<tr>"
            f'<td class="sic-table__rank">#{rank}</td>'
            f'<td class="sic-table__product">{product_id}</td>'
            f'<td class="sic-table__category" title="{escape(category)}">'
            f"{escape(category)}</td>"
            f'<td class="sic-table__brand">{escape(brand)}</td>'
            f'<td class="sic-table__num">{escape(price)}</td>'
            '<td><div class="sic-table__match">'
            '<span class="sic-table__match-track">'
            f'<span class="sic-table__match-fill" style="width:{match:.1f}%">'
            "</span></span>"
            f'<span class="sic-table__match-value">{match:.1f}%</span>'
            "</div></td>"
            f'<td class="sic-table__num">{score:.3f}</td>'
            "</tr>"
        )

    header = (
        "<thead><tr>"
        "<th>Rank</th>"
        "<th>Product</th>"
        "<th>Category</th>"
        "<th>Brand</th>"
        "<th class=\"sic-table__num\">Price</th>"
        "<th>Match</th>"
        "<th class=\"sic-table__num\">Cosine</th>"
        "</tr></thead>"
    )
    return (
        '<div class="sic-table-wrap">'
        f'<table class="sic-table">{header}<tbody>{"".join(rows)}</tbody></table>'
        "</div>"
    )


def render_empty_state(title: str, body: str) -> str:
    """Return a friendly empty-state block."""
    return (
        '<div class="sic-empty">'
        f"<strong>{escape(title)}</strong>"
        f"{escape(body)}"
        "</div>"
    )
