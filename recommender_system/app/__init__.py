"""Production-style demo layer for the content-based recommendation system.

The package contains two independently runnable processes:

``app.api``
    FastAPI service that loads the trained Module 2 artifacts and serves
    recommendation and profile endpoints.
``app.dashboard``
    Streamlit dashboard that consumes the API and renders the interactive demo.

Run both from the project root so that the ``model`` package and the trained
artifacts are importable and resolvable.
"""
