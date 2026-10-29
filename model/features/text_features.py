"""Catalogue text construction and TF-IDF encoding (Plan.md 1.3, 1.6, 1.9).

Two problems are solved here
----------------------------
1. **The ``'Unknown'`` trap.** 53.4% of products miss level-1/2 categories and
   17.1% have no usable metadata at all. If ``'Unknown'`` were a normal token,
   half the catalogue would share it and cosine similarity would claim that two
   unrelated products are alike. Unknown values are therefore dropped, and every
   remaining value is namespaced (``cat1_``, ``cat2_``, ``cat3_``, ``brand_``) so
   a brand can never collide with a category of the same spelling.

2. **Leakage.** The vocabulary and the IDF weights are fitted exclusively on the
   text of products that appear in the training split. Validation and test items
   are only *transformed*: a token that never occurred in training is invisible
   to the model, exactly as it would be in production.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

from model.config.config_schema import Module1Config
from model.core.logger import get_logger

_LOGGER = get_logger(__name__)

# Any run of characters that is not a lowercase letter or digit becomes a single
# underscore, giving one stable token per attribute value.
_NON_ALNUM = re.compile(r"[^a-z0-9]+")

# TfidfVectorizer splits on whitespace and keeps the already-clean tokens intact.
_TOKEN_PATTERN = r"(?u)\S+"


@dataclass
class TextFeatureArtifacts:
    """Fitted vectorizer, the catalogue-wide matrix and descriptive statistics."""

    vectorizer: TfidfVectorizer
    matrix: sp.csr_matrix
    stats: dict[str, Any]


def _clean_token(value: object) -> str:
    """Normalise one raw attribute value into a safe token.

    Args:
        value: Raw cell value from the catalogue.

    Returns:
        Lowercase token containing only letters, digits and underscores
        (empty string when the value carries no usable characters).
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return _NON_ALNUM.sub("_", str(value).strip().lower()).strip("_")


def build_metadata_text(products: pd.DataFrame, config: Module1Config) -> pd.Series:
    """Assemble the namespaced ``metadata_text`` document for every product.

    Args:
        products: Product catalogue loaded by :mod:`model.data.loader`.
        config: Validated Module 1 configuration.

    Returns:
        One whitespace-separated token document per product, aligned with the
        catalogue index. Products with no usable metadata yield ``""`` and will
        become zero text vectors, relying on the price dimension instead.

    Raises:
        KeyError: If a configured text field is absent from the catalogue.
    """
    unknown_value = config.text.unknown_value.strip().lower()
    parts: list[pd.Series] = []

    for column, prefix in config.text.field_prefixes.items():
        if column not in products.columns:
            raise KeyError(f"Configured text field {column!r} is missing from the catalogue.")

        cleaned = products[column].astype("string").fillna("").map(_clean_token)

        # Drop the sentinel before namespacing: the token must disappear, not
        # become ``cat1_unknown`` (which would recreate the similarity bridge).
        if config.text.drop_unknown:
            cleaned = cleaned.mask(cleaned == unknown_value, "")

        parts.append(cleaned.map(lambda token, p=prefix: f"{p}_{token}" if token else ""))

    # Optional extra token carrying taxonomy depth that the three levels cannot
    # express (103 products have a fourth segment in ``category_code``).
    if config.text.include_category_path and "category_code" in products.columns:
        path_tokens = products["category_code"].astype("string").fillna("").map(_clean_token)
        parts.append(path_tokens.map(lambda token: f"path_{token}" if token else ""))

    text = parts[0]
    for part in parts[1:]:
        text = text.str.cat(part, sep=" ")

    # Collapse the gaps left by omitted tokens so the document is compact.
    text = text.str.replace(r"\s+", " ", regex=True).str.strip().astype(str)

    empty_share = float((text.str.len() == 0).mean() * 100)
    _LOGGER.info(
        "Built metadata_text for %s products | %.2f%% have no usable text tokens",
        f"{len(text):,}",
        empty_share,
    )
    return text


def fit_transform_text(
    metadata_text: pd.Series,
    train_mask: pd.Series,
    config: Module1Config,
) -> TextFeatureArtifacts:
    """Fit TF-IDF on training products and transform the whole catalogue.

    Args:
        metadata_text: Documents produced by :func:`build_metadata_text`.
        train_mask: Boolean mask aligned with ``metadata_text`` marking products
            that appear in the training split.
        config: Validated Module 1 configuration.

    Returns:
        Fitted vectorizer, the CSR matrix for all products, and fit diagnostics.

    Raises:
        ValueError: If the training slice contains no products or no tokens, in
            which case an empty vocabulary would silently produce zero vectors.
    """
    if len(metadata_text) != len(train_mask):
        raise ValueError("metadata_text and train_mask must have the same length.")

    train_documents = metadata_text.loc[train_mask]
    if train_documents.empty:
        raise ValueError(
            "The training split contains no products; widen the train ratio or "
            "check the split boundaries."
        )

    non_empty_train = int((train_documents.str.len() > 0).sum())
    if non_empty_train == 0:
        raise ValueError(
            "No training product carries usable metadata; TF-IDF cannot be fitted. "
            "Check text.field_prefixes and text.drop_unknown."
        )

    tfidf = config.text.tfidf
    vectorizer = TfidfVectorizer(
        token_pattern=_TOKEN_PATTERN,
        lowercase=False,  # tokens are normalised in _clean_token already
        min_df=tfidf.min_df,
        max_features=tfidf.max_features,
        ngram_range=tfidf.ngram_range,
        sublinear_tf=tfidf.sublinear_tf,
        dtype=np.float64,
    )
    vectorizer.fit(train_documents.tolist())

    feature_names = vectorizer.get_feature_names_out()
    if len(feature_names) == 0:
        raise ValueError("TF-IDF produced an empty vocabulary on the training slice.")

    matrix = vectorizer.transform(metadata_text.tolist()).tocsr()

    # Rows without any in-vocabulary token stay all-zero. They are not an error
    # (17.1% of the catalogue is metadata-free) but must be visible in the report.
    row_nnz = np.diff(matrix.indptr)
    zero_rows_all = int((row_nnz == 0).sum())
    zero_rows_train = int((row_nnz[train_mask.to_numpy()] == 0).sum())

    order = np.argsort(vectorizer.idf_)[:20]
    most_common = [str(feature_names[index]) for index in order]

    stats: dict[str, Any] = {
        "vocabulary_size": int(matrix.shape[1]),
        "train_documents": int(len(train_documents)),
        "train_documents_with_text": non_empty_train,
        "zero_text_rows_catalogue": zero_rows_all,
        "zero_text_rows_catalogue_pct": round(zero_rows_all / len(metadata_text) * 100, 4),
        "zero_text_rows_train": zero_rows_train,
        "matrix_nnz": int(matrix.nnz),
        "most_common_tokens": most_common,
        "tfidf_parameters": {
            "min_df": tfidf.min_df,
            "max_features": tfidf.max_features,
            "ngram_range": list(tfidf.ngram_range),
            "sublinear_tf": tfidf.sublinear_tf,
        },
    }

    _LOGGER.info(
        "TF-IDF fitted on %s training documents: %s features | "
        "%s catalogue rows (%.2f%%) have no in-vocabulary text token",
        f"{len(train_documents):,}",
        f"{matrix.shape[1]:,}",
        f"{zero_rows_all:,}",
        stats["zero_text_rows_catalogue_pct"],
    )
    return TextFeatureArtifacts(vectorizer=vectorizer, matrix=matrix, stats=stats)
