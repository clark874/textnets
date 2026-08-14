"""Tests for the Chinese localization layer on textnets 0.10.5."""

from __future__ import annotations

import pandas as pd
import pytest

import textnets as tn
from textnets.corpus import TidyText


@pytest.fixture
def zh_docs() -> pd.Series:
    return pd.Series(
        {
            "doc1": "外交部就气候变化问题答记者问",
            "doc2": "外交部回应核污染水排海问题",
            "doc3": "气候变化是全球共同挑战",
        }
    )


def test_tidytext_from_records_and_dataframe_roundtrip():
    records = [
        {"label": "a", "term": "气候", "n": 2},
        {"label": "a", "term": "外交部", "n": 1},
        {"label": "b", "term": "气候", "n": 1},
    ]
    tidy = TidyText(records)
    frame = tidy.to_dataframe()
    assert list(frame.columns) == ["term", "n"]
    assert set(frame.index) == {"a", "b"}
    weighted = tn.corpus._tf_idf(frame, sublinear=False)
    assert set(weighted.columns) == {"term", "n", "term_weight"}


def test_official_edge_weight_is_term_weight(corpus):
    tokens = corpus.tokenized()
    official = tn.Textnet(tokens, min_docs=1)
    counted = tn.Textnet(tokens, min_docs=1, edge_weight="count")
    assert official.m.to_dataframe().shape == counted.m.to_dataframe().shape
    assert not official.m.to_dataframe().equals(counted.m.to_dataframe())


def test_top_n_terms_and_min_frequency(corpus):
    tokens = corpus.tokenized()
    trimmed = tn.Textnet(tokens, min_docs=1, top_n_terms=1)
    assert trimmed.graph.vcount() > 0
    rare = tn.Textnet(tokens, min_docs=1, min_term_frequency=10_000)
    assert rare.graph.ecount() == 0


def test_node_frequency_attributes(corpus):
    net = tn.Textnet(corpus.tokenized(), min_docs=1)
    exported = net.to_dataframe()
    assert {"id", "type", "total_frequency", "frequency_by_doc", "tfidf_by_doc"} <= set(
        exported.columns
    )
    assert exported.loc[exported["type"] == "term", "total_frequency"].notna().any()
    assert "frequency_by_doc" not in net.graph.vs.attributes()
    adj = net.to_adjacency_dataframe()
    assert adj.shape[0] > 0


def test_doc_projection_co_occurrence(corpus):
    net = tn.Textnet(corpus.tokenized(), min_docs=1)
    papers = net.project(node_type=tn.DOC)
    table = papers.doc_co_occurrence_dataframe()
    assert {"doc_a", "doc_b", "count", "terms"} <= set(table.columns)
    if not table.empty:
        assert (table["count"] >= 0).all()


def test_custom_tokenized_chinese(zh_docs):
    pytest.importorskip("jieba")
    corpus = tn.Corpus(zh_docs)
    tokens = corpus.custom_tokenized(custom_dict=["外交部", "核污染水", "气候变化"])
    terms = set(tokens.to_dataframe()["term"])
    assert {"外交部", "气候变化"} & terms
    net = tn.Textnet(tokens, min_docs=1, top_n_terms=5)
    assert net.to_dataframe()["id"].isin(["外交部", "气候变化", "核污染水"]).any()
