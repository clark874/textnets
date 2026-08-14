#!/usr/bin/env python3
"""Compare official term_weight edges with the 2024 count-edge reconstruction.

Rebuilds the env-card notebook's jieba TidyText, then builds matched Textnets
with edge_weight='term_weight' and edge_weight='count'.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

import jieba
import jieba.posseg as pseg
import pandas as pd
from scipy.stats import spearmanr

import textnets as tn

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "cn-audit" / "edge-weight-compare"
CSV = Path("/Users/moongoat/Downloads/按列合并的文档数据.csv")
DICT = Path("/Users/moongoat/Documents/writing/字典/20240524ht字典转jieba字典.txt")
STOP = Path("/Users/moongoat/Downloads/stop_words-master/original_lib/chinese/哈工大停用词表的副本.txt")
VALID_POS = {"n", "nr", "ns", "nt", "nz", "vn", "an", "i", "l", "t", "s", "f"}
EXTRA_STOP = {
    "地向",
    "让我们",
    "就像",
    "下午",
    "女士们",
    "先生们",
    "(",
    ")",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "0",
    "上午",
    "晚上",
    "早上",
    "中午",
    "凌晨",
    "今天",
    "明天",
    "后天",
    "昨天",
    "前天",
    "现在",
    " ",
}


def load_dict(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_stopwords(path: Path) -> set[str]:
    words = {re.sub("\n|\r", "", line) for line in path.read_text(encoding="utf-8").splitlines()}
    words.update(EXTRA_STOP)
    return words


def tokenize(text: str, custom_dict: list[str], stop_words: set[str]) -> list[str]:
    for word in custom_dict:
        jieba.add_word(word)
    allowed = set(custom_dict)
    tokens = [
        word
        for word, flag in pseg.lcut(str(text))
        if len(word) >= 2 and (flag in VALID_POS or word in allowed)
    ]
    url = re.compile(r"https?://\S+|www\.\S+")
    punct = re.compile(r"[^\w\s]")
    cleaned = []
    for token in tokens:
        if token in stop_words or token.isdigit() or url.match(token) or punct.match(token):
            continue
        cleaned.append(token)
    return cleaned


def tidy_from_series(docs: pd.Series, custom_dict: list[str], stop_words: set[str]) -> pd.DataFrame:
    rows = []
    for label, text in docs.items():
        counts = Counter(tokenize(text, custom_dict, stop_words))
        for term, n in counts.items():
            rows.append({"label": label, "term": term, "n": n})
    return pd.DataFrame(rows).set_index("label")


def pairwise_agreement(a: list, b: list) -> float:
    n = len(a)
    if n < 2:
        return float("nan")
    agree = 0
    total = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += 1
            agree += int((a[i] == a[j]) == (b[i] == b[j]))
    return agree / total


def spearman(a: pd.Series, b: pd.Series) -> dict:
    aligned = pd.concat([a, b], axis=1, keys=["a", "b"]).dropna()
    if len(aligned) < 3:
        return {"n": int(len(aligned)), "rho": None, "p": None}
    rho, p = spearmanr(aligned["a"], aligned["b"])
    return {"n": int(len(aligned)), "rho": None if math.isnan(rho) else float(rho), "p": None if math.isnan(p) else float(p)}


def edge_frame(net: tn.Textnet) -> pd.DataFrame:
    rows = []
    g = net.graph
    for edge in g.es:
        source = g.vs[edge.source]
        target = g.vs[edge.target]
        if source["type"] == "doc":
            doc, term = source["id"], target["id"]
        else:
            doc, term = target["id"], source["id"]
        rows.append({"doc": doc, "term": term, "weight": float(edge["weight"])})
    return pd.DataFrame(rows)


def compare_nets(name: str, left: tn.Textnet, right: tn.Textnet) -> dict:
    left_e = edge_frame(left).set_index(["doc", "term"])["weight"]
    right_e = edge_frame(right).set_index(["doc", "term"])["weight"]
    shared = left_e.index.intersection(right_e.index)
    only_left = left_e.index.difference(right_e.index)
    only_right = right_e.index.difference(left_e.index)
    left_nodes = pd.DataFrame({"id": left.nodes["id"], "type": left.nodes["type"]})
    right_nodes = pd.DataFrame({"id": right.nodes["id"], "type": right.nodes["type"]})
    left_pos = {(node["type"], node["id"]): i for i, node in enumerate(left.nodes)}
    right_pos = {(node["type"], node["id"]): i for i, node in enumerate(right.nodes)}
    shared_nodes = [key for key in left_pos if key in right_pos]
    left_cluster = [left.clusters.membership[left_pos[key]] for key in shared_nodes]
    right_cluster = [right.clusters.membership[right_pos[key]] for key in shared_nodes]

    def typed(series: pd.Series, net: tn.Textnet, node_type: str) -> pd.Series:
        values = {}
        for node, value in zip(net.nodes, series.to_numpy()):
            if node["type"] == node_type:
                values[node["id"]] = value
        return pd.Series(values)

    left_term_strength = typed(left.strength, left, "term")
    right_term_strength = typed(right.strength, right, "term")
    result = {
        "setting": name,
        "left_nodes": int(left.graph.vcount()),
        "right_nodes": int(right.graph.vcount()),
        "left_edges": int(left.graph.ecount()),
        "right_edges": int(right.graph.ecount()),
        "shared_edges": int(len(shared)),
        "only_term_weight": int(len(only_left)),
        "only_count": int(len(only_right)),
        "same_topology": bool(len(only_left) == 0 and len(only_right) == 0 and len(left_e) == len(right_e)),
        "edge_weight": spearman(left_e, right_e),
        "clusters_term_weight": int(len(set(left.clusters.membership))),
        "clusters_count": int(len(set(right.clusters.membership))),
        "cluster_pairwise_agreement": pairwise_agreement(left_cluster, right_cluster),
        "strength_docs": spearman(typed(left.strength, left, "doc"), typed(right.strength, right, "doc")),
        "strength_terms": spearman(left_term_strength, right_term_strength),
        "degree_docs": spearman(typed(left.degree, left, "doc"), typed(right.degree, right, "doc")),
        "degree_terms": spearman(typed(left.degree, left, "term"), typed(right.degree, right, "term")),
        "birank_docs": spearman(typed(left.birank, left, "doc"), typed(right.birank, right, "doc")),
        "birank_terms": spearman(typed(left.birank, left, "term"), typed(right.birank, right, "term")),
        "hits_docs": spearman(typed(left.hits, left, "doc"), typed(right.hits, right, "doc")),
        "hits_terms": spearman(typed(left.hits, left, "term"), typed(right.hits, right, "term")),
        "top10_terms_term_weight": left_term_strength.sort_values(ascending=False).head(10).index.tolist(),
        "top10_terms_count": right_term_strength.sort_values(ascending=False).head(10).index.tolist(),
    }
    result["top10_term_overlap"] = len(
        set(result["top10_terms_term_weight"]).intersection(result["top10_terms_count"])
    )
    return result


def compare_projection(name: str, left: tn.network.ProjectedTextnet, right: tn.network.ProjectedTextnet) -> dict:
    def edges(net):
        rows = []
        for edge in net.graph.es:
            a, b = sorted((net.graph.vs[edge.source]["id"], net.graph.vs[edge.target]["id"]))
            rows.append({"a": a, "b": b, "weight": float(edge["weight"])})
        return pd.DataFrame(rows).set_index(["a", "b"])["weight"]

    left_e, right_e = edges(left), edges(right)
    shared = left_e.index.intersection(right_e.index)
    left_ids = list(left.nodes["id"])
    right_pos = {node["id"]: i for i, node in enumerate(right.nodes)}
    keep = [i for i in left_ids if i in right_pos]
    left_pos = {node["id"]: i for i, node in enumerate(left.nodes)}
    return {
        "setting": name,
        "left_nodes": int(left.graph.vcount()),
        "right_nodes": int(right.graph.vcount()),
        "left_edges": int(left.graph.ecount()),
        "right_edges": int(right.graph.ecount()),
        "shared_edges": int(len(shared)),
        "same_topology": bool(set(left_e.index) == set(right_e.index)),
        "edge_weight": spearman(left_e, right_e),
        "clusters_term_weight": int(len(set(left.clusters.membership))),
        "clusters_count": int(len(set(right.clusters.membership))),
        "cluster_pairwise_agreement": pairwise_agreement(
            [left.clusters.membership[left_pos[i]] for i in keep],
            [right.clusters.membership[right_pos[i]] for i in keep],
        ),
        "strength": spearman(left.strength, right.strength),
        "betweenness": spearman(left.betweenness, right.betweenness),
        "top_nodes_term_weight": left.strength.sort_values(ascending=False).head(8).index.tolist(),
        "top_nodes_count": right.strength.sort_values(ascending=False).head(8).index.tolist(),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tn.params["seed"] = 42
    raw = pd.read_csv(CSV)
    docs = raw.set_index("文档标签")["文档内容"]
    custom_dict = load_dict(DICT)
    stop_words = load_stopwords(STOP)
    tidy = tidy_from_series(docs, custom_dict, stop_words)
    weighted = tn.corpus._tf_idf(tidy, sublinear=False)
    weighted.to_dataframe().to_csv(OUT / "tidytext.csv")

    specs = {
        "top5": {"min_docs": 1, "top_n_terms": 5},
        "all": {"min_docs": 1, "top_n_terms": None},
    }
    report = {
        "corpus": {
            "path": str(CSV),
            "n_docs": int(len(docs)),
            "labels": docs.index.tolist(),
            "n_tidy_rows": int(len(weighted.to_dataframe())),
            "n_terms": int(weighted.to_dataframe()["term"].nunique()),
            "dict_size": len(custom_dict),
            "stopwords": len(stop_words),
            "seed": 42,
        },
        "bipartite": [],
        "projections": [],
    }

    for name, kwargs in specs.items():
        tw = tn.Textnet(weighted, edge_weight="term_weight", **kwargs)
        ct = tn.Textnet(weighted, edge_weight="count", **kwargs)
        report["bipartite"].append(compare_nets(name, tw, ct))
        report["projections"].append(compare_projection(f"{name}-doc", tw.project(node_type=tn.DOC), ct.project(node_type=tn.DOC)))
        report["projections"].append(compare_projection(f"{name}-term", tw.project(node_type=tn.TERM), ct.project(node_type=tn.TERM)))
        edge_frame(tw).assign(scheme="term_weight", setting=name).to_csv(OUT / f"edges_{name}_term_weight.csv", index=False)
        edge_frame(ct).assign(scheme="count", setting=name).to_csv(OUT / f"edges_{name}_count.csv", index=False)

    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
