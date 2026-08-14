"""Implements the features relating to networks."""

from __future__ import annotations

import json
import os
import sqlite3
import warnings
from abc import ABC, abstractmethod
from collections import Counter
from enum import Flag
from functools import cached_property
from os import cpu_count
from pathlib import Path
from typing import Any, Callable, IO, Iterator, Literal
from warnings import warn
from collections import defaultdict

import igraph as ig
import leidenalg as la
import numpy as np
import pandas as pd
from scipy import LowLevelCallable
from scipy.integrate import quad
from toolz import memoize
from tqdm.auto import tqdm
from tqdm.contrib.concurrent import thread_map
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
import networkx as nx

import textnets as tn
from ._util import LiteFrame

from .corpus import TidyText
from .fca import FormalContext
from .viz import decorate_plot

try:
    from . import _ext  # type: ignore

    integrand = LowLevelCallable.from_cython(_ext, "df_integrand")
except ImportError:

    def integrand(x: float, degree: int) -> np.float64:
        """Fallback version of integrand function for the disparity filter."""
        return np.float64(1 - x) ** (degree - 2)

    warn("Could not import compiled extension, backbone extraction will be slow.")


#: Flag to distinguish node types.
NodeType = Flag("NodeType", [("TERM", True), ("DOC", False)])
TERM = NodeType.TERM
DOC = NodeType.DOC


def _make_top(prop, desc):
    """Helper function to create top_* methods for Textnet classes."""

    def method(cls, n=10):
        return getattr(cls, prop).sort_values(ascending=False).head(n)

    method.__doc__ = f"""
        Show nodes sorted by {desc}.

        Parameters
        ----------
        n : int, optional
            How many nodes to show (default: 10).

        Returns
        -------
        `pandas.Series`
            Ranked nodes.
        """
    return method


class TextnetBase(ABC):
    """
    Abstract base class for `Textnet` and `ProjectedTextnet`.

    Attributes
    ----------
    graph : `igraph.Graph`
        Direct access to the igraph object.
    """

    def __init__(self, graph: ig.Graph) -> None:
        self.graph: ig.Graph = graph

    @property
    def summary(self) -> str:
        """Summary of underlying graph."""
        return self.graph.summary()

    @property
    def nodes(self) -> ig.VertexSeq:
        """Iterate over nodes."""
        return self.graph.vs

    @property
    def edges(self) -> ig.EdgeSeq:
        """Iterate of edges."""
        return self.graph.es

    def vcount(self) -> int:
        """Return the number of vertices (nodes)."""
        return self.graph.vcount()

    def ecount(self) -> int:
        """Return the number of edges."""
        return self.graph.ecount()

    def save_graph(
        self,
        target: str | bytes | os.PathLike[Any] | IO,
        format: str | None = None,
    ) -> None:
        """
        Save the underlying graph.

        Parameters
        ----------
        target : str or path or file
            File or path that the graph should be written to.
        format : {"dot", "edgelist", "gml", "graphml", "pajek", ...}, optional
            Optionally specify the desired format (otherwise it is inferred
            from the file suffix).
        """
        self.graph.write(target, format)

    @cached_property
    def degree(self) -> pd.Series:
        """Unweighted node degree."""
        return pd.Series(self.graph.degree(), index=self.nodes["id"])

    @cached_property
    def strength(self) -> pd.Series:
        """Weighted node degree."""
        return pd.Series(self.graph.strength(weights="weight"), index=self.nodes["id"])

    @cached_property
    def node_types(self) -> list[NodeType]:
        """Return list of node types."""
        return [TERM if t == "term" else DOC for t in self.nodes["type"]]

    @abstractmethod
    def plot(self, **kwargs) -> ig.Plot:
        pass

    @abstractmethod
    def _partition_graph(self, resolution: float, seed: int) -> ig.VertexClustering:
        pass

    _partition: ig.VertexClustering | None = None

    @property
    def clusters(self) -> ig.VertexClustering:
        """
        Return graph partition.

        The partition is detected by the Leiden algorithm, unless a different
        partition that was supplied to the setter.
        """
        if self._partition is None:
            self._partition = self._partition_graph(
                resolution=tn.params["resolution_parameter"],
                seed=tn.params["seed"],
            )
        return self._partition

    @clusters.setter
    def clusters(
        self,
        value: ig.VertexClustering | ig.VertexDendrogram | dict[int, list[int]],
    ) -> None:
        if isinstance(value, ig.VertexClustering):
            self._partition = value
        elif isinstance(value, ig.VertexDendrogram):
            self._partition = value.as_clustering()
        elif isinstance(value, dict):
            sorted_node_community_map = dict(sorted(value.items()))
            part = ig.VertexClustering(
                self.graph,
                membership=[i[0] for i in sorted_node_community_map.values()],
            )
            self._partition = part
        elif isinstance(value, list):
            part = ig.VertexClustering(self.graph, membership=value)
            self._partition = part
        else:
            raise ValueError("No valid clusters supplied.")

    @clusters.deleter
    def clusters(self) -> None:
        self._partition = None

    @property
    def modularity(self) -> float:
        """Return modularity based on graph partition."""
        return self.graph.modularity(self.clusters, weights="weight")

    @property
    def cluster_strength(self) -> pd.Series:
        """Weighted node degree within each cluster's subgraph."""
        d = {}
        for c in self.clusters:
            subgraph = self.graph.subgraph(c)
            subgraph_strength = subgraph.strength(weights="weight")
            for n, strength in zip(c, subgraph_strength):
                node = self.nodes[n]
                d[node["id"]] = strength
        return pd.Series(d).reindex(self.strength.index)

    @property
    def cluster_local_cc(self) -> pd.Series:
        """Weighted local clustering coefficient within each cluster's subgraph."""
        d = {}
        for c in self.clusters:
            subgraph = self.graph.subgraph(c)
            subgraph_cc = subgraph.transitivity_local_undirected(weights="weight")
            for n, cc in zip(c, subgraph_cc):
                node = self.nodes[n]
                d[node["id"]] = cc
        return pd.Series(d).reindex(self.strength.index)

    def top_cluster_nodes(
            self, n: int = 10, rank_nodes_by: str = "cluster_strength"
        ) -> pd.DataFrame:
            """
            Show top nodes per cluster, ranked by a chosen metric.

            Parameters
            ----------
            n : int, optional
                How many nodes to show per cluster (default: 10)
            rank_nodes_by : str, optional
                Metric to rank nodes within each cluster by (default:
                cluster_strength).

            Returns
            -------
            `pandas.DataFrame`
                Clusters with representative nodes and additional information.
            """
            # 获取节点信息
            node_data = pd.DataFrame({
                "id": self.nodes["id"],
                "metric": getattr(self, rank_nodes_by),
                "cluster": self.clusters.membership,
                "type": ["doc" if t == "doc" else "term" for t in self.nodes["type"]],
                "degree": self.graph.degree(),
                "total_frequency": [self.nodes["total_frequency"][i] if "total_frequency" in self.nodes.attributes() else None for i in range(len(self.nodes))],
                "frequency_by_doc": [self.nodes["frequency_by_doc"][i] if "frequency_by_doc" in self.nodes.attributes() else None for i in range(len(self.nodes))],
                "tfidf_by_doc": [self.nodes["tfidf_by_doc"][i] if "tfidf_by_doc" in self.nodes.attributes() else None for i in range(len(self.nodes))]
            })

            # 按文档和词语分别排序
            node_data_docs = node_data[node_data['type'] == 'doc'].sort_values(by="metric", ascending=False)
            node_data_terms = node_data[node_data['type'] == 'term'].sort_values(by="metric", ascending=False)
            node_data = pd.concat([node_data_docs, node_data_terms])

            # 获取聚类信息
            cluster_info = (
                node_data.groupby("cluster")
                .agg({
                    "id": lambda x: list(x[:n]),
                    "metric": lambda x: list(x[:n]),
                    "type": lambda x: x.value_counts().to_dict(),
                    "degree": "mean",
                    "total_frequency": lambda x: list(x[:n]),
                    "frequency_by_doc": lambda x: list(x[:n]),
                    "tfidf_by_doc": lambda x: list(x[:n])
                })
                .rename(columns={"degree": "avg_degree", "metric": "top_metrics", "id": "nodes"})
                .reset_index()
            )

            # 添加聚类方法名称和其他信息
            cluster_info["method"] = "leiden"
            cluster_info["modularity"] = self.modularity
            cluster_info["cluster_strength"] = cluster_info["nodes"].apply(lambda x: [self.cluster_strength[node] for node in x])
            cluster_info["cluster_local_cc"] = cluster_info["nodes"].apply(lambda x: [self.cluster_local_cc[node] for node in x])

            return cluster_info

    @decorate_plot
    def _plot(
        self,
        **kwargs,
    ) -> ig.Plot:
        tn.init_seed()
        return ig.plot(self.graph, **kwargs)

    def __repr__(self) -> str:
        type_counts: Counter[Literal["doc", "term"]] = Counter(self.nodes["type"])
        return (
            f"""<{self.__class__.__name__} with {type_counts["doc"]} documents, """
            + f"""{type_counts["term"]} terms, and {self.ecount()} edges>"""
        )

    def _repr_html_(self) -> str:
        type_counts: Counter[Literal["doc", "term"]] = Counter(self.nodes["type"])
        return f"""
            <style scoped>
              .full-width {{ width: 100%; }}
            </style>
            <table class="full-width">
              <tr style="font-weight: 600;">
                <td style="text-align: left;">
                  <kbd>{self.__class__.__name__}</kbd>
                </td>
                <td style="color: dodgerblue;">
                  <svg width="1ex" height="1ex">
                    <rect width="1ex" height="1ex" fill="dodgerblue">
                  </svg>
                  Docs: {type_counts["doc"]}
                </td>
                <td style="color: orangered;">
                  <svg width="1ex" height="1ex">
                    <circle cx="50%" cy="50%" r="50%" fill="orangered">
                  </svg>
                  Terms: {type_counts["term"]}
                </td>
                <td style="color: darkgray;">
                  <svg width="2ex" height="1ex">
                    <line x1="0" y1="50%" x2="100%" y2="50%"
                      stroke="darkgray"
                      stroke-width="2">
                  </svg>
                  Edges: {self.ecount()}
                </td>
              </tr>
            </table>"""


for prop, desc in [("degree", "unweighted degree"), ("strength", "weighted degree")]:
    setattr(TextnetBase, f"top_{prop}", _make_top(prop, desc))


class Textnet(TextnetBase, FormalContext):
    """
    Textnet for the relational analysis of meanings.

    A textnet is a bipartite network of documents and terms. Links exist
    only between the two types of nodes. Documents have a tie with terms
    they contain; the tie is weighted by *tf-idf*.

    The bipartite network can be projected into two different kinds of
    single-mode networks: document-to-document, and term-to-term.

    Experimental: The underlying bipartite adjacency matrix can also be turned
    into a formal context, which can be used to construct a concept lattice.

    Parameters
    ----------
    data: TidyText or BiadjacencyMatrix
        * DataFrame of tokens with per-document counts, as created by
          `Corpus.tokenized` `Corpus.ngrams`, and `Corpus.noun_phrases`.
        * A bipartite adjacency matrix relating documents to terms.
    min_docs : int, optional
        Minimum number of documents a term must appear in to be included
        in the network (default: 2).
    connected : bool, optional
        Keep only the largest connected component of the network (default:
        False).
    remove_weak_edges : bool, optional
        Remove edges with weights far below average (default: False).
    doc_attrs : dict of dict, optional
        Additional attributes of document nodes.

    Raises
    ------
    ValueError
        If the supplied data is empty.
    """

    def __init__(
        self,
        data: TidyText | BiadjacencyMatrix | pd.DataFrame,
        min_docs: int = 2,
        min_term_frequency: int = 1,  # 新增：最低词频
        top_n_terms: int = None,  # 新增：每个主题的最高词数
        connected: bool = False,
        remove_weak_edges: bool = False,
        doc_attrs: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self._connected = connected
        self._doc_attrs = doc_attrs
        if data.empty:
            raise ValueError("Data is empty.")
        if isinstance(data, BiadjacencyMatrix):
            self._matrix = data
        elif isinstance(data, (TidyText, pd.DataFrame)):
            self._matrix = _matrix_from_tidy_text(data, min_docs, min_term_frequency, top_n_terms)  # 修改：加入top_n_terms参数
        if remove_weak_edges:
            pairs: pd.Series = self._matrix.stack()
            edge_weights: pd.Series = pairs[pairs > 0]
            iqr: float = edge_weights.quantile(0.75) - edge_weights.quantile(0.25)
            cutoff: float = edge_weights.median() - 1.5 * iqr
            if cutoff > 0:
                self._matrix = BiadjacencyMatrix(
                    self._matrix[self._matrix > cutoff].dropna(how="all").fillna(0)
                )
        # 新增部分：创建igraph图对象
        self._graph = self._create_graph()


    def _create_graph(self) -> ig.Graph:
        g = _graph_from_matrix(self._matrix)
        if self._doc_attrs is not None:
            for name, attr in self._doc_attrs.items():
                g.vs[name] = [attr.get(doc) for doc in g.vs["id"]]
        if self._connected:
            return giant_component(g)
        return g

    @cached_property
    def graph(self) -> ig.Graph:
        """Direct access to the underlying igraph object."""
        g = _graph_from_matrix(self._matrix)

        # 添加词频和 tf-idf 权重属性到节点
        term_frequencies = self._matrix.term_total_freq
        term_freq_dict = self._matrix.term_freq_dict
        term_tfidf_dict = self._matrix.term_tfidf_dict

        doc_count = len(self._matrix.index)

        for i, node in enumerate(g.vs):
            if node["type"] == "term":
                node_index = i - doc_count  # 词汇节点的索引从 doc_count 开始
                term = node["id"]
                node["total_frequency"] = term_frequencies.get(term, 0)
                node["frequency_by_doc"] = term_freq_dict.get(term, {})
                node["tfidf_by_doc"] = term_tfidf_dict.get(term, {})
                # 打印调试信息
                # print(f"词汇：{term}，总词频：{node['total_frequency']}，词频分布：{node['frequency_by_doc']}，tf-idf 分布：{node['tfidf_by_doc']}")

        # 将 doc_attrs 属性添加到节点
        if self._doc_attrs is not None:
            for name, attr in self._doc_attrs.items():
                g.vs[name] = [attr.get(doc) for doc in g.vs["id"]]

        if self._connected:
            return giant_component(g)
        return g



    @cached_property
    def m(self) -> BiadjacencyMatrix:
        """Weighted bipartite adjacency matrix of the bipartite graph."""
        if not self._connected:
            return self._matrix
        a = np.array(self.graph.get_biadjacency(types=self.node_types)[0]).astype(
            "float64"
        )
        a[a == 1] = self.edges["weight"]
        doc_count, _ = a.shape
        m = BiadjacencyMatrix(
            a, index=self.nodes["id"][:doc_count], columns=self.nodes["id"][doc_count:]
        )
        m.T.index.name = "term"
        return m


    def project(
        self,
        *,
        node_type: Literal["doc", "term"] | NodeType,
        connected: bool | None = False,
    ) -> ProjectedTextnet:
        """
        Project to one-mode network.

        Parameters
        ----------
        node_type : {NodeType.DOC, NodeType.TERM, "doc", "term"}
            Either `DOC` or `TERM`, depending on desired node type.
        connected : bool, optional
            Keep only the largest connected component of the projected network
            (default: False).

        Raises
        ------
        ValueError
            If no valid node type is specified.

        Returns
        -------
        `ProjectedTextnet`
            A one-mode textnet.
        """
        if not isinstance(node_type, NodeType) and node_type not in {"doc", "term"}:
            raise ValueError("No valid node_type specified.")
        graph_to_return = 0
        if node_type in (TERM, "term"):
            graph_to_return = 1
            sparse_array = self.m.to_sparse_array()
            weights = sparse_array.T @ sparse_array
        else:
            array = self.m.to_array()
            weights = array @ array.T
        g = self.graph.bipartite_projection(
            types=self.node_types, which=graph_to_return
        )
        pairs = [
            (s.index, t.index) for s, t in (g.es[i].vertex_tuple for i in g.es.indices)
        ]
        g.es.set_attribute_values("weight", [weights[i] for i in pairs])
        g.es["cost"] = [
            1 / pow(w, tn.params["tuning_parameter"]) for w in g.es["weight"]
        ]

        # 新增部分：计算文档之间的共现词详细信息并设置为节点属性
        if node_type in (DOC, "doc"):
            biadjacency_df = self.m.to_dataframe()
            doc_co_occurrence = defaultdict(dict)
            for i, j in pairs:
                doc_i, doc_j = g.vs[i]["id"], g.vs[j]["id"]
                common_terms_i = biadjacency_df.loc[doc_i].to_numpy().nonzero()[0]
                common_terms_j = biadjacency_df.loc[doc_j].to_numpy().nonzero()[0]
                common_terms = set(common_terms_i).intersection(set(common_terms_j))
                common_terms_details = [biadjacency_df.columns[term_index] for term_index in common_terms]
                count = len(common_terms)
                doc_co_occurrence[doc_i][doc_j] = {
                    "count": count,
                    "terms": common_terms_details
                }
                doc_co_occurrence[doc_j][doc_i] = {
                    "count": count,
                    "terms": common_terms_details
                }
            
            for doc, co_occurrences in doc_co_occurrence.items():
                g.vs.find(id=doc)["doc_co_occurrence"] = co_occurrences

        if connected:
            g = giant_component(g)
        return ProjectedTextnet(g)
    

    def save(self, target: os.PathLike[Any] | str) -> None:
        """
        Save a textnet to file.

        Parameters
        ----------
        target : str or path
            File to save the corpus to. If the file exists, it will be
            overwritten.
        """
        conn = sqlite3.connect(Path(target))
        meta = {"connected": self._connected, "doc_attrs": json.dumps(self._doc_attrs)}
        with conn, warnings.catch_warnings():
            warnings.simplefilter("ignore")  # catch warning from pandas.to_sql
            self.m.T.to_sql("textnet_im", conn, if_exists="replace")
            pd.Series(meta, name="values").to_sql(
                "textnet_meta", conn, if_exists="replace", index_label="keys"
            )

    @classmethod
    def load(cls, source: os.PathLike[Any] | str) -> Textnet:
        """
        Load a textnet from file.

        Parameters
        ----------
        source : str or path
            File to read the corpus from. This should be a file created by
            `Textnet.save`.

        Raises
        ------
        FileNotFoundError
            If the provided path does not exist.

        Returns
        -------
        `Textnet`
        """
        if not Path(source).exists():
            raise FileNotFoundError(f"File '{source}' does not exist.")
        conn = sqlite3.connect(Path(source))
        with conn as c:
            m = pd.read_sql("SELECT * FROM textnet_im", c, index_col="term")
            meta = pd.read_sql("SELECT * FROM textnet_meta", c, index_col="keys")[
                "values"
            ]
        connected = meta["connected"] == "1"
        doc_attrs = json.loads(meta["doc_attrs"])
        return cls(BiadjacencyMatrix(m.T), connected=connected, doc_attrs=doc_attrs)


    def plot(
        self,
        *,
        color_clusters: bool | ig.VertexClustering = False,
        show_clusters: bool | ig.VertexClustering = False,
        bipartite_layout: bool = False,
        sugiyama_layout: bool = False,
        circular_layout: bool = False,
        kamada_kawai_layout: bool = False,
        drl_layout: bool = False,
        node_opacity: float | None = None,
        edge_opacity: float | None = None,
        label_term_nodes: bool = False,
        label_doc_nodes: bool = False,
        label_nodes: bool = False,
        label_edges: bool = False,
        node_label_filter: Callable[[ig.Vertex], bool] | None = None,
        edge_label_filter: Callable[[ig.Edge], bool] | None = None,
        scale_nodes_by: str | None = None,
        scale_edges_by: str | None = None,
        **kwargs,
    ) -> ig.Plot:
        """
        Plot the bipartite graph.

        Parameters
        ----------
        color_clusters : bool or VertexClustering, optional
            Color nodes according to clusters detected by the Leiden algorithm
            (default: False). Alternately a clustering object generated by
            another community detection algorithm can be passed.
        show_clusters : bool or VertexClustering, optional
            Mark clusters detected by the Leiden algorithm (default: False).
            Alternately a clustering object generated by another community
            detection algorithm can be passed.
        bipartite_layout : bool, optional
            Use a bipartite graph layout (default: False; a
            weighted Fruchterman-Reingold layout is used unless
            another layout is specified).
        sugiyama_layout : bool, optional
            Use layered Sugiyama layout (default: False; a weighted
            Fruchterman-Reingold layout is used unless another layout is
            specified).
        circular_layout : bool, optional
            Use circular Reingold-Tilford layout (default: False; a weighted
            Fruchterman-Reingold layout is used unless another layout is
            specified).
        kamada_kawai_layout : bool, optional
            Use a layout created by the Kamada-Kawai algorithm (default: False;
            a weighted Fruchterman-Reingold layout is used unless another
            layout is specified).
        drl_layout : bool, optional
            Use the DrL layout, suitable for large networks (default: False;
            a weighted Fruchterman-Reingold layout is used unless another
            layout is specified).
        node_opacity : float, optional
            Opacity (between 0 and 1) to apply to nodes (default: no
            transparency).
        edge_opacity : float, optional
            Opacity (between 0 and 1) to apply to edges (default: no
            transparency).
        label_term_nodes : bool, optional
            Label term nodes (default: False).
        label_doc_nodes : bool, optional
            Label document nodes (default: False).
        label_nodes : bool, optional
            Label term and document nodes (default: False).
        label_edges : bool, optional
            Show edge weights in plot.
        node_label_filter : function, optional
            Function returning boolean value mapped to iterator of nodes to
            decide whether or not to suppress labels.
        edge_label_filter : function, optional
            Function returning boolean value mapped to iterator of edges to
            decide whether or not to suppress labels.
        scale_nodes_by : str, optional
            Name of centrality measure or node attribute to scale nodes by.
            Possible values: ``degree``, ``strength``, ``hits``, ``cohits``,
            ``birank`` or any node attribute (default: None).
        scale_edges_by : str, optional
                Name of edge attribute to scale edges by.
                Possible value: ``frequency_by_doc`` (default: None).

        Other Parameters
        ----------------
        target : str or file, optional
            File or path that the plot should be saved to (e.g., ``plot.png``).
        kwargs
            Additional arguments to pass to `igraph.drawing.plot`.

        Returns
        -------
        `igraph.drawing.Plot`
            The plot can be directly displayed in a Jupyter notebook or saved
            as an image file.
        """
        args = locals()
        del args["self"], args["kwargs"]
        kwargs.update(args)
        return self._plot(**kwargs)
    
    def to_networkx(self) -> nx.Graph:
        """
        将igraph图转换为networkx图
        """
        g = self.graph
        B = nx.Graph()

        # 添加节点
        for v in g.vs:
            B.add_node(v["id"], bipartite=v["type"])

        # 添加边
        for e in g.es:
            source = g.vs[e.source]["id"]
            target = g.vs[e.target]["id"]
            B.add_edge(source, target, weight=e["weight"])

        return B

    def plot_with_matplotlib(self, font_path: str, label_nodes: bool = False, label_font_size: int = 12, 
                            label_bbox: dict = None, scale_node_size_by_frequency: bool = False):
        """
        使用matplotlib绘制双模式网络图，并正确显示中文字体。

        Parameters
        ----------
        font_path : str
            指定中文字体路径。
        label_nodes : bool, optional
            是否显示节点标签，默认不显示。
        label_font_size : int, optional
            节点标签的字体大小，默认值为12。
        label_bbox : dict, optional
            节点标签的边框属性，默认值为None。
        scale_node_size_by_frequency : bool, optional
            是否根据词频调整节点大小，默认值为False。
        """
        # 设置中文字体
        font_prop = FontProperties(fname=font_path)
        plt.rcParams['font.sans-serif'] = [font_prop.get_name()]  # 用于显示中文
        plt.rcParams['axes.unicode_minus'] = False  # 用于显示负号

        # 将igraph图转换为networkx图
        B = self.to_networkx()

        # 获取节点位置
        pos = nx.spring_layout(B)
        plt.figure(figsize=(12, 8))

        # 获取文档和词汇节点
        docs = [n for n, d in B.nodes(data=True) if d['bipartite'] == 0]
        terms = [n for n, d in B.nodes(data=True) if d['bipartite'] == 1]

        # 计算节点大小
        if scale_node_size_by_frequency:
            term_sizes = [B.nodes[n]['frequency'] * 10 for n in terms]
        else:
            term_sizes = [100 for _ in terms]

        # 绘制节点
        nx.draw_networkx_nodes(B, pos, nodelist=terms, node_color='r', node_shape='o', node_size=term_sizes, alpha=0.8, label='Terms')
        nx.draw_networkx_nodes(B, pos, nodelist=docs, node_color='b', node_shape='s', node_size=100, alpha=0.8, label='Docs')

        # 绘制边
        edge_widths = [B[u][v]['weight'] for u, v in B.edges()]
        nx.draw_networkx_edges(B, pos, width=edge_widths, alpha=0.5)

        # 绘制标签
        if label_nodes:
            for node, (x, y) in pos.items():
                plt.text(x, y, node, fontsize=label_font_size, fontproperties=font_prop, ha='center', va='center', bbox=label_bbox)

        # 添加图例
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='Terms', markerfacecolor='r', markersize=10),
            Line2D([0], [0], marker='s', color='w', label='Docs', markerfacecolor='b', markersize=10)
        ]
        plt.legend(handles=legend_elements, scatterpoints=1)

        plt.axis('off')
        plt.show()



    @cached_property
    def hits(self) -> pd.Series:
        """HITS rank of nodes."""
        return bipartite_rank(self, normalizer="HITS")

    @cached_property
    def cohits(self) -> pd.Series:
        """CoHITS rank of nodes."""
        return bipartite_rank(self, normalizer="CoHITS")

    @cached_property
    def birank(self) -> pd.Series:
        """BiRank of nodes."""
        return bipartite_rank(self, normalizer="BiRank")

    @cached_property
    def bipartite_cc(self) -> pd.Series:
        """
        Calculate the unweighted bipartite clustering coefficient.

        Returns
        -------
        `pandas.Series`
            The clustering cofficients indexed by node label.

        Notes
        -----
        Adapted from the ``networkx`` implementation.

        References
        ----------
        :cite:`Latapy2008`
        """
        ccs = []
        for node in self.nodes:
            cc: float = 0
            fon = set(node.neighbors())
            son = {nn for nbr in fon for nn in nbr.neighbors()} - {node}
            for nn in son:
                nnn = set(nn.neighbors())
                cc += len(nnn & fon) / len(nnn | fon)
            if cc > 0:
                cc /= len(son)
            ccs.append(cc)
        return pd.Series(ccs, index=self.nodes["id"])

    def _partition_graph(self, resolution: float, seed: int) -> ig.VertexClustering:
        part, part0, part1 = la.CPMVertexPartition.Bipartite(
            self.graph, resolution_parameter_01=resolution, weights="weight"
        )
        opt = la.Optimiser()
        opt.set_rng_seed(seed)
        opt.optimise_partition_multiplex(
            [part, part0, part1], layer_weights=[1, -1, -1], n_iterations=-1
        )
        return part
    
    def to_dataframe(self):
        """将 Textnet 对象转换为 Pandas DataFrame。"""
        # 创建一个空的 DataFrame 来存储节点的属性
        g = self.graph  # 获取图对象
        attributes = g.vs.attributes()  # 获取所有属性名
        
        # 创建一个空的DataFrame来存储节点的属性
        df = pd.DataFrame({attr: g.vs[attr] for attr in attributes})
        
        return df

    def to_adjacency_dataframe(self):
        """将 Textnet 对象的双模邻接矩阵转换为 Pandas DataFrame。"""
        adjacency_matrix = self._matrix.to_array()
        df = pd.DataFrame(adjacency_matrix, index=self._matrix.index, columns=self._matrix.columns)
        return df

    def print_node_attributes(self):
            """打印图中所有节点的属性。"""
            for node in self.graph.vs:
                print(f"id：{node['id']}，type：{node['type']}")
                if node["type"] == "term":
                    print(f"total_frequency：{node['total_frequency']}")
                    print(f"frequency_by_doc：{node['frequency_by_doc']}")
                    print(f"tfidf_by_doc：{node['tfidf_by_doc']}")
                print()


for prop, desc in [
    ("hits", "HITS rank"),
    ("cohits", "CoHITS rank"),
    ("birank", "BiRank"),
    ("bipartite_cc", "bipartite clustering coefficient"),
]:
    setattr(Textnet, f"top_{prop}", _make_top(prop, desc))


class ProjectedTextnet(TextnetBase):
    """
    One-mode projection of a textnet.

    Created by calling `Textnet.project()` with the desired ``node_type``.

    Attributes
    ----------
    graph : `igraph.Graph`
        Direct access to the igraph object.
    """

    @cached_property
    def m(self) -> pd.DataFrame:
        """Weighted adjacency matrix."""
        d = self.graph.get_adjacency().data
        m = np.array(d, dtype=float)
        for i in self.edges.indices:
            edge = self.graph.es[i]
            source, target = edge.source, edge.target
            m[source, target] = m[target, source] = edge["weight"]
        return pd.DataFrame(m, index=self.nodes["id"], columns=self.nodes["id"])

    @cached_property
    def betweenness(self) -> pd.Series:
        """Weighted betweenness centrality."""
        return pd.Series(self.graph.betweenness(weights="cost"), index=self.nodes["id"])

    @cached_property
    def closeness(self) -> pd.Series:
        """Weighted closeness centrality."""
        return pd.Series(self.graph.closeness(weights="cost"), index=self.nodes["id"])

    @cached_property
    def harmonic(self) -> pd.Series:
        """Weighted harmonic centrality."""
        return pd.Series(
            self.graph.harmonic_centrality(weights="cost"), index=self.nodes["id"]
        )

    @cached_property
    def eigenvector_centrality(self) -> pd.Series:
        """Weighted eigenvector centrality."""
        return pd.Series(
            self.graph.eigenvector_centrality(weights="weight"), index=self.nodes["id"]
        )

    @cached_property
    def pagerank(self) -> pd.Series:
        """Weighted PageRank centrality."""
        return pd.Series(self.graph.pagerank(weights="weight"), index=self.nodes["id"])

    @property
    def spanning(self) -> pd.Series:
        """Textual spanning measure."""
        if any(self.node_types):
            warn("Textual spanning is only defined for document nodes.")
        if not self.graph.is_connected():
            warn(
                "Graph is disconnected. "
                "The textual spanning measure is not effective on disconnected graphs."
            )
        a = self.m.to_numpy()
        return pd.Series(textual_spanning(a), index=self.nodes["id"])

    def alpha_cut(self, alpha: float) -> ProjectedTextnet:
        """
        Return graph backbone.

        Parameters
        ----------
        alpha : float
            Threshold for edge elimination. Must be between 0 and 1. Edges with
            an alpha value above the specified threshold are removed.

        Returns
        -------
        `ProjectedTextnet`
            New textnet sans pruned edges.
        """
        if "alpha" not in self.graph.vertex_attributes():
            self.graph.es["alpha"] = list(disparity_filter(self.graph))
        pruned = self.graph.copy()
        pruned.delete_edges(pruned.es.select(alpha_ge=alpha))
        return ProjectedTextnet(giant_component(pruned))

    def plot(self, *, alpha: float | None = None, **kwargs) -> ig.Plot:
        """
        Plot the projected graph.

        Parameters
        ----------
        alpha : float, optional
            Threshold for edge elimination. Must be between 0 and 1. Edges with
            an alpha value above the specified threshold are removed. This is
            useful when plotting "hairball" graphs.
        scale_nodes_by : str, optional
            Name of centrality measure or node attribute to scale nodes by.
            Possible values: ``degree``, ``strength``, ``betweenness``,
            ``closeness``, ``eigenvector_centrality``, ``pagerank`` or any node
            attribute (default: None).

        Returns
        -------
        `igraph.drawing.Plot`
            The plot can be directly displayed in a Jupyter notebook or saved
            as an image file.

        Other Parameters
        ----------------
        target : str or file, optional
            File or path that the plot should be saved to (e.g., ``plot.png``).
        kwargs
            Additional arguments to pass to `igraph.drawing.plot`.
        """
        if alpha is not None:
            to_plot = self.alpha_cut(alpha)
        else:
            to_plot = self
        return to_plot._plot(**kwargs)

    def _partition_graph(self, resolution: float, seed: int) -> ig.VertexClustering:
        part = la.find_partition(
            self.graph,
            la.ModularityVertexPartition,
            weights="weight",
            n_iterations=-1,
            seed=seed,
        )
        return part

    def to_dataframe(self):
            """将 ProjectedTextnet 对象转换为 Pandas DataFrame。"""
            # 创建一个空的 DataFrame 来存储节点的属性
            g = self.graph  # 获取图对象
            attributes = g.vs.attributes()  # 获取所有属性名
            
            # 创建一个空的DataFrame来存储节点的属性
            df = pd.DataFrame({attr: g.vs[attr] for attr in attributes})
            
            return df

    def to_adjacency_dataframe(self):
        """将 ProjectedTextnet 对象的邻接矩阵转换为 Pandas DataFrame。"""
        adjacency_matrix = self.graph.get_adjacency(attribute='weight').data
        df = pd.DataFrame(adjacency_matrix, index=self.graph.vs['id'], columns=self.graph.vs['id'])
        return df

    def doc_co_occurrence_dataframe(self):
        """将文档之间的共现词数量转换为 Pandas DataFrame。"""
        doc_ids = [v["id"] for v in self.graph.vs if v["type"] == "doc"]
        data = []

        for doc1 in doc_ids:
            row = []
            for doc2 in doc_ids:
                co_occurrence = self.graph.vs.find(id=doc1)["doc_co_occurrence"].get(doc2, {"count": 0, "terms": []})
                row.append(f"{co_occurrence['count']} ({', '.join(co_occurrence['terms'])})")
            data.append(row)

        return pd.DataFrame(data, index=doc_ids, columns=doc_ids)

for prop, desc in [
    ("betweenness", "betweenness"),
    ("closeness", "closeness"),
    ("harmonic", "harmonic centrality"),
    ("pagerank", "PageRank centrality"),
    ("eigenvector_centrality", "eigenvector centrality"),
    ("spanning", "textual spanning"),
]:
    setattr(ProjectedTextnet, f"top_{prop}", _make_top(prop, desc))

ProjectedTextnet.top_ev = ProjectedTextnet.top_eigenvector_centrality  # type: ignore


def _matrix_from_tidy_text(
    tidy_text: TidyText | pd.DataFrame, min_docs: int, min_term_frequency: int, top_n_terms: int = None
) -> BiadjacencyMatrix:
    term_counts = tidy_text["term"].value_counts()
    frequent_terms = term_counts[term_counts >= min_term_frequency].index
    tidy_text = tidy_text[tidy_text["term"].isin(frequent_terms)]
    
    # 统计每个词汇的总词频
    term_total_freq = tidy_text.groupby("term")["n"].sum()

    # 根据 min_docs 过滤词汇
    term_doc_counts = tidy_text.groupby("term").count()["n"]
    terms_to_keep = term_doc_counts[term_doc_counts >= min_docs].index
    tidy_text = tidy_text[tidy_text["term"].isin(terms_to_keep)]

    # 创建词汇在不同文档中的词频和 tf-idf 值的字典
    term_freq_dict = {term: group["n"].to_dict() for term, group in tidy_text.groupby("term")}
    term_tfidf_dict = {term: group["term_weight"].to_dict() for term, group in tidy_text.groupby("term")}

    # 新增：过滤每个主题下词频最高的 n 个词，按照tf-idf排序
    if top_n_terms:
        top_terms_per_label = tidy_text.groupby('label', group_keys=False).apply(
            lambda x: x.nlargest(top_n_terms, ['n', 'term_weight'])
        ).reset_index()
        tidy_text = top_terms_per_label.set_index('label')

    # 使用词频计数而不是权重
    m = (
        tidy_text.groupby(["label", "term"])
        .size()
        .unstack(fill_value=0)
    )
    
    biadjacency_matrix = BiadjacencyMatrix(m.astype("float64"))
    biadjacency_matrix._data = m  # 添加此行

    # 将总词频和词汇在不同文档中的词频和 tf-idf 值存储为 BiadjacencyMatrix 的属性
    biadjacency_matrix.term_total_freq = term_total_freq
    biadjacency_matrix.term_freq_dict = term_freq_dict
    biadjacency_matrix.term_tfidf_dict = term_tfidf_dict

    # 打印调试信息
    print("term_freq_dict:", term_freq_dict)
    # print("term_tfidf_dict:", term_tfidf_dict)

    return biadjacency_matrix



def _graph_from_matrix(m: BiadjacencyMatrix) -> ig.Graph:
    g = ig.Graph.Biadjacency(m.to_numpy().tolist(), directed=False, weighted=True)
    g.vs["id"] = np.append(m.index, m.columns).tolist()
    g.es["cost"] = [1 / pow(w, tn.params["tuning_parameter"]) for w in g.es["weight"]]
    g.vs["type"] = ["term" if t else "doc" for t in g.vs["type"]]
    return g


def disparity_filter(graph: ig.Graph) -> Iterator[float]:
    """
    Compute significance scores of edge weights.

    Parameters
    ----------
    graph : Graph
        The one-mode graph to compute the significance scores for.

    Yields
    ------
    float
        Iterator of significance scores.

    Notes
    -----
    Provided the package was installed properly, a compiled extension will be
    used for a significant speedup.

    References
    ----------
    :cite:`Serrano2009`
    """
    tqdm_args = dict(disable=not tn.params["progress_bar"] or None, unit="edges")
    cores = cpu_count() or 1
    if cores > 1 and len(graph.es) >= cores:
        sig_ufunc = np.frompyfunc(_edge_significance, 1, 1)
        edge_chunks = np.array_split(pd.Series(list(graph.es)), cores)
        yield from pd.concat(thread_map(sig_ufunc, edge_chunks, **tqdm_args))
    else:
        for edge in tqdm(graph.es, **tqdm_args):
            yield _edge_significance(edge)


def _edge_significance(edge: ig.Edge) -> float:
    source, target = edge.vertex_tuple
    degree_t = target.degree()
    degree_s = source.degree()
    sum_weights_s = source.strength(weights="weight")
    norm_weight_s = edge["weight"] / sum_weights_s
    sum_weights_t = target.strength(weights="weight")
    norm_weight_t = edge["weight"] / sum_weights_t
    integral_s = _disparity_filter_integral(norm_weight_s, degree_s)
    integral_t = _disparity_filter_integral(norm_weight_t, degree_t)
    return min(1 - (degree_s - 1) * integral_s[0], 1 - (degree_t - 1) * integral_t[0])


@memoize
def _disparity_filter_integral(norm_weight: float, degree: int) -> float:
    return quad(integrand, 0, norm_weight, args=(degree))


def giant_component(g: ig.Graph) -> ig.Graph:
    """
    Return the subgraph corresponding to the giant component.

    Parameters
    ----------
    `igraph.Graph`
        The (possibly) disconnected graph.

    Returns
    -------
    `igraph.Graph`
        The graph consisting of just the largest connected component.
    """
    size = max(g.components().sizes())
    pos = g.components().sizes().index(size)
    return g.subgraph(g.components()[pos])


def bipartite_rank(
    net: Textnet,
    normalizer: Literal["HITS", "CoHITS", "BGRM", "BiRank"],
    alpha: float = 0.85,
    beta: float = 0.85,
    max_iter: int = -1,
    tolerance: float = 1.0e-4,
) -> pd.Series:
    """
    Calculate centralities of nodes in the bipartite network.

    Parameters
    ----------
    normalizer : string
        The normalizer to use: ``HITS``, ``CoHITS``, ``BGRM``, or ``BiRank``.
        See reference for details.
    alpha : float, optional
        Damping factor for the rows and columns.
    beta : float, optional
        Damping factor for the rows and columns.
    max_iter : int, optional
        Maximum number of iterations to run before reaching convergence
        (default: -1, meaning iterate until the errors are within the
        specified tolerance).
    tolerance : float, optional
        Error tolerance when checking for convergence.

    Raises
    ------
    ValueError
        If an invalid normalizer is specified.

    Returns
    -------
    `pandas.Series`
        The BiRank for both sets of nodes indexed by node label.

    Notes
    -----
    Adapted from the implementation by :cite:t:`Yang2020`.

    References
    ----------
    :cite:`He2017`
    """
    if normalizer not in ("HITS", "CoHITS", "BGRM", "BiRank"):
        raise ValueError(f"'{normalizer}' is not a valid normalization option.")

    W = net.m.to_numpy()
    Kd = np.array(W.sum(axis=1)).flatten()
    Kp = np.array(W.T.sum(axis=1)).flatten()

    Kd[np.where(Kd == 0)] += 1
    Kp[np.where(Kp == 0)] += 1

    Kd_ = np.diagflat(1 / Kd)
    Kp_ = np.diagflat(1 / Kp)

    if normalizer == "HITS":
        Sp = W.T
        Sd = W
    elif normalizer == "CoHITS":
        Sp = W.T.dot(Kd_)
        Sd = W.dot(Kp_)
    elif normalizer == "BGRM":
        Sp = Kp_.dot(W.T).dot(Kd_)
        Sd = Sp.T
    elif normalizer == "BiRank":
        Kd_bi = np.diagflat(1 / np.sqrt(Kd))
        Kp_bi = np.diagflat(1 / np.sqrt(Kp))
        Sp = Kp_bi.dot(W.T).dot(Kd_bi)
        Sd = Sp.T

    p0 = np.repeat(1 / Kp_.shape[0], Kp_.shape[0])
    p_last = p0.copy()
    d0 = np.repeat(1 / Kd_.shape[0], Kd_.shape[0])
    d_last = d0.copy()

    iter_count = 0
    continue_iter = iter_count < max_iter or max_iter < 0

    while continue_iter:
        p = alpha * (Sp.dot(d_last)) + (1 - alpha) * p0
        d = beta * (Sd.dot(p_last)) + (1 - beta) * d0

        if normalizer == "HITS":
            p = p / p.sum()
            d = d / d.sum()

        err_p = np.absolute(p - p_last).sum()
        err_d = np.absolute(d - d_last).sum()

        iter_count += 1
        if err_p < tolerance and err_d < tolerance:
            continue_iter = False
        else:
            continue_iter = iter_count < max_iter or max_iter < 0

        p_last = p
        d_last = d

    return pd.Series(np.append(d, p), index=net.nodes["id"])


def textual_spanning(m: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """
    Calculate textual spanning of documents in a similarity matrix.

    Parameters
    ----------
    m : array
        Document similarity matrix.
    alpha : float, optional
        Tuning parameter to change relevance of edge weights vs. document
        counts in calculating the spanning score (default: 1.0).

    Returns
    -------
    `numpy.array`
        Each node's textual spanning measure.

    Notes
    -----
    The textual spanning measure is not effective on disconnected graphs.

    References
    ----------
    :cite:`Stoltz2019`
    """
    np.fill_diagonal(m, 0)
    k = np.sum(m != 0, axis=1)
    den = k * (m.sum(axis=1) / k**alpha)
    ps = m.T / den[:, np.newaxis]
    eps = np.zeros_like(ps)
    eps[ps > 0] = ps[ps > 0] ** -1
    ps2 = eps @ ps
    sp = (ps + ps2) ** 2
    csp = sp.sum(axis=1)
    csp_norm = ((csp - csp.mean()) / csp.std(ddof=1)) * -1
    return csp_norm


class BiadjacencyMatrix(LiteFrame):
    """Matrix relating documents to terms."""

    def __init__(self, data):
        super().__init__(data)
        self._data = data

    def to_dataframe(self):
        return self._data

    @property
    def shape(self):
        return self._data.shape

    @property
    def columns(self):
        return self._data.columns

    @property
    def index(self):
        return self._data.index

    def applymap(self, func):
        return self._data.applymap(func)

