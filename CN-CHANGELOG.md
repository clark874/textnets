# textnets-cn 0.10.5+cn.1

First Chinese-localization edition on official `v0.10.5`.

## Default behavior

Official `0.10.5` behavior is unchanged unless a Chinese-localization argument is passed. In particular, bipartite edges still use `term_weight`. The 2024 site-packages graph, which rebuilt edges from occurrence counts, is available only as `Textnet(..., edge_weight="count")`.

## Added

- `Corpus.custom_tokenized()`: jieba segmentation with an optional custom dictionary
- `TidyText` record-list constructor and `to_dataframe()`
- `Textnet(min_term_frequency=..., top_n_terms=..., edge_weight=...)`
- term-node attributes: `total_frequency`, `frequency_by_doc`, `tfidf_by_doc`
- `Textnet.to_dataframe()`, `to_adjacency_dataframe()`, `print_node_attributes()`, `to_networkx()`
- document-projection `doc_co_occurrence` attributes and `doc_co_occurrence_dataframe()`

## Not ported

- `viz.py` debug prints
- `plot_with_matplotlib` (notebook used its own matplotlib wrappers; official plot already accepts CJK fonts via the `cjk` extra)

## Evidence

See `docs/cn-audit/inventory.json` and `docs/cn-audit/baselines.md`.
