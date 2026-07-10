"""
Loads the Phase 2 `artifacts/phase2/graph_weight_maps.joblib` bundle ONCE at
server startup.

IMPORTANT -- Phase 2 no longer builds or exports a `networkx.DiGraph` at all
(the current Phase 2 notebook runs its causal graph forensics entirely on the
C-backed `igraph` engine, and only ever persists a plain dict of *derived*
per-account metrics + community assignments -- there is no `transaction_graph.joblib`
anywhere in the real artifact tree). This service is rewritten around what
Phase 2 actually ships in `graph_weight_maps.joblib`:

    {
        "account_graph_metrics": pd.DataFrame indexed by account, columns
            [graph_out_degree, graph_pagerank, graph_hub_score,
             graph_authority_score, community_id],
        "community_sizes": {community_id: size},
        "modularity_score": float,
        "n_communities": int,
        "leiden_resolution": float,
        "trust_weights": {...},
        "risk_weights": {...},
    }

This is a FROZEN, static snapshot (built train-only, causally, offline) --
Phase 4 never mutates it. Accounts unseen in that snapshot (cold-start: only
ever observed in the validation tail or test partition, or brand-new since)
resolve to the same 0.0 / community_id -1 convention Phase 2 itself uses.

Live-serving still needs *some* incremental, in-memory bookkeeping for
freshly-observed transactions between requests -- but since there is no live
graph object to append an edge to anymore, this service instead keeps a
lightweight per-account transaction/degree counter of its own, entirely
separate from the frozen Phase 2 snapshot, periodically persisted to Phase
4's own `RUNTIME_STATE_DIR` (never written back into the Drive artifact tree).
"""
import os
import logging
import threading
from typing import Dict, Optional

import joblib
import numpy as np
import pandas as pd

from app.core.config import settings

logger = logging.getLogger("graph_service")

GRAPH_METRIC_COLS = ["graph_out_degree", "graph_pagerank", "graph_hub_score", "graph_authority_score"]


class GraphService:
    def __init__(self) -> None:
        self.account_graph_metrics: Optional[pd.DataFrame] = None
        self.community_sizes: Dict[int, int] = {}
        self.modularity_score: float = float("nan")
        self.n_communities: int = 0
        self.trust_weights: Dict[str, float] = {}
        self.risk_weights: Dict[str, float] = {}
        self._metric_min: Dict[str, float] = {}
        self._metric_max: Dict[str, float] = {}

        # Phase-4-only, in-memory live bookkeeping (never part of the frozen
        # Phase 2 export). Maps account -> {"out_edges": {receiver: {"weight":.., "tx_count":..}}}
        self._live_edges: Dict[str, Dict[str, Dict[str, float]]] = {}
        self._lock = threading.Lock()
        self._edges_since_persist = 0
        self.loaded = False

    # --- path helpers --------------------------------------------------------
    def _phase2_path(self) -> str:
        return os.path.join(settings.PHASE2_ARTIFACTS_DIR, settings.PHASE2_GRAPH_WEIGHT_MAPS_FILE)

    def _live_state_path(self) -> str:
        return os.path.join(settings.RUNTIME_STATE_DIR, settings.LIVE_GRAPH_STATE_FILE)

    def load(self) -> None:
        path = self._phase2_path()
        if os.path.isfile(path):
            weight_maps = joblib.load(path)
            self.account_graph_metrics = weight_maps.get("account_graph_metrics")
            self.community_sizes = weight_maps.get("community_sizes", {})
            self.modularity_score = float(weight_maps.get("modularity_score", float("nan")))
            self.n_communities = int(weight_maps.get("n_communities", 0))
            self.trust_weights = weight_maps.get("trust_weights", {})
            self.risk_weights = weight_maps.get("risk_weights", {})
            n_accounts = len(self.account_graph_metrics) if self.account_graph_metrics is not None else 0
            logger.info(
                "Loaded frozen Phase 2 graph_weight_maps.joblib: %d accounts, %d communities, "
                "modularity=%.4f.", n_accounts, self.n_communities, self.modularity_score,
            )
            self._fit_metric_bounds()
        else:
            self.account_graph_metrics = pd.DataFrame(columns=GRAPH_METRIC_COLS + ["community_id"])
            logger.warning(
                "No graph_weight_maps.joblib found at %s -- serving with an empty frozen graph "
                "snapshot (every account will resolve as cold-start).", path,
            )

        self._load_live_state()
        self.loaded = True

    def _fit_metric_bounds(self) -> None:
        """Per-column min/max across the frozen account snapshot -- used to
        min-max scale the composite `graph_risk` fusion signal. Phase 3's exact
        train-fit-slice bounds for this scaler are not persisted anywhere in
        `phase3_metadata_registry.json`, so this is a documented, closest-available
        approximation computed once, at load time, off the same frozen accounts."""
        if self.account_graph_metrics is None or self.account_graph_metrics.empty:
            return
        for col in GRAPH_METRIC_COLS:
            if col in self.account_graph_metrics.columns:
                self._metric_min[col] = float(self.account_graph_metrics[col].min())
                self._metric_max[col] = float(self.account_graph_metrics[col].max())

    def _load_live_state(self) -> None:
        path = self._live_state_path()
        if os.path.isfile(path):
            try:
                self._live_edges = joblib.load(path)
                logger.info("Restored live (Phase-4-only) graph bookkeeping from %s.", path)
            except Exception as exc:
                logger.warning("Failed to restore live graph state from %s (%s) -- starting fresh.", path, exc)
                self._live_edges = {}

    def persist(self) -> None:
        """Persists ONLY the Phase-4-owned live bookkeeping to RUNTIME_STATE_DIR.
        The frozen Phase 2 `graph_weight_maps.joblib` artifact is never
        overwritten, moved, or renamed."""
        joblib.dump(self._live_edges, self._live_state_path())
        self._edges_since_persist = 0
        logger.info("Persisted live graph bookkeeping (%d accounts tracked).", len(self._live_edges))

    def add_edge_incremental(self, sender: str, receiver: str, amount: float) -> None:
        """Append/update a single weighted edge in the Phase-4-only live
        tracker. O(1) -- never touches the frozen Phase 2 snapshot."""
        with self._lock:
            bucket = self._live_edges.setdefault(sender, {})
            edge = bucket.setdefault(receiver, {"weight": 0.0, "tx_count": 0})
            edge["weight"] += amount
            edge["tx_count"] += 1
            self._edges_since_persist += 1
            if self._edges_since_persist >= settings.GRAPH_PERSIST_EVERY_N_EDGES:
                self.persist()

    def _static_metrics_for(self, account_id: str) -> Dict[str, float]:
        """Frozen, structural signals for one account straight out of Phase 2's
        `account_graph_metrics`. Cold-start accounts (never a node in the
        pruned, leakage-free train-only graph) resolve to 0.0 / community_id -1,
        the same convention Phase 2 itself uses for unseen accounts."""
        if self.account_graph_metrics is None or account_id not in self.account_graph_metrics.index:
            return {
                "graph_out_degree": 0.0, "graph_pagerank": 0.0,
                "graph_hub_score": 0.0, "graph_authority_score": 0.0, "community_id": -1,
            }
        row = self.account_graph_metrics.loc[account_id]
        return {
            "graph_out_degree": float(row.get("graph_out_degree", 0.0)),
            "graph_pagerank": float(row.get("graph_pagerank", 0.0)),
            "graph_hub_score": float(row.get("graph_hub_score", 0.0)),
            "graph_authority_score": float(row.get("graph_authority_score", 0.0)),
            "community_id": int(row.get("community_id", -1)),
        }

    def _community_size(self, community_id: int) -> int:
        if community_id == -1:
            return 0
        return int(self.community_sizes.get(community_id, 0))

    def _minmax(self, col: str, value: float) -> float:
        lo = self._metric_min.get(col)
        hi = self._metric_max.get(col)
        if lo is None or hi is None or hi <= lo:
            return 0.0
        return float(np.clip((value - lo) / (hi - lo), 0.0, 1.0))

    def account_risk_snapshot(self, sender: str, receiver: str) -> Dict[str, float]:
        """Cheap, per-transaction structural signals for a (sender, receiver)
        pair, using the real Phase 2 feature-naming contract
        (`sender_graph_*` / `receiver_graph_*` / `*_community_*` /
        `is_bridge_transaction`) instead of the old, nonexistent
        `sender_graph_risk_score` / `sender_pagerank` / `sender_betweenness` /
        `sender_is_bridge_account` names. Also returns a composite `graph_risk`
        in [0, 1] -- the same signal name `phase3_metadata_registry.json`'s
        `fusion_weights` expects -- built the same way Phase 3 Block 7 derives
        it (mean of min-max-scaled sender/receiver graph metrics, nudged up on
        a bridge transaction)."""
        sender_metrics = self._static_metrics_for(sender)
        receiver_metrics = self._static_metrics_for(receiver)

        sender_community_size = self._community_size(sender_metrics["community_id"])
        receiver_community_size = self._community_size(receiver_metrics["community_id"])
        min_active = 1  # matches Phase 2's MIN_COMMUNITY_ACTIVE_SIZE floor for a single live check
        sender_active = sender_metrics["community_id"] != -1 and sender_community_size >= min_active
        receiver_active = receiver_metrics["community_id"] != -1 and receiver_community_size >= min_active
        different_communities = sender_metrics["community_id"] != receiver_metrics["community_id"]
        is_bridge_transaction = int(sender_active and receiver_active and different_communities)

        scaled_vals = []
        for prefix, metrics in (("sender", sender_metrics), ("receiver", receiver_metrics)):
            for col in GRAPH_METRIC_COLS:
                scaled_vals.append(self._minmax(col, metrics[col]))
        composite = float(np.mean(scaled_vals)) if scaled_vals else 0.0
        graph_risk = float(np.clip(0.85 * composite + 0.15 * is_bridge_transaction, 0.0, 1.0))

        return {
            "sender_graph_out_degree": sender_metrics["graph_out_degree"],
            "sender_graph_pagerank": sender_metrics["graph_pagerank"],
            "sender_graph_hub_score": sender_metrics["graph_hub_score"],
            "sender_graph_authority_score": sender_metrics["graph_authority_score"],
            "sender_community_id": float(sender_metrics["community_id"]),
            "sender_community_size": float(sender_community_size),
            "receiver_graph_out_degree": receiver_metrics["graph_out_degree"],
            "receiver_graph_pagerank": receiver_metrics["graph_pagerank"],
            "receiver_graph_hub_score": receiver_metrics["graph_hub_score"],
            "receiver_graph_authority_score": receiver_metrics["graph_authority_score"],
            "receiver_community_id": float(receiver_metrics["community_id"]),
            "receiver_community_size": float(receiver_community_size),
            "is_bridge_transaction": float(is_bridge_transaction),
            "graph_risk": graph_risk,
        }


graph_service = GraphService()
