import os
import logging
import threading
import gc
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError

from app.core.config import settings

logger = logging.getLogger("graph_service")

GRAPH_METRIC_COLS = ["graph_out_degree", "graph_pagerank", "graph_hub_score", "graph_authority_score"]


class GraphService:
    def __init__(self) -> None:
        self._account_metrics: Dict[str, Tuple[float, float, float, float, int]] = {}
        self.community_sizes: Dict[int, int] = {}
        self.modularity_score: float = float("nan")
        self.n_communities: int = 0
        self.trust_weights: Dict[str, float] = {}
        self.risk_weights: Dict[str, float] = {}
        self._metric_min: Dict[str, float] = {}
        self._metric_max: Dict[str, float] = {}

        self._live_edges: Dict[str, Dict[str, Dict[str, float]]] = {}
        self._lock = threading.Lock()
        self._edges_since_persist = 0
        
        # Flags whether the remote artifacts have been loaded or skipped due to RAM constraints
        self.loaded = False
        self.oom_fallback_active = False

        self.repo_id = "ff49/financialfraudmodel"
        self.token = os.getenv("HF_TOKEN")

    def _live_state_path(self) -> str:
        return os.path.join(settings.RUNTIME_STATE_DIR, settings.LIVE_GRAPH_STATE_FILE)

    def _fetch_phase2_artifact(self) -> Optional[str]:
        repo_filepath = os.path.join(
            settings.PHASE2_ARTIFACTS_SUBDIR, settings.PHASE2_GRAPH_WEIGHT_MAPS_FILE
        ).replace(os.sep, "/")
        try:
            return hf_hub_download(repo_id=self.repo_id, filename=repo_filepath, token=self.token)
        except (EntryNotFoundError, Exception) as exc:
            logger.warning("Could not fetch %s from HF Hub (%s).", repo_filepath, exc)
            return None

    def load(self) -> None:
        """Attempts to load the structural graph metrics from Hugging Face Hub dynamically."""
        if self.loaded:
            return

        with self._lock:
            if self.loaded:
                return
                
            try:
                logger.info("Lazy-loading Graph Service payload from HF Hub...")
                path = self._fetch_phase2_artifact()
                if path is not None:
                    weight_maps = joblib.load(path)
                    df = weight_maps.get("account_graph_metrics")
                    self.community_sizes = weight_maps.get("community_sizes", {})
                    self.modularity_score = float(weight_maps.get("modularity_score", float("nan")))
                    self.n_communities = int(weight_maps.get("n_communities", 0))
                    self.trust_weights = weight_maps.get("trust_weights", {})
                    self.risk_weights = weight_maps.get("risk_weights", {})

                    if df is not None and not df.empty:
                        self._fit_metric_bounds(df)
                        for col in GRAPH_METRIC_COLS:
                            if col not in df.columns:
                                df[col] = 0.0
                        if "community_id" not in df.columns:
                            df["community_id"] = -1
                            
                        for account_id, row in zip(
                            df.index,
                            df[GRAPH_METRIC_COLS + ["community_id"]].itertuples(index=False, name=None),
                        ):
                            self._account_metrics[str(account_id)] = (
                                float(row[0]), float(row[1]), float(row[2]), float(row[3]), int(row[4]),
                            )
                        n_accounts = len(self._account_metrics)
                        
                        # Clear memory allocations immediately
                        del df, weight_maps
                        gc.collect()
                        
                        logger.info(
                            "Successfully loaded frozen Phase 2 graph metrics: %d accounts.", n_accounts
                        )
                    else:
                        logger.warning("graph_weight_maps.joblib was empty.")
                else:
                    logger.warning("Artifact download returned None.")
                    
            except (MemoryError, Exception) as exc:
                # CRITICAL RESILIENCE RAILS: If Render hits memory limits during initialization,
                # catch the failure, clear out any partial variables, and drop back to safe execution mode.
                logger.error("RAM headroom exceeded during graph parsing (%s). Activating safe cold-start fallback.", exc)
                self._account_metrics.clear()
                self.community_sizes.clear()
                self.oom_fallback_active = True
                gc.collect()

            self._load_live_state()
            self.loaded = True

    def _fit_metric_bounds(self, df: pd.DataFrame) -> None:
        for col in GRAPH_METRIC_COLS:
            if col in df.columns:
                self._metric_min[col] = float(df[col].min())
                self._metric_max[col] = float(df[col].max())

    def _load_live_state(self) -> None:
        path = self._live_state_path()
        if os.path.isfile(path):
            try:
                self._live_edges = joblib.load(path)
                logger.info("Restored live graph bookkeeping from %s.", path)
            except Exception as exc:
                logger.warning("Failed to restore live graph state (%s) -- starting fresh.", exc)
                self._live_edges = {}

    def persist(self) -> None:
        joblib.dump(self._live_edges, self._live_state_path())
        self._edges_since_persist = 0
        logger.info("Persisted live graph bookkeeping (%d accounts tracked).", len(self._live_edges))

    def add_edge_incremental(self, sender: str, receiver: str, amount: float) -> None:
        with self._lock:
            bucket = self._live_edges.setdefault(sender, {})
            edge = bucket.setdefault(receiver, {"weight": 0.0, "tx_count": 0})
            edge["weight"] += amount
            edge["tx_count"] += 1
            self._edges_since_persist += 1
            if self._edges_since_persist >= settings.GRAPH_PERSIST_EVERY_N_EDGES:
                self.persist()

    def _static_metrics_for(self, account_id: str) -> Dict[str, float]:
        row = self._account_metrics.get(account_id)
        if row is None:
            return {
                "graph_out_degree": 0.0, "graph_pagerank": 0.0,
                "graph_hub_score": 0.0, "graph_authority_score": 0.0, "community_id": -1,
            }
        out_degree, pagerank, hub, authority, community_id = row
        return {
            "graph_out_degree": out_degree, "graph_pagerank": pagerank,
            "graph_hub_score": hub, "graph_authority_score": authority,
            "community_id": community_id,
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
        # Triggers lazy-loading step on request if it has not executed yet
        if not self.loaded:
            self.load()

        sender_metrics = self._static_metrics_for(sender)
        receiver_metrics = self._static_metrics_for(receiver)

        sender_community_size = self._community_size(sender_metrics["community_id"])
        receiver_community_size = self._community_size(receiver_metrics["community_id"])
        min_active = 1
        sender_active = sender_metrics["community_id"] != -1 and sender_community_size >= min_active
        receiver_active = receiver_metrics["community_id"] != -1 and receiver_community_size >= min_active
        different_communities = sender_metrics["community_id"] != receiver_metrics["community_id"]
        is_bridge_transaction = int(sender_active and receiver_active and different_communities)

        scaled_vals = []
        for prefix, metrics in (("sender", sender_metrics), ("receiver", receiver_metrics)):
            for col in GRAPH_METRIC_COLS:
                scaled_vals.append(self._minmax(col, metrics[col]))
                
        composite = float(np.mean(scaled_vals)) if scaled_vals else 0.0
        
        # If the fallback is active because the 341MB object couldn't load,
        # apply a safe fallback weight (5%) so graph risk is gracefully handled.
        if self.oom_fallback_active:
            graph_risk = float(0.05 + 0.15 * is_bridge_transaction)
        else:
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
