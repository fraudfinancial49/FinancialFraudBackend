import os
import json
import sqlite3
import logging
import threading
import time
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
from huggingface_hub import hf_hub_download

from app.core.config import settings

logger = logging.getLogger("graph_service")

GRAPH_METRIC_COLS = ["graph_out_degree", "graph_pagerank", "graph_hub_score", "graph_authority_score"]

# Slim, disk-backed artifact (built offline from graph_weight_maps.joblib).
GRAPH_SQLITE_FILE = "graph_metrics.sqlite"
RETRY_AFTER_SECONDS = 300


class GraphService:
    """Phase 2 graph metrics served from a small read-only SQLite file instead of
    a resident 341MB joblib/DataFrame. RAM cost is a few MB: only the 2 rows a
    transaction needs are read per request. If the artifact is unavailable the
    service silently reports `available=False` and the caller skips graph_risk."""

    def __init__(self) -> None:
        self.community_sizes: Dict[int, int] = {}
        self._metric_min: Dict[str, float] = {}
        self._metric_max: Dict[str, float] = {}

        self._conn: Optional[sqlite3.Connection] = None
        self._db_lock = threading.Lock()
        self._next_retry_at = 0.0
        self._loading = False

        self._live_edges: Dict[str, Dict[str, Dict[str, float]]] = {}
        self._lock = threading.Lock()
        self._edges_since_persist = 0
        self.loaded = False  # True only once the SQLite artifact is open

        self.repo_id = "ff49/financialfraudmodel"
        self.token = os.getenv("HF_TOKEN")
        self._load_live_state()

    def _live_state_path(self) -> str:
        return os.path.join(settings.RUNTIME_STATE_DIR, settings.LIVE_GRAPH_STATE_FILE)

    @property
    def available(self) -> bool:
        return self._ensure_ready()

    def _open_artifact(self) -> None:
        """Blocking download + open. Runs ONLY in the background thread."""
        try:
            repo_filepath = f"{settings.PHASE2_ARTIFACTS_SUBDIR}/{GRAPH_SQLITE_FILE}"
            path = hf_hub_download(repo_id=self.repo_id, filename=repo_filepath, token=self.token)
            conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, check_same_thread=False)
            meta = dict(conn.execute("SELECT key, value FROM meta").fetchall())
            bounds = json.loads(meta.get("bounds", "{}"))
            self._metric_min = {k: float(v[0]) for k, v in bounds.items()}
            self._metric_max = {k: float(v[1]) for k, v in bounds.items()}
            self.community_sizes = {int(k): int(v) for k, v in json.loads(meta.get("community_sizes", "{}")).items()}
            self._conn = conn
            self.loaded = True
            logger.info("Graph metrics artifact ready (SQLite, disk-backed).")
        except Exception as exc:
            self._next_retry_at = time.time() + RETRY_AFTER_SECONDS
            logger.warning("Graph metrics unavailable (%s) -- graph_risk bypassed, retry in %ds.",
                           exc, RETRY_AFTER_SECONDS)
        finally:
            self._loading = False

    def prewarm(self) -> None:
        """Starts the download in a background thread (non-blocking). Safe to call
        repeatedly; call once at startup so the file is ready before traffic."""
        if self._conn is not None or self._loading or time.time() < self._next_retry_at:
            return
        with self._db_lock:
            if self._conn is not None or self._loading:
                return
            self._loading = True
        threading.Thread(target=self._open_artifact, name="graph-prewarm", daemon=True).start()

    def _ensure_ready(self) -> bool:
        """Never blocks a request. True only if the artifact is already open;
        otherwise kicks off the background load and the caller bypasses graph_risk."""
        if self._conn is not None:
            return True
        self.prewarm()
        return False

    def _lookup(self, account_id: str) -> Optional[Tuple[float, float, float, float, int]]:
        if self._conn is None:
            return None
        try:
            with self._db_lock:
                row = self._conn.execute(
                    "SELECT out_degree, pagerank, hub, authority, community_id FROM accounts WHERE account_id = ?",
                    (account_id,),
                ).fetchone()
        except Exception:
            return None
        if row is None:
            return None
        return float(row[0]), float(row[1]), float(row[2]), float(row[3]), int(row[4])

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
        joblib.dump(self._live_edges, self._live_state_path())
        self._edges_since_persist = 0
        logger.info("Persisted live graph bookkeeping (%d accounts tracked).", len(self._live_edges))

    def add_edge_incremental(self, sender: str, receiver: str, amount: float) -> None:
        with self._lock:
            bucket = self._live_edges.setdefault(sender, {})
            edge = bucket.setdefault(receiver, {"weight": 0.0, "tx_count": 0, "confirmed_fraud_count": 0})
            edge["weight"] += amount
            edge["tx_count"] += 1
            self._edges_since_persist += 1
            if self._edges_since_persist >= settings.GRAPH_PERSIST_EVERY_N_EDGES:
                self.persist()

    def record_confirmed_fraud_edge(self, sender: str, receiver: str) -> None:
        """Real-time feedback loop: called immediately when an admin confirms a
        transaction as fraud, NOT deferred to a batch job. Marks this specific
        sender->receiver edge as fraud-confirmed so `account_risk_snapshot` boosts
        graph_risk for that pair on every future transaction between them."""
        with self._lock:
            bucket = self._live_edges.setdefault(sender, {})
            edge = bucket.setdefault(receiver, {"weight": 0.0, "tx_count": 0, "confirmed_fraud_count": 0})
            edge["confirmed_fraud_count"] = edge.get("confirmed_fraud_count", 0) + 1
            # Confirmed-fraud signals are rare and high-value -- persist immediately
            # rather than waiting for the batch edge-count threshold.
            self.persist()

    def _confirmed_fraud_boost(self, sender: str, receiver: str) -> float:
        edge = self._live_edges.get(sender, {}).get(receiver)
        if not edge:
            return 0.0
        fraud_count = edge.get("confirmed_fraud_count", 0)
        return min(0.5, 0.2 * fraud_count) if fraud_count else 0.0

    def _static_metrics_for(self, account_id: str) -> Dict[str, float]:
        row = self._lookup(account_id)
        if row is None:
            # Safe runtime fallback boundaries for feature compliance
            # Avoids looking up empty keys or causing structural mapping errors
            if account_id.startswith("M"):  # Merchant account structural defaults
                return {
                    "graph_out_degree": 1.0, "graph_pagerank": 0.00001,
                    "graph_hub_score": 0.001, "graph_authority_score": 0.001, "community_id": 15,
                }
            return {  # Regular account structural defaults
                "graph_out_degree": 2.0, "graph_pagerank": 0.00001,
                "graph_hub_score": 0.001, "graph_authority_score": 0.001, "community_id": 12,
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
        val = self.community_sizes.get(community_id)
        if val is None:
            return 450 if community_id == 12 else 120
        return int(val)

    def _minmax(self, col: str, value: float) -> float:
        lo = self._metric_min.get(col)
        hi = self._metric_max.get(col)
        if lo is None or hi is None or hi <= lo:
            return 0.0
        return float(np.clip((value - lo) / (hi - lo), 0.0, 1.0))

    def account_risk_snapshot(self, sender: str, receiver: str) -> Dict[str, Optional[float]]:
        graph_ready = self._ensure_ready()
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
        fraud_boost = self._confirmed_fraud_boost(sender, receiver)
        base_risk = 0.85 * composite + 0.15 * is_bridge_transaction
        graph_risk: Optional[float] = float(np.clip(base_risk + fraud_boost, 0.0, 1.0))
        if not graph_ready:
            # Silent bypass: no real Phase 2 data, so don't inject a fake constant
            # into fusion. Admin-confirmed fraud on this exact edge is still real.
            graph_risk = float(min(1.0, fraud_boost)) if fraud_boost > 0 else None

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
