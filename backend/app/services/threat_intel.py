"""
Offline/batch threat-intelligence job. Never runs on the request path — only
via `POST /admin/run-attacker-profiling`. Aggregates honeypot telemetry per
`browser_fingerprint`, standardizes behavioral features, and runs K-Means to
assign a human-readable cluster label.
"""
import logging
from datetime import datetime
from typing import Dict, List

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.db import models

logger = logging.getLogger("threat_intel")

CLUSTER_LABELS = ["Automated Bot", "Slow Prober", "Credential Stuffer", "Opportunistic Tester"]


def _aggregate_sessions(db: Session) -> List[Dict[str, object]]:
    rows = []
    fingerprints = (
        db.query(models.HoneypotSession.browser_fingerprint)
        .filter(models.HoneypotSession.browser_fingerprint.isnot(None))
        .distinct()
        .all()
    )
    for (fp,) in fingerprints:
        sessions = db.query(models.HoneypotSession).filter(
            models.HoneypotSession.browser_fingerprint == fp
        ).all()
        durations, event_counts = [], []
        for s in sessions:
            n_events = db.query(func.count(models.HoneypotEvent.id)).filter(
                models.HoneypotEvent.session_id == s.id
            ).scalar() or 0
            event_counts.append(n_events)
            if s.closed_at and s.started_at:
                durations.append((s.closed_at - s.started_at).total_seconds())
        rows.append({
            "browser_fingerprint": fp,
            "simulated_ip": sessions[-1].simulated_ip if sessions else None,
            "total_sessions": len(sessions),
            "avg_session_duration_seconds": float(np.mean(durations)) if durations else 0.0,
            "avg_events_per_session": float(np.mean(event_counts)) if event_counts else 0.0,
        })
    return rows


def run_attacker_profiling(db: Session, n_clusters: int = 4) -> Dict[str, object]:
    rows = _aggregate_sessions(db)
    if len(rows) < n_clusters:
        logger.warning(
            "Only %d attacker fingerprint(s) with honeypot history — skipping K-Means "
            "(need >= %d) and labeling all as 'Opportunistic Tester'.", len(rows), n_clusters
        )
        for row in rows:
            _upsert_profile(db, row, "Opportunistic Tester", threat_score=25.0)
        return {"status": "skipped_insufficient_data", "n_profiles_updated": len(rows)}

    X = np.array([
        [r["total_sessions"], r["avg_session_duration_seconds"], r["avg_events_per_session"]]
        for r in rows
    ])
    X_scaled = StandardScaler().fit_transform(X)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_ids = kmeans.fit_predict(X_scaled)

    # Rank clusters by mean events-per-session descending -> map to descriptive labels.
    cluster_order = np.argsort(-np.array([
        X[cluster_ids == c, 2].mean() if (cluster_ids == c).any() else 0.0
        for c in range(n_clusters)
    ]))
    label_map = {int(cluster_order[i]): CLUSTER_LABELS[i % len(CLUSTER_LABELS)] for i in range(n_clusters)}

    for row, cluster_id in zip(rows, cluster_ids):
        label = label_map[int(cluster_id)]
        threat_score = min(100.0, 20.0 + row["avg_events_per_session"] * 5.0 + row["total_sessions"] * 2.0)
        _upsert_profile(db, row, label, threat_score)

    return {"status": "completed", "n_profiles_updated": len(rows), "n_clusters": n_clusters}


def _upsert_profile(db: Session, row: Dict[str, object], label: str, threat_score: float) -> None:
    profile = db.query(models.AttackerProfile).filter(
        models.AttackerProfile.browser_fingerprint == row["browser_fingerprint"]
    ).first()
    if profile is None:
        profile = models.AttackerProfile(browser_fingerprint=row["browser_fingerprint"])
    profile.simulated_ip = row["simulated_ip"]
    profile.total_sessions = row["total_sessions"]
    profile.avg_session_duration_seconds = row["avg_session_duration_seconds"]
    profile.avg_events_per_session = row["avg_events_per_session"]
    profile.cluster_label = label
    profile.threat_score = threat_score
    profile.last_seen_at = datetime.utcnow()
    profile.updated_at = datetime.utcnow()
    db.add(profile)
    db.commit()
