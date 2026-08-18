"""Moves money between `Balance` rows. The only place balances are ever mutated."""
import logging
from datetime import datetime
from sqlalchemy.orm import Session
from app.db import models

logger = logging.getLogger("ledger_service")

def settle_transaction(db: Session, name_orig: str, name_dest: str, amount: float) -> None:
    # 1. Deduct from sender
    sender = db.query(models.Balance).filter(models.Balance.account_id == name_orig).first()
    if sender is not None:
        sender.amount -= amount
        sender.updated_at = datetime.utcnow()

    # 2. Credit to receiver
    receiver = db.query(models.Balance).filter(models.Balance.account_id == name_dest).first()
    if receiver is not None:
        receiver.amount += amount
        receiver.updated_at = datetime.utcnow()

    if sender is None and receiver is None:
        logger.info("settle_transaction: neither %s nor %s has a Balance row — nothing to update.", name_orig, name_dest)
        
    db.commit()
