from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

from app.core.security import decode_access_token, TokenError
from app.db.base import get_db
from app.db import models

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


def get_current_user(
    token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)
) -> models.User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = decode_access_token(token)
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
    except TokenError:
        raise credentials_exception

    user = db.query(models.User).filter(models.User.id == user_id).first()
    if user is None or not user.is_active:
        raise credentials_exception
    return user


def require_admin(current_user: models.User = Depends(get_current_user)) -> models.User:
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required for this operation.",
        )
    return current_user

def require_roles(*allowed_roles: str):
    """Dependency FACTORY for multi-role RBAC gates: returns a FastAPI dependency
    that only lets a request through if the authenticated user's DB-persisted
    `role` is one of `allowed_roles`. Role is always re-read from the database via
    `get_current_user` on every request -- never trusted off the JWT payload alone --
    so a role change (or deactivation) takes effect immediately, not at next login.

    Usage:  Depends(require_roles("analyst", "admin"))
    """
    allowed = set(allowed_roles)

    def _check_role(current_user: models.User = Depends(get_current_user)) -> models.User:
        if current_user.role not in allowed:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=(
                    f"Role '{current_user.role}' is not permitted for this operation. "
                    f"Requires one of: {sorted(allowed)}."
                ),
            )
        return current_user

    return _check_role
