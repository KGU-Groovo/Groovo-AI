from jose import JWTError, jwt
from app.config import settings


class TokenPayload:
    def __init__(self, session_id: str, user_id: int):
        self.session_id = session_id
        self.user_id = user_id


def verify_ws_token(token: str) -> TokenPayload:
    """Spring Boot가 발급한 ws_token 검증"""
    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret,
            algorithms=[settings.jwt_algorithm],
        )
        session_id: str = payload.get("session_id")
        user_id: int = payload.get("user_id")
        if not session_id or user_id is None:
            raise ValueError("token payload missing fields")
        return TokenPayload(session_id=session_id, user_id=user_id)
    except JWTError as e:
        raise ValueError(f"invalid token: {e}")
