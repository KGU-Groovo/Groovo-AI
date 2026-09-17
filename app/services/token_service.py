from jose import JWTError, jwt
from app.config import settings


class TokenPayload:
    def __init__(self, session_id: str, user_id: int):
        self.session_id = session_id
        self.user_id = user_id


def verify_ws_token(token: str) -> TokenPayload:
    """Spring Boot가 발급한 ws_token 검증.

    Spring Boot(JwtProvider.create)는 session_id를 별도 claim이 아니라
    subject(`sub`)로 넣고, user_id는 `userId`(카멜케이스) claim으로 넣는다.
    """
    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret,
            algorithms=[settings.jwt_algorithm],
        )
    except JWTError as e:
        raise ValueError(f"invalid token: {e}")

    session_id: str = payload.get("sub")
    raw_user_id = payload.get("userId")
    if not session_id or raw_user_id is None:
        raise ValueError("token payload missing fields")

    try:
        user_id = int(raw_user_id)
    except (TypeError, ValueError):
        raise ValueError("token payload missing fields")

    return TokenPayload(session_id=session_id, user_id=user_id)
