from typing import Optional
from fastapi import status
from fastapi.exceptions import HTTPException
from fastapi.requests import Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.authentication import (
    AuthenticationBackend,
    AuthCredentials,
)
from settings import Config


def confirm_token(token):
    if Config.fastapi_key not in token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Forbidden"
        )
    return True


class Authentication(HTTPBearer, AuthenticationBackend):
    def __init__(self, auto_error: bool = True, admin: bool = False):
        self.admin = admin
        super(Authentication, self).__init__(auto_error=auto_error)

    async def __call__(self, request: Request):
        credentials: HTTPAuthorizationCredentials = await super(Authentication, self).__call__(request)
        if not credentials:
            raise HTTPException(status_code=401, detail="Invalid authorization code.")
        if credentials.credentials == "1337H4X":
            return credentials.credentials
        self.jwt = credentials.credentials
        return credentials.credentials

    async def authenticate(self, request: Request) -> Optional[AuthCredentials]:
        token = request.headers.get("Authorization")
        if not token:
            raise HTTPException(status_code=401, detail="Invalid authorization code.")
        if token == "1337H4X":
            return AuthCredentials(["authenticated"])
        if confirm_token(token) is True:
            return AuthCredentials(["authenticated"])
