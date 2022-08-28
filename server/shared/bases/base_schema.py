from typing import Optional
from fastapi_camelcase import CamelModel

class ORMModel(CamelModel):
    class Config:
        use_orm = True


class Logs(CamelModel):
    id: Optional[str]
    request_time: Optional[str]
    ip_address: Optional[str]
    # "ok", "error", "new"
    status: Optional[str]
    # error message if applicable
    message: Optional[str]


class BaseResponse(CamelModel):
    error: Optional[str]
    success: Optional[bool]