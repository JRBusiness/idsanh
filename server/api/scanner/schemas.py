from typing import Optional
from pydantic import BaseModel

from server.shared.bases.base_schema import BaseResponse


class ScannerResponse(BaseResponse):
    parsed_result: Optional[dict]


class RequestSchema(BaseModel):
    img_data: str
