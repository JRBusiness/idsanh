from typing import Optional

from fastapi import File
from fastapi_camelcase import CamelModel
from pydantic import BaseModel
from starlette.responses import FileResponse

from server.shared.bases.base_schema import BaseResponse


class ScannerResponse(BaseResponse):
    parsed_result: Optional[dict]


class RequestSchema(BaseModel):
    img_data: str
