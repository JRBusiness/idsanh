import sys
import structlog
import uvicorn
from ddtrace import patch_all, tracer
from fastapi import FastAPI, HTTPException, Security
from fastapi.security.api_key import APIKeyHeader
from starlette.status import HTTP_401_UNAUTHORIZED

from server.api.test.schemas import RequestSchema, ScannerResponse
from server.shared.helpers.id_module import IDHandler
from server.logs import LogHandler
from settings import Config

config = Config()
variable = LogHandler(config.app_name)
structlog.configure(
    processors=[
        variable.event_dict,
        structlog.processors.JSONRenderer()
    ]
)
patch_all()
structlog.PrintLoggerFactory(sys.stdout)
log = structlog.get_logger()
envi = "Development"
api_key_header_auth = APIKeyHeader(name="Scan-OCR", auto_error=True)
app_name = "Scan"
app = FastAPI()


def get_api_key(api_key_header: str = Security(api_key_header_auth)):
    if api_key_header != config.api_key:
        return HTTPException(status_code=HTTP_401_UNAUTHORIZED, detail="Invalid API Key", )


id_file = "server/api/models/ID.pkl"
proto_text = "server/api/models/config.prototxt"
caffe_model = "server/api/models/trained.caffemodel"


@app.post(
    '/scan_id',
    # dependencies=[Depends(api_key_header_auth)]
    response_model=ScannerResponse
)
@tracer.wrap()
async def scan_id(request: RequestSchema):
    handler = IDHandler(proto_text, caffe_model, id_file)
    result = handler.main(
        request.img_data,
    )
    log.info(result)
    return ScannerResponse(success=True, parsed_result=result.response)


def run():
    uvicorn.run(
        "views:app",
        host="localhost",
        debug=config.debug,
        port=7990,
        workers=1
    )


if __name__ == '__main__':
    run()
