import structlog
import sys

from fastapi import Depends, APIRouter
from ddtrace import patch_all, tracer

from server.api.scanner.schemas import ScannerResponse, RequestSchema
from server.shared.bases.auth import Authentication
from settings import Config
from server.shared.helpers.barcode_module import BarcodeHandler
from server.shared.helpers.id_module import IDHandler
from server.shared.helpers.ins_module_deprecated import InsuranceHandler
from server.logs import LogHandler
from server.shared.excpetions.exceptions import ExternalApiException

variable = LogHandler(Config.app_name)
structlog.configure(processors=[variable.event_dict, structlog.processors.JSONRenderer()])
patch_all()
structlog.PrintLoggerFactory(sys.stdout)
log = structlog.get_logger()
app_name = "Scan"
id_model = "server/api/models/ID.pkl"
is_model = "server/api/models/insurance.pkl"
proto_text = "server/api/models/config.prototxt"
caffe_model = "server/api/models/trained.caffemodel"
media_type = "image/jpeg"


router = APIRouter(
    dependencies=[Depends(Authentication())],
    tags=["id_scanner"],
)


@router.get('/')
@tracer.wrap()
def home():
    try:
        result = {app_name: Config.version}
        log.info(result)
        return result
    except ExternalApiException as e:
        log.exception(e)


@router.post(
    '/scan_id',
    response_model=ScannerResponse,
)
@tracer.wrap()
async def scan_id(request: RequestSchema):
    handler = IDHandler(proto_text, caffe_model, id_model)
    result = handler.main(
        request.img_data,
    )
    log.info(result)
    return ScannerResponse(success=True, parsed_result=result.response)


@router.post(
    '/scan_insurance',
    response_model=ScannerResponse,
)
@tracer.wrap()
async def scan_insurance(request: RequestSchema):
    handler = InsuranceHandler(is_model)
    result = handler.main(
        request.img_data,
    )
    log.info(result)
    return ScannerResponse(success=True, parsed_result=result.response)


@router.post(
    '/barcode',
    response_model=ScannerResponse,
)
@tracer.wrap()
def scan_barcode(request: RequestSchema):
    handler = BarcodeHandler()
    result = handler.decode_bar(
        request.img_data,
    )
    log.info(result)
    return ScannerResponse(success=True, parsed_result=result.response)


@router.post(
    '/scan_id_alt',
    response_model=ScannerResponse,
)
@tracer.wrap()
async def scan_id_main(request: RequestSchema):
    handler = IDHandler(proto_text, caffe_model, id_model)
    result = handler.id_main(
        request.img_data,
    )
    log.info(result)
    return ScannerResponse(success=True, parsed_result=result.response)