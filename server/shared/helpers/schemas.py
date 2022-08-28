from enum import Enum
from typing import Optional, Union, List, Dict
from pydantic import BaseModel


class AddressEnum(Enum):
    address1 = "address1"
    full_address = "address2"


class NameEnum(Enum):
    first_name: str = "first_name"
    full_name: str = "full_name"
    first_middle: str = "first_middle"
    name: str = "name"
    member_name: str = "member_name"


class ZipCodeEnum(Enum):
    zip: str = "zip"
    zip_code: str = "zip_code"
    zipcode: str = "zipcode"
    postal_code: str = "postal_code"


class IdNumberEnum(Enum):
    no: str = "no"
    license_number: str = "license_number"
    mem_id: str = "mem_id"
    member_id: str = "member_id"


class ExpireEnum(Enum):
    exp: str = "exp"
    expiry: str = "expiry"


class IssEnum(Enum):
    issued: str = "issued"
    iss: str = "iss"


class IdClassEnum(Enum):
    class_type: str = "class_type"
    DL_class: str = "DL_class"


class IsrProviderEnum(Enum):
    provider: str = "provider"
    insurance_provider: str = "insurance_provider"


class PhysicianEnum(Enum):
    physician_name: str = "physician_name"
    pcp_name: str = "pcp_name"


class DDEnum(Enum):
    document: str = "document"
    dd: str = "dd"


class DemographicInfo(BaseModel):
    face_photo: str
    name: NameEnum
    last_name: Optional[str]
    dob: str
    eyes: str
    hair: str
    weight: Optional[str]
    height: Optional[str]


class IdentificationType(BaseModel):
    address: AddressEnum
    address2: Optional[str]
    city_state_zip: Optional[str]
    city: Optional[str]
    state: Optional[str]
    zip_code: ZipCodeEnum
    id_type: Optional[str]
    state_name: Optional[str]
    country: Optional[str]
    no: IdNumberEnum
    exp: ExpireEnum
    class_type: IdClassEnum
    restrictions: Optional[str]
    iss: IssEnum
    dd: DDEnum


class Response(BaseModel):
    success: bool
    error: Optional[str]
    response: Optional[Union[dict, str]]


class BarcodeScannerResult(BaseModel):
    Demographic: DemographicInfo
    IdentificationType: IdentificationType


class IdScannerResult(BaseModel):
    Demographic: DemographicInfo
    IdentificationType: IdentificationType


class InsuranceScannerResult(BaseModel):
    name: NameEnum
    dob: Optional[str]
    sex: Optional[str]
    iss: IssEnum
    ins_provider: IsrProviderEnum
    sub_id: Optional[str]
    mem_id: IdNumberEnum
    care_provider: Optional[str]
    group_number: Optional[str]
    insurance_id: Optional[str]
    physician_id: Optional[str]
    pcp_phone: Optional[str]
    physician_name: PhysicianEnum
    chip_number: Optional[str]
    rxbin: Optional[str]
    rxpcn: Optional[str]
    rxgrp: Optional[str]
    plan: Optional[str]
    plan_name: Optional[str]
    plan_no: Optional[str]
    cms: Optional[str]
    behavior_health_plan: Optional[str]
    cover_type: Optional[str]
    deductible_innetwork: Optional[str]
    coinsurance_innetwork: Optional[str]
    care_network: Optional[str]
    pharmacy_use_phone: Optional[str]
    service_phone: Optional[str]
    nurse_helpline: Optional[str]
    crrisis_phone: Optional[str]
    aetna_site: Optional[str]
    deductable_primary_hmo: Optional[str]
    deductable_specialist_hmo: Optional[str]
    deductable_urgent_care_hmo: Optional[str]
    deductable_er_hmo: Optional[str]
    deductable_hospital_hmo: Optional[str]
    deductable_annual_allow_hmo: Optional[str]
    deductable_primary_oon: Optional[str]
    deductable_specialist_oon: Optional[str]
    deductable_urgent_care_oon: Optional[str]
    deductable_er_oon: Optional[str]
    deductable_hospital_oon: Optional[str]
    deductable_annual_allow_oon: Optional[str]
    deductible_rx: Optional[str]
    pcp_copay: Optional[str]
    spec_copay: Optional[str]
    er_copay: Optional[str]
    rx_copay: Optional[str]
    visit_pay: Optional[str]
    dental: Optional[str]
    er_pay: Optional[str]


class TemplateResponse(BaseModel):
    pixels: Dict[Optional[str], List[int]]
    width: int
    height: int
