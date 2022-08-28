
from pydantic import BaseModel
from sqlalchemy.dialects.postgresql import UUID, ARRAY

from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy_mixins import AllFeaturesMixin
from typing import TypeVar, Union, List
from fastapi.logger import logger
from server.shared.bases.base_schema import BaseResponse

Base = declarative_base()
ModelType = TypeVar('ModelType', bound=Base)


class SafeException(BaseModel):
    pass


def safe(function):
    def run(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except Exception as e:
            logger.error(e)
            return SafeException(success=False, error=f"Caught Error: {e.args[0]}")
    return run


class CrudModel(AllFeaturesMixin):
    __abstract__ = True
    @safe
    def save(self) -> ModelType:
        self.session.add(self)
        self.session.commit()
        return self

    @classmethod
    def read(cls, *_, many=False, **kwargs) -> Union[List[ModelType], ModelType]:
        return cls.where().filter_by(**kwargs).all() if many else cls.where().filter_by(**kwargs).first()

    @classmethod
    @safe
    def update(cls, *_, object_id=None, **kwargs) -> BaseResponse:
        result = cls.where(id=object_id).update(**kwargs).save()
        return BaseResponse(success=True, response=result)

    @classmethod
    @safe
    def delete(cls, *_, **kwargs):
        result = cls.where(**kwargs).save()
        return BaseResponse(success=True, response=result)


class ModelMixin(Base, CrudModel):
    __abstract__ = True

    def __init__(self, *args, **kwargs):
        super(Base, self).__init__(*args, **kwargs)

    @classmethod
    def user_claims(cls, claim: UUID):
        user = cls.where(id=claim).first()
        return user.to_dict() if user else None



