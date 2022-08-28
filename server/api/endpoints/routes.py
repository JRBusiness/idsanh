from fastapi import FastAPI
from fastapi.routing import APIRoute

from server import app
from server.api.endpoints.urls import URLs


def use_route_names_as_operation_ids(app: FastAPI) -> None:
    for route in app.routes:
        if isinstance(route, APIRoute):
            method = list(route.methods)[0].lower()
            route.operation_id = f"{route.tags[0]}_{route.name}_{method}"


def add_routes() -> FastAPI:
    for route in URLs.include:
        exec(f"from server.api.{route}.views import router as {route}")
        app.include_router(eval(route))
    return app
