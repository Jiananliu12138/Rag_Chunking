from __future__ import annotations

from concurrent import futures

import grpc

from app.core.logging_config import logger
from app.core.path_setup import ensure_paths
from app.rpc.generated import evaluation_service_pb2_grpc as pb2_grpc
from app.rpc.servicer import ComponentEvalRpcServicer, EvalRpcServicer


class RpcSettings:
    def __init__(self, host: str = "127.0.0.1", port: int = 50051, max_workers: int = 4) -> None:
        self.host = host
        self.port = port
        self.max_workers = max_workers

    @property
    def bind_address(self) -> str:
        return f"{self.host}:{self.port}"


def get_rpc_settings() -> RpcSettings:
    import os

    return RpcSettings(
        host=os.getenv("MC_RPC_HOST", "127.0.0.1"),
        port=int(os.getenv("MC_RPC_PORT", "50051")),
        max_workers=int(os.getenv("MC_RPC_MAX_WORKERS", "4")),
    )


def build_server(settings: RpcSettings | None = None) -> tuple[grpc.Server, RpcSettings]:
    resolved_settings = settings or get_rpc_settings()
    ensure_paths()

    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=resolved_settings.max_workers),
    )
    pb2_grpc.add_EvalRpcServiceServicer_to_server(
        EvalRpcServicer(),
        server,
    )
    pb2_grpc.add_ComponentEvalRpcServiceServicer_to_server(
        ComponentEvalRpcServicer(),
        server,
    )
    server.add_insecure_port(resolved_settings.bind_address)
    return server, resolved_settings


def serve(settings: RpcSettings | None = None) -> None:
    server, resolved_settings = build_server(settings)
    logger.info("Starting gRPC evaluation server on %s", resolved_settings.bind_address)
    server.start()
    server.wait_for_termination()
