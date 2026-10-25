"""Bounded byte-exact body evidence, separate from normalized model records."""

from __future__ import annotations

import base64
import hashlib
import time
import zlib
from typing import Any

import httpx

MAX_BODY_BYTES = 8 * 1024 * 1024
MAX_BODY_CHUNKS = 65536


class CaptureLimitError(RuntimeError):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class DecodedBody:
    """Bounded incremental normalization of identity, gzip or zlib-wrapped deflate."""

    def __init__(self, encoding: str | None, max_bytes: int):
        self.encoding = (encoding or "identity").strip().lower()
        self.max_bytes = max_bytes
        self.body = bytearray()
        self.decoder = (zlib.decompressobj(16 + zlib.MAX_WBITS) if self.encoding in {"gzip", "x-gzip"}
                        else zlib.decompressobj() if self.encoding == "deflate" else None)

    def receive(self, chunk: bytes) -> None:
        if self.encoding not in {"identity", "gzip", "x-gzip", "deflate"}:
            raise httpx.DecodingError("unsupported upstream content encoding")
        remaining = self.max_bytes - len(self.body)
        try:
            # max_length prevents a small compressed chunk from allocating a huge decoded body.
            decoded = self.decoder.decompress(chunk, remaining + 1) if self.decoder else chunk
        except zlib.error:
            raise httpx.DecodingError("invalid compressed upstream body") from None
        self.body.extend(decoded[:remaining])
        if len(decoded) > remaining:
            raise CaptureLimitError("decoded_byte_limit")
        if self.decoder and self.decoder.unused_data:
            raise httpx.DecodingError("unexpected data after compressed upstream body")

    def finish(self) -> None:
        if self.encoding not in {"identity", "gzip", "x-gzip", "deflate"}:
            raise httpx.DecodingError("unsupported upstream content encoding")
        if self.decoder and not self.decoder.eof:
            raise httpx.DecodingError("compressed upstream body ended before EOF")

    def text(self) -> str:
        return self.body.decode("utf-8", "replace")


class BodyCapture:
    """Capture HTTP delivery chunks; their timestamps are not token generation times.

    The inline envelope travels through the existing authenticated records endpoint and
    run-bundle collection. It is written when a request finishes or is interrupted;
    abrupt process death before finalization can still lose an in-flight capture.
    """

    def __init__(self, request_body: bytes, *, max_bytes: int = MAX_BODY_BYTES, max_chunks: int = MAX_BODY_CHUNKS):
        if max_bytes < 1 or max_chunks < 1:
            raise ValueError("capture limits must be positive")
        if len(request_body) > max_bytes:
            raise CaptureLimitError("request_byte_limit")
        self.max_bytes, self.max_chunks = max_bytes, max_chunks
        self._started_ns = time.monotonic_ns()
        self._digest = hashlib.sha256()
        self._captured_digest = hashlib.sha256()
        self._byte_count = 0
        self._captured_count = 0
        self._chunk_count = 0
        self._chunks: list[dict[str, Any]] = []
        self._response: dict[str, Any] = {
            "body_representation": None, "content_type": None, "content_encoding": None,
            "transport_complete": False, "termination": "pending", "error_type": None,
            "capture_complete": True, "truncation_reason": None,
        }
        self._request = {
            "body_base64": base64.b64encode(request_body).decode("ascii"),
            "byte_count": len(request_body), "sha256": hashlib.sha256(request_body).hexdigest(),
            "dispatch_at": time.time(),
        }

    def response(self, *, content_type: str | None, content_encoding: str | None, encoded: bool) -> None:
        self._response.update({
            "body_representation": "content_encoded_http_body" if encoded else "decoded_http_body",
            "content_type": content_type, "content_encoding": content_encoding,
        })

    def receive(self, chunk: bytes) -> None:
        received_at, elapsed_ns = time.time(), time.monotonic_ns() - self._started_ns
        self._digest.update(chunk)
        self._byte_count += len(chunk)
        self._chunk_count += 1
        reason = ("chunk_limit" if self._chunk_count > self.max_chunks else
                  "byte_limit" if self._byte_count > self.max_bytes else None)
        kept = chunk[:max(0, self.max_bytes - self._captured_count)] if self._chunk_count <= self.max_chunks else b""
        if kept:
            self._chunks.append({
                "sequence": len(self._chunks), "received_at": received_at, "elapsed_ns": elapsed_ns,
                "body_base64": base64.b64encode(kept).decode("ascii"), "byte_count": len(kept),
            })
            self._captured_count += len(kept)
            self._captured_digest.update(kept)
        if reason:
            self.truncate(reason)
            raise CaptureLimitError(reason)

    def truncate(self, reason: str) -> None:
        self._response.update(capture_complete=False, truncation_reason=reason)

    def finish(self, termination: str, error: BaseException | None = None) -> None:
        if self._response["termination"] != "pending":
            return
        self._response.update({
            "transport_complete": termination == "eof", "termination": termination,
            "error_type": type(error).__name__ if error is not None else None, "finished_at": time.time(),
        })

    def record(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "timestamp_semantics": "gateway_receive_time_not_generation_time",
            "chunk_semantics": "http_body_delivery_not_token_or_sse_event",
            "limits": {"max_bytes": self.max_bytes, "max_chunks": self.max_chunks},
            "request": self._request,
            "response": {**self._response, "chunks": self._chunks, "byte_count": self._byte_count,
                         "sha256": self._digest.hexdigest(), "captured_byte_count": self._captured_count,
                         "captured_sha256": self._captured_digest.hexdigest(),
                         "received_chunk_count": self._chunk_count},
        }
