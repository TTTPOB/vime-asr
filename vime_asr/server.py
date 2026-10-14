"""Single-user vime live transcription adapter (no audio/text logging)."""
import argparse
import asyncio
import base64
import binascii
import ctypes as C
import codecs
import json
import logging
import resource
import hmac
from pathlib import Path
import os
import time
import uuid
from dataclasses import dataclass, field

os.environ["OPENBLAS_NUM_THREADS"] = "4"

import numpy as np
import soxr
from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosed

LOG = logging.getLogger("vime")
TOKEN_CALLBACK = C.CFUNCTYPE(None, C.c_char_p)


class Native:
    def __init__(self, library, model_dir, ar_weights=None, unfixed_chunks=0):
        if ar_weights is None:
            os.environ.pop("VIME_AR_WEIGHTS", None)
        else:
            os.environ["VIME_AR_WEIGHTS"] = str(Path(ar_weights).resolve())
        os.environ["VIME_STREAM_UNFIXED_CHUNKS"] = str(unfixed_chunks)
        os.environ["QWEN_STREAM_EOF_COALESCE"] = "1"
        self.lib = C.CDLL(library)
        specs = {
            "vime_load": ([C.c_char_p], C.c_void_p),
            "vime_model_free": ([C.c_void_p], None),
            "vime_live_new": ([], C.c_void_p),
            "vime_append": ([C.c_void_p, C.POINTER(C.c_float), C.c_int], C.c_int),
            "vime_eof": ([C.c_void_p], None),
            "vime_run_stream": ([C.c_void_p, C.c_void_p, TOKEN_CALLBACK], C.c_void_p),
            "vime_tokens": ([C.c_void_p], C.c_size_t),
            "vime_first_token_at": ([C.c_void_p], C.c_double),
            "vime_text_free": ([C.c_void_p], None),
            "vime_live_free": ([C.c_void_p], None),
        }
        for name, (args, result) in specs.items():
            fn = getattr(self.lib, name)
            fn.argtypes, fn.restype = args, result
        self.ctx = self.lib.vime_load(model_dir.encode())
        if not self.ctx:
            raise RuntimeError("model load failed")

    def new(self):
        live = self.lib.vime_live_new()
        if not live:
            raise MemoryError("live allocation failed")
        return live

    def append(self, live, samples):
        samples = np.ascontiguousarray(samples, dtype=np.float32)
        if self.lib.vime_append(live, samples.ctypes.data_as(C.POINTER(C.c_float)), len(samples)):
            raise RuntimeError("native append failed")

    def eof(self, live):
        self.lib.vime_eof(live)

    def run(self, live, on_piece):
        # Keep the ctypes callback alive until native inference returns.
        callback = TOKEN_CALLBACK(on_piece)
        text = self.lib.vime_run_stream(self.ctx, live, callback)
        if not text:
            raise RuntimeError("native inference failed")
        try:
            return (C.string_at(text).decode("utf-8"), self.lib.vime_tokens(live),
                    self.lib.vime_first_token_at(live))
        finally:
            self.lib.vime_text_free(text)

    def free(self, live):
        self.lib.vime_live_free(live)

    def close(self):
        self.lib.vime_model_free(self.ctx)


@dataclass(eq=False)
class Recording:
    live: object
    ws: object
    resampler: object = field(default_factory=lambda: soxr.ResampleStream(
        24000, 16000, 1, dtype="float32", quality="HQ"))
    lock: object = field(default_factory=asyncio.Lock)
    task: object = None
    discarded: bool = False
    finished: bool = False
    committed: bool = False
    item_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    chunks: int = 0
    started: float = field(default_factory=time.monotonic)
    commit_time: float = 0.0


async def send(ws, event):
    await ws.send(json.dumps(event, ensure_ascii=False))


async def error(ws, message):
    await send(ws, {"type": "error", "error": {"message": message}})


class BusyError(ValueError):
    pass


class Service:
    def __init__(self, native, token=None):
        self.native = native
        self.token = token
        self.active = None
        self.tasks = set()

    async def begin(self, ws):
        if self.active is not None:
            raise BusyError("recognizer busy; retry after current worker exits")
        # Claim before yielding, including allocation time.
        self.active = True
        try:
            rec = Recording(await asyncio.to_thread(self.native.new), ws)
        except BaseException:
            self.active = None
            raise
        self.active = rec
        return rec

    def start(self, rec):
        rec.task = asyncio.create_task(self.finish(rec))
        self.tasks.add(rec.task)
        rec.task.add_done_callback(self.tasks.discard)

    def feed(self, rec, pcm=None, last=False):
        data = np.empty(0, dtype=np.float32) if pcm is None else (
            np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0)
        samples = rec.resampler.resample_chunk(data, last=last)
        if len(samples):
            self.native.append(rec.live, samples)
        if last:
            self.native.eof(rec.live)

    async def append(self, rec, pcm):
        async with rec.lock:
            if rec.finished or rec.committed or rec.discarded:
                raise ValueError("recording is no longer accepting audio")
            await asyncio.to_thread(self.feed, rec, pcm)
            rec.chunks += 1

    async def commit(self, rec):
        async with rec.lock:
            if rec.finished or rec.committed or rec.discarded:
                raise ValueError("recording cannot be committed")
            rec.committed = True
            rec.commit_time = time.monotonic()
            # Acknowledgement precedes resampler flush, EOF and final delivery.
            await send(rec.ws, {"type": "input_audio_buffer.committed", "item_id": rec.item_id})
            await asyncio.to_thread(self.feed, rec, None, True)

    async def cancel(self, rec):
        rec.discarded = True
        async with rec.lock:
            if not rec.finished:
                # Cancellation drops the resampler tail, but releases the native wait.
                await asyncio.to_thread(self.native.eof, rec.live)
                if rec.task is None:
                    await asyncio.to_thread(self.native.free, rec.live)
                    rec.finished = True
                    self.active = None

    async def forward_deltas(self, rec, pieces):
        # Byte-level tokens may split a UTF-8 character across callbacks.
        decoder = codecs.getincrementaldecoder("utf-8")()
        while (piece := await pieces.get()) is not None:
            if not rec.discarded:
                delta = decoder.decode(piece)
                if delta:
                    await send(rec.ws, {
                        "type": "conversation.item.input_audio_transcription.delta",
                        "item_id": rec.item_id, "delta": delta})
        if not rec.discarded:
            delta = decoder.decode(b"", final=True)
            if delta:
                await send(rec.ws, {
                    "type": "conversation.item.input_audio_transcription.delta",
                    "item_id": rec.item_id, "delta": delta})

    async def finish(self, rec):
        tokens = 0
        first_token_at = 0.0
        failed = False
        transcript = ""
        loop = asyncio.get_running_loop()
        pieces = asyncio.Queue()

        def on_piece(piece):
            # The native worker only schedules bytes; all WS sends stay on the loop.
            if not rec.discarded:
                loop.call_soon_threadsafe(pieces.put_nowait, piece)

        forwarding = asyncio.create_task(self.forward_deltas(rec, pieces))
        try:
            try:
                transcript, tokens, first_token_at = await asyncio.to_thread(
                    self.native.run, rec.live, on_piece)
            finally:
                pieces.put_nowait(None)
            # Drain callback bytes before sending the authoritative final.
            await forwarding
        except ConnectionClosed:
            rec.discarded = True
        except Exception as exc:
            failed = True
            LOG.error("streaming failure type=%s", type(exc).__name__)
        finally:
            if not forwarding.done():
                forwarding.cancel()
            await asyncio.gather(forwarding, return_exceptions=True)
        # Serialize destruction against outstanding producer/EOF calls.
        async with rec.lock:
            await asyncio.to_thread(self.native.free, rec.live)
            rec.finished = True
            self.active = None
        now = time.monotonic()
        LOG.info("worker elapsed_ms=%.0f final_wait_ms=%.0f chunks=%d callbacks=%d first_token_start_ms=%s first_token_commit_ms=%s rss_peak_kib=%d cancelled=%s failed=%s",
                 (now-rec.started)*1000, (now-rec.commit_time)*1000 if rec.commit_time else 0,
                 rec.chunks, tokens,
                 round((first_token_at-rec.started)*1000) if first_token_at else "none",
                 round((first_token_at-rec.commit_time)*1000) if first_token_at and rec.commit_time else "none",
                 resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                 rec.discarded, failed)
        if not rec.discarded:
            try:
                if failed or not rec.committed:
                    await error(rec.ws, "native recognition failed")
                else:
                    await send(rec.ws, {
                        "type": "conversation.item.input_audio_transcription.completed",
                        "item_id": rec.item_id, "transcript": transcript})
            except ConnectionClosed:
                pass

    async def handler(self, ws):
        rec = None
        try:
            async for raw in ws:
                try:
                    event = json.loads(raw)
                    kind = event["type"]
                    if kind == "session.update":
                        if rec is not None:
                            raise ValueError("session already configured")
                        session = event["session"]
                        inp = session["audio"]["input"]
                        if (session["type"] != "transcription" or
                            inp["format"] != {"type": "audio/pcm", "rate": 24000} or
                            inp["turn_detection"] is not None or
                            not isinstance(inp["transcription"]["model"], str)):
                            raise ValueError("unsupported transcription session")
                        rec = await self.begin(ws)
                        await send(ws, {"type": "session.updated", "session": session})
                        self.start(rec)
                    elif kind == "input_audio_buffer.clear":
                        if rec is not None:
                            await self.cancel(rec)
                    elif kind in ("input_audio_buffer.append", "input_audio_buffer.commit"):
                        if rec is None:
                            raise ValueError("session.update required")
                        if kind.endswith("append"):
                            pcm = base64.b64decode(event["audio"], validate=True)
                            if not 2 <= len(pcm) <= 48000 or len(pcm) % 2:
                                raise ValueError("append requires 1..24000 PCM16 samples")
                            await self.append(rec, pcm)
                        else:
                            await self.commit(rec)
                    else:
                        raise ValueError("unsupported event")
                except BusyError:
                    await error(ws, "recognizer busy; retry after current worker exits")
                except (ValueError, KeyError, TypeError, binascii.Error):
                    await error(ws, "invalid request")
                except Exception:
                    await error(ws, "adapter operation failed")
                    if rec is not None:
                        await self.cancel(rec)
        except ConnectionClosed:
            pass
        finally:
            if rec is not None:
                await self.cancel(rec)

    def health(self, connection, request):
        if request.path == "/health":
            body = json.dumps({"ready": True, "busy": self.active is not None}) + "\n"
            return connection.respond(200, body)
        if self.token is not None:
            values = request.headers.get_all("Authorization")
            expected = "Bearer " + self.token
            if len(values) != 1 or not hmac.compare_digest(values[0].encode(), expected.encode()):
                response = connection.respond(401, "Unauthorized\n")
                response.headers["WWW-Authenticate"] = "Bearer"
                return response


async def main(args):
    token = Path(args.token_file).read_text().strip()
    if not token or any(char.isspace() for char in token):
        raise ValueError("token file must contain one nonempty token")
    start = time.monotonic()
    native = await asyncio.to_thread(Native, args.library, args.model_dir,
                                     args.ar_weights, args.unfixed_chunks)
    LOG.info("model load_ms=%.0f rss_peak_kib=%d", (time.monotonic()-start)*1000,
             resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    service = Service(native, token)
    try:
        async with serve(service.handler, args.host, args.port, max_size=1024*1024,
                         process_request=service.health):
            LOG.info("listening host=%s port=%d", args.host, args.port)
            await asyncio.Future()
    finally:
        # Server context closes sockets first; handler finally sets EOF.
        if service.tasks:
            await asyncio.gather(*service.tasks)
        await asyncio.to_thread(native.close)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18766)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--library", default=str(
        Path(__file__).resolve().parents[1] / "build/native/libvime_qwen.so"))
    parser.add_argument("--ar-weights", help="Offline-generated decoder AR weights")
    parser.add_argument("--unfixed-chunks", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--token-file", required=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    # Avoid third-party connection logs; only adapter metrics are logged.
    logging.getLogger("websockets").setLevel(logging.CRITICAL)
    try:
        asyncio.run(main(parser.parse_args()))
    except KeyboardInterrupt:
        pass
