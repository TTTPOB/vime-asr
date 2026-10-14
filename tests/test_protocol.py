"""Focused wire tests; fake inference does not establish ASR quality."""
import asyncio
import base64
import json
import threading
import time
import unittest

import numpy as np
import soxr
from websockets.asyncio.client import connect
from websockets.asyncio.server import serve
from websockets.exceptions import InvalidStatus

from vime_asr.server import Service

SESSION = {"type": "transcription", "audio": {"input": {
    "format": {"type": "audio/pcm", "rate": 24000},
    "transcription": {"model": "qwen3-asr-0.6b"}, "turn_detection": None}}}


class FakeNative:
    def __init__(self):
        self.release = threading.Event()
        self.lives = []
        self.pieces = []

    def new(self):
        live = {"eof": threading.Event(), "audio_ready": threading.Event(),
                "freed": False, "samples": [], "first": 0.0}
        self.lives.append(live)
        return live

    def append(self, live, samples):
        assert not live["freed"]
        live["samples"].append(samples.copy())
        live["first"] = live["first"] or time.monotonic()
        live["audio_ready"].set()

    def eof(self, live):
        assert not live["freed"]
        live["eof"].set()

    def run(self, live, on_piece):
        live["on_piece"] = on_piece
        if self.pieces:
            assert live["audio_ready"].wait(5)
            for piece in self.pieces:
                on_piece(piece)
        assert live["eof"].wait(5)
        assert self.release.wait(5)
        return "最终完整文本", 3, live["first"]

    def free(self, live):
        assert live["eof"].is_set()
        live["freed"] = True


async def emit(ws, kind, **kwargs):
    await ws.send(json.dumps({"type": kind, **kwargs}))


async def recv(ws):
    return json.loads(await asyncio.wait_for(ws.recv(), 2))


class ProtocolTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.native = FakeNative()
        self.service = Service(self.native)
        self.server = await serve(self.service.handler, "127.0.0.1", 0,
                                  process_request=self.service.health)
        self.uri = "ws://127.0.0.1:%d" % self.server.sockets[0].getsockname()[1]

    async def asyncTearDown(self):
        self.native.release.set()
        self.server.close()
        await self.server.wait_closed()
        if self.service.tasks:
            await asyncio.gather(*self.service.tasks)

    async def configure(self, ws):
        await emit(ws, "session.update", session=SESSION)
        self.assertEqual(await recv(ws), {"type": "session.updated", "session": SESSION})

    async def test_bearer_handshake(self):
        self.service.token = "test-only-token"
        for headers in (None, {"Authorization": "Bearer wrong"}):
            with self.assertRaises(InvalidStatus) as caught:
                async with connect(self.uri, additional_headers=headers):
                    pass
            self.assertEqual(caught.exception.response.status_code, 401)
        async with connect(self.uri, additional_headers={"Authorization": "Bearer test-only-token"}):
            pass
        port = self.server.sockets[0].getsockname()[1]
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        writer.write(b"GET /health HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
        await writer.drain()
        self.assertIn(b"200 OK", await reader.read())
        writer.close()
        await writer.wait_closed()

    async def test_ack_commit_flush_final_and_busy(self):
        async with connect(self.uri) as ws:
            await self.configure(ws)
            async with connect(self.uri) as other:
                await emit(other, "session.update", session=SESSION)
                busy = await recv(other)
                self.assertEqual(busy["type"], "error")
                self.assertIn("busy", busy["error"]["message"])
            pcm = np.arange(24000, dtype="<i2").tobytes()
            for offset in range(0, len(pcm), 16000):
                await emit(ws, "input_audio_buffer.append",
                           audio=base64.b64encode(pcm[offset:offset+16000]).decode())
            await emit(ws, "input_audio_buffer.commit")
            ack = await recv(ws)
            self.assertEqual(ack["type"], "input_audio_buffer.committed")
            self.assertTrue(ack["item_id"])
            # ACK arrives while inference is still blocked, before any final.
            self.assertFalse(self.native.release.is_set())
            # The HTTP endpoint remains responsive while the worker blocks.
            port = self.server.sockets[0].getsockname()[1]
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            writer.write(b"GET /health HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
            await writer.drain()
            reply = await asyncio.wait_for(reader.read(), 2)
            writer.close()
            await writer.wait_closed()
            self.assertIn(b"200 OK", reply)
            self.assertEqual(json.loads(reply.split(b"\r\n\r\n", 1)[1]),
                             {"ready": True, "busy": True})
            self.native.release.set()
            final = await recv(ws)
            self.assertEqual(final, {
                "type": "conversation.item.input_audio_transcription.completed",
                "item_id": ack["item_id"], "transcript": "最终完整文本"})
            live = self.native.lives[0]
            self.assertTrue(live["freed"])
            self.assertEqual(sum(len(x) for x in live["samples"]), 16000)
            reference = soxr.ResampleStream(24000, 16000, 1, dtype="float32", quality="HQ")
            expected = reference.resample_chunk(
                np.frombuffer(pcm, dtype="<i2").astype(np.float32)/32768.0, last=True)
            np.testing.assert_allclose(np.concatenate(live["samples"]), expected, atol=1e-7)

    async def test_live_delta_utf8_and_authoritative_final(self):
        # Split two Chinese characters across byte-level token callbacks.
        self.native.pieces = [b"\xe4", b"\xbd", b"\xa0\xe5\xa5", b"\xbd"]
        async with connect(self.uri) as ws:
            await self.configure(ws)
            await emit(ws, "input_audio_buffer.append",
                       audio=base64.b64encode(bytes(48000)).decode())
            deltas = [await recv(ws), await recv(ws)]
            self.assertEqual([event["delta"] for event in deltas], ["你", "好"])
            self.assertTrue(all(event["type"] ==
                "conversation.item.input_audio_transcription.delta" for event in deltas))
            item_id = deltas[0]["item_id"]
            self.assertTrue(all(event["item_id"] == item_id for event in deltas))
            self.assertFalse(self.native.lives[0]["eof"].is_set())
            await emit(ws, "input_audio_buffer.commit")
            self.assertEqual((await recv(ws))["item_id"], item_id)
            self.native.release.set()
            final = await recv(ws)
            self.assertEqual(final["type"], "conversation.item.input_audio_transcription.completed")
            self.assertEqual(final["item_id"], item_id)
            self.assertEqual(final["transcript"], "最终完整文本")

    async def test_clear_and_disconnect_keep_busy_until_worker_exits(self):
        for disconnect in (False, True):
            self.native.release.clear()
            ws = await connect(self.uri)
            await self.configure(ws)
            await emit(ws, "input_audio_buffer.commit")
            await recv(ws)
            if disconnect:
                await ws.close()
            else:
                await emit(ws, "input_audio_buffer.clear")
                # A subsequent error proves clear was processed, without polling.
                await emit(ws, "input_audio_buffer.commit")
                self.assertEqual((await recv(ws))["type"], "error")
            async with connect(self.uri) as other:
                await emit(other, "session.update", session=SESSION)
                self.assertIn("busy", (await recv(other))["error"]["message"])
            self.assertIsNotNone(self.service.active)
            # A late native callback must not revive a cancelled recording.
            self.native.lives[-1]["on_piece"]("迟到文字".encode())
            self.native.release.set()
            await asyncio.gather(*self.service.tasks)
            self.assertIsNone(self.service.active)
            self.assertTrue(self.native.lives[-1]["freed"])
            if not disconnect:
                with self.assertRaises(asyncio.TimeoutError):
                    await asyncio.wait_for(ws.recv(), 0.05)
                await ws.close()


if __name__ == "__main__":
    unittest.main()
