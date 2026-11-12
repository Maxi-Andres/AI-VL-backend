"""robot_ear.py — the G1's own microphone, held by the gateway, not by a page.

    robot mic --relay--> executor --HTTP stream--> Ear --WS binary--> iacore /ws/ear
    iacore's events (listening / speech / heard / wake / command / barge) --> every page watching

Until 2026-11-12 the page's "robot mic" switch WAS the stream: no page open, no ear. Now the
gateway holds ONE Ear per robot, and it listens while anything holds it: a page's switch
("page"), or a walk that someone is blocking ("blocked", app.py's walk watch) — so the robot can
be told "Bender, no podés pasar, date la vuelta" with nobody looking at the app. What it hears
after its name goes to the owner's `on_command` (app.py: the agent), not to each page: two
pages once both answered every command (2026-10-19); now they only watch.

Muting: while the robot speaks the ear must not answer its own voice. Each source of speech
mutes it under its own name (a page that plays through the robot, the gateway's own replies);
it is muted while any of them is, and still hears its name ("barge") only if all of them allow
it. Audio only flows to iacore; pages get text.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging

import httpx

logger = logging.getLogger("uvicorn.error")

EAR_TIMEOUT = httpx.Timeout(connect=3.0, read=10.0, write=5.0, pool=3.0)
QUEUE_MAX = 200          # a page this far behind loses events, never slows the ear


class Ear:
    def __init__(self, robot, executor_url, connect_ear, on_command=None, on_wake=None):
        self.robot = robot
        self.executor_url = executor_url          # a callable: tests point it elsewhere late
        self.connect_ear = connect_ear
        self.on_command = on_command
        self.on_wake = on_wake
        self.holds: set[str] = set()
        self.subs: set[asyncio.Queue] = set()
        self.listening = False
        self.task: asyncio.Task | None = None
        self._ear = None
        self._why_off = ""
        self._mutes: dict[str, tuple[bool, bool]] = {}
        self._tasks: set[asyncio.Task] = set()

    # -- watchers ------------------------------------------------------- #
    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(QUEUE_MAX)
        self.subs.add(q)
        if self.listening:
            q.put_nowait({"type": "listening", "robot": self.robot})
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self.subs.discard(q)

    def publish(self, event: dict) -> None:
        for q in list(self.subs):
            with contextlib.suppress(asyncio.QueueFull):
                q.put_nowait(event)

    # -- who keeps it open ------------------------------------------------ #
    def running(self) -> bool:
        return self.task is not None and not self.task.done()

    def hold(self, why: str) -> None:
        self.holds.add(why)
        if not self.running():
            self._why_off = ""
            self.task = asyncio.create_task(self._run(), name=f"ear-{self.robot}")

    async def release(self, why: str) -> None:
        self.holds.discard(why)
        if not self.holds:
            await self._stop("the robot's microphone was switched off")

    async def off(self, why: str) -> None:
        """The switch: off for every page and every reason at once."""
        self.holds.clear()
        await self._stop(why)

    async def _stop(self, why: str) -> None:
        task = self.task
        if task is not None and not task.done():
            self._why_off = why
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task

    # -- muting ------------------------------------------------------------- #
    async def mute(self, source: str, on: bool, barge: bool = True) -> None:
        if on:
            self._mutes[source] = (True, barge)
        else:
            self._mutes.pop(source, None)
        await self._send_mute()

    async def _send_mute(self) -> None:
        if self._ear is None:
            return
        active = list(self._mutes.values())
        on = bool(active)
        barge = all(b for _on, b in active) if active else False
        with contextlib.suppress(Exception):
            await self._ear.send(json.dumps({"type": "mute", "on": on, "barge": barge}))

    # -- the stream ---------------------------------------------------------- #
    def _dispatch(self, event: dict) -> None:
        kind, text = event.get("type"), str(event.get("text") or "").strip()
        if kind in ("command", "barge") and text and self.on_command:
            self._spawn(self.on_command(self.robot, text))
        elif kind in ("wake", "barge") and not text and self.on_wake:
            self._spawn(self.on_wake(self.robot))

    def _spawn(self, coro) -> None:
        """Run an owner's callback without waiting for it; kept so it is not collected."""
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _run(self) -> None:
        final = {"type": "error", "error": "the robot's microphone stream ended"}
        try:
            async with (
                httpx.AsyncClient(timeout=EAR_TIMEOUT) as ec,
                ec.stream("GET", f"{self.executor_url()}/audio/mic/stream",
                          params={"robot": self.robot}) as r,
            ):
                if r.status_code != 200:
                    raw = (await r.aread())[:500]
                    try:
                        detail = json.loads(raw).get("error") or raw.decode("utf-8", "replace")
                    except ValueError:
                        detail = raw.decode("utf-8", "replace")
                    final = {"type": "error", "error": detail, "status": r.status_code}
                    return
                try:
                    ear_cm = self.connect_ear()
                    ear = await ear_cm.__aenter__()
                except Exception as e:
                    final = {"type": "error", "error": f"speech-to-text (iacore) unreachable: {e}"}
                    return
                try:
                    self._ear = ear
                    await self._send_mute()
                    self.listening = True
                    self.publish({"type": "listening", "robot": self.robot})

                    async def mic_to_ear() -> None:
                        async for chunk in r.aiter_bytes():
                            await ear.send(chunk)

                    async def ear_to_watchers() -> None:
                        async for msg in ear:
                            try:
                                event = json.loads(msg if isinstance(msg, str) else msg.decode())
                            except ValueError:
                                continue
                            if isinstance(event, dict):
                                self.publish(event)
                                self._dispatch(event)

                    tasks = {asyncio.create_task(mic_to_ear()),
                             asyncio.create_task(ear_to_watchers())}
                    try:
                        done, _pending = await asyncio.wait(tasks,
                                                            return_when=asyncio.FIRST_COMPLETED)
                    finally:
                        for t in tasks:
                            t.cancel()
                    for t in done:
                        if not t.cancelled() and t.exception() is not None:
                            final = {"type": "error",
                                     "error": f"the robot's ear stopped: {t.exception()}"}
                finally:
                    self._ear = None
                    await ear_cm.__aexit__(None, None, None)
        except asyncio.CancelledError:
            final = {"type": "error",
                     "error": self._why_off or "the robot's microphone was switched off"}
            raise
        except Exception as e:
            final = {"type": "error", "error": f"robot executor unreachable: {e}"}
        finally:
            self.listening = False
            self.holds.clear()
            self.publish(final)
