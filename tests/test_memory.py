"""Each robot's memory: "guardá esta imagen" and "acordate que…" keep things, the Memory page
lists and deletes them, and the notes reach the assistant.

The real app, database and sign-in (marker `real_auth`, as test_fleet.py); iacore is a
MockTransport that records what it was sent.
"""
from __future__ import annotations

import base64
import json

import httpx
import pytest
from fastapi.testclient import TestClient

import app as gateway
import auth
import memory

pytestmark = pytest.mark.real_auth
OWNER = {"org_name": "Acme Logistics", "name": "Ana Owner", "email": "ana@acme.example",
         "password": "correct horse battery"}

JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 200
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 200


@pytest.mark.parametrize("said, kind, note", [
    ("Guardá esta imagen", "image", ""),
    ("Bender, sacá una foto", "image", ""),
    ("guardame la foto por favor", "image", ""),
    ("Save this picture", "image", ""),
    ("Huesos, take a photo", "image", ""),
    ("Acordate que la llave está en el cajón", "note", "la llave está en el cajón"),
    ("Bender, recordá que Mateo llega a las 3.", "note", "Mateo llega a las 3"),
    ("Remember that the meeting is on Friday", "note", "the meeting is on Friday"),
    ("Acuérdate de que la llave está en el cajón", "note", "la llave está en el cajón"),
    ("Recuerda que hoy cierra temprano", "note", "hoy cierra temprano"),
    ("Sacad una foto a esto y acordátelo", "image", ""),
    ("Sacale una foto", "image", ""),
    ("Tomá una foto de la pantalla", "image", ""),
    ("Fotografiá esto", "image", ""),
    ("Capturá esto", "image", ""),
    ("¿Podés guardar esta foto?", "image", ""),
    ("Can you take a picture?", "image", ""),
    ("Lo guardamos por todo este control.", "image", ""),
    ("A notar que hoy es el cumpleaños de Mateo.", "note", "hoy es el cumpleaños de Mateo"),
    ("Tomá nota de que el lunes no hay oficina", "note", "el lunes no hay oficina"),
    ("Make a note that the printer needs toner", "note", "the printer needs toner"),
    ("Che, acordate que el martes viene Mateo", "note", "el martes viene Mateo"),
    ("Bueno Bender, sacá una foto", "image", ""),
    ("Ok, remember that the door code is 1234", "note", "the door code is 1234"),
    ("Guardá esto", "image", ""),
    ("Keep this", "image", ""),
    ("¿Me podés recordar que mañana hay reunión?", "note", "mañana hay reunión"),
])
def test_what_asks_to_keep_something(said, kind, note):
    assert memory.command(said) == (kind, note)


@pytest.mark.parametrize("said", ["¿Qué imagen guardaste?", "¿Qué ves en la foto?", "Bien, anotalo",
                                  "¿Guardaste la foto?", "¿Te acordás de lo que te dije?", "Toma nota",
                                  "Sacá la mano de ahí", "Guardá silencio",
                                  "No guardes esto", "Don't save this picture", "Never remember that",
                                  "Describe the picture", "¿Te acordás de algo?", "Hola"])
def test_what_does_not(said):
    assert memory.command(said) is None


@pytest.fixture(scope="module")
def tc():
    with TestClient(gateway.app) as tc:
        if tc.get("/api/auth/state").json()["setup_required"]:
            code = auth.SETUP_CODE_PATH.read_text().strip()
            r = tc.post("/api/auth/setup", json={**OWNER, "setup_code": code})
        else:
            r = tc.post("/api/auth/login", json={"email": OWNER["email"],
                                                 "password": OWNER["password"]})
        assert r.status_code == 200, r.text
        yield tc


@pytest.fixture
def iacore(tc, monkeypatch):
    got = []

    def handle(req: httpx.Request) -> httpx.Response:
        got.append(json.loads(req.content))
        return httpx.Response(200, content=b"En el cajon.")

    monkeypatch.setattr(gateway, "client", httpx.AsyncClient(
        base_url="http://iacore.test", transport=httpx.MockTransport(handle)))
    return got


def _robot(tc, slot="g1", name="Bender"):
    r = tc.post("/api/robots", json={"name": name, "slot": slot})
    assert r.status_code in (200, 409), r.text


def test_a_spoken_picture_and_note_are_kept_and_listed(tc, iacore):
    _robot(tc)
    frame = base64.b64encode(JPEG).decode()
    r = tc.post("/api/vlm/stream", json={"image": frame, "prompt": "Guardá esta imagen", "robot": "g1"})
    assert r.text == "Listo, guardé la imagen en mi memoria."
    r = tc.post("/api/vlm/stream", json={"image": frame, "prompt": "Acordate que la llave está en el cajón",
                                         "robot": "g1"})
    assert r.text == "Listo, me lo acuerdo: la llave está en el cajón."
    assert iacore == [], "no model to keep a file"
    listed = tc.get("/api/memory", params={"robot": "g1"}).json()
    kinds = [(m["kind"], m["text"]) for m in listed["items"][:2]]
    assert kinds == [("note", "la llave está en el cajón"), ("image", "")]
    pic = listed["items"][1]
    got = tc.get(f"/api/memory/{pic['id']}/image")
    assert got.status_code == 200 and got.content == JPEG and got.headers["content-type"] == "image/jpeg"


def test_the_notes_go_to_the_assistant(tc, iacore):
    _robot(tc)
    tc.post("/api/memory", json={"robot": "g1", "kind": "note", "text": "Mateo llega a las 3"})
    tc.post("/api/vlm/stream", json={"image": "aGk=", "prompt": "¿A qué hora llega Mateo?", "robot": "g1"})
    assert "Mateo llega a las 3" in iacore[0]["robot_notes"]


def test_english_is_answered_in_english(tc, iacore):
    _robot(tc)
    r = tc.post("/api/vlm/stream", json={"image": base64.b64encode(PNG).decode(),
                                         "prompt": "Save this picture", "robot": "g1"})
    assert r.text == "Done, I saved the picture to my memory."


def test_nothing_is_kept_without_a_robot_or_a_real_picture(tc, iacore):
    assert tc.post("/api/vlm/stream", json={"prompt": "Guardá esta imagen", "robot": "nope"}).text == \
        "No hay un robot elegido donde guardarlo."
    _robot(tc)
    bad = base64.b64encode(b"<svg onload=alert(1)>").decode()
    assert tc.post("/api/vlm/stream", json={"image": bad, "prompt": "Guardá esta imagen",
                                            "robot": "g1"}).text == "No pude guardar esa imagen."
    assert tc.post("/api/memory", json={"robot": "g1", "kind": "image", "image": bad}).status_code == 422


def test_deleting_takes_the_file_too(tc, iacore):
    _robot(tc)
    item = tc.post("/api/memory", json={"robot": "g1", "kind": "image",
                                        "image": base64.b64encode(JPEG).decode()}).json()
    assert (memory.MEMORY_DIR / item["id"]).is_file()
    assert tc.delete(f"/api/memory/{item['id']}").status_code == 200
    assert not (memory.MEMORY_DIR / item["id"]).exists()
    assert tc.get(f"/api/memory/{item['id']}/image").status_code == 404
    assert tc.delete(f"/api/memory/{item['id']}").status_code == 404
