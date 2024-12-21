import asyncio
import json
import os
from aiohttp import web
import aiohttp_cors
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class FileChangeHandler(FileSystemEventHandler):
    def __init__(self, websocket):
        self.websocket = websocket

    def on_modified(self, event):
        if event.src_path == os.path.abspath("trades.json"):
            asyncio.run_coroutine_threadsafe(self.send_new_data(), asyncio.get_event_loop())

    async def send_new_data(self):
        with open("trades.json", "r") as f:
            data = json.load(f)
            await self.websocket.send(json.dumps(data))

async def send_initial_data(websocket):
    with open("trades.json", "r") as f:
        data = json.load(f)
        await websocket.send(json.dumps(data))

async def handle_client(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)

    print(f"Client connected: {request.remote}")
    await send_initial_data(ws)

    event_handler = FileChangeHandler(ws)
    observer = Observer()
    observer.schedule(event_handler, path='.', recursive=False)
    observer.start()

    try:
        async for msg in ws:
            if msg.type == web.WSMsgType.TEXT:
                print(f"Message received: {msg.data}")
            elif msg.type == web.WSMsgType.ERROR:
                print(f"WebSocket connection closed with exception: {ws.exception()}")
    except Exception as e:
        print(f"Exception occurred: {e}")
    finally:
        observer.stop()
        observer.join()

    print(f"Client disconnected: {request.remote}")
    return ws

app = web.Application()
cors = aiohttp_cors.setup(app, defaults={
    "*": aiohttp_cors.ResourceOptions(
        allow_credentials=True,
        expose_headers="*",
        allow_headers="*"
    )
})

app.router.add_get('/ws', handle_client)

# Configure CORS on all routes.
for route in list(app.router.routes()):
    cors.add(route)

if __name__ == "__main__":
    web.run_app(app, port=6789)