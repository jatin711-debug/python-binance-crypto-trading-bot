import asyncio
import json
import os
from aiohttp import web
import aiohttp_cors
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Define paths
script_directory = os.path.dirname(__file__)
trades_json_path = os.path.join(script_directory, 'trades.json')

class FileChangeHandler(FileSystemEventHandler):
    def __init__(self, websocket, loop):
        self.websocket = websocket
        self.loop = loop

    def on_modified(self, event):
        if event.src_path == trades_json_path:
            asyncio.run_coroutine_threadsafe(self.send_new_data(), self.loop)

    async def send_new_data(self):
        with open(trades_json_path, "r") as f:
            data = json.load(f)
            await self.websocket.send_json(data)

async def send_initial_data(websocket):
    with open(trades_json_path, "r") as f:
        data = json.load(f)
        await websocket.send_json(data)

async def handle_client(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)

    print(f"Client connected: {request.remote}")
    await send_initial_data(ws)

    loop = asyncio.get_event_loop()
    event_handler = FileChangeHandler(ws, loop)
    observer = Observer()
    observer.schedule(event_handler, path=script_directory, recursive=False)
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