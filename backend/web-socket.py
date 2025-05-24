import asyncio
import json
import os
import sqlite3
from aiohttp import web
import aiohttp_cors
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Define paths
script_directory = os.path.dirname(__file__)
db_path = os.path.join(script_directory, 'trades.db')

# Function to fetch trade data from SQLite database
def get_trade_data_from_db(limit=None):
    """Fetch trade data from SQLite database with optional limit."""
    try:
        # Connect to the SQLite database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Construct query based on whether a limit is provided
        if limit:
            cursor.execute('SELECT * FROM trades ORDER BY timestamp DESC LIMIT ?', (limit,))
        else:
            cursor.execute('SELECT * FROM trades ORDER BY timestamp DESC')

        trades = cursor.fetchall()

        # Close connection
        conn.close()

        # Convert trades into a list of dictionaries
        trade_data = []
        for trade in trades:
            trade_data.append({
                'id': trade[0],
                'timestamp': trade[1],
                'current_price': trade[2],
                'lstm_prediction': trade[3],
                'rf_prediction': trade[4],
                'ensemble_prediction': trade[5],
                'action': trade[6],
                'average_sentiment': trade[7] # Added average_sentiment
            })
        return trade_data
    except Exception as e:
        print(f"Error fetching trade data from DB: {e}")
        return []

class FileChangeHandler(FileSystemEventHandler):
    def __init__(self, websocket, loop):
        self.websocket = websocket
        self.loop = loop

    def on_modified(self, event):
        if event.src_path == db_path:
            asyncio.run_coroutine_threadsafe(self.send_new_data(), self.loop)

    async def send_new_data(self):
        data = get_trade_data_from_db()
        await self.websocket.send_json(data)

async def send_initial_data(websocket, num_records=None):
    """Send the initial trade data or parameterized data."""
    data = get_trade_data_from_db(num_records)
    await websocket.send_json(data)

async def handle_client(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)

    print(f"Client connected: {request.remote}")

    # Send initial data with optional parameter
    # Default to None if not provided by the client
    num_records = None  # You can modify this based on what the client sends
    await send_initial_data(ws, num_records)

    loop = asyncio.get_event_loop()
    event_handler = FileChangeHandler(ws, loop)
    observer = Observer()
    observer.schedule(event_handler, path=script_directory, recursive=False)
    observer.start()

    try:
        async for msg in ws:
            if msg.type == web.WSMsgType.TEXT:
                print(f"Message received: {msg.data}")
                try:
                    # Parse the message as JSON
                    request_data = json.loads(msg.data)

                    # Check for 'num_records' in the request
                    num_records = request_data.get("num_records", None)
                    if num_records:
                        # If provided, send the requested number of records
                        await send_initial_data(ws, num_records)

                except json.JSONDecodeError:
                    print("Received message is not valid JSON")
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
