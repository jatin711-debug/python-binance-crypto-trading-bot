import React, { useEffect, useState } from 'react';
import './App.css';
import TradeTable from './TradeTable';
import TradeChart from './TradeChart';

function App() {
  const [data, setData] = useState([]);

  useEffect(() => {
    const ws = new WebSocket('ws://localhost:6789/ws');

    ws.onopen = () => {
      console.log('Connected to WebSocket server');
    };

    ws.onmessage = (event) => {
      const receivedData = JSON.parse(event.data);
      setData(receivedData);
    };

    ws.onclose = () => {
      console.log('Disconnected from WebSocket server');
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };

    const keepAliveInterval = setInterval(() => {
      if (ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: 'keep-alive' }));
      }
    }, 30000); // Send a keep-alive message every 30 seconds

    return () => {
      clearInterval(keepAliveInterval);
      ws.close();
    };
  }, []);

  return (
    <div className="App">
      <header className="App-header">
        <h1>Trade Data</h1>
        <TradeTable data={data} />
        <TradeChart data={data.slice(0, 20)} />
      </header>
    </div>
  );
}

export default App;