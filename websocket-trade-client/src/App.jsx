import React, { useEffect, useState } from 'react';
import './App.css';

function App() {
  const [data, setData] = useState([]);

  useEffect(() => {
    const ws = new WebSocket('ws://localhost:6789/ws');

    ws.onopen = () => {
      console.log('Connected to WebSocket server');
    };

    ws.onmessage = (event) => {
      const receivedData = JSON.parse(event.data);
      setData(receivedData.data); // assuming the format is {"data": [...]}
    };

    ws.onclose = () => {
      console.log('Disconnected from WebSocket server');
    };

    return () => {
      ws.close();
    };
  }, []);

  return (
    <div className="App">
      <header className="App-header">
        <h1>Trade Data</h1>
        <ul>
          {data.map((trade, index) => (
            <li key={index}>
              <p>Timestamp: {trade.timestamp}</p>
              <p>Current Price: {trade.current_price}</p>
              <p>LSTM Prediction: {trade.lstm_prediction}</p>
              <p>RF Prediction: {trade.rf_prediction}</p>
              <p>Ensemble Prediction: {trade.ensemble_prediction}</p>
              <p>Action: {trade.action}</p>
            </li>
          ))}
        </ul>
      </header>
    </div>
  );
}

export default App;