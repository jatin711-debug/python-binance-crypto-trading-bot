import React, { useState } from 'react';

const TradeTable = ({ data }) => {
  // Ensure data is an array before attempting to map over it
  const sortedData = Array.isArray(data) ? [...data].sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp)) : [];
  const [visibleRows, setVisibleRows] = useState(10);

  const handleLoadMore = () => {
    setVisibleRows(prevVisibleRows => prevVisibleRows + 10);
  };

  return (
    <div className="overflow-x-auto">
      <table className="min-w-full divide-y divide-gray-200">
        <thead className="bg-gray-50">
          <tr>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Timestamp</th>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Current Price</th>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">LSTM Prediction</th>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">RF Prediction</th>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Ensemble Prediction</th>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Action</th>
          </tr>
        </thead>
        <tbody className="bg-white divide-y divide-gray-200">
          {sortedData.slice(0, visibleRows).map((trade, index) => (
            <tr key={index}>
              <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{trade.timestamp}</td>
              <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{trade.current_price}</td>
              <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{trade.lstm_prediction}</td>
              <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{trade.rf_prediction}</td>
              <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{trade.ensemble_prediction}</td>
              <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{trade.action}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {visibleRows < sortedData.length && (
        <div className="flex justify-center mt-4">
          <button
            className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-700"
            onClick={handleLoadMore}
          >
            Load More
          </button>
        </div>
      )}
    </div>
  );
};

export default TradeTable;
