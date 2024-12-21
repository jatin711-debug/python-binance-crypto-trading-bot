import React from 'react';
import { Line } from 'react-chartjs-2';
import { Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend } from 'chart.js';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend);

const TradeChart = ({ data }) => {
    // Prepare data for the chart
    const chartData = {
        labels: data.map(trade => trade.timestamp).reverse(),
        datasets: [
            {
                label: 'Current Price',
                data: data.map(trade => trade.current_price).reverse(),
                borderColor: 'rgba(75, 192, 192, 1)',
                backgroundColor: 'rgba(75, 192, 192, 0.2)',
                fill: true,
            },
            {
                label: 'LSTM Prediction',
                data: data.map(trade => trade.lstm_prediction).reverse(),
                borderColor: 'rgba(255, 99, 132, 1)',
                backgroundColor: 'rgba(255, 99, 132, 0.2)',
                fill: true,
            },
            {
                label: 'RF Prediction',
                data: data.map(trade => trade.rf_prediction).reverse(),
                borderColor: 'rgba(54, 162, 235, 1)',
                backgroundColor: 'rgba(54, 162, 235, 0.2)',
                fill: true,
            },
            {
                label: 'Ensemble Prediction',
                data: data.map(trade => trade.ensemble_prediction).reverse(),
                borderColor: 'rgba(255, 206, 86, 1)',
                backgroundColor: 'rgba(255, 206, 86, 0.2)',
                fill: true,
            },
        ],
    };

    const options = {
        responsive: true,
        plugins: {
            legend: {
                position: 'top',
            },
            title: {
                display: true,
                text: 'Trade Data Over Time',
            },
        },
    };

    return <Line data={chartData} options={options} />;
};

export default TradeChart;