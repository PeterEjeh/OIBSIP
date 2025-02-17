import { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { ArrowUp, ArrowDown, RefreshCcw } from 'lucide-react';

const StockPredictionDashboard = () => {
  const [predictions, setPredictions] = useState([]);
  const [stockData, setStockData] = useState({
    open_value: '',
    high_value: '',
    low_value: '',
    turnover: '',
    change_prev_close_percentage: '',
    last_value: '',
    symbol: 'CBX'
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [latestPrediction, setLatestPrediction] = useState(null);

  // Fetch prediction history
  const fetchPredictionHistory = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/predictor/history/');
      const data = await response.json();
      setPredictions(data.reverse()); // Reverse to show newest first
    } catch (err) {
      setError('Failed to fetch prediction history');
    }
  };

  useEffect(() => {
    fetchPredictionHistory();
  }, []);

  // Handle form submission
  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    try {
      const response = await fetch('http://localhost:8000/api/predictor/predict/', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(stockData),
      });

      if (!response.ok) throw new Error('Prediction request failed');

      const data = await response.json();
      setLatestPrediction(data);
      fetchPredictionHistory();
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  // Handle input changes
  const handleChange = (e) => {
    setStockData({
      ...stockData,
      [e.target.name]: e.target.value
    });
  };

  return (
    <div className="min-h-screen bg-gray-50 p-8">
      <div className="max-w-6xl mx-auto">
        <h1 className="text-3xl font-bold mb-8">Stock Market Forecast</h1>
        
        {/* Input Form */}
        <div className="bg-white rounded-lg shadow p-6 mb-8">
          <form onSubmit={handleSubmit} className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Symbol</label>
              <input
                type="text"
                name="symbol"
                value={stockData.symbol}
                onChange={handleChange}
                className="w-full p-2 border rounded"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Open Value</label>
              <input
                type="number"
                name="open_value"
                value={stockData.open_value}
                onChange={handleChange}
                className="w-full p-2 border rounded"
                step="0.01"
                required
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">High Value</label>
              <input
                type="number"
                name="high_value"
                value={stockData.high_value}
                onChange={handleChange}
                className="w-full p-2 border rounded"
                step="0.01"
                required
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Low Value</label>
              <input
                type="number"
                name="low_value"
                value={stockData.low_value}
                onChange={handleChange}
                className="w-full p-2 border rounded"
                step="0.01"
                required
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Turnover</label>
              <input
                type="number"
                name="turnover"
                value={stockData.turnover}
                onChange={handleChange}
                className="w-full p-2 border rounded"
                required
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Last Value</label>
              <input
                type="number"
                name="last_value"
                value={stockData.last_value}
                onChange={handleChange}
                className="w-full p-2 border rounded"
                step="0.01"
                required
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Change %</label>
              <input
                type="number"
                name="change_prev_close_percentage"
                value={stockData.change_prev_close_percentage}
                onChange={handleChange}
                className="w-full p-2 border rounded"
                step="0.01"
                required
              />
            </div>
            <div className="md:col-span-3">
              <button
                type="submit"
                disabled={loading}
                className="w-full bg-blue-600 text-white py-2 px-4 rounded hover:bg-blue-700 disabled:bg-blue-300"
              >
                {loading ? (
                  <RefreshCcw className="animate-spin inline-block mr-2" size={20} />
                ) : 'Get Prediction'}
              </button>
            </div>
          </form>
        </div>

        {/* Latest Prediction */}
        {latestPrediction && (
          <div className="bg-white rounded-lg shadow p-6 mb-8">
            <h2 className="text-xl font-semibold mb-4">Latest Prediction</h2>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="p-4 bg-gray-50 rounded">
                <div className="text-sm text-gray-600">Current Price</div>
                <div className="text-2xl font-bold">${latestPrediction.current_price.toFixed(2)}</div>
              </div>
              <div className="p-4 bg-gray-50 rounded">
                <div className="text-sm text-gray-600">Predicted Price</div>
                <div className="text-2xl font-bold">${latestPrediction.predicted_price.toFixed(2)}</div>
              </div>
              <div className="p-4 bg-gray-50 rounded">
                <div className="text-sm text-gray-600">Change</div>
                <div className="text-2xl font-bold flex items-center">
                  {latestPrediction.predicted_price > latestPrediction.current_price ? (
                    <ArrowUp className="text-green-500 mr-1" />
                  ) : (
                    <ArrowDown className="text-red-500 mr-1" />
                  )}
                  {Math.abs(((latestPrediction.predicted_price - latestPrediction.current_price) / latestPrediction.current_price) * 100).toFixed(2)}%
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Error Display */}
        {error && (
          <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded mb-8">
            {error}
          </div>
        )}

        {/* Prediction History Chart */}
        {predictions.length > 0 && (
          <div className="bg-white rounded-lg shadow p-6">
            <h2 className="text-xl font-semibold mb-4">Prediction History</h2>
            <div className="h-96">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={predictions}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="date" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Line type="monotone" dataKey="actual_price" stroke="#2563eb" name="Actual Price" />
                  <Line type="monotone" dataKey="predicted_price" stroke="#16a34a" name="Predicted Price" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default StockPredictionDashboard;