
 
import React, { useState } from 'react';

import axios from 'axios';
 
function App() {

  const [query, setQuery] = useState('');

  const [loading, setLoading] = useState(false);

  const [responseData, setResponseData] = useState(null);

  const [error, setError] = useState(null);
 
  const handleQuerySubmit = async () => {

    setLoading(true);

    setError(null);

    setResponseData(null);
 
    try {

      const res = await axios.post('http://localhost:5000/api/chat', {

        message: query,

      });

      setResponseData(res.data.data);

    } catch (err) {

      setError('Error: Could not connect to the backend');

    } finally {

      setLoading(false);

    }

  };
 
  return (
<div className="min-h-screen bg-gray-100 p-8">
<h1 className="text-2xl font-bold mb-4 text-center">AI Supply Chain Assistant</h1>
<div className="flex justify-center mb-4">
<input

          type="text"

          className="border p-2 w-2/3"

          placeholder="e.g., Forecast demand for household items"

          value={query}

          onChange={(e) => setQuery(e.target.value)}

        />
<button

          onClick={handleQuerySubmit}

          className="ml-2 px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
>

          Run
</button>
</div>
 
      {loading && <p className="text-center text-gray-500">Running optimization...</p>}

      {error && <p className="text-center text-red-500">{error}</p>}
 
      {responseData && (
<div className="bg-white p-6 rounded shadow">
<h2 className="text-xl font-semibold mb-2">📊 Results</h2>
 
          {responseData.messages?.map((msg, idx) => (
<p key={idx} className="mb-2 text-gray-800">➡️ {msg}</p>

          ))}
 
          <hr className="my-4" />
 
          <h3 className="font-bold text-lg">🔮 Forecast Summary</h3>
<pre className="bg-gray-100 p-2 overflow-x-auto text-sm">{JSON.stringify(responseData.forecast_results?.forecast_data?.summary, null, 2)}</pre>
 
          <h3 className="font-bold text-lg mt-4">📦 Inventory Recommendations</h3>
<pre className="bg-gray-100 p-2 overflow-x-auto text-sm">{JSON.stringify(responseData.inventory_recommendations?.optimization_results, null, 2)}</pre>
 
          <h3 className="font-bold text-lg mt-4">🚚 Route Optimization</h3>
<pre className="bg-gray-100 p-2 overflow-x-auto text-sm">{JSON.stringify(responseData.route_optimization?.route_data?.summary, null, 2)}</pre>
</div>

      )}
</div>

  );

}
 
export default App;

 
 


 