import React, { useState } from 'react';
import axios from 'axios';

const QueryForm = () => {
  const [query, setQuery] = useState('');
  const [quizQuestions, setQuizQuestions] = useState('');

  const handleQueryChange = (e) => {
    setQuery(e.target.value);
  };

  const handleSubmit = async () => {
    if (!query.trim()) {
      setQuizQuestions('Please enter a query.');
      return;
    }

    try {
      const response = await axios.post('http://localhost:8000/query', { query });
      setQuizQuestions(response.data.quiz_questions);
    } catch (error) {
      setQuizQuestions('Error generating quiz: ' + (error.response?.data?.detail || error.message));
    }
  };

  return (
    <div className="bg-white p-6 rounded-lg shadow-md">
      <h2 className="text-2xl font-semibold mb-4">Generate Quiz</h2>
      <input
        type="text"
        value={query}
        onChange={handleQueryChange}
        placeholder="Enter your query (e.g., 'What is photosynthesis?')"
        className="w-full p-2 border rounded mb-4"
      />
      <button
        onClick={handleSubmit}
        className="bg-green-500 text-white px-4 py-2 rounded hover:bg-green-600"
      >
        Generate Quiz
      </button>
      {quizQuestions && (
        <div className="mt-4 p-4 bg-gray-50 rounded">
          <h3 className="text-lg font-medium">Quiz Questions:</h3>
          <pre className="text-sm whitespace-pre-wrap">{quizQuestions}</pre>
        </div>
      )}
    </div>
  );
};

export default QueryForm;