import React from 'react';
import UploadPDF from './UploadPDF';
import QueryForm from './QueryForm';
import './App.css';

function App() {
  return (
    <div className="min-h-screen bg-gray-100 flex flex-col items-center justify-center p-4">
      <h1 className="text-4xl font-bold text-blue-600 mb-8">QuizMaster</h1>
      <div className="w-full max-w-2xl space-y-8">
        <UploadPDF />
        <QueryForm />
      </div>
    </div>
  );
}

export default App;