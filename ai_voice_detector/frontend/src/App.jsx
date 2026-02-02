import React, { useState } from "react";

// Point to the running backend; adjust port if the API runs elsewhere.
const API_BASE = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";
const API_URL = API_BASE.endsWith("/detect-voice")
  ? API_BASE
  : `${API_BASE.replace(/\/$/, "")}/detect-voice`;

export default function App() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const toBase64 = (blob) =>
    new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result.split(",")[1]);
      reader.onerror = reject;
      reader.readAsDataURL(blob);
    });

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) {
      setError("Choose an MP3 file first.");
      return;
    }
    setError(null);
    setResult(null);
    setLoading(true);
    try {
      const audioBase64 = await toBase64(file);
      const resp = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ audio: audioBase64 }),
      });
      if (!resp.ok) throw new Error(`Server error ${resp.status}`);
      const data = await resp.json();
      setResult(data);
    } catch (err) {
      const message = err?.message || "Request failed";
      if (message.includes("Failed to fetch")) {
        setError(
          "Could not reach the API. Ensure the FastAPI server is running at http://127.0.0.1:8000 and try again."
        );
      } else {
        setError(message);
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="page">
      <div className="card">
        <h1>AI Voice Detector</h1>
        <p className="muted">Upload an MP3; the API will classify it.</p>
        <form onSubmit={handleSubmit} className="form">
          <label className="file-input">
            <span>Select audio (MP3)</span>
            <input
              type="file"
              accept="audio/mpeg, audio/mp3"
              onChange={(e) => setFile(e.target.files?.[0] || null)}
            />
          </label>
          <button type="submit" disabled={loading}>
            {loading ? "Analyzing..." : "Detect Voice"}
          </button>
        </form>
        {error && <div className="alert error">{error}</div>}
        {result && (
          <div className="result">
            <div className="pill">{result.classification}</div>
            <div className="confidence">Confidence: {result.confidence}</div>
            <div className="language">
              Language: {result.language} 
              {result.language !== "Unknown" && ` (${(result.language_confidence * 100).toFixed(0)}%)`}
              {result.language === "Unknown" && " - Not in supported languages (en, hi, ta, te, ml)"}
            </div>
            <div className="explanation">{result.explanation}</div>
          </div>
        )}
      </div>
    </div>
  );
}
