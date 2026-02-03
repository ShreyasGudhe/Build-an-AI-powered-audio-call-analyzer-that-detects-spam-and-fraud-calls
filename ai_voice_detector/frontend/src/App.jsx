import React, { useState, useEffect } from "react";

const API_BASE = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

export default function App() {
  const [file, setFile] = useState(null);
  const [callerNumber, setCallerNumber] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [analysisMode, setAnalysisMode] = useState("comprehensive");
  const [alertHistory, setAlertHistory] = useState([]);
  const [showHistory, setShowHistory] = useState(false);
  
  const [isRecording, setIsRecording] = useState(false);
  const [liveResult, setLiveResult] = useState(null);
  const [mediaRecorder, setMediaRecorder] = useState(null);

  useEffect(() => {
    if ("Notification" in window && Notification.permission === "default") {
      Notification.requestPermission();
    }
  }, []);

  const toBase64 = (blob) =>
    new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result.split(",")[1]);
      reader.onerror = reject;
      reader.readAsDataURL(blob);
    });

  const startLiveRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream);
      const chunks = [];
      
      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunks.push(e.data);
      };
      
      recorder.onstop = async () => {
        const audioBlob = new Blob(chunks, { type: 'audio/webm' });
        await analyzeLiveRecording(audioBlob);
        stream.getTracks().forEach(track => track.stop());
      };
      
      recorder.start();
      setMediaRecorder(recorder);
      setIsRecording(true);
      setLiveResult(null);
      setError(null);
    } catch (err) {
      setError("Microphone access denied. Please enable microphone permissions.");
    }
  };

  const stopLiveRecording = () => {
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
      mediaRecorder.stop();
      setIsRecording(false);
    }
  };

  const playAlertSound = () => {
    const audioContext = new (window.AudioContext || window.webkitAudioContext)();
    const oscillator = audioContext.createOscillator();
    const gainNode = audioContext.createGain();
    
    oscillator.connect(gainNode);
    gainNode.connect(audioContext.destination);
    oscillator.frequency.value = 800;
    oscillator.type = 'sine';
    gainNode.gain.value = 0.3;
    
    oscillator.start(audioContext.currentTime);
    oscillator.stop(audioContext.currentTime + 0.2);
    
    setTimeout(() => {
      const osc2 = audioContext.createOscillator();
      osc2.connect(gainNode);
      osc2.frequency.value = 600;
      osc2.start(audioContext.currentTime);
      osc2.stop(audioContext.currentTime + 0.2);
    }, 250);
  };

  const showBrowserNotification = (title, message, riskLevel) => {
    if ("Notification" in window && Notification.permission === "granted") {
      new Notification(title, {
        body: message,
        icon: riskLevel === 'CRITICAL' || riskLevel === 'HIGH' ? '🚨' : '⚠️',
        badge: '🛡️',
        tag: 'fraud-alert',
        requireInteraction: riskLevel === 'CRITICAL'
      });
    }
  };

  const analyzeLiveRecording = async (audioBlob) => {
    setLoading(true);
    try {
      const audioBase64 = await toBase64(audioBlob);
      const payload = { audio: audioBase64 };
      if (callerNumber) payload.caller_number = callerNumber;
      
      const resp = await fetch(`${API_BASE}/analyze-call`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      
      if (!resp.ok) throw new Error(`Server error: ${resp.statusText}`);
      
      const data = await resp.json();
      setLiveResult(data);
      
      const riskLevel = data.fraud_detection?.risk_level;
      const hasPatterns = data.fraud_detection?.detected_patterns?.length > 0;
      const isAI = data.voice_detection?.classification === 'AI-generated';
      const isFraud = data.fraud_detection?.classification === 'Fraud' || data.classification === 'Fraud';
      
      if (riskLevel || hasPatterns || isAI || isFraud) {
        playAlertSound();
        
        const alertTitle = riskLevel === 'CRITICAL' ? '🚨 CRITICAL FRAUD ALERT!' : 
                          riskLevel === 'HIGH' ? '⚠️ HIGH FRAUD RISK DETECTED' :
                          isFraud ? '⚠️ FRAUD DETECTED' : '⚡ FRAUD WARNING';
        
        const alertMessage = hasPatterns ? 
          `Detected: ${data.fraud_detection.detected_patterns.slice(0, 2).join(', ')}` :
          isFraud ? 'Fraud keywords detected in call' :
          isAI ? 'AI-generated voice detected' : 'Suspicious patterns found';
        
        showBrowserNotification(alertTitle, alertMessage, riskLevel);
        
        const alert = {
          time: new Date().toLocaleString(),
          number: callerNumber || 'Live Recording',
          reason: hasPatterns ? data.fraud_detection.detected_patterns.join(', ') : 
                  isFraud ? 'Fraud keywords detected' :
                  isAI ? 'AI-generated voice' : 'Suspicious activity',
          risk: riskLevel || 'HIGH'
        };
        setAlertHistory(prev => [alert, ...prev]);
      }
    } catch (err) {
      setError(err.message || "Failed to analyze recording");
    } finally {
      setLoading(false);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) {
      setError("Choose an audio file first.");
      return;
    }
    setError(null);
    setResult(null);
    setLoading(true);
    
    try {
      const audioBase64 = await toBase64(file);
      const endpoint = {
        voice: "/detect-voice",
        fraud: "/detect-fraud",
        comprehensive: "/analyze-call",
        conversation: "/analyze-conversation"
      }[analysisMode];
      
      const payload = { audio: audioBase64 };
      if (callerNumber && analysisMode !== "voice") {
        payload.caller_number = callerNumber;
      }
      
      const resp = await fetch(`${API_BASE}${endpoint}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      
      if (!resp.ok) throw new Error(`Server error ${resp.status}`);
      const data = await resp.json();
      setResult(data);
      
      if (analysisMode !== "voice") {
        const riskLevel = data.fraud_detection?.risk_level || data.risk_level;
        const hasPatterns = data.fraud_detection?.detected_patterns?.length > 0 || data.detected_patterns?.length > 0;
        const isAI = data.voice_detection?.classification === 'AI-generated' || data.classification === 'AI-generated';
        const isFraud = data.fraud_detection?.classification === 'Fraud' || data.classification === 'Fraud';
        
        if (riskLevel || hasPatterns || isAI || isFraud) {
          playAlertSound();
          
          const alertTitle = riskLevel === 'CRITICAL' ? '🚨 CRITICAL FRAUD ALERT!' : 
                            riskLevel === 'HIGH' ? '⚠️ HIGH FRAUD RISK DETECTED' :
                            isFraud ? '⚠️ FRAUD DETECTED' : '⚡ FRAUD WARNING';
          
          const patterns = data.fraud_detection?.detected_patterns || data.detected_patterns || [];
          const alertMessage = hasPatterns ? 
            `Detected: ${patterns.slice(0, 2).join(', ')}` :
            isFraud ? 'Fraud keywords detected in audio' :
            isAI ? 'AI-generated voice detected' : 'Suspicious patterns found';
          
          showBrowserNotification(alertTitle, alertMessage, riskLevel);
        }
        
        fetchAlertHistory();
      }
    } catch (err) {
      const message = err?.message || "Request failed";
      if (message.includes("Failed to fetch")) {
        setError("Could not reach the API. Ensure the FastAPI server is running at http://127.0.0.1:8000");
      } else {
        setError(message);
      }
    } finally {
      setLoading(false);
    }
  };

  const fetchAlertHistory = async () => {
    try {
      const resp = await fetch(`${API_BASE}/alert-history?limit=10`);
      if (resp.ok) {
        const data = await resp.json();
        setAlertHistory(data.alerts || []);
      }
    } catch (err) {
      console.error("Failed to fetch alert history:", err);
    }
  };

  const getRiskColor = (riskLevel) => {
    const colors = {
      CRITICAL: "#dc2626",
      HIGH: "#ea580c",
      MEDIUM: "#f59e0b",
      LOW: "#16a34a"
    };
    return colors[riskLevel] || "#6b7280";
  };

  const getRiskIcon = (riskLevel) => {
    const icons = {
      CRITICAL: "🚨",
      HIGH: "⚠️",
      MEDIUM: "⚡",
      LOW: "✓"
    };
    return icons[riskLevel] || "ℹ️";
  };

  return (
    <div className="page">
      <div className="card" style={{ maxWidth: "1000px" }}>
        <h1>🛡️ AI Call Fraud Analyzer</h1>
        <p className="muted">
          Advanced real-time fraud detection • Upload audio to analyze voice patterns, detect spam calls with AI-powered keyword detection
        </p>

        <div className="live-recording-section">
          <h3>🎙️ Live Call Recording & Analysis</h3>
          <p className="muted" style={{ marginBottom: '16px' }}>
            Record calls in real-time and get instant fraud detection, voice type identification, and language detection
          </p>
          
          <div className="live-controls">
            <input
              type="text"
              placeholder="📞 Caller Number (optional)"
              value={callerNumber}
              onChange={(e) => setCallerNumber(e.target.value)}
              className="caller-input"
              disabled={isRecording}
            />
            
            {!isRecording ? (
              <button 
                onClick={startLiveRecording} 
                className="record-btn start"
                disabled={loading}
              >
                🎙️ Start Recording
              </button>
            ) : (
              <button 
                onClick={stopLiveRecording} 
                className="record-btn stop"
              >
                ⏹️ Stop & Analyze
              </button>
            )}
          </div>
          
          {isRecording && (
            <div className="recording-indicator">
              <span className="pulse-dot"></span>
              <span>Recording in progress...</span>
            </div>
          )}
        </div>

        <div className="divider"><span>OR</span></div>

        <div className="mode-selector">
          <button
            className={`mode-btn ${analysisMode === "comprehensive" ? "active" : ""}`}
            onClick={() => setAnalysisMode("comprehensive")}
          >
            🔍 Full Analysis
          </button>
          <button
            className={`mode-btn ${analysisMode === "fraud" ? "active" : ""}`}
            onClick={() => setAnalysisMode("fraud")}
          >
            🚨 Fraud Detection
          </button>
          <button
            className={`mode-btn ${analysisMode === "voice" ? "active" : ""}`}
            onClick={() => setAnalysisMode("voice")}
          >
            🎤 Voice Analysis
          </button>
          <button
            className={`mode-btn ${analysisMode === "conversation" ? "active" : ""}`}
            onClick={() => setAnalysisMode("conversation")}
          >
            💬 Conversation (Both Sides)
          </button>
        </div>

        <form onSubmit={handleSubmit} className="form">
          <label className="file-input">
            <span>📁 {file ? file.name : 'Select Audio File (MP3, WAV, M4A...)'}</span>
            <input
              type="file"
              accept="audio/*"
              onChange={(e) => setFile(e.target.files?.[0] || null)}
            />
          </label>
          
          {analysisMode !== "voice" && (
            <input
              type="text"
              placeholder="📞 Caller Number (optional)"
              value={callerNumber}
              onChange={(e) => setCallerNumber(e.target.value)}
              className="caller-input"
            />
          )}
          
          <button type="submit" disabled={loading} className="analyze-btn">
            {loading ? "⚡ Analyzing..." : `🚀 ${analysisMode === "comprehensive" ? "Full Analysis" : analysisMode === "fraud" ? "Detect Fraud" : analysisMode === "conversation" ? "Analyze Conversation" : "Analyze Voice"}`}
          </button>
        </form>

        {error && <div className="alert error">{error}</div>}

        {liveResult && (
          <div className="live-results">
            {(liveResult.fraud_detection?.classification === 'Fraud' || 
              liveResult.classification === 'Fraud' ||
              liveResult.fraud_detection?.risk_level || 
              liveResult.fraud_detection?.detected_patterns?.length > 0 ||
              liveResult.voice_detection?.classification === 'AI-generated') && (
              <div className="immediate-alert-banner">
                <div className="alert-icon-pulse">🚨</div>
                <div className="alert-content">
                  <h3>⚠️ FRAUD ALERT DETECTED!</h3>
                  <p>
                    {(liveResult.fraud_detection?.classification === 'Fraud' || liveResult.classification === 'Fraud') && 'Fraud keywords detected • '}
                    {liveResult.fraud_detection?.risk_level && `Risk: ${liveResult.fraud_detection.risk_level} • `}
                    {liveResult.fraud_detection?.detected_patterns?.length > 0 && 
                      `Patterns: ${liveResult.fraud_detection.detected_patterns.slice(0, 2).join(', ')}`}
                    {liveResult.voice_detection?.classification === 'AI-generated' && ' • AI Voice Detected'}
                  </p>
                </div>
              </div>
            )}
            
            <h3>🎙️ Live Recording Analysis Results</h3>
            
            {liveResult.overall_assessment && (
              <div className={`assessment-banner ${liveResult.overall_assessment.is_suspicious ? "danger" : "safe"}`}>
                <h2>
                  {liveResult.overall_assessment.is_suspicious ? "⚠️ FRAUD DETECTED" : "✓ Call Appears Safe"}
                </h2>
                <p>{liveResult.overall_assessment.summary || 'Analysis complete'}</p>
              </div>
            )}

            <div className="live-result-grid">
              {liveResult.fraud_detection && (
                <div className="result-section">
                  <h4>🚨 Fraud Detection</h4>
                  <div className={`pill ${liveResult.fraud_detection.classification === 'Fraud' ? "fraud" : "safe"}`}>
                    {liveResult.fraud_detection.classification === 'Fraud' ? "⚠️ FRAUD" : "✓ SAFE"}
                  </div>
                  {liveResult.fraud_detection.risk_level && (
                    <div className={`risk-badge ${liveResult.fraud_detection.risk_level.toLowerCase()}`}>
                      {getRiskIcon(liveResult.fraud_detection.risk_level)} {liveResult.fraud_detection.risk_level}
                    </div>
                  )}
                  <div className="confidence">Confidence: {liveResult.fraud_detection.confidence}</div>
                  <p className="explanation">{liveResult.fraud_detection.explanation}</p>
                  
                  {liveResult.fraud_detection.detected_patterns?.length > 0 && (
                    <div className="detected-patterns">
                      <h4>Detected Keywords:</h4>
                      <ul>
                        {liveResult.fraud_detection.detected_patterns.map((p, i) => (
                          <li key={i}>{p}</li>
                        ))}
                      </ul>
                    </div>
                  )}
                  
                  {liveResult.fraud_detection.transcribed_text && (
                    <div className="transcript">
                      <strong>Transcript:</strong>
                      <p>{liveResult.fraud_detection.transcribed_text}</p>
                    </div>
                  )}
                </div>
              )}

              {liveResult.voice_detection && (
                <div className="result-section">
                  <h4>🎤 Voice Type</h4>
                  <div className={`pill ${liveResult.voice_detection.classification === 'AI-generated' ? "ai" : "human"}`}>
                    {liveResult.voice_detection.classification === 'AI-generated' ? "🤖 AI" : "👤 Human"}
                  </div>
                  <div className="confidence">Confidence: {liveResult.voice_detection.confidence}</div>
                </div>
              )}

              {(liveResult.transcription || liveResult.fraud_detection?.language_name) && (
                <div className="result-section">
                  <h4>🌍 Language</h4>
                  <div className="language-info">
                    <span className="language-name">
                      {liveResult.transcription?.language || liveResult.fraud_detection?.language_name || 'Unknown'}
                    </span>
                    <div className="confidence">
                      {((liveResult.transcription?.confidence || liveResult.fraud_detection?.language_confidence || 0) * 100).toFixed(0)}% confidence
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {result && (
          <div className="results-container">
            {analysisMode !== "voice" && (
              result.fraud_detection?.classification === 'Fraud' || 
              result.classification === 'Fraud' ||
              result.fraud_detection?.risk_level || 
              result.risk_level ||
              result.fraud_detection?.detected_patterns?.length > 0 ||
              result.detected_patterns?.length > 0 ||
              result.voice_detection?.classification === 'AI-generated' ||
              result.classification === 'AI-generated'
            ) && (
              <div className="immediate-alert-banner">
                <div className="alert-icon-pulse">🚨</div>
                <div className="alert-content">
                  <h3>⚠️ FRAUD ALERT DETECTED!</h3>
                  <p>
                    {(result.fraud_detection?.classification === 'Fraud' || result.classification === 'Fraud') && 'Fraud keywords detected • '}
                    {(result.fraud_detection?.risk_level || result.risk_level) && 
                      `Risk: ${result.fraud_detection?.risk_level || result.risk_level} • `}
                    {(result.fraud_detection?.detected_patterns?.length > 0 || result.detected_patterns?.length > 0) && 
                      `Keywords: ${(result.fraud_detection?.detected_patterns || result.detected_patterns)?.slice(0, 2).join(', ')}`}
                    {(result.voice_detection?.classification === 'AI-generated' || result.classification === 'AI-generated') && 
                      ' • AI Voice'}
                  </p>
                </div>
              </div>
            )}
            
            {analysisMode === "comprehensive" && result.overall_assessment && (
              <div className={`assessment-banner ${result.overall_assessment.is_suspicious ? "danger" : "safe"}`}>
                <h2>
                  {result.overall_assessment.is_suspicious ? "⚠️ SUSPICIOUS CALL" : "✓ Safe"}
                </h2>
                <p>{result.overall_assessment.summary}</p>
              </div>
            )}

            {analysisMode === "comprehensive" && result.fraud_detection && (
              <>
                <div className="result-section">
                  <h3>🚨 Fraud Analysis</h3>
                  <div className={`pill ${result.fraud_detection.classification === 'Fraud' ? "fraud" : "safe"}`}>
                    {result.fraud_detection.classification === 'Fraud' ? "⚠️ FRAUD DETECTED" : "✓ SAFE CALL"}
                  </div>
                  {result.fraud_detection.risk_level && (
                    <div className="risk-badge" style={{ backgroundColor: getRiskColor(result.fraud_detection.risk_level) }}>
                      {getRiskIcon(result.fraud_detection.risk_level)} {result.fraud_detection.risk_level}
                    </div>
                  )}
                  <div className="confidence">Confidence: {result.fraud_detection.confidence}</div>
                  <p className="explanation">{result.fraud_detection.explanation}</p>
                  
                  {result.fraud_detection.detected_patterns?.length > 0 && (
                    <div className="detected-patterns">
                      <h4>Fraud Keywords Detected:</h4>
                      <ul>
                        {result.fraud_detection.detected_patterns.map((pattern, i) => (
                          <li key={i}>{pattern}</li>
                        ))}
                      </ul>
                    </div>
                  )}
                  
                  {result.fraud_detection.transcribed_text && (
                    <div className="transcript">
                      <strong>Full Transcript:</strong>
                      <p>{result.fraud_detection.transcribed_text}</p>
                    </div>
                  )}
                </div>

                {result.voice_detection && (
                  <div className="result-section">
                    <h3>🎤 Voice Analysis</h3>
                    <div className={`pill ${result.voice_detection.classification === "AI-generated" ? "ai" : "human"}`}>
                      {result.voice_detection.classification}
                    </div>
                    <div className="confidence">Confidence: {result.voice_detection.confidence}</div>
                    <p className="explanation">{result.voice_detection.explanation}</p>
                  </div>
                )}

                {result.transcription?.text && (
                  <div className="result-section">
                    <h3>📝 Transcription</h3>
                    <div className="language-tag">
                      Language: {result.transcription.language} 
                      ({Math.round(result.transcription.confidence * 100)}%)
                    </div>
                    <div className="transcript-text">{result.transcription.text}</div>
                  </div>
                )}
              </>
            )}

            {analysisMode === "fraud" && (
              <div className="result-section">
                <div className={`pill ${result.classification === 'Fraud' ? "fraud" : "safe"}`}>
                  {result.classification === 'Fraud' ? "⚠️ FRAUD DETECTED" : "✓ SAFE CALL"}
                </div>
                {result.risk_level && (
                  <div className="risk-badge" style={{ backgroundColor: getRiskColor(result.risk_level) }}>
                    {getRiskIcon(result.risk_level)} {result.risk_level}
                  </div>
                )}
                <div className="confidence">Confidence: {result.confidence}</div>
                <p className="explanation">{result.explanation}</p>
                
                {result.detected_patterns?.length > 0 && (
                  <div className="detected-patterns">
                    <h4>Fraud Keywords Found:</h4>
                    <ul>
                      {result.detected_patterns.map((pattern, i) => (
                        <li key={i}>{pattern}</li>
                      ))}
                    </ul>
                  </div>
                )}
                
                {result.transcribed_text && (
                  <div className="transcript">
                    <strong>Transcript:</strong>
                    <p>{result.transcribed_text}</p>
                  </div>
                )}
              </div>
            )}

            {analysisMode === "conversation" && result.conversation_summary && (
              <div className="conversation-results">
                <div className={`conversation-banner ${result.conversation_summary.is_fraud ? "danger" : "safe"}`}>
                  <h2>{result.conversation_summary.formatted_summary || 
                       (result.conversation_summary.is_fraud ? "⚠️ FRAUD DETECTED IN CONVERSATION" : "✓ Conversation Appears Safe")}</h2>
                  <p className="recommendation">{result.conversation_summary.recommendation}</p>
                </div>
                
                <div className="speakers-grid">
                  {result.speaker_1 && (
                    <div className="speaker-card">
                      <h4>👤 Speaker 1 (Caller)</h4>
                      <div className="speaker-summary">{result.speaker_1.formatted_summary}</div>
                      
                      <div className="speaker-details">
                        <div className={`pill ${result.speaker_1.voice_type === 'AI-generated' ? "ai" : "human"}`}>
                          {result.speaker_1.voice_type === 'AI-generated' ? "🤖 AI Voice" : "👤 Human Voice"}
                        </div>
                        <div className={`pill ${result.speaker_1.classification === 'Fraud' ? "fraud" : "safe"}`}>
                          {result.speaker_1.classification === 'Fraud' ? "⚠️ FRAUD" : "✓ SAFE"}
                        </div>
                        {result.speaker_1.risk_level && (
                          <div className={`risk-badge ${result.speaker_1.risk_level.toLowerCase()}`}>
                            {getRiskIcon(result.speaker_1.risk_level)} {result.speaker_1.risk_level}
                          </div>
                        )}
                      </div>
                      
                      {result.speaker_1.detected_patterns && result.speaker_1.detected_patterns.length > 0 && (
                        <div className="fraud-keywords">
                          <strong>⚠️ Detected Keywords:</strong>
                          <div className="keyword-tags">
                            {result.speaker_1.detected_patterns.map((kw, i) => (
                              <span key={i} className="keyword-tag">{kw}</span>
                            ))}
                          </div>
                        </div>
                      )}
                      
                      {result.speaker_1.transcribed_text && (
                        <div className="transcript">
                          <strong>Transcript:</strong>
                          <p>{result.speaker_1.transcribed_text}</p>
                        </div>
                      )}
                      
                      {result.speaker_1.speaking_turns && (
                        <div className="speaking-stats">
                          <span>🔊 {result.speaker_1.speaking_turns} turns</span>
                          {result.speaker_1.duration_seconds && (
                            <span> • ⏱️ {result.speaker_1.duration_seconds.toFixed(1)}s</span>
                          )}
                        </div>
                      )}
                    </div>
                  )}
                  
                  {result.speaker_2 && (
                    <div className="speaker-card">
                      <h4>👤 Speaker 2 (Receiver)</h4>
                      <div className="speaker-summary">{result.speaker_2.formatted_summary}</div>
                      
                      <div className="speaker-details">
                        <div className={`pill ${result.speaker_2.voice_type === 'AI-generated' ? "ai" : "human"}`}>
                          {result.speaker_2.voice_type === 'AI-generated' ? "🤖 AI Voice" : "👤 Human Voice"}
                        </div>
                        <div className={`pill ${result.speaker_2.classification === 'Fraud' ? "fraud" : "safe"}`}>
                          {result.speaker_2.classification === 'Fraud' ? "⚠️ FRAUD" : "✓ SAFE"}
                        </div>
                        {result.speaker_2.risk_level && (
                          <div className={`risk-badge ${result.speaker_2.risk_level.toLowerCase()}`}>
                            {getRiskIcon(result.speaker_2.risk_level)} {result.speaker_2.risk_level}
                          </div>
                        )}
                      </div>
                      
                      {result.speaker_2.detected_patterns && result.speaker_2.detected_patterns.length > 0 && (
                        <div className="fraud-keywords">
                          <strong>⚠️ Detected Keywords:</strong>
                          <div className="keyword-tags">
                            {result.speaker_2.detected_patterns.map((kw, i) => (
                              <span key={i} className="keyword-tag">{kw}</span>
                            ))}
                          </div>
                        </div>
                      )}
                      
                      {result.speaker_2.transcribed_text && (
                        <div className="transcript">
                          <strong>Transcript:</strong>
                          <p>{result.speaker_2.transcribed_text}</p>
                        </div>
                      )}
                      
                      {result.speaker_2.speaking_turns && (
                        <div className="speaking-stats">
                          <span>🔊 {result.speaker_2.speaking_turns} turns</span>
                          {result.speaker_2.duration_seconds && (
                            <span> • ⏱️ {result.speaker_2.duration_seconds.toFixed(1)}s</span>
                          )}
                        </div>
                      )}
                    </div>
                  )}
                </div>
                
                {result.conversation_summary.ai_voices_detected && (
                  <div className="ai-warning">
                    ⚠️ AI-generated voice detected from: {result.conversation_summary.ai_voice_from?.map(s => s.replace('_', ' ')).join(' and ')}
                  </div>
                )}
              </div>
            )}

            {analysisMode === "voice" && (
              <div className="result-section">
                <div className={`pill ${result.classification === 'Fraud' ? "fraud" : result.classification === 'Safe Call' ? "safe" : ""}`}>
                  {result.classification}
                </div>
                <div className="confidence">Confidence: {result.confidence}</div>
                <div className="language">
                  Language: {result.language}
                  {result.language !== "Unknown" && ` (${Math.round(result.language_confidence * 100)}%)`}
                </div>
                <p className="explanation">{result.explanation}</p>
                
                {result.transcribed_text && (
                  <div className="transcript">
                    <strong>Transcript:</strong>
                    <p>{result.transcribed_text}</p>
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {analysisMode !== "voice" && (
          <div className="history-section">
            <button 
              className="history-toggle"
              onClick={() => {
                setShowHistory(!showHistory);
                if (!showHistory) fetchAlertHistory();
              }}
            >
              {showHistory ? "Hide" : "Show"} Recent Alerts ({alertHistory.length})
            </button>
            
            {showHistory && alertHistory.length > 0 && (
              <div className="alert-history">
                {alertHistory.map((alert, i) => (
                  <div key={i} className="history-item">
                    <div className="history-header">
                      <span className="history-risk" style={{ color: getRiskColor(alert.risk || alert.risk_level) }}>
                        {getRiskIcon(alert.risk || alert.risk_level)} {alert.risk || alert.risk_level}
                      </span>
                      <span className="history-time">{alert.time}</span>
                    </div>
                    <div className="history-details">
                      {alert.number && <span className="history-number">{alert.number}</span>}
                      <span className="history-reason">{alert.reason}</span>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
