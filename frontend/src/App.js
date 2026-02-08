import React, { useState, useCallback, useEffect } from 'react';
import { useDropzone } from 'react-dropzone';
import axios from 'axios';
import './App.css';

function App() {
  const [activeTab, setActiveTab] = useState('detection');
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [history, setHistory] = useState([]);
  const [stats, setStats] = useState({
    totalPredictions: 0,
    highRiskCount: 0,
  });

  // Load history from localStorage on mount
  useEffect(() => {
    const saved = localStorage.getItem('predictionHistory');
    if (saved) {
      try {
        const parsed = JSON.parse(saved);
        setHistory(parsed);
        setStats({
          totalPredictions: parsed.length,
          highRiskCount: parsed.filter(p => p.risk_level === 'High').length,
        });
      } catch (e) {
        console.error('Failed to parse history:', e);
      }
    }
  }, []);

  const onDrop = useCallback(acceptedFiles => {
    const file = acceptedFiles[0];
    if (file) {
      setSelectedFile(file);
      setPreview(URL.createObjectURL(file));
      setResult(null);
      setError(null);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.jpeg', '.jpg', '.png', '.bmp', '.tiff', '.webp']
    },
    multiple: false
  });

  const analyzeImage = async () => {
    if (!selectedFile) return;

    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await axios.post('/analyze', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      console.log('✅ API RESPONSE:', response.data);
      console.log('📊 Prediction:', response.data.prediction);
      console.log('📈 Analysis:', response.data.analysis);
      console.log('🖼️ Grad-CAM present:', !!response.data.gradcam_image);

      setResult(response.data);

      // Save to history
      const newEntry = {
        timestamp: new Date().toISOString(),
        filename: selectedFile.name,
        class_name: response.data.prediction.class_name,
        confidence: response.data.prediction.confidence,
        risk_level: response.data.analysis.risk_level,
        fault_area: response.data.analysis.fault_area_percent,
        severity: response.data.analysis.severity_score,
      };

      const updatedHistory = [newEntry, ...history].slice(0, 50); // Keep last 50
      setHistory(updatedHistory);
      localStorage.setItem('predictionHistory', JSON.stringify(updatedHistory));

      // Update stats
      setStats({
        totalPredictions: updatedHistory.length,
        highRiskCount: updatedHistory.filter(p => p.risk_level === 'High').length,
      });
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to analyze image. Please try again.');
      console.error('❌ ERROR:', err);
      console.error('Response:', err.response?.data);
    } finally {
      setLoading(false);
    }
  };

  const resetAnalysis = () => {
    setSelectedFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
  };

  const getRiskColor = (riskLevel) => {
    switch (riskLevel?.toLowerCase()) {
      case 'low':
        return '#10b981';
      case 'medium':
        return '#f59e0b';
      case 'high':
        return '#ef4444';
      default:
        return '#6b7280';
    }
  };

  const getRiskIcon = (riskLevel) => {
    switch (riskLevel?.toLowerCase()) {
      case 'low':
        return '🟢';
      case 'medium':
        return '🟡';
      case 'high':
        return '🔴';
      default:
        return '⚪';
    }
  };

  return (
    <div className="App">
      {/* Header */}
      <header className="header">
        <div className="header-content">
          <div className="header-title">
            <h1>🌞 Solar Panel Fault Detection System</h1>
            <p>AI-Powered Monitoring & Maintenance Assistant</p>
          </div>
          <div className="header-stats">
            <div className="stat-chip">
              <span className="stat-label">Model Accuracy</span>
              <span className="stat-value">93.38%</span>
            </div>
          </div>
        </div>
      </header>

      {/* Navigation Tabs */}
      <nav className="nav-tabs">
        <button
          className={`tab ${activeTab === 'dashboard' ? 'active' : ''}`}
          onClick={() => setActiveTab('dashboard')}
        >
          📊 Dashboard
        </button>
        <button
          className={`tab ${activeTab === 'detection' ? 'active' : ''}`}
          onClick={() => setActiveTab('detection')}
        >
          🔍 Detection (RGB)
        </button>
        <button
          className="tab disabled"
          disabled
        >
          🌡️ Thermal (Coming Soon)
        </button>
      </nav>

      {/* Main Content */}
      <main className="main-content">
        {activeTab === 'dashboard' && (
          <div className="dashboard-page">
            <h2>System Overview</h2>
            <div className="dashboard-grid">
              <div className="dashboard-card">
                <div className="card-icon">📈</div>
                <div className="card-content">
                  <h3>{stats.totalPredictions}</h3>
                  <p>Total Predictions</p>
                </div>
              </div>
              <div className="dashboard-card">
                <div className="card-icon">🎯</div>
                <div className="card-content">
                  <h3>93.38%</h3>
                  <p>Model Accuracy</p>
                </div>
              </div>
              <div className="dashboard-card">
                <div className="card-icon">🏷️</div>
                <div className="card-content">
                  <h3>6</h3>
                  <p>Fault Types</p>
                </div>
              </div>
              <div className="dashboard-card alert-card">
                <div className="card-icon">🔴</div>
                <div className="card-content">
                  <h3>{stats.highRiskCount}</h3>
                  <p>High Risk Alerts</p>
                </div>
              </div>
            </div>

            <div className="info-section">
              <h3>Detected Fault Classes</h3>
              <div className="fault-classes">
                <span className="fault-badge">Bird Droppings</span>
                <span className="fault-badge">Clean Panel</span>
                <span className="fault-badge">Dusty</span>
                <span className="fault-badge">Electrical Damage</span>
                <span className="fault-badge">Physical Damage</span>
                <span className="fault-badge">Snow Covered</span>
              </div>
            </div>

            {history.length > 0 && (
              <div className="info-section">
                <h3>Recent Predictions</h3>
                <div className="history-list">
                  {history.slice(0, 10).map((entry, idx) => (
                    <div key={idx} className="history-item">
                      <div className="history-item-header">
                        <span className="history-filename">{entry.filename}</span>
                        <span className={`history-risk ${entry.risk_level.toLowerCase()}`}>
                          {getRiskIcon(entry.risk_level)} {entry.risk_level}
                        </span>
                      </div>
                      <div className="history-item-details">
                        <span>{entry.class_name}</span>
                        <span>•</span>
                        <span>{(entry.confidence * 100).toFixed(1)}% confident</span>
                        <span>•</span>
                        <span>{new Date(entry.timestamp).toLocaleDateString()}</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {activeTab === 'detection' && (
          <div className="detection-page">
            {!preview ? (
              <div className="upload-section">
                <div {...getRootProps()} className={`dropzone ${isDragActive ? 'active' : ''}`}>
                  <input {...getInputProps()} />
                  <div className="dropzone-content">
                    <div className="upload-icon">📤</div>
                    {isDragActive ? (
                      <p>Drop the solar panel image here...</p>
                    ) : (
                      <>
                        <p><strong>Drag & Drop</strong> Solar Panel Image</p>
                        <p className="or-text">or</p>
                        <button type="button" className="browse-btn">Browse Files</button>
                        <p className="hint">Supported: JPG, PNG, BMP, TIFF, WEBP (Max 20MB)</p>
                      </>
                    )}
                  </div>
                </div>
              </div>
            ) : (
              <div className="analysis-layout">
                {/* Alert Banner (only for High Risk and not Clean) */}
                {result && result.analysis.risk_level === 'High' && result.prediction.class_name !== 'Clean' && (
                  <div className="alert-banner">
                    <span className="alert-icon">⚠️</span>
                    <div className="alert-content">
                      <strong>CRITICAL FAULT DETECTED</strong>
                      <p>{result.alert.message}</p>
                    </div>
                  </div>
                )}

                {/* Two Column Layout */}
                <div className="two-column-layout">
                  {/* LEFT COLUMN: Images */}
                  <div className="left-column">
                    <div className="image-card">
                      <h3>Original Image</h3>
                      <img src={preview} alt="Solar panel" className="panel-image" />
                    </div>

                    {result && result.gradcam_image && (
                      <div className="image-card">
                        <h3>Grad-CAM Visualization</h3>
                        <p className="gradcam-hint">Heatmap showing AI's attention areas</p>
                        <img
                          src={`data:image/png;base64,${result.gradcam_image}`}
                          alt="Grad-CAM heatmap"
                          className="panel-image"
                        />
                        <div className="heatmap-legend">
                          <span>🔵 Low Attention</span>
                          <span>🟡 Medium</span>
                          <span>🔴 High Attention</span>
                        </div>
                      </div>
                    )}

                    {!result && !loading && (
                      <div className="action-buttons">
                        <button onClick={analyzeImage} className="analyze-btn">
                          🔍 Analyze Panel
                        </button>
                        <button onClick={resetAnalysis} className="secondary-btn">
                          ↺ Upload New
                        </button>
                      </div>
                    )}
                  </div>

                  {/* RIGHT COLUMN: Results */}
                  <div className="right-column">
                    {loading && (
                      <div className="loading-card">
                        <div className="spinner"></div>
                        <p>Analyzing image with AI...</p>
                      </div>
                    )}

                    {error && (
                      <div className="error-card">
                        <h3>❌ Analysis Failed</h3>
                        <p>{error}</p>
                        <button onClick={resetAnalysis} className="retry-btn">
                          Try Again
                        </button>
                      </div>
                    )}

                    {result && (
                      <>
                        {/* Prediction Card */}
                        <div className="result-card">
                          <h3>🎯 Detection Result</h3>
                          <div className="prediction-details">
                            <div className="fault-type-display">
                              <label>Fault Type:</label>
                              <h2 className="fault-name">{result.prediction.class_name}</h2>
                            </div>
                            <div className="confidence-display">
                              <label>Confidence Score:</label>
                              <div className="confidence-bar-container">
                                <div
                                  className="confidence-bar-fill"
                                  style={{ width: `${(result.prediction.confidence * 100).toFixed(1)}%` }}
                                >
                                  <span className="confidence-value">
                                    {(result.prediction.confidence * 100).toFixed(1)}%
                                  </span>
                                </div>
                              </div>
                            </div>
                          </div>
                        </div>

                        {/* Analysis Metrics */}
                        <div className="result-card">
                          <h3>📊 Analysis Metrics</h3>
                          <div className="metrics-grid">
                            <div className="metric-item">
                              <label>Fault Area:</label>
                              <p className="metric-value">{result.analysis.fault_area_percent.toFixed(1)}%</p>
                              <span className="metric-source">via {result.analysis.fault_area_source || 'grad-cam'}</span>
                            </div>
                            <div className="metric-item">
                              <label>Severity Score:</label>
                              <p className="metric-value">{result.analysis.severity_score.toFixed(1)}</p>
                            </div>
                          </div>
                        </div>

                        {/* Classification Breakdown */}
                        {result.class_probabilities && (
                          <div className="result-card">
                            <h3>📈 Classification Breakdown</h3>
                            <p className="card-hint">Probability distribution across all fault types</p>
                            <div className="probability-bars">
                              {Object.entries(result.class_probabilities)
                                .sort(([, a], [, b]) => b - a)
                                .map(([className, prob]) => (
                                  <div key={className} className="prob-bar-item">
                                    <div className="prob-label">
                                      <span className="prob-class-name">
                                        {className.replace(/_/g, ' ').replace(' generateds', '').replace(' generated', '')}
                                      </span>
                                      <span className="prob-value">{(prob * 100).toFixed(1)}%</span>
                                    </div>
                                    <div className="prob-bar-bg">
                                      <div
                                        className="prob-bar-fill"
                                        style={{
                                          width: `${(prob * 100).toFixed(1)}%`,
                                          background: className === result.prediction.class_name
                                            ? 'linear-gradient(90deg, #3b82f6, #2563eb)'
                                            : '#cbd5e1'
                                        }}
                                      />
                                    </div>
                                  </div>
                                ))}
                            </div>
                          </div>
                        )}

                        {/* Segmentation Mask (when available) */}
                        {result.segmentation_mask && result.segmentation_available && (
                          <div className="result-card">
                            <h3>🎯 Pixel-Level Segmentation</h3>
                            <p className="card-hint">Precise fault localization from U-Net model</p>
                            <img
                              src={`data:image/png;base64,${result.segmentation_mask}`}
                              alt="Segmentation mask"
                              className="mask-image"
                            />
                          </div>
                        )}

                        {/* Risk Level */}
                        <div className="result-card">
                          <h3>⚠️ Risk Assessment</h3>
                          <div
                            className="risk-badge-large"
                            style={{
                              backgroundColor: getRiskColor(result.analysis.risk_level),
                              borderColor: getRiskColor(result.analysis.risk_level)
                            }}
                          >
                            {getRiskIcon(result.analysis.risk_level)}
                            <span>{result.analysis.risk_level} Risk</span>
                          </div>
                        </div>

                        {/* Maintenance Recommendation */}
                        <div className="result-card suggestion-card">
                          <h3>🔧 Maintenance Recommendation</h3>
                          <p className="suggestion-text">
                            {result.analysis.maintenance_suggestion}
                          </p>
                        </div>

                        <button onClick={resetAnalysis} className="new-analysis-btn">
                          ↺ Analyze Another Panel
                        </button>
                      </>
                    )}
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {activeTab === 'thermal' && (
          <div className="thermal-page">
            <div className="placeholder-card">
              <div className="placeholder-icon">🌡️</div>
              <h2>Thermal Imaging Module</h2>
              <p>Thermal segmentation module will be available in the next version.</p>
              <p className="placeholder-hint">
                This feature will enable infrared thermal analysis for advanced fault detection.
              </p>
            </div>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="footer">
        <p>Powered by ResNet18 + Grad-CAM • 93.38% Validation Accuracy • Real-time Analysis</p>
      </footer>
    </div>
  );
}

export default App;
