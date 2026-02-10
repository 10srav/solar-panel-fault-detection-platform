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
  const [diagnosticOpen, setDiagnosticOpen] = useState(false);

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

  const analyzeImage = async (inputType = 'rgb') => {
    if (!selectedFile) return;

    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('input_type', inputType);

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

      // Save to history (handle both RGB and thermal responses)
      const isThermal = response.data.input_type === 'thermal';
      const newEntry = {
        timestamp: new Date().toISOString(),
        filename: selectedFile.name,
        class_name: isThermal ? 'Thermal Fault' : response.data.prediction?.class_name || 'Unknown',
        confidence: isThermal ? null : (response.data.prediction?.confidence || 0),
        risk_level: response.data.analysis?.risk_level || 'Unknown',
        fault_area: response.data.analysis?.fault_area_percent || 0,
        severity: response.data.analysis?.severity_score || 0,
        input_type: response.data.input_type || 'rgb',
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
    setDiagnosticOpen(false);
  };

  const getRiskColor = (riskLevel) => {
    switch (riskLevel?.toLowerCase()) {
      case 'low':
        return '#10b981';
      case 'medium':
        return '#f59e0b';
      case 'high':
        return '#ef4444';
      case 'critical':
        return '#991b1b';
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
      case 'critical':
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
          onClick={() => { resetAnalysis(); setActiveTab('dashboard'); }}
        >
          📊 Dashboard
        </button>
        <button
          className={`tab ${activeTab === 'detection' ? 'active' : ''}`}
          onClick={() => { resetAnalysis(); setActiveTab('detection'); }}
        >
          🔍 Detection (RGB)
        </button>
        <button
          className={`tab ${activeTab === 'thermal' ? 'active' : ''}`}
          onClick={() => { resetAnalysis(); setActiveTab('thermal'); }}
        >
          🌡️ Thermal Segmentation
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
                        <span>{entry.confidence != null ? `${(entry.confidence * 100).toFixed(1)}% confident` : `${entry.fault_area?.toFixed(1)}% fault area`}</span>
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
                {/* Alert Banner (only for High Risk and not Clean — RGB only) */}
                {result && result.prediction && result.analysis?.risk_level === 'High' && result.prediction?.class_name !== 'Clean' && (
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

                    {result && result.prediction && (
                      <>
                        {/* Prediction Card */}
                        <div className="result-card">
                          <h3>🎯 Detection Result</h3>
                          <div className="prediction-details">
                            <div className="fault-type-display">
                              <label>Fault Type:</label>
                              <h2 className="fault-name">{result.prediction?.class_name}</h2>
                            </div>
                            <div className="confidence-display">
                              <label>Confidence Score:</label>
                              <div className="confidence-bar-container">
                                <div
                                  className="confidence-bar-fill"
                                  style={{ width: `${((result.prediction?.confidence || 0) * 100).toFixed(1)}%` }}
                                >
                                  <span className="confidence-value">
                                    {((result.prediction?.confidence || 0) * 100).toFixed(1)}%
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
                              <p className="metric-value">{result.analysis?.fault_area_percent?.toFixed(1) || '0.0'}%</p>
                              <span className="metric-source">via {result.analysis?.fault_area_source || 'grad-cam'}</span>
                            </div>
                            <div className="metric-item">
                              <label>Severity Score:</label>
                              <p className="metric-value">{result.analysis?.severity_score?.toFixed(1) || '0.0'}</p>
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
                                          background: className === result.prediction?.class_name
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
                              backgroundColor: getRiskColor(result.analysis?.risk_level),
                              borderColor: getRiskColor(result.analysis?.risk_level)
                            }}
                          >
                            {getRiskIcon(result.analysis?.risk_level)}
                            <span>{result.analysis?.risk_level || 'Unknown'} Risk</span>
                          </div>
                        </div>

                        {/* Maintenance Recommendation */}
                        <div className="result-card suggestion-card">
                          <h3>🔧 Maintenance Recommendation</h3>
                          <p className="suggestion-text">
                            {result.analysis?.maintenance_suggestion || 'No recommendation available'}
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
          <div className="detection-page">
            {!preview ? (
              <div className="upload-section">
                <div {...getRootProps()} className={`dropzone ${isDragActive ? 'active' : ''}`}>
                  <input {...getInputProps()} />
                  <div className="dropzone-content">
                    <div className="upload-icon">🌡️</div>
                    {isDragActive ? (
                      <p>Drop the thermal image here...</p>
                    ) : (
                      <>
                        <p><strong>Drag & Drop</strong> Thermal Image</p>
                        <p className="or-text">or</p>
                        <button type="button" className="browse-btn">Browse Files</button>
                        <p className="hint">Upload thermal/infrared image for advisory segmentation analysis</p>
                      </>
                    )}
                  </div>
                </div>
              </div>
            ) : (
              <div className="analysis-layout">
                {/* Two Column Layout */}
                <div className="two-column-layout">
                  {/* LEFT COLUMN: Images */}
                  <div className="left-column">
                    <div className="image-card">
                      <h3>Thermal Image</h3>
                      <img src={preview} alt="Thermal scan" className="panel-image" />
                    </div>

                    {result && result.segmentation_mask && (
                      <div className="image-card">
                        <h3>Segmentation Overlay</h3>
                        <p className="gradcam-hint">Red regions indicate predicted thermal anomalies</p>
                        <img
                          src={`data:image/png;base64,${result.segmentation_mask}`}
                          alt="Thermal segmentation"
                          className="panel-image"
                        />
                      </div>
                    )}

                    {!result && !loading && (
                      <div className="action-buttons">
                        <button onClick={() => analyzeImage('thermal')} className="analyze-btn">
                          🌡️ Analyze Thermal Image
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
                        <p>Analyzing thermal image with U-Net...</p>
                      </div>
                    )}

                    {error && (
                      <div className="error-card">
                        <h3>Analysis Failed</h3>
                        <p>{error}</p>
                        <button onClick={resetAnalysis} className="retry-btn">
                          Try Again
                        </button>
                      </div>
                    )}

                    {result && (
                      <>
                        {/* Fault Area */}
                        <div className="result-card">
                          <h3>Thermal Fault Area</h3>
                          <div className="thermal-area-display">
                            <div className="area-percentage">
                              {result.analysis.fault_area_percent.toFixed(1)}%
                            </div>
                            <p className="area-label">of panel flagged by segmentation</p>
                            <div className="area-bar-bg">
                              <div
                                className="area-bar-fill"
                                style={{
                                  width: `${Math.min(result.analysis.fault_area_percent, 100)}%`,
                                  background: result.analysis.fault_area_percent >= 12
                                    ? 'linear-gradient(90deg, #ef4444, #dc2626)'
                                    : result.analysis.fault_area_percent >= 6
                                    ? 'linear-gradient(90deg, #f59e0b, #d97706)'
                                    : result.analysis.fault_area_percent >= 3
                                    ? 'linear-gradient(90deg, #fb923c, #f97316)'
                                    : 'linear-gradient(90deg, #10b981, #059669)'
                                }}
                              />
                            </div>
                          </div>
                        </div>

                        {/* Analysis Metrics */}
                        <div className="result-card">
                          <h3>Analysis Metrics</h3>
                          <div className="metrics-grid">
                            <div className="metric-item">
                              <label>Severity Score:</label>
                              <p className="metric-value">{result.analysis.severity_score.toFixed(1)}</p>
                            </div>
                            <div className="metric-item">
                              <label>Heat Difference:</label>
                              <p className="metric-value">
                                {result.analysis.heat_difference != null
                                  ? `${result.analysis.heat_difference.toFixed(1)}`
                                  : 'N/A'}
                                {result.analysis.heat_difference != null && result.analysis.heat_difference < 10 && (
                                  <span style={{fontSize: '0.75rem', color: '#6b7280', marginLeft: '4px'}}>(low contrast)</span>
                                )}
                              </p>
                            </div>
                            <div className="metric-item">
                              <label>Image Mean Intensity:</label>
                              <p className="metric-value">
                                {result.analysis.image_mean_intensity != null
                                  ? result.analysis.image_mean_intensity.toFixed(1)
                                  : 'N/A'}
                              </p>
                            </div>
                            <div className="metric-item">
                              <label>Mask Mean Intensity:</label>
                              <p className="metric-value">
                                {result.analysis.mask_mean_intensity != null
                                  ? result.analysis.mask_mean_intensity.toFixed(1)
                                  : 'N/A'}
                              </p>
                            </div>
                          </div>
                        </div>

                        {/* Risk Badge */}
                        <div className="result-card">
                          <h3>Risk Assessment</h3>
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
                          <p className="card-hint" style={{marginTop: '8px', fontSize: '0.8rem', color: '#6b7280'}}>
                            Advisory only — verify with on-site inspection
                          </p>
                        </div>

                        {/* Maintenance Suggestion */}
                        <div className="result-card suggestion-card">
                          <h3>Maintenance Suggestion</h3>
                          <p className="suggestion-text">
                            {result.analysis.maintenance_suggestion}
                          </p>
                        </div>

                        <button onClick={resetAnalysis} className="new-analysis-btn">
                          ↺ Analyze Another Thermal Image
                        </button>
                      </>
                    )}
                  </div>
                </div>

                {/* Diagnostic Panel — collapsible, read-only, does NOT affect alerts/risk */}
                {result && result.segmentation_mask && (
                  <div className="diagnostic-section">
                    <button
                      className={`diagnostic-toggle ${diagnosticOpen ? 'open' : ''}`}
                      onClick={() => setDiagnosticOpen(!diagnosticOpen)}
                    >
                      <span className="diagnostic-toggle-icon">{diagnosticOpen ? '▾' : '▸'}</span>
                      <span>Segmentation Details (Diagnostic View)</span>
                    </button>

                    {diagnosticOpen && (
                      <div className="diagnostic-panel">
                        <div className="diagnostic-grid">
                          {/* Column 1: Original Image */}
                          <div className="diagnostic-cell">
                            <img
                              src={preview}
                              alt="Original thermal"
                              className="diagnostic-image"
                            />
                            <p className="diagnostic-caption">Original Image</p>
                          </div>

                          {/* Column 2: Ground Truth */}
                          <div className="diagnostic-cell">
                            <div className="diagnostic-placeholder">
                              <p>Ground truth not available for live inference</p>
                            </div>
                            <p className="diagnostic-caption">Ground Truth Mask</p>
                          </div>

                          {/* Column 3: Predicted Mask */}
                          <div className="diagnostic-cell">
                            <img
                              src={`data:image/png;base64,${result.segmentation_mask}`}
                              alt="Predicted mask"
                              className="diagnostic-image"
                            />
                            <p className="diagnostic-caption">Predicted Mask (U-Net)</p>
                          </div>

                          {/* Column 4: Overlay */}
                          <div className="diagnostic-cell">
                            <img
                              src={`data:image/png;base64,${result.segmentation_mask}`}
                              alt="Overlay visualization"
                              className="diagnostic-image"
                            />
                            <p className="diagnostic-caption">Overlay</p>
                          </div>
                        </div>

                        <div className="diagnostic-legend">
                          <span className="legend-item"><span className="legend-dot legend-normal"></span> Normal</span>
                          <span className="legend-item"><span className="legend-dot legend-anomaly"></span> Anomaly</span>
                          <span className="legend-item"><span className="legend-dot legend-high"></span> High-Confidence Region</span>
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </div>
            )}
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
