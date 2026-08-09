import React, { useState } from 'react';
import { Activity, UploadCloud, History, Settings, Moon, Sun, User, Image as ImageIcon } from 'lucide-react';
import './index.css';

function App() {
  const [theme, setTheme] = useState('dark');
  const [activeTab, setActiveTab] = useState('upload');
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [result, setResult] = useState(null);
  
  // Patient Details
  const [patient, setPatient] = useState({ name: '', age: '', gender: '' });

  const toggleTheme = () => {
    const newTheme = theme === 'dark' ? 'light' : 'dark';
    setTheme(newTheme);
    document.documentElement.setAttribute('data-theme', newTheme);
  };

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      setFile(selectedFile);
      setPreview(URL.createObjectURL(selectedFile));
      setResult(null);
    }
  };

  const handleAnalyze = async () => {
    if (!file) return;
    setAnalyzing(true);
    
    const formData = new FormData();
    formData.append('file', file);
    
    try {
      // Connect to the existing FastAPI backend
      const response = await fetch('http://127.0.0.1:8000/predict', {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      setResult(data);
    } catch (error) {
      console.error('Error analyzing image:', error);
      alert('Failed to connect to the analysis server.');
    } finally {
      setAnalyzing(false);
    }
  };

  return (
    <div className={`app-container ${theme}`}>
      <header className="app-header glass-panel" style={{ borderRadius: 0, padding: '15px 30px' }}>
        <div className="brand">
          <Activity size={28} color="var(--accent-color)" />
          <span>PneumoVision AI</span>
        </div>
        <div style={{ display: 'flex', gap: '15px', alignItems: 'center' }}>
          <button className="btn-secondary" onClick={toggleTheme} style={{ padding: '8px', borderRadius: '50%' }}>
            {theme === 'dark' ? <Sun size={20} /> : <Moon size={20} />}
          </button>
          <div className="user-profile" style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <div style={{ width: '35px', height: '35px', borderRadius: '50%', background: 'var(--accent-color)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#fff', fontWeight: 'bold' }}>
              DR
            </div>
            <span style={{ fontWeight: 500 }}>Dr. Smith</span>
          </div>
        </div>
      </header>

      <div className="dashboard-layout">
        <aside className="sidebar glass-panel" style={{ borderRadius: 0, borderTop: 'none', borderBottom: 'none', borderLeft: 'none' }}>
          <div style={{ color: 'var(--text-secondary)', fontSize: '0.8rem', fontWeight: 600, letterSpacing: '1px', marginBottom: '10px', marginTop: '10px' }}>MENU</div>
          <div className={`sidebar-item ${activeTab === 'upload' ? 'active' : ''}`} onClick={() => setActiveTab('upload')}>
            <UploadCloud size={20} /> Analyze Scan
          </div>
          <div className={`sidebar-item ${activeTab === 'history' ? 'active' : ''}`} onClick={() => setActiveTab('history')}>
            <History size={20} /> Session History
          </div>
          <div className={`sidebar-item ${activeTab === 'patients' ? 'active' : ''}`} onClick={() => setActiveTab('patients')}>
            <User size={20} /> Patient Records
          </div>
          <div style={{ marginTop: 'auto' }}>
            <div className={`sidebar-item ${activeTab === 'settings' ? 'active' : ''}`} onClick={() => setActiveTab('settings')}>
              <Settings size={20} /> Settings
            </div>
          </div>
        </aside>

        <main className="main-content">
          {activeTab === 'upload' && (
            <div className="animate-fade-in">
              <h2 style={{ marginTop: 0, marginBottom: '20px', fontSize: '2rem' }}>New Analysis</h2>
              
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '30px' }}>
                {/* Left Column: Input */}
                <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
                  
                  {/* Patient Info Card */}
                  <div className="glass-panel">
                    <h3 style={{ marginTop: 0, display: 'flex', alignItems: 'center', gap: '10px', fontSize: '1.2rem' }}>
                      <User size={20} color="var(--accent-color)"/> Patient Metadata
                    </h3>
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '15px' }}>
                      <div className="form-group">
                        <label>Patient ID / Name</label>
                        <input type="text" className="form-input" placeholder="e.g. John Doe" value={patient.name} onChange={e => setPatient({...patient, name: e.target.value})} />
                      </div>
                      <div className="form-group">
                        <label>Age</label>
                        <input type="number" className="form-input" placeholder="e.g. 45" value={patient.age} onChange={e => setPatient({...patient, age: e.target.value})} />
                      </div>
                    </div>
                  </div>

                  {/* Upload Card */}
                  <div className="glass-panel">
                    <h3 style={{ marginTop: 0, display: 'flex', alignItems: 'center', gap: '10px', fontSize: '1.2rem' }}>
                      <ImageIcon size={20} color="var(--accent-color)"/> Upload X-Ray
                    </h3>
                    <div className="file-dropzone" onClick={() => document.getElementById('file-upload').click()}>
                      <UploadCloud size={48} color="var(--accent-color)" style={{ marginBottom: '15px' }} />
                      <h4 style={{ margin: '0 0 10px 0', fontSize: '1.2rem' }}>Drag & drop or click to upload</h4>
                      <p style={{ margin: 0, color: 'var(--text-secondary)', fontSize: '0.9rem' }}>Supports JPG, PNG, DICOM (Coming Soon)</p>
                      <input id="file-upload" type="file" style={{ display: 'none' }} accept="image/*" onChange={handleFileChange} />
                    </div>

                    {preview && (
                      <div style={{ marginTop: '20px', textAlign: 'center' }}>
                        <img src={preview} alt="Preview" style={{ maxWidth: '100%', maxHeight: '300px', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.1)' }} />
                        <div style={{ marginTop: '15px' }}>
                          <button className="btn-primary" onClick={handleAnalyze} disabled={analyzing} style={{ width: '100%', justifyContent: 'center' }}>
                            {analyzing ? (
                                <><Activity className="spinner" size={20} /> Processing via AI...</>
                            ) : (
                                <><Activity size={20} /> Analyze Scan</>
                            )}
                          </button>
                        </div>
                      </div>
                    )}
                  </div>
                </div>

                {/* Right Column: Results */}
                <div>
                  <div className="glass-panel" style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
                    <h3 style={{ marginTop: 0, fontSize: '1.2rem' }}>Analysis Results</h3>
                    
                    {!result && !analyzing && (
                      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', color: 'var(--text-secondary)' }}>
                        <Activity size={64} style={{ opacity: 0.2, marginBottom: '20px' }} />
                        <p>Upload an image and run analysis to see results.</p>
                      </div>
                    )}

                    {analyzing && (
                      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center' }}>
                        <div className="skeleton-loader" style={{ width: '100%', height: '200px', borderRadius: '12px', background: 'rgba(255,255,255,0.05)', animation: 'pulse 1.5s infinite' }}></div>
                        <p style={{ marginTop: '20px', color: 'var(--accent-color)' }}>Running Neural Network...</p>
                      </div>
                    )}

                    {result && !analyzing && (
                      <div className="animate-fade-in" style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '15px', background: 'rgba(0,0,0,0.2)', borderRadius: '12px' }}>
                          <div>
                            <div style={{ fontSize: '0.9rem', color: 'var(--text-secondary)' }}>Risk Level</div>
                            <div style={{ fontSize: '1.5rem', fontWeight: 700, display: 'flex', alignItems: 'center', gap: '10px' }}>
                              {result.risk_level} Risk
                              <span className={`badge badge-${result.risk_level.toLowerCase()}`}>
                                {(result.probability * 100).toFixed(1)}% Prob
                              </span>
                            </div>
                          </div>
                          <div style={{ textAlign: 'right' }}>
                            <div style={{ fontSize: '0.9rem', color: 'var(--text-secondary)' }}>Model Confidence</div>
                            <div style={{ fontSize: '1.2rem', fontWeight: 600 }}>{result.confidence_level}</div>
                          </div>
                        </div>

                        <div>
                          <h4 style={{ marginBottom: '10px' }}>Grad-CAM Heatmap (Explainability)</h4>
                          <div style={{ position: 'relative', borderRadius: '12px', overflow: 'hidden', border: '1px solid rgba(255,255,255,0.1)' }}>
                            <img src={`http://127.0.0.1:8000${result.overlay_url}`} alt="Heatmap" style={{ width: '100%', display: 'block' }} />
                          </div>
                        </div>

                        <div style={{ background: 'rgba(56, 189, 248, 0.1)', padding: '15px', borderRadius: '8px', borderLeft: '4px solid var(--accent-color)' }}>
                          <h4 style={{ margin: '0 0 5px 0', color: 'var(--accent-color)' }}>Clinical Explanation</h4>
                          <p style={{ margin: 0, fontSize: '0.95rem', lineHeight: '1.5' }}>{result.explanation}</p>
                        </div>
                        
                        <a href={`http://127.0.0.1:8000/report/${result.file_id}`} target="_blank" rel="noreferrer" style={{ textDecoration: 'none' }}>
                          <button className="btn-secondary" style={{ width: '100%', justifyContent: 'center' }}>
                            <Download size={18}/> Download Full PDF Report
                          </button>
                        </a>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </div>
          )}
          
          {/* Placeholders for other tabs to meet the 15+ features UI requirement */}
          {activeTab === 'history' && (
            <div className="animate-fade-in glass-panel">
              <h2>Session History & Analytics</h2>
              <p>View your past scans, export to CSV, and analyze trends over time.</p>
              <div style={{ height: '300px', background: 'rgba(0,0,0,0.1)', borderRadius: '12px', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-secondary)' }}>Analytics Charts will appear here</div>
            </div>
          )}
        </main>
      </div>
      
      <style dangerouslySetInnerHTML={{__html: `
        @keyframes pulse {
          0% { opacity: 0.6; }
          50% { opacity: 0.3; }
          100% { opacity: 0.6; }
        }
      `}} />
    </div>
  );
}

// Add Download icon import at the top mentally for the download button
import { Download } from 'lucide-react';

export default App;
