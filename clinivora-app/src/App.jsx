import { useState } from 'react'
import './App.css'

function App() {
  const [formData, setFormData] = useState({
    age: 65,
    gender: 0,
    heart_rate_avg: 90,
    sys_bp_avg: 120,
    dias_bp_avg: 80,
    mean_bp_avg: 90,
    resp_rate_avg: 18,
    temp_avg: 37.2,
    spo2_avg: 98,
    wbc_max: 12,
    creatinine_max: 1.1,
    heart_rate_range: 20,
    sys_bp_range: 30,
    resp_rate_range: 5,
    glucose_max: 110,
    lactate_max: 1.5,
    urineoutput_sum: 1500
  })

  const [loading, setLoading] = useState(false)
  const [results, setResults] = useState(null)

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: parseFloat(e.target.value) || 0
    })
  }

  const handlePredict = async (e) => {
    e.preventDefault()
    setLoading(true)
    try {
      const res = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData)
      })
      const data = await res.json()
      setResults(data)
    } catch (err) {
      console.error(err)
      alert("Failed to connect to FastAPI backend. Ensure the server is running on port 8000.")
    } finally {
      setLoading(false)
    }
  }

  const getRiskColor = (prob) => {
    if (prob < 0.3) return '#10b981' // Green
    if (prob < 0.6) return '#f59e0b' // Yellow
    return '#ef4444' // Red
  }

  return (
    <div className="dashboard-container">
      {/* Navbar Integration */}
      <nav className="navbar">
        <div className="navbar-left">
          <h1>Clinivora</h1>
        </div>
        <div className="navbar-right">
          <span>Multi-Disease Early Screening Engine</span>
        </div>
      </nav>

      <main className="main-content">
        <section className="input-panel card-panel">
          <h2>Patient Physiology</h2>
          <form onSubmit={handlePredict} className="form-grid">
            <div className="input-group">
              <label>Age</label>
              <input type="number" name="age" value={formData.age} onChange={handleChange} />
            </div>
            <div className="input-group">
              <label>Heart Rate (Avg)</label>
              <input type="number" name="heart_rate_avg" value={formData.heart_rate_avg} onChange={handleChange} />
            </div>
            <div className="input-group">
              <label>Systolic BP (Avg)</label>
              <input type="number" name="sys_bp_avg" value={formData.sys_bp_avg} onChange={handleChange} />
            </div>
            <div className="input-group">
              <label>Resp Rate (Avg)</label>
              <input type="number" name="resp_rate_avg" value={formData.resp_rate_avg} onChange={handleChange} />
            </div>
            <div className="input-group">
              <label>SpO2 (Avg)</label>
              <input type="number" name="spo2_avg" value={formData.spo2_avg} onChange={handleChange} />
            </div>
            <div className="input-group">
              <label>WBC (Max)</label>
              <input type="number" name="wbc_max" value={formData.wbc_max} onChange={handleChange} />
            </div>
            <div className="input-group">
              <label>Creatinine (Max)</label>
              <input type="number" name="creatinine_max" value={formData.creatinine_max} onChange={handleChange} />
            </div>
            <div className="input-group">
              <label>Glucose (Max)</label>
              <input type="number" name="glucose_max" value={formData.glucose_max} onChange={handleChange} />
            </div>
            
            <button type="submit" className="btn-predict" disabled={loading}>
              {loading ? 'Analyzing...' : 'Run Causal DAG Model'}
            </button>
          </form>
        </section>

        <section className="results-panel card-panel">
          <h2>Risk Assessment Dashboard</h2>
          {results ? (
            <div className="meters-container">
              <RiskMeter name="Diabetes Risk (Baseline)" prob={results.Diabetes_Risk} color={getRiskColor(results.Diabetes_Risk)} />
              <div className="arrow">↓</div>
              <RiskMeter name="CHF Risk (Progressive)" prob={results.CHF_Risk} color={getRiskColor(results.CHF_Risk)} />
              <div className="arrow">↓</div>
              <RiskMeter name="Sepsis Risk (Acute)" prob={results.Sepsis_Risk} color={getRiskColor(results.Sepsis_Risk)} />
              <div className="arrow">↓</div>
              <RiskMeter name="AKI Risk (Complication)" prob={results.AKI_Risk} color={getRiskColor(results.AKI_Risk)} />
            </div>
          ) : (
            <div className="empty-state">
              <p>Enter patient data and run the model to see the causal risk cascade.</p>
            </div>
          )}
        </section>
      </main>
    </div>
  )
}

function RiskMeter({ name, prob, color }) {
  const percentage = (prob * 100).toFixed(1)
  return (
    <div className="risk-meter">
      <div className="meter-header">
        <span>{name}</span>
        <span style={{ color }}>{percentage}%</span>
      </div>
      <div className="progress-bar-bg">
        <div 
          className="progress-bar-fill" 
          style={{ width: `${percentage}%`, backgroundColor: color }}
        />
      </div>
    </div>
  )
}

export default App
