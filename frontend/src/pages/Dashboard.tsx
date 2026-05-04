import React, { useEffect, useState, useCallback } from 'react';
import { 
  Play, 
  Square, 
  RotateCcw, 
  Activity, 
  TrendingUp, 
  Target, 
  Cpu, 
  Settings2,
  ShieldAlert,
  UploadCloud
} from 'lucide-react';
import { streamService, type StreamState } from '../services/api';
import DriftChart from '../components/DriftChart';
import MetricsCard from '../components/MetricsCard';
import FeatureImportance from '../components/FeatureImportance';

const Dashboard: React.FC = () => {
  const [data, setData] = useState<StreamState[]>([]);
  const [currentState, setCurrentState] = useState<StreamState | null>(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const [modelType, setModelType] = useState('linear');
  const [driftEnabled, setDriftEnabled] = useState(true);
  const [speed, setSpeed] = useState(0.01);
  const [totalRows, setTotalRows] = useState(0);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [isUploading, setIsUploading] = useState(false);
  const fileInputRef = React.useRef<HTMLInputElement>(null);

  const fetchState = useCallback(async () => {
    try {
      const stateRes = await streamService.getState();
      if (stateRes.data) {
        setCurrentState(stateRes.data);
        if (stateRes.data.model_type) setModelType(stateRes.data.model_type);
        setDriftEnabled(!!stateRes.data.drift_handling_enabled);
        
        // Auto-stop when backend finishes streaming
        if (stateRes.data.is_streaming !== undefined) {
          if (!stateRes.data.is_streaming && isStreaming) {
            setIsStreaming(false); // Stop UI
          }
        }

        // Efficiently append new data point locally
        setData(prevData => {
          const lastData = prevData[prevData.length - 1];
          if (!lastData || lastData.timestamp !== stateRes.data.timestamp) {
             const newData = [...prevData, stateRes.data];
             return newData.slice(-200); // Prevent memory bloat
          }
          return prevData;
        });
      }
    } catch (err) {
      console.error('Error fetching state:', err);
    }
  }, [isStreaming]);

  const loadHistory = useCallback(async () => {
    try {
      const historyRes = await streamService.getHistory();
      if (historyRes.data) {
        // Only keep the last 200 items initially to prevent initial stutter
        setData(historyRes.data.slice(-200));
      }
    } catch (err) {
      console.error('Error fetching history:', err);
    }
  }, []);

  const fetchInfo = useCallback(async () => {
    try {
      const res = await streamService.getInfo();
      if (res.data && typeof res.data.total_rows === 'number') {
        setTotalRows(res.data.total_rows);
      }
    } catch (err) {
      console.error('Error fetching info:', err);
    }
  }, []);

  useEffect(() => {
    fetchInfo();
    loadHistory();
    let interval: ReturnType<typeof setInterval>;
    
    if (isStreaming) {
      interval = setInterval(() => {
        loadHistory(); // Get the latest 200 points to show fast-moving data smoothly
        fetchState();
        fetchInfo(); // Poll info to keep current index updated
      }, 100);
    } else {
      // Fetch once when stopped to get final state
      fetchState();
      fetchInfo();
    }
    
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [fetchState, fetchInfo, loadHistory, isStreaming]);

  const handleStart = async () => {
    try {
      await streamService.startStream();
      setIsStreaming(true);
    } catch (err) {
      console.error('Failed to start stream:', err);
    }
  };

  const handleStop = async () => {
    try {
      await streamService.stopStream();
      setIsStreaming(false);
    } catch (err) {
      console.error('Failed to stop stream:', err);
    }
  };

  const handleRestart = async () => {
    try {
      await streamService.restartStream();
      setIsStreaming(true);
      setCurrentIndex(0);
      setData([]); // Clear data locally for immediate feedback
    } catch (err) {
      console.error('Failed to restart stream:', err);
    }
  };

  const handleSliderChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const newIndex = parseInt(e.target.value);
    setCurrentIndex(newIndex);
    setIsStreaming(false); // Pause on slide
    try {
      await streamService.setIndex(newIndex);
      await loadHistory(); // Refresh charts cleanly
      await fetchState();
    } catch (err) {
      console.error('Failed to set index:', err);
    }
  };

  const handleModelChange = async (e: React.ChangeEvent<HTMLSelectElement>) => {
    const newModel = e.target.value;
    try {
      await streamService.setModel(newModel);
      setModelType(newModel);
    } catch (err) {
      console.error('Failed to switch model:', err);
    }
  };

  const handleDriftToggle = async () => {
    const newVal = !driftEnabled;
    try {
      await streamService.toggleDrift(newVal);
      setDriftEnabled(newVal);
    } catch (err) {
      console.error('Failed to toggle drift:', err);
    }
  };

  const handleSpeedChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const newSpeed = parseFloat(e.target.value);
    setSpeed(newSpeed);
    try {
      await streamService.updateConfig(newSpeed);
    } catch (err) {
      console.error('Failed to update speed:', err);
    }
  };

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setIsUploading(true);
    try {
      await streamService.uploadDataset(file);
      await fetchInfo();
      await loadHistory();
      setCurrentIndex(0);
      setIsStreaming(false);
      setData([]); // Clear charts
    } catch (err) {
      console.error('Failed to upload dataset:', err);
      alert('Failed to upload dataset. Make sure it is a valid CSV.');
    } finally {
      setIsUploading(false);
      // Reset input value to allow uploading the same file again
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  };

  return (
    <div className="min-h-screen p-6 lg:p-10 max-w-7xl mx-auto space-y-8">
      {/* Header */}
      <header className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold text-white tracking-tight flex items-center gap-3 group">
            <Cpu className="text-indigo-500 w-8 h-8 group-hover:text-purple-400 transition-colors duration-500" />
            AI Drift Monitor
          </h1>
          <p className="text-gray-400 mt-1 font-medium tracking-wide">Real-time concept drift detection & adaptive learning</p>
        </div>

        <div className="flex items-center gap-3 glass p-2 shadow-2xl shadow-indigo-500/10">
          {/* Hidden File Input */}
          <input 
            type="file" 
            accept=".csv" 
            ref={fileInputRef} 
            onChange={handleFileUpload} 
            className="hidden" 
          />
          <button 
            onClick={() => fileInputRef.current?.click()}
            disabled={isUploading}
            className="flex items-center gap-2 px-4 py-2.5 bg-white/5 hover:bg-white/10 border border-white/10 text-white rounded-lg transition-all duration-300 font-semibold disabled:opacity-50"
            title="Upload Custom CSV Dataset"
          >
            <UploadCloud className={`w-4 h-4 ${isUploading ? 'animate-bounce' : ''}`} />
            <span className="hidden sm:inline">{isUploading ? 'Uploading...' : 'Upload CSV'}</span>
          </button>

          {!isStreaming ? (
            <button 
              onClick={handleStart}
              className="flex items-center gap-2 px-5 py-2.5 bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-500 hover:to-purple-500 text-white rounded-lg transition-all duration-300 font-semibold shadow-lg shadow-indigo-500/30 hover:scale-105 hover:shadow-indigo-500/50"
            >
              <Play className="w-4 h-4" /> Start
            </button>
          ) : (
            <button 
              onClick={handleStop}
              className="flex items-center gap-2 px-5 py-2.5 bg-gradient-to-r from-red-600 to-rose-600 hover:from-red-500 hover:to-rose-500 text-white rounded-lg transition-all duration-300 font-semibold shadow-lg shadow-red-500/30 hover:scale-105 hover:shadow-red-500/50"
            >
              <Square className="w-4 h-4" /> Stop
            </button>
          )}
          <button 
            onClick={handleRestart}
            className="p-2.5 text-gray-400 hover:text-white hover:bg-white/10 rounded-lg transition-all duration-300 hover:rotate-180"
            title="Restart Stream"
          >
            <RotateCcw className="w-5 h-5" />
          </button>
        </div>
      </header>

      {/* Metrics Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricsCard 
          label="MAE" 
          value={currentState?.MAE != null ? currentState.MAE.toFixed(4) : '0.0000'} 
          icon={Activity} 
          color="bg-blue-500" 
        />
        <MetricsCard 
          label="RMSE" 
          value={currentState?.RMSE != null ? currentState.RMSE.toFixed(4) : '0.0000'} 
          icon={TrendingUp} 
          color="bg-purple-500" 
        />
        <MetricsCard 
          label="R² Score" 
          value={currentState?.R2 != null ? currentState.R2.toFixed(4) : '0.0000'} 
          icon={Target} 
          color="bg-emerald-500" 
        />
        <div className={`glass-card hover-glow p-4 flex items-center space-x-4 border-l-4 transition-all duration-500 ${currentState?.drift_detected ? 'border-red-500 shadow-[0_0_15px_rgba(239,68,68,0.3)]' : 'border-emerald-500'}`}>
           <div className={`p-3 rounded-xl transition-colors duration-500 ${currentState?.drift_detected ? 'bg-red-500' : 'bg-emerald-500'} bg-opacity-20`}>
            <ShieldAlert className={`w-6 h-6 ${currentState?.drift_detected ? 'text-red-500 animate-pulse' : 'text-emerald-400'}`} />
          </div>
          <div>
            <p className="text-gray-400 text-xs uppercase tracking-wider font-medium">Drift Status</p>
            <p className={`text-xl font-bold transition-colors duration-500 ${currentState?.drift_detected ? 'text-red-500' : 'text-emerald-400'}`}>
              {currentState?.drift_detected 
                ? `${currentState.drift_type?.toUpperCase() || 'DRIFT'} DETECTED` 
                : 'STABLE'}
            </p>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Main Chart */}
        <div className="lg:col-span-2 space-y-6">
          <div className="relative">
             {!isStreaming && currentIndex >= totalRows && totalRows > 0 && (
                <div className="absolute inset-0 z-10 flex items-center justify-center bg-black/60 backdrop-blur-sm rounded-xl animate-fade-in border border-indigo-500/30">
                  <div className="bg-indigo-900/40 p-6 rounded-2xl border border-indigo-400/30 text-center shadow-[0_0_30px_rgba(79,70,229,0.3)]">
                     <Target className="w-12 h-12 text-indigo-400 mx-auto mb-3" />
                     <h3 className="text-2xl font-bold text-white mb-2">Training Complete</h3>
                     <p className="text-indigo-200">The dataset has been fully processed.</p>
                     <button onClick={handleRestart} className="mt-4 px-6 py-2 bg-indigo-500 hover:bg-indigo-400 text-white font-semibold rounded-lg transition-colors">
                        Restart Training
                     </button>
                  </div>
                </div>
             )}
             <DriftChart data={data} />
          </div>
          
          {/* Timeline Slider */}
          <div className="glass-card p-6 space-y-4">
            <div className="flex justify-between items-center">
              <div className="flex items-center gap-3">
                <button 
                  onClick={isStreaming ? handleStop : handleStart}
                  className="p-2 rounded-full bg-white/5 hover:bg-white/10 text-white transition-all"
                >
                  {isStreaming ? <Square className="w-4 h-4 fill-white" /> : <Play className="w-4 h-4 fill-white" />}
                </button>
                <h4 className="text-sm font-semibold text-gray-300">Timeline Navigation</h4>
              </div>
              <div className="text-[11px] font-mono text-gray-500">
                SAMPLE: {currentIndex.toLocaleString()} / {totalRows.toLocaleString()}
              </div>
            </div>
              <input 
                type="range" 
                min="0" 
                max={totalRows} 
                value={currentIndex}
                onChange={handleSliderChange}
                className="w-full h-2 bg-gray-800 rounded-lg appearance-none cursor-pointer accent-indigo-500 hover:accent-indigo-400 transition-all focus:outline-none focus:ring-2 focus:ring-indigo-500/50"
              />
              <div className="flex justify-between text-[10px] text-gray-600 font-medium tracking-widest">
                <span>START</span>
                <span>END</span>
              </div>
          </div>
        </div>

        {/* Sidebar Controls */}
        <div className="space-y-6">
          {/* Feature Importance Panel */}
          <FeatureImportance features={currentState?.top_features || []} />

          <div className="glass-card p-6 space-y-6">
            <h3 className="text-lg font-semibold text-white flex items-center gap-2">
              <Settings2 className="w-5 h-5 text-indigo-400" />
              Controls
            </h3>

            {/* Model Selector */}
            <div className="space-y-2">
              <label className="text-sm text-gray-400">Online Model Type</label>
              <select 
                value={modelType}
                onChange={handleModelChange}
                className="w-full bg-[#1a1a1e] border border-white/10 text-white rounded-lg p-2.5 outline-none focus:ring-2 focus:ring-indigo-500"
              >
                <option value="linear">Linear Regression</option>
                <option value="tree">Hoeffding Tree</option>
                <option value="adaptive_tree">Adaptive Hoeffding Tree</option>
                <option value="ensemble">Weighted Ensemble</option>
              </select>
            </div>

            {/* Drift Toggle */}
            <div className="flex items-center justify-between p-3 rounded-xl bg-white/5 border border-white/5">
              <div className="space-y-0.5">
                <p className="text-sm font-medium text-white">Drift Handling</p>
                <p className="text-xs text-gray-500">Enable adaptive retraining</p>
              </div>
              <button 
                onClick={handleDriftToggle}
                className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors focus:outline-none ${driftEnabled ? 'bg-indigo-600' : 'bg-gray-700'}`}
              >
                <span className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${driftEnabled ? 'translate-x-6' : 'translate-x-1'}`} />
              </button>
            </div>

            {/* Speed Slider */}
            <div className="space-y-3">
              <div className="flex justify-between items-center">
                <label className="text-sm text-gray-400">Streaming Speed</label>
                <span className="text-xs font-mono text-indigo-400">{speed}s</span>
              </div>
              <input 
                type="range" 
                min="0.01" 
                max="1.0" 
                step="0.01" 
                value={speed}
                onChange={handleSpeedChange}
                className="w-full h-1.5 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-indigo-500"
              />
              <div className="flex justify-between text-[10px] text-gray-500 uppercase tracking-tighter">
                <span>Fast</span>
                <span>Slow</span>
              </div>
            </div>
          </div>

          {/* Status Panel */}
          <div className="glass-card p-6 bg-gradient-to-br from-indigo-500/5 to-purple-500/5 border-indigo-500/20 relative overflow-hidden group">
            <div className="absolute top-0 left-0 w-1 h-full bg-indigo-500 group-hover:bg-purple-500 transition-colors duration-500"></div>
            <h4 className="text-sm font-semibold text-indigo-300 uppercase tracking-widest mb-4">System Log</h4>
            <div className="space-y-3 font-mono text-[11px]">
              <div className="flex justify-between text-gray-500">
                <span>Model:</span>
                <span className="text-gray-300">{(modelType || '').replace('_', ' ')}</span>
              </div>
              <div className="flex justify-between text-gray-500">
                <span>Buffer:</span>
                <span className="text-gray-300">{data.length} pts</span>
              </div>
              <div className="flex justify-between text-gray-500">
                <span>Status:</span>
                <span className={isStreaming ? "text-emerald-400 font-bold animate-pulse" : (currentIndex >= totalRows && totalRows > 0 ? "text-indigo-400 font-bold" : "text-gray-400 font-bold")}>
                  {isStreaming ? "ACTIVE" : (currentIndex >= totalRows && totalRows > 0 ? "COMPLETE" : "IDLE")}
                </span>
              </div>
              <div className="mt-4 pt-4 border-t border-white/5">
                <p className="text-gray-500 italic flex items-center gap-2">
                  <span className={`w-2 h-2 rounded-full ${isStreaming ? 'bg-emerald-500' : 'bg-gray-600'}`}></span>
                  Connected to FastAPI
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
