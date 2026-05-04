import axios from 'axios';

const API_BASE_URL = 'http://127.0.0.1:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
});

export interface StreamState {
  timestamp: string | null;
  y_true: number | null;
  y_pred: number | null;
  error: number | null;
  drift_detected: boolean;
  drift_type: string | null;
  model_type: string;
  drift_handling_enabled: boolean;
  MAE: number;
  RMSE: number;
  R2: number;
  top_features: { name: string; importance: number }[];
  is_streaming: boolean;
}

export const streamService = {
  startStream: () => api.get('/start-stream'),
  stopStream: () => api.get('/stop-stream'),
  restartStream: () => api.get('/restart-stream'),
  getState: () => api.get<StreamState>('/get-state'),
  getHistory: (limit = 200) => api.get<StreamState[]>(`/get-history?limit=${limit}`),
  getInfo: () => api.get<{ total_rows: number }>('/get-info'),
  setModel: (modelType: string) => api.post('/set-model', { model_type: modelType }),
  toggleDrift: (enabled: boolean) => api.post('/toggle-drift', { enabled }),
  updateConfig: (speed: number) => api.post('/update-config', { stream_speed: speed }),
  setIndex: (index: number) => api.post('/set-index', { index }),
  uploadDataset: (file: File) => {
    const formData = new FormData();
    formData.append('file', file);
    return api.post('/upload-dataset', formData, {
      headers: { 'Content-Type': 'multipart/form-data' }
    });
  }
};
