import React from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  AreaChart,
  Area,
} from 'recharts';
import type { StreamState } from '../services/api';

interface DriftChartProps {
  data: StreamState[];
}

const DriftChart: React.FC<DriftChartProps> = ({ data }) => {
  // Limit to last 200 points for performance
  const displayData = data.slice(-200);

  // Extract indices where drift or retraining occurred
  const events = displayData
    .map((d, i) => {
      if (d.drift_detected) {
        return {
          index: i,
          type: d.drift_handling_enabled ? 'retrain' : 'drift',
          timestamp: d.timestamp
        };
      }
      return null;
    })
    .filter((e) => e !== null) as { index: number; type: string; timestamp: string | null }[];

  return (
    <div className="space-y-6">
      {/* Prediction Chart */}
      <div className="w-full h-[350px] glass-card p-4">
        <div className="flex justify-between items-center mb-4">
          <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-widest">Prediction vs Actual</h3>
          <div className="flex gap-4 text-[10px] uppercase">
             <div className="flex items-center gap-1.5"><span className="w-2 h-2 rounded-full bg-indigo-500"></span> Predicted</div>
             <div className="flex items-center gap-1.5"><span className="w-2 h-2 rounded-full bg-gray-400"></span> Actual</div>
          </div>
        </div>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={displayData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#ffffff05" vertical={false} />
            <XAxis dataKey="timestamp" hide />
            <YAxis stroke="#444" tick={{ fill: '#666', fontSize: 10 }} domain={['auto', 'auto']} />
            <Tooltip
              contentStyle={{ backgroundColor: '#0a0a0c', border: '1px solid #ffffff10', borderRadius: '12px' }}
              itemStyle={{ fontSize: '11px' }}
              labelStyle={{ display: 'none' }}
            />
            <Line
              type="monotone"
              dataKey="y_true"
              stroke="#94a3b8"
              strokeWidth={1}
              dot={false}
              name="Actual"
              isAnimationActive={false}
            />
            <Line
              type="monotone"
              dataKey="y_pred"
              stroke="#6366f1"
              strokeWidth={2}
              dot={false}
              name="Predicted"
              isAnimationActive={false}
            />
            
            {events.map((event, idx) => (
              <ReferenceLine
                key={idx}
                x={event.timestamp || event.index}
                stroke={event.type === 'retrain' ? '#10b981' : '#ef4444'}
                strokeWidth={event.type === 'retrain' ? 2 : 1}
                strokeDasharray={event.type === 'retrain' ? 'none' : '3 3'}
                label={{ 
                  value: event.type === 'retrain' ? 'RETAIN' : 'DRIFT', 
                  fill: event.type === 'retrain' ? '#10b981' : '#ef4444', 
                  fontSize: 9, 
                  position: 'insideTopLeft' 
                }}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Error Chart */}
      <div className="w-full h-[200px] glass-card p-4">
        <h3 className="text-sm font-semibold mb-4 text-gray-400 uppercase tracking-widest">Absolute Error</h3>
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={displayData}>
            <defs>
              <linearGradient id="colorError" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#f43f5e" stopOpacity={0.3}/>
                <stop offset="95%" stopColor="#f43f5e" stopOpacity={0}/>
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="#ffffff05" vertical={false} />
            <XAxis dataKey="timestamp" hide />
            <YAxis stroke="#444" tick={{ fill: '#666', fontSize: 10 }} />
            <Tooltip
              contentStyle={{ backgroundColor: '#0a0a0c', border: '1px solid #ffffff10', borderRadius: '12px' }}
              itemStyle={{ fontSize: '11px', color: '#f43f5e' }}
              labelStyle={{ display: 'none' }}
            />
            <Area
              type="monotone"
              dataKey="error"
              stroke="#f43f5e"
              fillOpacity={1}
              fill="url(#colorError)"
              isAnimationActive={false}
            />
            {events.map((event, idx) => (
              <ReferenceLine
                key={idx}
                x={event.timestamp || event.index}
                stroke="#ffffff10"
              />
            ))}
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export default DriftChart;
