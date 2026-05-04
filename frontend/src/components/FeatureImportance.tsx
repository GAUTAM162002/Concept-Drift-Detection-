import React from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from 'recharts';

interface FeatureImportanceProps {
  features: { name: string; importance: number }[];
}

const FeatureImportance: React.FC<FeatureImportanceProps> = ({ features }) => {
  return (
    <div className="glass-card p-6 h-full min-h-[300px]">
      <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-widest mb-6">Top Feature Influence</h3>
      
      {features.length > 0 ? (
        <ResponsiveContainer width="100%" height={240}>
          <BarChart
            data={features}
            layout="vertical"
            margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="#ffffff05" horizontal={true} vertical={false} />
            <XAxis type="number" hide />
            <YAxis 
              dataKey="name" 
              type="category" 
              width={100}
              tick={{ fill: '#94a3b8', fontSize: 11 }}
              axisLine={false}
              tickLine={false}
            />
            <Tooltip
              contentStyle={{ backgroundColor: '#0a0a0c', border: '1px solid #ffffff10', borderRadius: '12px' }}
              itemStyle={{ fontSize: '11px', color: '#6366f1' }}
              cursor={{ fill: 'transparent' }}
            />
            <Bar dataKey="importance" radius={[0, 4, 4, 0]} barSize={20}>
              {features.map((_, index) => (
                <Cell key={`cell-${index}`} fill={`rgba(99, 102, 241, ${1 - index * 0.15})`} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      ) : (
        <div className="flex flex-col items-center justify-center h-[200px] text-gray-600">
          <p className="text-sm italic">Initializing model weights...</p>
        </div>
      )}
      
      <div className="mt-4 pt-4 border-t border-white/5">
        <p className="text-[10px] text-gray-500 uppercase tracking-tighter text-center">
          Powered by Linear Surrogate Model
        </p>
      </div>
    </div>
  );
};

export default FeatureImportance;
