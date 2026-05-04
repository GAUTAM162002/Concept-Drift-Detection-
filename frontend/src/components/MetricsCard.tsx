import React from 'react';

interface MetricsCardProps {
  label: string;
  value: string | number;
  icon: React.ElementType;
  color: string;
}

const MetricsCard: React.FC<MetricsCardProps> = ({ label, value, icon: Icon, color }) => {
  return (
    <div className="glass-card hover-glow p-4 flex items-center space-x-4 transition-all duration-300 hover:-translate-y-1">
      <div className={`p-3 rounded-xl ${color} bg-opacity-20`}>
        <Icon className={`w-6 h-6 ${color.replace('bg-', 'text-')}`} />
      </div>
      <div>
        <p className="text-gray-400 text-xs uppercase tracking-wider font-medium">{label}</p>
        <p className="text-2xl font-bold text-white">{value}</p>
      </div>
    </div>
  );
};

export default MetricsCard;
