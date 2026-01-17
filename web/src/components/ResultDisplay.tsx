'use client';

import { useStore } from '@/store/useStore';
import { useCC200Regions } from '@/lib/useCC200Regions';
import WindowPredictions from './WindowPredictions';

export default function ResultDisplay() {
  const { 
    predictionResult, 
    isLoading, 
    error,
    connectionFilter,
    setConnectionFilter,
    setSelectedConnection,
    selectedConnection,
  } = useStore();
  const { regions: CC200_REGIONS } = useCC200Regions();

  if (error) {
    return (
      <div className="card border-red-500/30">
        <div className="flex items-center gap-3 text-red-400">
          <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          <span>{error}</span>
        </div>
      </div>
    );
  }

  if (isLoading) {
    return (
      <div className="card">
        <div className="flex flex-col items-center justify-center py-8">
          <div className="relative w-16 h-16">
            <div className="absolute inset-0 rounded-full border-4 border-cyan-500/20"></div>
            <div className="absolute inset-0 rounded-full border-4 border-transparent border-t-cyan-500 animate-spin"></div>
          </div>
          <p className="mt-4 text-slate-400">正在分析 fMRI 数据...</p>
        </div>
      </div>
    );
  }

  if (!predictionResult) {
    return (
      <div className="card">
        <div className="flex flex-col items-center justify-center py-8 text-slate-500">
          <svg className="w-12 h-12 mb-3 opacity-50" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
          </svg>
          <p>上传 fMRI 数据后开始预测</p>
        </div>
      </div>
    );
  }

  const { probability, confidence, abnormalConnections, processingTime } = predictionResult;
  const isASD = probability > 0.5;
  const percentProbability = (probability * 100).toFixed(1);

  const filteredConnections = abnormalConnections.filter(conn => {
    if (connectionFilter === 'all') return true;
    return conn.type === connectionFilter;
  });

  const hyperCount = abnormalConnections.filter(c => c.type === 'hyper').length;
  const hypoCount = abnormalConnections.filter(c => c.type === 'hypo').length;

  return (
    <div className="space-y-4">
      <div className={`card ${isASD ? 'border-pink-500/30' : 'border-emerald-500/30'}`}>
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-slate-200">预测结果</h3>
          <span className="text-xs text-slate-500">{processingTime.toFixed(0)}ms</span>
        </div>

        <div className="flex items-center gap-6 mb-6">
          <div className="relative w-24 h-24">
            <svg className="w-24 h-24 transform -rotate-90">
              <circle cx="48" cy="48" r="40" stroke="currentColor" strokeWidth="8" fill="none" className="text-white/10" />
              <circle cx="48" cy="48" r="40" stroke="currentColor" strokeWidth="8" fill="none"
                strokeDasharray={`${probability * 251.2} 251.2`}
                className={isASD ? 'text-pink-500' : 'text-emerald-500'}
                strokeLinecap="round"
              />
            </svg>
            <div className="absolute inset-0 flex items-center justify-center">
              <span className={`text-2xl font-bold ${isASD ? 'text-pink-400' : 'text-emerald-400'}`}>
                {percentProbability}%
              </span>
            </div>
          </div>

          <div className="flex-1">
            <div className={`
              inline-flex items-center gap-2 px-3 py-1.5 rounded-full text-sm font-medium mb-2
              ${isASD ? 'bg-pink-500/20 text-pink-400' : 'bg-emerald-500/20 text-emerald-400'}
            `}>
              {isASD ? 'ASD 风险较高' : 'ASD 风险较低'}
            </div>
            <p className="text-sm text-slate-400">
              置信度: <span className="text-slate-300">{(confidence * 100).toFixed(1)}%</span>
            </p>
          </div>
        </div>

        <div className="grid grid-cols-3 gap-3 p-3 bg-white/5 rounded-xl">
          <div className="text-center">
            <p className="text-2xl font-bold text-slate-200">{abnormalConnections.length}</p>
            <p className="text-xs text-slate-500">异常连接</p>
          </div>
          <div className="text-center">
            <p className="text-2xl font-bold text-red-400">{hyperCount}</p>
            <p className="text-xs text-slate-500">过度连接</p>
          </div>
          <div className="text-center">
            <p className="text-2xl font-bold text-blue-400">{hypoCount}</p>
            <p className="text-xs text-slate-500">连接不足</p>
          </div>
        </div>
      </div>

      <div className="card">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-sm font-medium text-slate-300">异常连接</h3>
          <div className="flex gap-1 p-1 bg-white/5 rounded-lg">
            {(['all', 'hyper', 'hypo'] as const).map((filter) => (
              <button
                key={filter}
                onClick={() => setConnectionFilter(filter)}
                className={`
                  px-3 py-1 text-xs rounded-md transition-all
                  ${connectionFilter === filter 
                    ? 'bg-cyan-500/20 text-cyan-400' 
                    : 'text-slate-500 hover:text-slate-300'
                  }
                `}
              >
                {filter === 'all' ? '全部' : filter === 'hyper' ? '过度' : '不足'}
              </button>
            ))}
          </div>
        </div>

        <div className="space-y-2 max-h-64 overflow-y-auto pr-2">
          {filteredConnections.map((conn, idx) => {
            const region1 = CC200_REGIONS[conn.region1];
            const region2 = CC200_REGIONS[conn.region2];
            const isSelected = selectedConnection === conn;

            return (
              <button
                key={idx}
                onClick={() => setSelectedConnection(isSelected ? null : conn)}
                className={`
                  w-full p-3 rounded-xl text-left transition-all
                  ${isSelected 
                    ? 'bg-cyan-500/20 border border-cyan-500/50' 
                    : 'bg-white/5 border border-transparent hover:bg-white/10'
                  }
                `}
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <span className={`w-2 h-2 rounded-full ${conn.type === 'hyper' ? 'bg-red-400' : 'bg-blue-400'}`} />
                    <span className="text-xs text-slate-400">
                      {conn.type === 'hyper' ? '过度连接' : '连接不足'}
                    </span>
                  </div>
                  <span className="text-xs font-mono text-slate-500">
                    {(conn.strength * 100).toFixed(0)}%
                  </span>
                </div>
                <div className="mt-2 text-sm">
                  <span className="text-slate-300">{region1?.name}</span>
                  <span className="text-slate-600 mx-2">↔</span>
                  <span className="text-slate-300">{region2?.name}</span>
                </div>
              </button>
            );
          })}
        </div>
      </div>

      {/* 窗口预测结果 */}
      <WindowPredictions />
    </div>
  );
}
