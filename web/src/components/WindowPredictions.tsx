'use client';

import { useStore } from '@/store/useStore';
import { WindowPrediction } from '@/lib/mock-data';

export default function WindowPredictions() {
  const { predictionResult } = useStore();

    // #endregion

  if (!predictionResult || !predictionResult.windowPredictions || predictionResult.windowPredictions.length === 0) {
    return null;
  }

  const windowPredictions = predictionResult.windowPredictions;
  const asdWindows = windowPredictions.filter(w => w.prediction === 1);
  const tcWindows = windowPredictions.filter(w => w.prediction === 0);

  return (
    <div className="card">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-medium text-slate-300">窗口预测结果</h3>
        <div className="flex items-center gap-4 text-xs">
          <div className="flex items-center gap-1.5">
            <span className="w-3 h-3 rounded-full bg-pink-500" />
            <span className="text-slate-400">ASD ({asdWindows.length})</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-3 h-3 rounded-full bg-emerald-500" />
            <span className="text-slate-400">正常 ({tcWindows.length})</span>
          </div>
        </div>
      </div>

      {/* 窗口列表 */}
      <div className="space-y-2 max-h-80 overflow-y-auto pr-2">
        <div className="grid grid-cols-11 gap-1.5">
          {windowPredictions.map((window) => {
            const isASD = window.prediction === 1;
            const probability = (window.asd_probability * 100).toFixed(0);

            return (
              <div
                key={window.window_index}
                className={`
                  relative p-2 rounded-lg text-center transition-all
                  ${isASD 
                    ? 'bg-pink-500/20 border border-pink-500/50' 
                    : 'bg-emerald-500/20 border border-emerald-500/50'
                  }
                  hover:scale-105 cursor-pointer
                `}
                title={`窗口 ${window.window_index}: ${isASD ? 'ASD' : '正常'} (${probability}%)`}
              >
                <div className="text-xs font-mono text-slate-400 mb-1">
                  {window.window_index}
                </div>
                <div className={`
                  text-lg font-bold
                  ${isASD ? 'text-pink-400' : 'text-emerald-400'}
                `}>
                  {isASD ? 'A' : 'T'}
                </div>
                <div className="text-[10px] text-slate-500 mt-1">
                  {probability}%
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* 统计信息 */}
      <div className="mt-4 pt-4 border-t border-white/10">
        <div className="grid grid-cols-2 gap-3">
          <div className="p-3 bg-pink-500/10 rounded-lg border border-pink-500/20">
            <div className="text-xs text-slate-400 mb-1">预测为 ASD 的窗口</div>
            <div className="text-lg font-bold text-pink-400">{asdWindows.length}</div>
            <div className="text-xs text-slate-500 mt-1">
              {asdWindows.length > 0 && (
                <>
                  窗口: {asdWindows.map(w => w.window_index).join(', ')}
                </>
              )}
            </div>
          </div>
          <div className="p-3 bg-emerald-500/10 rounded-lg border border-emerald-500/20">
            <div className="text-xs text-slate-400 mb-1">预测为正常的窗口</div>
            <div className="text-lg font-bold text-emerald-400">{tcWindows.length}</div>
            <div className="text-xs text-slate-500 mt-1">
              {tcWindows.length > 0 && (
                <>
                  窗口: {tcWindows.map(w => w.window_index).join(', ')}
                </>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
