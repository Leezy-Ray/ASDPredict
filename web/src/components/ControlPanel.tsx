'use client';

import { useStore } from '@/store/useStore';

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:5000';

interface BackendSample {
  id: number;
  original_id: number;
  type: 'asd' | 'control';
  label: number;
  name: string;
  description: string;
}

export default function ControlPanel() {
  const { 
    fmriData, 
    isLoading, 
    setLoading, 
    setPredictionResult, 
    setFmriData,
    setError,
    showAllRegions,
    setShowAllRegions,
    showAllConnections,
    setShowAllConnections,
    reset,
    selectedSampleId,
    setSelectedSampleId,
    incrementRefreshKey,
  } = useStore();

  const handlePredict = async () => {
        // #endregion
    
    if (!fmriData) {
            // #endregion
      setError('请先上传或选择fMRI数据');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      // 检查是否有选中的样本ID（从SampleSelector选择）
      // 如果有，使用 analyze-sample 接口；否则使用 /api/predict 接口
      
            // #endregion
      
      let response;
      let result;
      
      // 如果有选中的样本，使用 analyze-sample 接口
      if (selectedSampleId !== null && selectedSampleId !== undefined) {
                // #endregion
        
        response = await fetch(`${BACKEND_URL}/api/analyze-sample`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            sample_id: selectedSampleId,
            window_size: 32,
            stride: 16,
            threshold: 0.5,
            top_k: 50
          })
        });

        if (!response.ok) {
          const error = await response.json();
          throw new Error(error.error || '分析请求失败');
        }

        const analyzeData = await response.json();

        if (!analyzeData.success) {
          throw new Error(analyzeData.error || '分析失败');
        }

        // 计算平均ASD概率作为整体预测
        const windowProbs = analyzeData.predictions.window_predictions.map((w: any) => w.asd_probability);
        const meanProb = windowProbs.reduce((a: number, b: number) => a + b, 0) / windowProbs.length;

        // 转换异常连接格式（前端使用1-based索引）
        const abnormalConnections = analyzeData.abnormal_connections.connections.map((conn: any) => ({
          region1: conn.region1 + 1,
          region2: conn.region2 + 1,
          type: conn.type,
          strength: conn.strength
        }));

        // 转换窗口预测格式
        const windowPredictions = analyzeData.predictions.window_predictions.map((w: any) => ({
          window_index: w.window_index,
          prediction: w.prediction,
          asd_probability: w.asd_probability,
          logits: w.logits
        }));

        result = {
          probability: meanProb,
          confidence: Math.min(0.95, Math.abs(meanProb - 0.5) * 2),
          abnormalConnections,
          windowPredictions,
          processingTime: 0
        };
      } else {
        // 使用 mock 预测接口
                // #endregion
        
        response = await fetch('/api/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ fmriData, useMock: true }),
        });

        if (!response.ok) {
          const error = await response.json();
          throw new Error(error.error || '预测请求失败');
        }

        result = await response.json();
      }

            // #endregion
      console.log('[ControlPanel] Setting prediction result', result);
      setPredictionResult(result);
    } catch (err) {
            // #endregion
      setError(err instanceof Error ? err.message : '未知错误');
    } finally {
      setLoading(false);
            // #endregion
    }
  };

  const handleRefreshData = async () => {
    // 刷新数据：只重新加载样本列表，不调用分析接口
    setError(null);
    setLoading(true);

    try {
      // 从后端获取样本列表
      const response = await fetch(`${BACKEND_URL}/api/samples?n_asd=5&n_control=5`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();

      if (!data.success || !data.samples || data.samples.length === 0) {
        throw new Error(data.error || '无法获取样本列表');
      }

      // 随机选择一个样本，只设置fMRI数据，不进行分析
      const samples: BackendSample[] = data.samples;
      const randomSample = samples[Math.floor(Math.random() * samples.length)];

      // 获取样本数据（不进行分析）
      const sampleDataResponse = await fetch(`${BACKEND_URL}/api/sample-data?sample_id=${randomSample.id}`, {
        method: 'GET',
        headers: { 'Content-Type': 'application/json' },
      });

      if (!sampleDataResponse.ok) {
        // 如果是 404，可能是接口不存在，需要重启后端
        if (sampleDataResponse.status === 404) {
          const errorText = await sampleDataResponse.text();
          if (errorText.includes('404') || errorText.includes('Not Found')) {
            throw new Error('接口不存在，请重启后端服务以加载新接口');
          }
        }
        const errorData = await sampleDataResponse.json().catch(() => ({}));
        throw new Error(errorData.error || `HTTP error! status: ${sampleDataResponse.status}`);
      }

      const sampleData = await sampleDataResponse.json();
      if (!sampleData.success) {
        throw new Error(sampleData.error || '获取样本数据失败');
      }

      if (sampleData.data && Array.isArray(sampleData.data)) {
        // 转换数据格式为二维数组
        const fmriData = Array.isArray(sampleData.data) ? sampleData.data : [];
        setFmriData(fmriData, `${randomSample.type === 'asd' ? 'ASD' : 'Control'}-${randomSample.original_id}.csv`);
        
        // 设置选中的样本ID，以便预测时使用 analyze-sample 接口
        setSelectedSampleId(randomSample.id);
        
        // 更新 refreshKey 以触发 SampleSelector 重新加载样本列表
        incrementRefreshKey();
      } else {
        throw new Error('无效的数据格式');
      }

      // 清除之前的预测结果
      setPredictionResult(null);

    } catch (err: any) {
      console.error('Error refreshing data:', err);
      if (err.message && err.message.includes('Failed to fetch')) {
        setError('无法连接到后端服务器，请确保后端服务已启动（端口5000）');
      } else if (err.message && err.message.includes('接口不存在')) {
        setError('接口不存在，请重启后端服务以加载新接口 /api/sample-data');
      } else {
        setError(err.message || '刷新数据失败');
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-4">
      <div className="flex gap-3">
        <button
          onClick={handlePredict}
          disabled={!fmriData || isLoading}
          className={`flex-1 btn-primary flex items-center justify-center gap-2 ${(!fmriData || isLoading) ? 'opacity-50 cursor-not-allowed' : ''}`}
        >
          {isLoading ? (
            <>
              <svg className="w-5 h-5 animate-spin" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
              </svg>
              分析中...
            </>
          ) : (
            <>
              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
              </svg>
              开始预测
            </>
          )}
        </button>

        <button 
          onClick={handleRefreshData} 
          disabled={isLoading}
          className={`btn-secondary ${isLoading ? 'opacity-50 cursor-not-allowed' : ''}`} 
          title="刷新数据"
        >
          <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
          </svg>
        </button>
      </div>

      <div className="card">
        <h3 className="text-sm font-medium text-slate-300 mb-3">显示选项</h3>
        
        <label className="flex items-center justify-between cursor-pointer group">
          <span className="text-sm text-slate-400 group-hover:text-slate-300 transition-colors">
            显示所有脑区
          </span>
          <div className="relative">
            <input
              type="checkbox"
              checked={showAllRegions}
              onChange={(e) => setShowAllRegions(e.target.checked)}
              className="sr-only"
            />
            <div className={`w-10 h-5 rounded-full transition-colors ${showAllRegions ? 'bg-cyan-500' : 'bg-slate-600'}`}>
              <div className={`absolute top-0.5 left-0.5 w-4 h-4 rounded-full bg-white transition-transform ${showAllRegions ? 'translate-x-5' : 'translate-x-0'}`} />
            </div>
          </div>
        </label>
        
        <label className="flex items-center justify-between cursor-pointer group">
          <span className="text-sm text-slate-400 group-hover:text-slate-300 transition-colors">
            显示所有连接
          </span>
          <div className="relative">
            <input
              type="checkbox"
              checked={showAllConnections}
              onChange={(e) => setShowAllConnections(e.target.checked)}
              className="sr-only"
            />
            <div className={`w-10 h-5 rounded-full transition-colors ${showAllConnections ? 'bg-cyan-500' : 'bg-slate-600'}`}>
              <div className={`absolute top-0.5 left-0.5 w-4 h-4 rounded-full bg-white transition-transform ${showAllConnections ? 'translate-x-5' : 'translate-x-0'}`} />
            </div>
          </div>
        </label>
      </div>

      <div className="card">
        <h3 className="text-sm font-medium text-slate-300 mb-3">脑叶图例</h3>
        <div className="grid grid-cols-2 gap-2 text-xs">
          <div className="flex items-center gap-2">
            <span className="w-3 h-3 rounded-full bg-cyan-500" />
            <span className="text-slate-400">额叶</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="w-3 h-3 rounded-full bg-purple-500" />
            <span className="text-slate-400">顶叶</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="w-3 h-3 rounded-full bg-amber-500" />
            <span className="text-slate-400">颞叶</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="w-3 h-3 rounded-full bg-emerald-500" />
            <span className="text-slate-400">枕叶</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="w-3 h-3 rounded-full bg-pink-500" />
            <span className="text-slate-400">皮层下</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="w-3 h-3 rounded-full bg-indigo-500" />
            <span className="text-slate-400">小脑</span>
          </div>
        </div>
      </div>
    </div>
  );
}
