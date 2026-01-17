'use client';

import { useState, useEffect } from 'react';
import { useStore } from '@/store/useStore';

interface BackendSample {
  id: number;
  original_id: number;
  type: 'asd' | 'control';
  label: number;
  name: string;
  description: string;
}

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:5000';

export default function SampleSelector() {
  const { setFmriData, fmriFileName, setLoading, setError, refreshKey, setSelectedSampleId, selectedSampleId: storeSelectedSampleId } = useStore();
  const [samples, setSamples] = useState<BackendSample[]>([]);
  const [isLoadingSamples, setIsLoadingSamples] = useState(true);
  const [localSelectedSampleId, setLocalSelectedSampleId] = useState<number | null>(null);

  const loadSamples = () => {
    // 从后端获取样本列表
        // #endregion
    
    const abortController = new AbortController();
    const timeoutId = setTimeout(() => abortController.abort(), 10000); // 10秒超时
    
    fetch(`${BACKEND_URL}/api/samples?n_asd=5&n_control=5`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
      signal: abortController.signal
    })
      .then(res => {
                // #endregion
        
        if (!res.ok) {
          throw new Error(`HTTP error! status: ${res.status}`);
        }
        return res.json();
      })
      .then(data => {
                // #endregion
        
        if (data.success) {
                    // #endregion
          setSamples(data.samples);
        } else {
          console.error('Failed to load samples:', data.error);
          setError(`加载样本列表失败: ${data.error || '未知错误'}`);
        }
      })
      .catch(err => {
                // #endregion
        
        console.error('Error loading samples:', err);
        if (err.name === 'AbortError') {
          setError('连接超时，请检查后端服务器是否运行');
        } else if (err.message.includes('Failed to fetch')) {
          setError('无法连接到后端服务器，请确保后端服务已启动（端口5000）');
        } else {
          setError(`连接错误: ${err.message}`);
        }
      })
      .finally(() => {
        clearTimeout(timeoutId);
        setIsLoadingSamples(false);
      });
  };

  useEffect(() => {
    // 当 refreshKey 变化时重新加载样本列表
    setLocalSelectedSampleId(null); // 清除本地选中状态
    setSelectedSampleId(null); // 清除 store 中的选中状态
    loadSamples();
  }, [refreshKey, setError, setSelectedSampleId]);

  useEffect(() => {
    // #region agent log
    if (samples.length > 0) {
      fetch('http://127.0.0.1:7244/ingest/1be70a65-d779-4250-81e8-2fff034b0cfe',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'SampleSelector.tsx:useEffect-samples',message:'Samples state updated',data:{samples_count:samples.length,first_sample:samples[0] ? {id:samples[0].id,original_id:samples[0].original_id,name:samples[0].name} : null,all_names:samples.map(s=>s.name)},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'I'})}).catch(()=>{});
    }
    // #endregion
  }, [samples]);

  const handleSelectSample = async (sample: BackendSample) => {
    setLocalSelectedSampleId(sample.id);
    setSelectedSampleId(sample.id); // 保存到 store，用于预测时调用 analyze-sample
    setLoading(true);
    setError(null);

    try {
      // 只获取样本数据，不进行分析（分析在点击"开始预测"时进行）
      const response = await fetch(`${BACKEND_URL}/api/sample-data?sample_id=${sample.id}`, {
        method: 'GET',
        headers: { 'Content-Type': 'application/json' },
      });

      if (!response.ok) {
        // 如果是 404，可能是接口不存在，需要重启后端
        if (response.status === 404) {
          const errorText = await response.text();
          if (errorText.includes('404') || errorText.includes('Not Found')) {
            throw new Error('接口不存在，请重启后端服务以加载新接口');
          }
        }
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
      }

      const data = await response.json();

      if (!data.success) {
        throw new Error(data.error || '获取样本数据失败');
      }

      // 设置fMRI数据
      if (data.data && Array.isArray(data.data)) {
        setFmriData(data.data, `${sample.type === 'asd' ? 'ASD' : 'Control'}-${sample.original_id}.csv`);
      } else {
        throw new Error('无效的数据格式');
      }

      // 清除之前的预测结果（选择新样本时）
      const { setPredictionResult } = useStore.getState();
      setPredictionResult(null);

    } catch (err: any) {
      console.error('Error loading sample data:', err);
      if (err.message && err.message.includes('Failed to fetch')) {
        setError('无法连接到后端服务器，请确保后端服务已启动（端口5000）');
      } else if (err.message && err.message.includes('接口不存在')) {
        setError('接口不存在，请重启后端服务以加载新接口 /api/sample-data');
      } else {
        setError(err.message || '加载样本数据失败');
      }
    } finally {
      setLoading(false);
    }
  };

  if (isLoadingSamples) {
    return (
      <div className="w-full">
        <h3 className="text-sm font-medium text-slate-400 mb-3">或选择示例数据</h3>
        <div className="text-sm text-slate-500">加载样本列表中...</div>
      </div>
    );
  }

  if (samples.length === 0) {
    return (
      <div className="w-full">
        <h3 className="text-sm font-medium text-slate-400 mb-3">或选择示例数据</h3>
        <div className="text-sm text-red-400">无法加载样本列表</div>
      </div>
    );
  }

  return (
    <div className="w-full">
      <h3 className="text-sm font-medium text-slate-400 mb-3">或选择示例数据</h3>
      <div className="grid grid-cols-2 gap-3">
        {samples.map((sample) => (
          <button
            key={sample.id}
            onClick={() => handleSelectSample(sample)}
            disabled={isLoadingSamples}
            className={`
              p-4 rounded-xl text-left transition-all duration-300 border
              ${localSelectedSampleId === sample.id
                ? 'bg-cyan-500/20 border-cyan-500/50'
                : 'bg-white/5 border-white/10 hover:bg-white/10 hover:border-white/20'
              }
              disabled:opacity-50 disabled:cursor-not-allowed
            `}
          >
            <div className="flex items-start gap-3">
              <div className={`
                w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0
                ${sample.type === 'asd' 
                  ? 'bg-pink-500/20 text-pink-400' 
                  : 'bg-emerald-500/20 text-emerald-400'
                }
              `}>
                {sample.type === 'asd' ? (
                  <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                  </svg>
                ) : (
                  <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                )}
              </div>
              <div className="min-w-0">
                <p className="font-medium text-slate-200 text-sm truncate">
                  {sample.type === 'asd' ? 'ASD' : 'Control'}-{sample.original_id}
                </p>
                <p className="text-xs text-slate-500 mt-1 line-clamp-2">{sample.description}</p>
              </div>
            </div>
          </button>
        ))}
      </div>
    </div>
  );
}
