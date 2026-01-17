'use client';

import { useCallback, useState } from 'react';
import Papa from 'papaparse';
import { useStore } from '@/store/useStore';

export default function FileUploader() {
  const { setFmriData, setError, fmriFileName } = useStore();
  const [isDragging, setIsDragging] = useState(false);

  const validateAndParseData = useCallback((data: number[][], fileName: string) => {
    // 支持100-200个时间点（根据实际数据调整）
    if (data.length < 100 || data.length > 200) {
      setError(`时间点数量错误: 期望 100-200, 实际 ${data.length}`);
      return;
    }

    if (!data[0] || !Array.isArray(data[0]) || data[0].length !== 200) {
      setError(`脑区数量错误: 期望 200, 实际 ${data[0]?.length || 0}`);
      return;
    }

    for (let i = 0; i < data.length; i++) {
      for (let j = 0; j < data[i].length; j++) {
        if (typeof data[i][j] !== 'number' || isNaN(data[i][j])) {
          setError(`数据格式错误: 位置 [${i}][${j}] 不是有效数字`);
          return;
        }
      }
    }

    setFmriData(data, fileName);
  }, [setFmriData, setError]);

  const handleFile = useCallback((file: File) => {
    const fileName = file.name;
    const extension = fileName.split('.').pop()?.toLowerCase();

    if (extension === 'csv') {
      Papa.parse(file, {
        complete: (results) => {
          const data = results.data as string[][];
          const numericData = data
            .filter(row => row.length > 0 && row.some(cell => cell.trim() !== ''))
            .map(row => row.map(cell => parseFloat(cell)));
          
          validateAndParseData(numericData, fileName);
        },
        error: (error) => {
          setError(`CSV 解析错误: ${error.message}`);
        },
      });
    } else if (extension === 'json') {
      const reader = new FileReader();
      reader.onload = (e) => {
        try {
          const data = JSON.parse(e.target?.result as string);
          if (Array.isArray(data)) {
            validateAndParseData(data, fileName);
          } else if (data.data && Array.isArray(data.data)) {
            validateAndParseData(data.data, fileName);
          } else {
            setError('JSON 格式错误: 需要二维数组');
          }
        } catch {
          setError('JSON 解析错误');
        }
      };
      reader.readAsText(file);
    } else {
      setError('不支持的文件格式，请上传 CSV 或 JSON 文件');
    }
  }, [validateAndParseData, setError]);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  }, [handleFile]);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleInputChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) handleFile(file);
  }, [handleFile]);

  return (
    <div className="w-full">
      <label
        className={`
          relative flex flex-col items-center justify-center w-full h-40
          border-2 border-dashed rounded-2xl cursor-pointer
          transition-all duration-300
          ${isDragging 
            ? 'border-cyan-400 bg-cyan-500/10' 
            : 'border-white/20 hover:border-cyan-500/50 hover:bg-white/5'
          }
          ${fmriFileName ? 'border-emerald-500/50 bg-emerald-500/10' : ''}
        `}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
      >
        <input
          type="file"
          className="hidden"
          accept=".csv,.json"
          onChange={handleInputChange}
        />
        
        <div className="flex flex-col items-center gap-3 p-6">
          {fmriFileName ? (
            <>
              <div className="w-12 h-12 rounded-full bg-emerald-500/20 flex items-center justify-center">
                <svg className="w-6 h-6 text-emerald-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                </svg>
              </div>
              <div className="text-center">
                <p className="text-emerald-400 font-medium">{fmriFileName}</p>
                <p className="text-sm text-slate-400 mt-1">已加载 fMRI 数据</p>
              </div>
            </>
          ) : (
            <>
              <div className={`
                w-12 h-12 rounded-full flex items-center justify-center
                ${isDragging ? 'bg-cyan-500/20' : 'bg-white/5'}
                transition-colors duration-300
              `}>
                <svg className={`w-6 h-6 ${isDragging ? 'text-cyan-400' : 'text-slate-400'}`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                </svg>
              </div>
              <div className="text-center">
                <p className="text-slate-300">
                  <span className="text-cyan-400 font-medium">点击上传</span> 或拖拽文件
                </p>
                <p className="text-sm text-slate-500 mt-1">
                  CSV, JSON (100-200×200)
                </p>
              </div>
            </>
          )}
        </div>
      </label>
    </div>
  );
}
