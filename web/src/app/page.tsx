'use client';

import dynamic from 'next/dynamic';
import FileUploader from '@/components/FileUploader';
import SampleSelector from '@/components/SampleSelector';
import ResultDisplay from '@/components/ResultDisplay';
import ControlPanel from '@/components/ControlPanel';
import RegionInfo from '@/components/RegionInfo';

const BrainViewer = dynamic(() => import('@/components/BrainViewer'), {
  ssr: false,
  loading: () => (
    <div className="w-full h-full min-h-[500px] rounded-2xl bg-slate-800/50 flex items-center justify-center">
      <div className="flex flex-col items-center gap-4">
        <div className="w-12 h-12 border-4 border-cyan-500/30 border-t-cyan-500 rounded-full animate-spin" />
        <p className="text-slate-400">加载 3D 可视化...</p>
      </div>
    </div>
  ),
});

export default function Home() {
  return (
    <main className="min-h-screen p-4 md:p-6 lg:p-8">
      <header className="mb-8">
        <div className="flex items-center gap-4 mb-2">
          <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-cyan-500 to-pink-500 flex items-center justify-center">
            <svg className="w-7 h-7 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
            </svg>
          </div>
          <div>
            <h1 className="text-2xl md:text-3xl font-bold gradient-text">
              ASD 脑连接预测系统
          </h1>
            <p className="text-slate-400 text-sm md:text-base">
              基于 fMRI 数据的自闭症谱系障碍风险评估与脑区异常连接可视化
            </p>
          </div>
        </div>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        <div className="lg:col-span-3 space-y-6">
          <section className="card">
            <h2 className="text-lg font-semibold text-slate-200 mb-4 flex items-center gap-2">
              <svg className="w-5 h-5 text-cyan-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
              </svg>
              数据输入
            </h2>
            <div className="space-y-4">
              <FileUploader />
              <div className="relative">
                <div className="absolute inset-0 flex items-center">
                  <div className="w-full border-t border-white/10" />
                </div>
                <div className="relative flex justify-center">
                  <span className="px-3 bg-slate-800 text-xs text-slate-500">或</span>
                </div>
              </div>
              <SampleSelector />
            </div>
          </section>
          <ControlPanel />
        </div>

        <div className="lg:col-span-6">
          <div className="card p-0 overflow-hidden relative">
            <BrainViewer />
            <RegionInfo />
          </div>
        </div>

        <div className="lg:col-span-3">
          <ResultDisplay />
        </div>
      </div>

      <footer className="mt-12 text-center text-sm text-slate-600">
        <p>CC200 脑区模板 · Three.js 3D 可视化 · <span className="text-slate-500">仅供研究参考</span></p>
      </footer>
      </main>
  );
}
