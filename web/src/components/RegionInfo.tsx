'use client';

import { useStore } from '@/store/useStore';
import { LOBE_COLORS } from '@/lib/cc200-regions';

export default function RegionInfo() {
  const { hoveredRegion, selectedRegion, setSelectedRegion } = useStore();
  
  const region = hoveredRegion || selectedRegion;

  if (!region) return null;

  const lobeColor = LOBE_COLORS[region.lobe];
  const lobeNames: Record<string, string> = {
    frontal: '额叶', parietal: '顶叶', temporal: '颞叶',
    occipital: '枕叶', subcortical: '皮层下', cerebellum: '小脑',
  };
  const hemisphereNames: Record<string, string> = {
    left: '左半球', right: '右半球', midline: '中线',
  };

  return (
    <div className="absolute top-4 left-4 glass rounded-xl p-4 max-w-xs">
      <div className="flex items-start justify-between gap-3">
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-full" style={{ backgroundColor: lobeColor }} />
          <h4 className="text-sm font-medium text-slate-200">{region.name}</h4>
        </div>
        {selectedRegion && (
          <button onClick={() => setSelectedRegion(null)} className="text-slate-500 hover:text-slate-300 transition-colors">
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        )}
      </div>
      
      <div className="mt-3 space-y-2 text-xs">
        <div className="flex justify-between">
          <span className="text-slate-500">脑区 ID</span>
          <span className="text-slate-300 font-mono">{region.id}</span>
        </div>
        <div className="flex justify-between">
          <span className="text-slate-500">脑叶</span>
          <span className="text-slate-300">{lobeNames[region.lobe]}</span>
        </div>
        <div className="flex justify-between">
          <span className="text-slate-500">半球</span>
          <span className="text-slate-300">{hemisphereNames[region.hemisphere]}</span>
        </div>
        <div className="flex justify-between">
          <span className="text-slate-500">MNI 坐标</span>
          <span className="text-slate-300 font-mono">({region.x}, {region.y}, {region.z})</span>
        </div>
      </div>
    </div>
  );
}
