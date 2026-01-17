import { create } from 'zustand';
import { PredictionResult, AbnormalConnection } from '@/lib/mock-data';
import { BrainRegion } from '@/lib/cc200-regions';

interface AppState {
  fmriData: number[][] | null;
  fmriFileName: string | null;
  predictionResult: PredictionResult | null;
  isLoading: boolean;
  error: string | null;
  selectedRegion: BrainRegion | null;
  hoveredRegion: BrainRegion | null;
  selectedConnection: AbnormalConnection | null;
  showAllRegions: boolean;
  showAllConnections: boolean;
  connectionFilter: 'all' | 'hyper' | 'hypo';
  autoRotate: boolean;
  refreshKey: number; // 用于触发样本列表刷新
  selectedSampleId: number | null; // 当前选中的样本ID（用于调用 analyze-sample）
  
  setFmriData: (data: number[][] | null, fileName?: string) => void;
  setPredictionResult: (result: PredictionResult | null) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  setSelectedRegion: (region: BrainRegion | null) => void;
  setHoveredRegion: (region: BrainRegion | null) => void;
  setSelectedConnection: (connection: AbnormalConnection | null) => void;
  setShowAllRegions: (show: boolean) => void;
  setShowAllConnections: (show: boolean) => void;
  setConnectionFilter: (filter: 'all' | 'hyper' | 'hypo') => void;
  setAutoRotate: (rotate: boolean) => void;
  setSelectedSampleId: (sampleId: number | null) => void;
  incrementRefreshKey: () => void; // 增加 refreshKey 以触发样本列表刷新
  reset: () => void;
}

const initialState = {
  fmriData: null,
  fmriFileName: null,
  predictionResult: null,
  isLoading: false,
  error: null,
  selectedRegion: null,
  hoveredRegion: null,
  selectedConnection: null,
  showAllRegions: true,
  showAllConnections: true, // 默认显示所有连接
  connectionFilter: 'all' as const,
  autoRotate: true,
  refreshKey: 0,
  selectedSampleId: null,
};

export const useStore = create<AppState>((set) => ({
  ...initialState,

  setFmriData: (data, fileName) => set({ 
    fmriData: data, 
    fmriFileName: fileName || null,
    predictionResult: null,
    error: null,
  }),

  setPredictionResult: (result) => {
    console.log('[Store] setPredictionResult called', {
      hasResult: !!result,
      hasAbnormalConnections: !!result?.abnormalConnections,
      abnormalConnectionsCount: result?.abnormalConnections?.length,
      hasWindowPredictions: !!result?.windowPredictions,
      windowPredictionsCount: result?.windowPredictions?.length,
      resultKeys: Object.keys(result || {})
    });
    set({ 
      predictionResult: result,
      isLoading: false,
    });
  },

  setLoading: (loading) => set({ isLoading: loading }),

  setError: (error) => set({ 
    error, 
    isLoading: false,
  }),

  setSelectedRegion: (region) => set({ selectedRegion: region }),
  setHoveredRegion: (region) => set({ hoveredRegion: region }),
  setSelectedConnection: (connection) => set({ selectedConnection: connection }),
  setShowAllRegions: (show) => set({ showAllRegions: show }),
  setShowAllConnections: (show) => set({ showAllConnections: show }),
  setConnectionFilter: (filter) => set({ connectionFilter: filter }),
  setAutoRotate: (rotate) => set({ autoRotate: rotate }),
  setSelectedSampleId: (sampleId) => set({ selectedSampleId: sampleId }),
  incrementRefreshKey: () => {
    const currentState = useStore.getState();
    set({ refreshKey: currentState.refreshKey + 1 });
  },
  reset: () => {
    const currentState = useStore.getState();
    set({ 
      ...initialState, 
      refreshKey: currentState.refreshKey + 1 // 增加 refreshKey 以触发样本列表刷新
    });
  },
}));
