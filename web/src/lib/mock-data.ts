// Mock data for ASD prediction results

export interface AbnormalConnection {
  region1: number;
  region2: number;
  strength: number;
  type: 'hyper' | 'hypo';
}

export interface WindowPrediction {
  window_index: number;
  prediction: number; // 0 = TC (正常), 1 = ASD
  asd_probability: number;
  logits?: number[];
}

export interface PredictionResult {
  probability: number;
  confidence: number;
  abnormalConnections: AbnormalConnection[];
  processingTime: number;
  windowPredictions?: WindowPrediction[];
}

export function generateMockConnections(isASD: boolean): AbnormalConnection[] {
  const connections: AbnormalConnection[] = [];
  
  if (isASD) {
    connections.push(
      { region1: 3, region2: 55, strength: 0.85, type: 'hypo' },
      { region1: 28, region2: 75, strength: 0.78, type: 'hypo' },
      { region1: 20, region2: 57, strength: 0.72, type: 'hypo' },
      { region1: 45, region2: 77, strength: 0.69, type: 'hypo' },
      { region1: 90, region2: 170, strength: 0.82, type: 'hypo' },
      { region1: 110, region2: 185, strength: 0.75, type: 'hypo' },
      { region1: 98, region2: 100, strength: 0.68, type: 'hypo' },
      { region1: 118, region2: 120, strength: 0.71, type: 'hypo' },
      { region1: 0, region2: 3, strength: 0.76, type: 'hyper' },
      { region1: 25, region2: 28, strength: 0.73, type: 'hyper' },
      { region1: 6, region2: 7, strength: 0.65, type: 'hyper' },
      { region1: 31, region2: 32, strength: 0.62, type: 'hyper' },
      { region1: 61, region2: 81, strength: 0.79, type: 'hypo' },
      { region1: 18, region2: 43, strength: 0.74, type: 'hypo' },
      { region1: 160, region2: 175, strength: 0.67, type: 'hypo' },
      { region1: 168, region2: 100, strength: 0.58, type: 'hypo' },
      { region1: 183, region2: 120, strength: 0.55, type: 'hypo' },
      { region1: 171, region2: 90, strength: 0.63, type: 'hyper' },
      { region1: 186, region2: 110, strength: 0.61, type: 'hyper' },
    );
  } else {
    connections.push(
      { region1: 10, region2: 50, strength: 0.35, type: 'hypo' },
      { region1: 35, region2: 70, strength: 0.32, type: 'hypo' },
      { region1: 130, region2: 145, strength: 0.28, type: 'hyper' },
    );
  }

  return connections;
}

export function generateMockPrediction(fmriData?: number[][]): PredictionResult {
  const processingTime = 500 + Math.random() * 1000;
  
  let probability: number;
  
  if (fmriData) {
    const sum = fmriData.flat().reduce((a, b) => a + b, 0);
    const normalized = (sum % 100) / 100;
    probability = 0.3 + normalized * 0.5;
  } else {
    probability = 0.65 + Math.random() * 0.2;
  }

  const isASD = probability > 0.5;
  const confidence = 0.75 + Math.random() * 0.2;

  // 生成窗口预测数据（模拟11个窗口）
  const nWindows = 11;
  const windowPredictions: WindowPrediction[] = [];
  for (let i = 0; i < nWindows; i++) {
    // 随机生成预测，但整体趋势与 probability 一致
    const windowProb = probability + (Math.random() - 0.5) * 0.1;
    const prediction = windowProb > 0.5 ? 1 : 0;
    windowPredictions.push({
      window_index: i,
      prediction,
      asd_probability: Math.max(0, Math.min(1, windowProb)),
      logits: [Math.random() * 0.5, Math.random() * 0.5 + 0.2],
    });
  }

  return {
    probability,
    confidence,
    abnormalConnections: generateMockConnections(isASD),
    windowPredictions,
    processingTime,
  };
}

export function generateSampleFMRI(type: 'asd' | 'control' | 'random'): number[][] {
  const timePoints = 100;
  const regions = 200;
  const data: number[][] = [];

  for (let t = 0; t < timePoints; t++) {
    const row: number[] = [];
    for (let r = 0; r < regions; r++) {
      let value: number;
      
      switch (type) {
        case 'asd':
          value = Math.sin(t * 0.1 + r * 0.05) * 0.5 + 
                  Math.random() * 0.8 + 
                  (r < 50 ? 0.2 : -0.1);
          break;
        case 'control':
          value = Math.sin(t * 0.1 + r * 0.05) * 0.3 + 
                  Math.random() * 0.4;
          break;
        default:
          value = Math.random() * 2 - 1;
      }
      
      row.push(parseFloat(value.toFixed(4)));
    }
    data.push(row);
  }

  return data;
}

export const SAMPLE_DATASETS = [
  {
    id: 'sample-asd-1',
    name: 'ASD 样本 1',
    description: '典型 ASD 患者的 fMRI 数据',
    type: 'asd' as const,
  },
  {
    id: 'sample-asd-2',
    name: 'ASD 样本 2',
    description: 'ASD 患者数据，社交脑网络异常',
    type: 'asd' as const,
  },
  {
    id: 'sample-control-1',
    name: '对照组样本 1',
    description: '健康对照组的 fMRI 数据',
    type: 'control' as const,
  },
  {
    id: 'sample-control-2',
    name: '对照组样本 2',
    description: '另一例健康对照组数据',
    type: 'control' as const,
  },
];
