import { NextRequest, NextResponse } from 'next/server';
import { generateMockPrediction, PredictionResult } from '@/lib/mock-data';

export async function POST(request: NextRequest) {
  try {
    console.log('[API] POST /api/predict - Request received');
    const body = await request.json();
    const { fmriData, useMock = true } = body;

    console.log('[API] Request body parsed', { 
      hasFmriData: !!fmriData, 
      fmriDataLength: fmriData?.length,
      useMock 
    });

    if (!fmriData || !Array.isArray(fmriData)) {
      console.log('[API] Invalid data format');
      return NextResponse.json(
        { error: 'Invalid fMRI data format' },
        { status: 400 }
      );
    }

    // 支持100-200个时间点（根据实际数据调整）
    if (fmriData.length < 100 || fmriData.length > 200) {
      console.log('[API] Invalid time points', { length: fmriData.length });
      return NextResponse.json(
        { error: `Invalid time points: expected 100-200, got ${fmriData.length}` },
        { status: 400 }
      );
    }

    if (!fmriData[0] || !Array.isArray(fmriData[0]) || fmriData[0].length !== 200) {
      console.log('[API] Invalid regions', { firstRowLength: fmriData[0]?.length });
      return NextResponse.json(
        { error: `Invalid regions: expected 200, got ${fmriData[0]?.length || 0}` },
        { status: 400 }
      );
    }

    let result: PredictionResult;

    if (useMock) {
      console.log('[API] Using mock prediction');
      // Simulate processing delay - use setTimeout instead of blocking loop
      await new Promise(resolve => setTimeout(resolve, 500));
      
      result = generateMockPrediction(fmriData);
      console.log('[API] Mock prediction generated', { 
        hasResult: !!result,
        hasAbnormalConnections: !!result.abnormalConnections,
        abnormalConnectionsCount: result.abnormalConnections?.length,
        hasWindowPredictions: !!result.windowPredictions,
        windowPredictionsCount: result.windowPredictions?.length,
        resultKeys: Object.keys(result || {})
      });
    } else {
      try {
        const pythonResponse = await fetch('http://localhost:8000/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ fmri_data: fmriData }),
        });

        if (!pythonResponse.ok) {
          throw new Error('Python backend error');
        }

        result = await pythonResponse.json();
      } catch {
        console.warn('Python backend unavailable, using mock data');
        result = generateMockPrediction(fmriData);
      }
    }

    console.log('[API] Returning result', { 
      hasResult: !!result,
      hasWindowPredictions: !!result.windowPredictions 
    });
    return NextResponse.json(result);
  } catch (error) {
    console.error('[API] Prediction error:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}

export async function GET() {
  return NextResponse.json({ 
    status: 'ok',
    message: 'ASD Prediction API is running',
  });
}
