'use client';

import { Suspense } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, PerspectiveCamera } from '@react-three/drei';
import { useStore } from '@/store/useStore';
import BrainModel from './BrainModel';
import Connections from './Connections';
import AllConnections from './AllConnections';
import ConvexHull from './ConvexHull';

function LoadingFallback() {
  return (
    <mesh>
      <sphereGeometry args={[0.5, 16, 16]} />
      <meshStandardMaterial color="#06b6d4" wireframe />
    </mesh>
  );
}

export default function BrainViewer() {
  const { autoRotate, setAutoRotate, predictionResult } = useStore();

  return (
    <div className="relative w-full h-full min-h-[500px] rounded-2xl overflow-hidden">
      <div className="absolute inset-0 bg-gradient-to-br from-slate-900 via-indigo-950/80 to-slate-900" />
      
      <div 
        className="absolute inset-0 opacity-10"
        style={{
          backgroundImage: `
            linear-gradient(rgba(6, 182, 212, 0.3) 1px, transparent 1px),
            linear-gradient(90deg, rgba(6, 182, 212, 0.3) 1px, transparent 1px)
          `,
          backgroundSize: '50px 50px',
        }}
      />

      <Canvas className="!absolute inset-0">
        <PerspectiveCamera makeDefault position={[0, 0, 1.5]} fov={60} />
        
        <ambientLight intensity={0.4} />
        <directionalLight position={[10, 10, 5]} intensity={0.8} />
        <directionalLight position={[-10, -10, -5]} intensity={0.3} color="#f472b6" />
        <pointLight position={[0, 5, 0]} intensity={0.5} color="#06b6d4" />

        <group>
          <ConvexHull />
          <BrainModel />
          <AllConnections />
          {predictionResult && <Connections />}
        </group>

        <OrbitControls
          autoRotate={autoRotate}
          autoRotateSpeed={0.5}
          enablePan={true}
          minDistance={0.5}
          maxDistance={3}
          enableDamping
          dampingFactor={0.05}
        />
      </Canvas>

      <div className="absolute bottom-4 left-4 right-4 flex items-center justify-between">
        <button
          onClick={() => setAutoRotate(!autoRotate)}
          className={`p-2 rounded-lg transition-all ${autoRotate ? 'bg-cyan-500/20 text-cyan-400' : 'bg-white/10 text-slate-400 hover:bg-white/20'}`}
          title={autoRotate ? '停止旋转' : '自动旋转'}
        >
          <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
          </svg>
        </button>

        <div className="flex items-center gap-4 text-xs">
          <div className="flex items-center gap-1.5">
            <span className="w-3 h-3 rounded-full bg-red-400" />
            <span className="text-slate-400">过度连接</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-3 h-3 rounded-full bg-blue-400" />
            <span className="text-slate-400">连接不足</span>
          </div>
        </div>
      </div>

     
    </div>
  );
}
