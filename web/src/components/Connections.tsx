'use client';

import { useMemo, useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import { useStore } from '@/store/useStore';
import { useCC200Regions } from '@/lib/useCC200Regions';
import { AbnormalConnection } from '@/lib/mock-data';
import { getMappedPosition } from './BrainModel';

function ConnectionLine({ 
  connection,
  isSelected,
  regions,
}: { 
  connection: AbnormalConnection;
  isSelected: boolean;
  regions: ReturnType<typeof useCC200Regions>['regions'];
}) {
  const lineRef = useRef<THREE.Line>(null);
  const { setSelectedConnection } = useStore();

  // 注意：connection.region1/region2 可能是 1-based（从 SampleSelector 转换），需要转换为 0-based
  const region1Index = connection.region1 > 0 ? connection.region1 - 1 : connection.region1;
  const region2Index = connection.region2 > 0 ? connection.region2 - 1 : connection.region2;
  const region1 = regions[region1Index];
  const region2 = regions[region2Index];
  
  // #region agent log
  if (!region1 || !region2) {
    fetch('http://127.0.0.1:7244/ingest/1be70a65-d779-4250-81e8-2fff034b0cfe',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'Connections.tsx:ConnectionLine',message:'Invalid region indices',data:{region1:connection.region1,region2:connection.region2,region1Index,region2Index,hasRegion1:!!region1,hasRegion2:!!region2,regionsLength:regions.length},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'F'})}).catch(()=>{});
  }
  // #endregion

  const { points, color } = useMemo(() => {
    const startPos = getMappedPosition(region1);
    const endPos = getMappedPosition(region2);
    const start = new THREE.Vector3(...startPos);
    const end = new THREE.Vector3(...endPos);
    
    const mid = new THREE.Vector3().addVectors(start, end).multiplyScalar(0.5);
    const distance = start.distanceTo(end);
    mid.y += distance * 0.3;

    const curve = new THREE.QuadraticBezierCurve3(start, mid, end);
    const curvePoints = curve.getPoints(32);
    // 所有异常连接都显示为红色
    const lineColor = '#ef4444';

    return { points: curvePoints, color: lineColor };
  }, [region1, region2]);

  const geometry = useMemo(() => {
    return new THREE.BufferGeometry().setFromPoints(points);
  }, [points]);

  useFrame((state) => {
    if (lineRef.current && lineRef.current.material) {
      const material = lineRef.current.material as THREE.LineBasicMaterial;
      if (isSelected) {
        const pulse = Math.sin(state.clock.elapsedTime * 3) * 0.3 + 0.7;
        material.opacity = pulse;
      } else {
        material.opacity = 0.6;
      }
    }
  });

  return (
    <line
      ref={lineRef}
      geometry={geometry}
      onClick={(e) => {
        e.stopPropagation();
        setSelectedConnection(isSelected ? null : connection);
      }}
      onPointerOver={() => { document.body.style.cursor = 'pointer'; }}
      onPointerOut={() => { document.body.style.cursor = 'auto'; }}
    >
      <lineBasicMaterial color={color} transparent opacity={0.8} linewidth={isSelected ? 5 : 4} />
    </line>
  );
}

function ConnectionParticle({ 
  connection, 
  isSelected,
  regions,
}: { 
  connection: AbnormalConnection; 
  isSelected: boolean;
  regions: ReturnType<typeof useCC200Regions>['regions'];
}) {
  const particleRef = useRef<THREE.Mesh>(null);
  
  // 注意：connection.region1/region2 可能是 1-based（从 SampleSelector 转换），需要转换为 0-based
  const region1Index = connection.region1 > 0 ? connection.region1 - 1 : connection.region1;
  const region2Index = connection.region2 > 0 ? connection.region2 - 1 : connection.region2;
  const region1 = regions[region1Index];
  const region2 = regions[region2Index];

  const curve = useMemo(() => {
    const startPos = getMappedPosition(region1);
    const endPos = getMappedPosition(region2);
    const start = new THREE.Vector3(...startPos);
    const end = new THREE.Vector3(...endPos);
    const mid = new THREE.Vector3().addVectors(start, end).multiplyScalar(0.5);
    const distance = start.distanceTo(end);
    mid.y += distance * 0.3;
    return new THREE.QuadraticBezierCurve3(start, mid, end);
  }, [region1, region2]);

  // 所有异常连接都显示为红色
  const color = '#ef4444';

  useFrame((state) => {
    if (particleRef.current && isSelected) {
      const t = (state.clock.elapsedTime * 0.5) % 1;
      const point = curve.getPoint(t);
      particleRef.current.position.copy(point);
      const scale = 0.008 + Math.sin(state.clock.elapsedTime * 5) * 0.003;
      particleRef.current.scale.setScalar(scale * 100);
    }
  });

  if (!isSelected) return null;

  return (
    <mesh ref={particleRef}>
      <sphereGeometry args={[0.01, 8, 8]} />
      <meshBasicMaterial color={color} />
    </mesh>
  );
}

export default function Connections() {
  const { predictionResult, selectedConnection, connectionFilter } = useStore();
  const { regions } = useCC200Regions();

  const filteredConnections = useMemo(() => {
    if (!predictionResult) return [];
        // #endregion
    return predictionResult.abnormalConnections.filter(conn => {
      if (connectionFilter === 'all') return true;
      return conn.type === connectionFilter;
    });
  }, [predictionResult, connectionFilter, regions.length]);

  if (regions.length === 0) {
        // #endregion
    return null;
  }

  return (
    <group>
      {filteredConnections.map((connection, idx) => {
        const isSelected = selectedConnection === connection;
        return (
          <group key={idx}>
            <ConnectionLine connection={connection} isSelected={isSelected} regions={regions} />
            <ConnectionParticle connection={connection} isSelected={isSelected} regions={regions} />
          </group>
        );
      })}
    </group>
  );
}
