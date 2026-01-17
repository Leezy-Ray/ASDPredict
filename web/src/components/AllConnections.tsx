'use client';

import { useMemo, useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import { useStore } from '@/store/useStore';
import { useCC200Regions } from '@/lib/useCC200Regions';
import { getMappedPosition } from './BrainModel';

// 所有区域之间的连接线（使用LineSegments优化性能）
export default function AllConnections() {
  const { showAllConnections } = useStore();
  const { regions: CC200_REGIONS } = useCC200Regions();
  const lineRef = useRef<THREE.LineSegments>(null);

  // 生成所有区域之间的连接
  const { positions, colors } = useMemo(() => {
    if (!showAllConnections || CC200_REGIONS.length === 0) {
      return { positions: new Float32Array(0), colors: new Float32Array(0) };
    }

    const positionArray: number[] = [];
    const colorArray: number[] = [];
    
    // 为每个区域对创建连接（全连接）
    for (let i = 0; i < CC200_REGIONS.length; i++) {
      for (let j = i + 1; j < CC200_REGIONS.length; j++) {
        const region1 = CC200_REGIONS[i];
        const region2 = CC200_REGIONS[j];
        
        // 获取坐标位置
        const startPos = getMappedPosition(region1);
        const endPos = getMappedPosition(region2);
        
        // 添加起点和终点
        positionArray.push(startPos[0], startPos[1], startPos[2]);
        positionArray.push(endPos[0], endPos[1], endPos[2]);
        
        // 根据脑叶设置颜色（使用两个区域中第一个的颜色）
        const lobeColor = getLobeColor(region1.lobe);
        const r = parseInt(lobeColor.slice(1, 3), 16) / 255;
        const g = parseInt(lobeColor.slice(3, 5), 16) / 255;
        const b = parseInt(lobeColor.slice(5, 7), 16) / 255;
        
        // 起点颜色
        colorArray.push(r, g, b);
        // 终点颜色（稍微淡一些）
        colorArray.push(r * 0.7, g * 0.7, b * 0.7);
      }
    }
    
    return {
      positions: new Float32Array(positionArray),
      colors: new Float32Array(colorArray),
    };
  }, [showAllConnections, CC200_REGIONS]);

  const geometry = useMemo(() => {
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    return geo;
  }, [positions, colors]);

  useFrame(() => {
    // 可以添加动画效果
  });

  if (!showAllConnections || positions.length === 0) {
    return null;
  }

  return (
    <lineSegments ref={lineRef} geometry={geometry}>
      <lineBasicMaterial
        vertexColors
        transparent
        opacity={0.15}
        linewidth={1}
      />
    </lineSegments>
  );
}

// 获取脑叶颜色
function getLobeColor(lobe: string): string {
  const colors: Record<string, string> = {
    frontal: '#06b6d4',
    parietal: '#8b5cf6',
    temporal: '#f59e0b',
    occipital: '#10b981',
    subcortical: '#f472b6',
    cerebellum: '#6366f1',
  };
  return colors[lobe] || '#ffffff';
}
