'use client';

import { useMemo } from 'react';
import * as THREE from 'three';
import { ConvexGeometry } from 'three/examples/jsm/geometries/ConvexGeometry.js';
import { useCC200Regions } from '@/lib/useCC200Regions';
import { getMappedPosition } from './BrainModel';

// 使用凸包连接最外层的点，形成大脑轮廓
export default function ConvexHull() {
  const { regions: CC200_REGIONS } = useCC200Regions();

  // 生成凸包几何体
  const convexGeometry = useMemo(() => {
    if (CC200_REGIONS.length === 0) {
      return null;
    }

    // 将所有CC200坐标点转换为Three.js Vector3数组
    const points = CC200_REGIONS.map(region => {
      const pos = getMappedPosition(region);
      return new THREE.Vector3(pos[0], pos[1], pos[2]);
    });

    // 使用ConvexGeometry生成凸包
    const geometry = new ConvexGeometry(points);
    return geometry;
  }, [CC200_REGIONS]);

  if (!convexGeometry) {
    return null;
  }

  return (
    <group>
      {/* 凸包表面 */}
      <mesh geometry={convexGeometry}>
        <meshStandardMaterial
          color="#1e293b"
          transparent
          opacity={0.3}
          side={THREE.DoubleSide}
          roughness={0.7}
          metalness={0.1}
        />
      </mesh>
      {/* 凸包线框，显示边缘 */}
      <mesh geometry={convexGeometry}>
        <meshBasicMaterial
          color="#06b6d4"
          transparent
          opacity={0.4}
          wireframe
          side={THREE.DoubleSide}
        />
      </mesh>
    </group>
  );
}
