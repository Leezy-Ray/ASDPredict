'use client';

import { useRef, useMemo } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import { useStore } from '@/store/useStore';
import { LOBE_COLORS } from '@/lib/cc200-regions';
import { useCC200Regions } from '@/lib/useCC200Regions';
import type { BrainRegion } from '@/lib/cc200-regions';

// 简化的坐标映射：直接将MNI坐标转换为Three.js坐标
// MNI空间范围大约：x: -70 to 70, y: -100 to 70, z: -50 to 80 (单位：mm)
// 我们需要将其缩放到适合Three.js场景的大小
const MNI_SCALE = 0.01; // 将毫米转换为Three.js单位

export function getMappedPosition(region: { x: number; y: number; z: number }): [number, number, number] {
  // 直接使用MNI坐标，进行缩放和坐标轴转换
  // Three.js: Y向上，Z向前
  // MNI: X左右，Y前后，Z上下
  return [
    region.x * MNI_SCALE,      // X保持不变
    region.z * MNI_SCALE,      // MNI的Z对应Three.js的Y（向上）
    -region.y * MNI_SCALE,     // MNI的Y对应Three.js的Z（向前，需要翻转）
  ];
}

function BrainRegionPoint({ 
  region, 
  isHighlighted,
  isInvolved,
}: { 
  region: BrainRegion;
  isHighlighted: boolean;
  isInvolved: boolean;
}) {
  const meshRef = useRef<THREE.Mesh>(null);
  const { setHoveredRegion, setSelectedRegion, hoveredRegion } = useStore();
  
  // 直接使用映射后的坐标
  const position = useMemo(() => getMappedPosition(region), [region]);
  const color = LOBE_COLORS[region.lobe];
  const isHovered = hoveredRegion?.id === region.id;

  useFrame(() => {
    if (meshRef.current) {
      const targetScale = isHovered ? 1.5 : isHighlighted ? 1.3 : isInvolved ? 1.2 : 1;
      meshRef.current.scale.lerp(new THREE.Vector3(targetScale, targetScale, targetScale), 0.1);
    }
  });

  return (
    <mesh
      ref={meshRef}
      position={position}
      onPointerOver={(e) => {
        e.stopPropagation();
        setHoveredRegion(region);
        document.body.style.cursor = 'pointer';
      }}
      onPointerOut={() => {
        setHoveredRegion(null);
        document.body.style.cursor = 'auto';
      }}
      onClick={(e) => {
        e.stopPropagation();
        setSelectedRegion(region);
      }}
    >
      <sphereGeometry args={[0.02, 16, 16]} />
      <meshStandardMaterial
        color={color}
        emissive={color}
        emissiveIntensity={isHovered ? 1.0 : isHighlighted ? 0.7 : isInvolved ? 0.5 : 0.3}
        transparent
        opacity={isInvolved ? 1 : 0.8}
        roughness={0.2}
        metalness={0.8}
      />
    </mesh>
  );
}

export default function BrainModel() {
  const { predictionResult, selectedConnection, showAllRegions } = useStore();
  const { regions: CC200_REGIONS } = useCC200Regions();

  const involvedRegions = useMemo(() => {
    if (!predictionResult) return new Set<number>();
    const ids = new Set<number>();
    predictionResult.abnormalConnections.forEach(conn => {
      ids.add(conn.region1);
      ids.add(conn.region2);
    });
    return ids;
  }, [predictionResult]);

  const highlightedRegions = useMemo(() => {
    if (!selectedConnection) return new Set<number>();
    return new Set([selectedConnection.region1, selectedConnection.region2]);
  }, [selectedConnection]);

  // 计算所有点的中心，用于自动调整相机位置
  const center = useMemo(() => {
    if (CC200_REGIONS.length === 0) return [0, 0, 0];
    let sumX = 0, sumY = 0, sumZ = 0;
    CC200_REGIONS.forEach(region => {
      const pos = getMappedPosition(region);
      sumX += pos[0];
      sumY += pos[1];
      sumZ += pos[2];
    });
    return [
      sumX / CC200_REGIONS.length,
      sumY / CC200_REGIONS.length,
      sumZ / CC200_REGIONS.length,
    ];
  }, [CC200_REGIONS]);

  return (
    <group>
      {/* 绘制所有CC200区域点 */}
      {CC200_REGIONS.map((region) => {
        const isInvolved = involvedRegions.has(region.id);
        const isHighlighted = highlightedRegions.has(region.id);
        
        if (!showAllRegions && !isInvolved && predictionResult) {
          return null;
        }

        return (
          <BrainRegionPoint
            key={region.id}
            region={region}
            isHighlighted={isHighlighted}
            isInvolved={isInvolved}
          />
        );
      })}
    </group>
  );
}
