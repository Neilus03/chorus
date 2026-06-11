import { useEffect, useMemo, useRef, type RefObject } from 'react';
import { Canvas, useThree } from '@react-three/fiber';
import type { ThreeEvent } from '@react-three/fiber';
import { GizmoHelper, GizmoViewport, Grid, OrbitControls } from '@react-three/drei';
import type { OrbitControls as OrbitControlsImpl } from 'three-stdlib';
import * as THREE from 'three';
import { gatherFloat3 } from '../lib/subsample';
import { useStore } from '../state/store';

type PointCloudProps = {
  positions: Float32Array;
  colors: Uint8Array;
  sourceIndices: Uint32Array;
  pointSize: number;
};

function FitCamera({ positions, controlsRef }: { positions: Float32Array; controlsRef: RefObject<OrbitControlsImpl | null> }) {
  const camera = useThree(s => s.camera);

  useEffect(() => {
    const fit = () => {
      if (!positions.length) return;
      let minX = Number.POSITIVE_INFINITY;
      let minY = Number.POSITIVE_INFINITY;
      let minZ = Number.POSITIVE_INFINITY;
      let maxX = Number.NEGATIVE_INFINITY;
      let maxY = Number.NEGATIVE_INFINITY;
      let maxZ = Number.NEGATIVE_INFINITY;
      for (let i = 0; i < positions.length; i += 3) {
        const x = positions[i];
        const y = positions[i + 1];
        const z = positions[i + 2];
        if (x < minX) minX = x;
        if (y < minY) minY = y;
        if (z < minZ) minZ = z;
        if (x > maxX) maxX = x;
        if (y > maxY) maxY = y;
        if (z > maxZ) maxZ = z;
      }
      const center = new THREE.Vector3();
      const size = new THREE.Vector3();
      const box = new THREE.Box3(
        new THREE.Vector3(minX, minY, minZ),
        new THREE.Vector3(maxX, maxY, maxZ),
      );
      box.getCenter(center);
      box.getSize(size);
      const radius = Math.max(size.x, size.y, size.z, 1);
      camera.position.set(center.x + radius, center.y - radius, center.z + radius * 0.8);
      camera.near = Math.max(radius / 1000, 0.001);
      camera.far = radius * 20;
      if (camera instanceof THREE.PerspectiveCamera) camera.updateProjectionMatrix();
      if (controlsRef.current) {
        controlsRef.current.target.copy(center);
        controlsRef.current.update();
      } else {
        camera.lookAt(center);
      }
    };
    fit();
    window.addEventListener('chorus-fit-camera', fit);
    return () => window.removeEventListener('chorus-fit-camera', fit);
  }, [camera, controlsRef, positions]);
  return null;
}

function PointCloud({ positions, colors, sourceIndices, pointSize }: PointCloudProps) {
  const setSelected = useStore(s => s.setSelected);
  const invalidate = useThree(s => s.invalidate);
  const geometry = useMemo(() => {
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    g.setAttribute('color', new THREE.Uint8BufferAttribute(new Uint8Array(positions.length), 3, true));
    g.computeBoundingSphere();
    return g;
  }, [positions]);

  useEffect(() => {
    const attr = geometry.getAttribute('color') as THREE.BufferAttribute;
    (attr.array as Uint8Array).set(colors);
    attr.needsUpdate = true;
    invalidate();
  }, [colors, geometry, invalidate]);

  useEffect(() => () => geometry.dispose(), [geometry]);

  const onPointerDown = (event: ThreeEvent<PointerEvent>) => {
    event.stopPropagation();
    if (event.index === undefined) return;
    const sourceIndex = sourceIndices[event.index];
    if (sourceIndex !== undefined) setSelected({ sourceIndex });
  };

  return (
    <points geometry={geometry} onPointerDown={onPointerDown}>
      <pointsMaterial size={pointSize} sizeAttenuation={false} vertexColors />
    </points>
  );
}

type Props = {
  sourceIndices: Uint32Array | null;
  colors: Uint8Array | null;
};

export default function Viewport({ sourceIndices, colors }: Props) {
  const scene = useStore(s => s.scene);
  const pointSize = useStore(s => s.pointSize);
  const controlsRef = useRef<OrbitControlsImpl | null>(null);
  const positions = useMemo(() => {
    if (!scene || !sourceIndices) return null;
    return gatherFloat3(scene.points, sourceIndices);
  }, [scene, sourceIndices]);
  const raycasterRef = useRef<THREE.Raycaster | null>(null);

  return (
    <div className="viewport-canvas">
      {!scene || !positions || !colors || !sourceIndices ? (
        <div className="viewport-empty">Open a scene or inspection bundle.</div>
      ) : (
        <Canvas
          frameloop="demand"
          dpr={[1, 1.5]}
          camera={{ position: [2, -2, 2], fov: 55, near: 0.01, far: 1000 }}
          gl={{ antialias: false, powerPreference: 'high-performance', preserveDrawingBuffer: true }}
          onCreated={({ gl, raycaster }) => {
            gl.setClearColor('#080b10');
            raycaster.params.Points = { threshold: 0.06 };
            raycasterRef.current = raycaster;
          }}
        >
          <ambientLight intensity={0.8} />
          <PointCloud positions={positions} colors={colors} sourceIndices={sourceIndices} pointSize={pointSize} />
          <Grid args={[20, 20]} cellColor="#29313d" sectionColor="#3f4b5d" fadeDistance={50} fadeStrength={1.5} />
          <OrbitControls ref={controlsRef} makeDefault />
          <GizmoHelper alignment="bottom-right" margin={[78, 78]}>
            <GizmoViewport axisColors={['#ef4444', '#22c55e', '#60a5fa']} labelColor="#e6edf3" />
          </GizmoHelper>
          <FitCamera positions={positions} controlsRef={controlsRef} />
        </Canvas>
      )}
    </div>
  );
}
