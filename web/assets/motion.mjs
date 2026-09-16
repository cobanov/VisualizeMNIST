// Camera presentation only. Tensor values and their source/destination indices stay intact.
export function easeInOut(t) {
  t = Math.max(0, Math.min(1, t));
  return t * t * t * (t * (t * 6 - 15) + 10);
}

// Center the projected silhouette, accounting for each corner's perspective depth.
export function perspectiveCenter(points, distance) {
  return ['x', 'y'].map(axis => {
    let low = Math.min(...points.map(p => p[axis]));
    let high = Math.max(...points.map(p => p[axis]));
    for (let i = 0; i < 32; i++) {
      const center = (low + high) / 2;
      const projected = points.map(p => (p[axis] - center) / (distance - p.z));
      if (Math.min(...projected) + Math.max(...projected) > 0) low = center;
      else high = center;
    }
    return (low + high) / 2;
  });
}
