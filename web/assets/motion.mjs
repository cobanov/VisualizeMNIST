// Presentation time only. Tensor values and their source/destination indices stay intact.
export function easeInOut(t) {
  t = Math.max(0, Math.min(1, t));
  return t * t * t * (t * (t * 6 - 15) + 10);
}
