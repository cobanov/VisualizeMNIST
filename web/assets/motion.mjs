// Presentation time only. Tensor values and their source/destination indices stay intact.
export function easeInOut(t) {
  t = Math.max(0, Math.min(1, t));
  return t * t * t * (t * (t * 6 - 15) + 10);
}

export const cycleSeconds = {input: 1, conv: 1.65, dense: 1.4, flatten: 2.8, softmax: 1.8};

export function transferMotion(phase, order = 0) {
  const local = (phase - .08 - order * .12) / .72;
  return {
    travel: easeInOut((local - .12) / .7),
    visibility: easeInOut(local / .14) * (1 - easeInOut((local - .82) / .18)),
    arrival: easeInOut((phase - .72) / .18) * (1 - easeInOut((phase - .93) / .07)),
  };
}
