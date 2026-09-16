// Presentation time only. Tensor values and their source/destination indices stay intact.
export function easeInOut(t) {
  t = Math.max(0, Math.min(1, t));
  return t * t * t * (t * (t * 6 - 15) + 10);
}

// One complete channel/vector pass, independent of its number of cells.
export const cycleSeconds = {input: 1, conv: 4.2, dense: 3.2, flatten: 2.8, softmax: 2};
export function flowAt(phase, count) {
  const cells=[];
  if(phase<0||phase>=1)return cells;
  for(let index=0;index<count;index++){
    const local=(phase-.76*index/Math.max(1,count-1))/.24;
    if(local>=0&&local<1)cells.push({index,phase:local});
  }
  return cells;
}

export function transferMotion(phase, order = 0) {
  const local = (phase - .08 - order * .12) / .72;
  return {
    travel: easeInOut((local - .12) / .7),
    visibility: easeInOut(local / .14) * (1 - easeInOut((local - .82) / .18)),
    arrival: easeInOut((phase - .72) / .18) * (1 - easeInOut((phase - .93) / .07)),
  };
}

// A single input-to-prediction pass. Completed playback stays at the final stage.
export function playbackAt(seconds, layers) {
  for(let index=1;index<layers.length;index++){
    const duration=cycleSeconds[layers[index].op];
    if(seconds<duration)return {index,phase:seconds/duration,done:false};
    seconds-=duration;
  }
  return {index:layers.length-1,phase:1,done:true};
}
