// Scalar inspection (GELU uses an erf approximation within float32 tolerance) for the compact residual network and patch Transformer.
export const activate = (x, kind) => kind === 'relu' ? Math.max(0,x) : kind === 'gelu' ? .5*x*(1+erf(x/Math.SQRT2)) : x;
function erf(x) {
  const sign=Math.sign(x),a=Math.abs(x),t=1/(1+.3275911*a);
  return sign*(1-(((((1.061405429*t-1.453152027)*t)+1.421413741)*t-.284496736)*t+.254829592)*t*Math.exp(-a*a));
}
export function patchIndex(patch,y,x,size=7,grid=4) {
  return (Math.floor(patch/grid)*size+y)*(grid*size)+(patch%grid)*size+x;
}
export function tokenTerms(input,weights,row,inputWidth,outputWidth,column) {
  return Array.from({length:inputWidth},(_,i)=>({i:row*inputWidth+i,value:input[row*inputWidth+i],weight:weights[i*outputWidth+column],contribution:input[row*inputWidth+i]*weights[i*outputWidth+column]}));
}
export function normValue(input,row,width,column,gamma,beta,epsilon=1e-5) {
  const values=input.subarray(row*width,(row+1)*width);
  const mean=values.reduce((s,v)=>s+v,0)/width;
  const variance=values.reduce((s,v)=>s+(v-mean)**2,0)/width;
  return {mean,variance,value:(values[column]-mean)/Math.sqrt(variance+epsilon)*gamma[column]+beta[column]};
}
export function attentionRow(q,k,head,query,tokens,headDim) {
  const scores=Array.from({length:tokens},(_,key)=>{
    let sum=0;
    for(let d=0;d<headDim;d++)sum+=q[(head*tokens+query)*headDim+d]*k[(head*tokens+key)*headDim+d];
    return sum/Math.sqrt(headDim);
  });
  const max=Math.max(...scores),exp=scores.map(x=>Math.exp(x-max)),sum=exp.reduce((a,b)=>a+b,0);
  return {scores,probabilities:exp.map(v=>v/sum)};
}
