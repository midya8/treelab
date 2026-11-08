/* Stored response geometry only. No re-fitting or new model inference. MIT. */
(function(root){
 'use strict';
 function select(report,row,model,feature){
  if(!Number.isInteger(row)||!Number.isInteger(feature)||!['selected_cart','bagging_32'].includes(model))throw new Error('Invalid slice selection');
  const slice=report.sweeps.find(s=>s.row_id===row&&s.model===model&&s.feature===feature);
  if(!slice)throw new Error('Slice not found');return slice;
 }
 function segment(slice,index){
  if(!Number.isInteger(index)||index<0||index>=slice.segments.length)throw new Error('Invalid segment');return slice.segments[index];
 }
 function locate(slice,x){
  if(typeof x!=='number'||!Number.isFinite(x))throw new Error('Axis value must be finite');
  return slice.segments.findIndex(s=>(x>s.lower||(s.lower_inclusive&&x===s.lower))&&x<=s.upper);
 }
 function witness(slice,index){const s=segment(slice,index),row=slice.row.slice();row[slice.feature]=s.witness;return {row,feature:slice.feature,axis_value:s.witness,prediction:s.prediction,segment:index,model:slice.model,source_row:slice.row_id,domain:slice.domain.slice(),meaning:slice.meaning};}
 function interval(s){return `${s.lower_inclusive?'[':'('}${s.lower}, ${s.upper}]`;}
 function path(slice,width=840,height=300){
  if(!(width>0&&height>0))throw new Error('Invalid plot dimensions');
  const [lo,hi]=slice.domain,min=Math.min(...slice.segments.map(s=>s.prediction)),max=Math.max(...slice.segments.map(s=>s.prediction)),pad=Math.max((max-min)*.12,.1),a=min-pad,b=max+pad;
  const x=v=>hi===lo?width/2:(v-lo)/(hi-lo)*width,y=v=>height-(v-a)/(b-a)*height;
  return {points:slice.segments.flatMap(s=>[[x(s.lower),y(s.prediction)],[x(s.upper),y(s.prediction)]]),range:[a,b],domain:[lo,hi]};
 }
 const api={select,segment,locate,witness,interval,path};root.TreeGeometry=api;if(typeof module==='object')module.exports=api;
})(globalThis);
