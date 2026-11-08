'use strict';
(()=>{
 const $=id=>document.getElementById(id),data=globalThis.TREELAB_COUNTERFACTUAL;
 const policies={all_features:'All features may change',alcohol_only:'Only alcohol may change',fixed_alcohol:'Alcohol is immutable',all_fixed:'Every feature is immutable'};
 let current;
 const num=x=>String(Number(Number(x).toPrecision(8)));
 const text=(tag,value)=>{const e=document.createElement(tag);e.textContent=value;return e;};
 const body=()=>JSON.stringify(current,null,2);
 function render(){
  const r=data.report.cases.find(c=>String(c.row_id)===$('cf-row').value&&String(c.target[0])===$('cf-target').value&&c.policy===$('cf-policy').value);
  if(!r)throw Error('Missing frozen request');current=r;
  $('cf-status').dataset.status=r.status;$('cf-status').textContent=r.status==='optimal'?'An optimum exists under these constraints.':'No feasible target input under these constraints.';
  $('cf-base').textContent=num(r.base_prediction);$('cf-output').textContent=r.solution?num(r.solution.prediction):'Not attainable';$('cf-cost').textContent=r.solution?num(r.solution.cost):'No solution';
  $('cf-counts').textContent=r.feasible_target_leaves+' feasible target leaves / '+r.target_leaves+' target-output leaves / '+r.leaf_count+' total leaves.';
  $('cf-explanation').textContent=r.solution?policies[r.policy]+'. Best leaf ID '+r.solution.leaf+'; '+r.solution.changed_features.length+' changed features. Bars show each feature\'s fraction of total displayed weighted cost.':r.target_leaves===0?'No leaf in this stored model outputs a value in the requested band.':'Target-output leaves exist, but every one conflicts with the bounds or immutable features.';
  $('cf-witness').textContent=body();$('cf-copy-status').textContent='';$('cf-changes').replaceChildren();
  if(r.solution){
   const changes=r.solution.changed_features;
   if(!changes.length)$('cf-changes').append(text('p','The stored input already satisfies the target and constraints: exact cost 0.'));
   for(const i of changes){
    const line=text('div','');line.className='change-row';const label=text('strong',data.report.features[i]),values=text('div',String(r.row[i])+' to '+String(r.solution.input[i]));values.className='change-value';
    const cost=r.weights[i]*Math.abs(r.solution.input[i]-r.row[i]),area=text('div',''),track=text('div',''),fill=text('div','');track.className='change-track';fill.className='change-fill';fill.style.width=(r.solution.cost?Math.min(100,100*cost/r.solution.cost):0)+'%';track.append(fill);const caption=text('div','Weighted change '+num(cost));caption.className='change-weight';area.append(track,caption);line.append(label,values,area);$('cf-changes').append(line);
   }
  }
  $('cf-candidates').replaceChildren();$('cf-empty').textContent=r.candidates.length?'':'No candidate witness. Infeasible results are preserved, not replaced by a relaxed target.';
  for(const c of r.candidates){const li=text('li','');li.append(text('strong','Leaf '+c.leaf+' / output '+num(c.prediction)+' / cost '+num(c.cost)));const detail=text('details','');detail.append(text('summary','Full candidate, inclusive box and exact rational cost'),text('pre',JSON.stringify(c,null,2)));li.append(detail);$('cf-candidates').append(li);}
 }
 try{
  if(!data?.report?.cases?.length)throw Error('Bundled report missing');
  for(const row of data.report.source_rows){const o=text('option',String(row));o.value=row;$('cf-row').append(o);}
  for(const id of ['cf-row','cf-target','cf-policy'])$(id).addEventListener('change',render);
  $('cf-target').value='7';
  $('cf-provenance').textContent=JSON.stringify({report:data.source_path,report_sha256:data.source_sha256,model:data.model_path,model_sha256:data.model_sha256},null,2);
  $('cf-copy').addEventListener('click',async()=>{try{await navigator.clipboard.writeText(body());$('cf-copy-status').textContent='Request and result copied.';}catch{$('cf-copy-status').textContent='Copy is unavailable. Select the JSON above or use Download.';}});
  $('cf-download').addEventListener('click',()=>{const url=URL.createObjectURL(new Blob([body()+'\n'],{type:'application/json'})),a=document.createElement('a');a.href=url;a.download='treelab-counterfactual-row'+current.row_id+'-target'+current.target[0]+'-'+current.policy+'.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);});render();
 }catch(e){$('cf-error').hidden=false;$('cf-error').textContent='Cannot display the frozen report: '+e.message;}
})();
