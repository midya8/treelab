'use strict';
(()=>{
 const $=id=>document.getElementById(id),data=globalThis.TREELAB_FOREST;let current;
 const num=x=>String(Number(Number(x).toPrecision(8))),rat=x=>x?num(Number(x.numerator)/Number(x.denominator)):'Not available';
 const text=(tag,v)=>{const e=document.createElement(tag);e.textContent=v;return e;},body=()=>JSON.stringify(current,null,2);
 function render(){
  const r=data.report.cases.find(c=>String(c.row_id)===$('fs-row').value&&String(c.target[0])===$('fs-target').value&&c.policy===$('fs-policy').value&&String(c.max_states)===$('fs-budget').value);if(!r)throw Error('Missing frozen search');current=r;
  const labels={optimal:'Global optimum under these constraints.',infeasible:'No target input under these constraints.',budget_exhausted:'Budget reached. The search is unfinished.'};
  $('fs-status').textContent=labels[r.status];$('fs-status').dataset.status=r.status;
  $('fs-meaning').textContent=r.status==='optimal'?'A verified witness and a frontier proof agree at the same exact cost.':r.status==='infeasible'?'The frontier is exhausted without a feasible witness. This is a completed result, not a sampling failure.':r.solution?'A verified witness gives an upper bound. A better input may remain in the frontier.':'No witness was found within this search budget. This does not establish that the target is unreachable.';
  $('fs-lower').textContent=rat(r.objective_lower_exact);$('fs-upper').textContent=rat(r.objective_upper_exact);$('fs-gap').textContent=rat(r.absolute_gap_exact);
  $('fs-bounds').textContent=JSON.stringify({lower:r.objective_lower_exact,upper:r.objective_upper_exact,gap:r.absolute_gap_exact,complete:r.complete,expanded_states:r.expanded_states,frontier_states:r.frontier_states,generated_states:r.generated_states,peak_frontier_states:r.peak_frontier_states,seed_evaluations:r.seed_evaluations,feasible_seeds:r.feasible_seeds,pruned:r.pruned},null,2);
  $('fs-witness').textContent=body();$('fs-copy-status').textContent='';$('fs-changes').replaceChildren();$('fs-members').replaceChildren();
  $('fs-witness-summary').textContent=r.solution?'Stored row output '+num(r.base_prediction)+'; witness output '+num(r.solution.prediction)+'; '+r.solution.changed_features.length+' changed features. Every displayed member is evaluated at this same full input.':'Stored row output '+num(r.base_prediction)+'. No witness to display. Null is not replaced by an invented or relaxed target.';
  if(r.solution){
   for(const i of r.solution.changed_features){const e=text('div','');e.className='fs-change';e.append(text('strong',data.report.features[i]),text('code',String(r.row[i])+' to '+String(r.solution.input[i])));$('fs-changes').append(e);}
   for(let i=0;i<r.members;i++)$('fs-members').append(text('li','Tree '+(i+1)+': output '+r.solution.member_predictions[i]+'; leaf ID '+r.solution.member_leaf_ids[i]));
  }
  $('fs-history').replaceChildren();
  for(const c of data.report.cases.filter(c=>c.row_id===r.row_id&&c.target[0]===r.target[0]&&c.policy===r.policy)){
   const li=text('li',''),b=text('button','Budget '+c.max_states+' / '+c.status.replaceAll('_',' ')),s=text('small','Expanded '+c.expanded_states+'; lower '+rat(c.objective_lower_exact)+'; upper '+rat(c.objective_upper_exact)+'; gap '+rat(c.absolute_gap_exact));b.type='button';b.append(s);b.addEventListener('click',()=>{$('fs-budget').value=c.max_states;render();});li.append(b);$('fs-history').append(li);
  }
 }
 try{
  if(!data?.report?.cases?.length)throw Error('Bundled report missing');for(const id of data.report.source_rows){const o=text('option',String(id));o.value=id;$('fs-row').append(o);}
  $('fs-budget').value='4096';
  for(const id of ['fs-row','fs-target','fs-policy','fs-budget'])$(id).addEventListener('change',render);
  const provenance={};for(const [k,v] of Object.entries(data))if(k!=='report')provenance[k]=v;$('fs-provenance').textContent=JSON.stringify(provenance,null,2);
  $('fs-copy').addEventListener('click',async()=>{try{await navigator.clipboard.writeText(body());$('fs-copy-status').textContent='Search JSON copied.';}catch{$('fs-copy-status').textContent='Copy is unavailable. Select the JSON above or use Download.';}});
  $('fs-download').addEventListener('click',()=>{const url=URL.createObjectURL(new Blob([body()+'\n'],{type:'application/json'})),a=document.createElement('a');a.href=url;a.download='treelab-forest-row'+current.row_id+'-target'+current.target[0]+'-'+current.policy+'-budget'+current.max_states+'.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);});render();
 }catch(e){$('fs-error').hidden=false;$('fs-error').textContent='Cannot display the frozen report: '+e.message;}
})();
