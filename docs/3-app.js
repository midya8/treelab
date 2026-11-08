(function(){
 'use strict';
 const $=id=>document.getElementById(id),G=globalThis.TreeGeometry,D=globalThis.TREELAB_DATA;let selected,index=0;
 const number=x=>String(Number(x.toPrecision(8)));
 function fact(label,value){const d=document.createElement('div'),a=document.createElement('span'),b=document.createElement('strong');a.textContent=label;b.textContent=value;d.append(a,b);return d;}
 function option(value,label){const o=document.createElement('option');o.value=value;o.textContent=label;return o;}
 function svg(tag,attrs,text){const n=document.createElementNS('http://www.w3.org/2000/svg',tag);for(const[k,v]of Object.entries(attrs))n.setAttribute(k,v);if(text!==undefined)n.textContent=text;return n;}
 function layout(){const width=Math.max(260,$('chart').clientWidth),height=width<600?290:360;return {width,height,left:58,top:28,plotWidth:width-78,plotHeight:height-100};}
 function chart(){
  const L=layout(),plot=G.path(selected,L.plotWidth,L.plotHeight),root=$('chart-content');root.replaceChildren();$('chart').setAttribute('viewBox',`0 0 ${L.width} ${L.height}`);
  for(let i=0;i<3;i++){const y=L.top+L.plotHeight-i/2*L.plotHeight,value=plot.range[0]+i/2*(plot.range[1]-plot.range[0]);root.append(svg('line',{x1:L.left,x2:L.left+L.plotWidth,y1:y,y2:y,stroke:'#eae9ea'}),svg('text',{x:L.left-8,y:y+5,'text-anchor':'end','font-size':14,fill:'#585254'},String(Number(value.toPrecision(4)))));}
  root.append(svg('polyline',{points:plot.points.map(([x,y])=>`${x+L.left},${y+L.top}`).join(' '),fill:'none',stroke:'#251f21','stroke-width':2.5,id:'curve'}));
  for(let i=0;i<3;i++){const value=plot.domain[0]+i/2*(plot.domain[1]-plot.domain[0]);root.append(svg('text',{x:L.left+i/2*L.plotWidth,y:L.top+L.plotHeight+25,'text-anchor':'middle','font-size':14,fill:'#585254'},String(Number(value.toPrecision(5)))));}
  root.append(svg('text',{x:L.left+L.plotWidth/2,y:L.height-10,'text-anchor':'middle','font-size':14,fill:'#585254'},selected.feature_name),svg('text',{x:L.left,y:16,'font-size':14,fill:'#585254'},'Model output / score units'));
  $('chart-title').textContent=selected.feature_name+' response / '+selected.model+' / row '+selected.row_id;
  $('chart-desc').textContent=`${selected.segments.length} complete leaf segments from ${selected.domain[0]} to ${selected.domain[1]}. Not uncertainty or a causal effect. Exact values follow in the interval inspector.`;
 }

 function inspect(i){
  index=i;const s=G.segment(selected,index),w=G.witness(selected,index);$('segment').value=index;$('segment-label').textContent=`${index+1} of ${selected.segments.length}`;
  $('segment-info').replaceChildren(fact('Exact interval',G.interval(s)),fact('Representable axis witness',String(s.witness)),fact('Member-mean output',String(s.prediction)));
  $('previous').disabled=index===0;$('next').disabled=index===selected.segments.length-1;$('witness').textContent=JSON.stringify(w,null,2);$('copy-status').textContent='';
  $('members').replaceChildren(...s.member_predictions.map((v,j)=>{const li=document.createElement('li');li.textContent=`Tree ${j+1}: output ${v}; leaf ID ${s.member_leaf_ids[j]}`;return li;}));
  for(const old of $('chart-content').querySelectorAll('#highlight,#marker'))old.remove();const L=layout(),plot=G.path(selected,L.plotWidth,L.plotHeight),[lo,hi]=selected.domain,x=v=>hi===lo?L.left+L.plotWidth/2:L.left+(v-lo)/(hi-lo)*L.plotWidth,y=L.top+L.plotHeight-(s.prediction-plot.range[0])/(plot.range[1]-plot.range[0])*L.plotHeight;
  $('chart-content').append(svg('line',{id:'highlight',x1:x(s.lower),x2:x(s.upper),y1:y,y2:y,stroke:'#73a89a','stroke-width':6}),svg('circle',{id:'marker',cx:x(s.witness),cy:y,r:4,fill:'#73a89a'}));
 }
 function render(){
  selected=G.select(D.report,Number($('row').value),$('model').value,Number($('feature').value));$('domain').textContent=`[${selected.domain[0]}, ${selected.domain[1]}]`;$('base').textContent=number(selected.base_prediction);$('counts').textContent=`${selected.segments.length} / ${selected.members}`;
  $('axis-title').textContent=`${selected.feature_name}; only this coordinate varies. All other ${selected.row.length-1} inputs are fixed.`;$('extrema-summary').textContent=`Attainable minimum ${number(selected.minimum.prediction)}; maximum ${number(selected.maximum.prediction)}. Each button selects an actual common-axis witness.`;
  $('segment').max=selected.segments.length-1;const observed=G.locate(selected,selected.row[selected.feature]);$('observed').disabled=observed<0;
  $('all-segments').replaceChildren(...selected.segments.map((s,i)=>{const li=document.createElement('li'),button=document.createElement('button');button.type='button';button.textContent=`${G.interval(s)} → ${s.prediction}`;button.addEventListener('click',()=>inspect(i));li.append(button);return li;}));chart();inspect(observed<0?0:observed);
 }
 try{
  if(!D||!G)throw new Error('Bundled report or geometry code is unavailable');
  $('row').replaceChildren(...D.report.source_rows.map(x=>option(x,'Row '+x)));$('feature').replaceChildren(...D.report.features.map((x,i)=>option(i,x)));$('feature').value='10';
  $('source-path').textContent=D.source_path;$('source-hash').textContent=D.source_sha256;
  for(const id of ['row','model','feature'])$(id).addEventListener('change',render);
  $('segment').addEventListener('input',()=>inspect(Number($('segment').value)));$('previous').addEventListener('click',()=>inspect(Math.max(0,index-1)));$('next').addEventListener('click',()=>inspect(Math.min(selected.segments.length-1,index+1)));
  $('minimum').addEventListener('click',()=>inspect(selected.minimum.segment));$('maximum').addEventListener('click',()=>inspect(selected.maximum.segment));$('observed').addEventListener('click',()=>inspect(G.locate(selected,selected.row[selected.feature])));
  $('copy').addEventListener('click',async()=>{try{await navigator.clipboard.writeText($('witness').textContent);$('copy-status').textContent='Witness JSON copied.';}catch(e){$('copy-status').textContent='Copy is unavailable. Select the JSON above or use Download witness JSON.';}});
  window.addEventListener('resize',()=>{if(selected){chart();inspect(index);}});
  $('download').addEventListener('click',()=>{const blob=new Blob([$('witness').textContent+'\n'],{type:'application/json'}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=`treelab-row${selected.row_id}-${selected.model}-feature${selected.feature}-segment${index}.json`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);});render();
 }catch(e){$('error').hidden=false;$('error').textContent='Explorer could not start: '+e.message;}
})();
