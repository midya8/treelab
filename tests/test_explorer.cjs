'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto');
require('../docs/data.js');const G=require('../docs/geometry.js'),D=globalThis.TREELAB_DATA,raw=fs.readFileSync(path.join(__dirname,'../treelab/artifacts/sweep-report.json'));
test('all 66 complete slices, witnesses, membership values and exact source checksum',()=>{
 assert.equal(D.source_sha256,crypto.createHash('sha256').update(raw).digest('hex'));assert.deepEqual(D.report,JSON.parse(raw));assert.equal(D.report.sweeps.length,66);
 let segments=0;
 for(const s of D.report.sweeps){assert.equal(G.select(D.report,s.row_id,s.model,s.feature),s);const p=G.path(s);assert.equal(p.points.length,2*s.segments.length);
  for(let i=0;i<s.segments.length;i++){const a=G.segment(s,i),w=G.witness(s,i);segments++;assert.equal(G.locate(s,a.witness),i);assert.equal(G.locate(s,a.upper),i);assert.equal(G.locate(s,a.lower),a.lower_inclusive?i:i-1);assert.equal(w.row[s.feature],a.witness);assert.equal(w.prediction,a.prediction);assert.equal(a.member_predictions.length,s.members);assert.equal(a.member_leaf_ids.length,s.members);
   for(let j=0;j<s.row.length;j++)if(j!==s.feature)assert.equal(w.row[j],s.row[j]);
   assert.match(G.interval(a),a.lower_inclusive?/^\[/:/^\(/);assert.ok(G.interval(a).endsWith(']'));assert.ok(p.points[2*i][0]<=p.points[2*i+1][0]);
  }
  assert.equal(G.witness(s,s.minimum.segment).prediction,s.minimum.prediction);assert.equal(G.witness(s,s.maximum.segment).prediction,s.maximum.prediction);assert.equal(G.locate(s,s.domain[0]-1),-1);assert.equal(G.locate(s,s.domain[1]+1),-1);
 }assert.equal(segments,1028);
});
test('selection, interval and chart validation plus singleton domain',()=>{
 const s=D.report.sweeps[0];for(const i of [-1,1.2,s.segments.length,true])assert.throws(()=>G.segment(s,i));for(const x of [NaN,Infinity,'1'])assert.throws(()=>G.locate(s,x));assert.throws(()=>G.select(D.report,999,'bagging_32',0));assert.throws(()=>G.select(D.report,1,'bad',0));assert.throws(()=>G.select(D.report,true,'bagging_32',0));assert.throws(()=>G.path(s,0,3));const one={domain:[1,1],segments:[{lower:1,upper:1,lower_inclusive:true,witness:1,prediction:2}]};assert.equal(G.locate(one,1),0);assert.deepEqual(G.path(one).points,[[420,150],[420,150]]);
});
