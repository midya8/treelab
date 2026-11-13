'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),os=require('node:os'),path=require('node:path'),{spawn}=require('node:child_process');
const delay=ms=>new Promise(resolve=>setTimeout(resolve,ms));
test('real Chrome: all 36 counterfactual requests, complete candidates, copy fallback, download and mobile layouts', {timeout:90000}, async()=>{
 const profile=fs.mkdtempSync(path.join(os.tmpdir(),'treelab-browser-'));
 const browser=spawn(process.env.CHROME_BIN||'google-chrome',['--headless','--no-sandbox','--disable-gpu','--disable-extensions','--disable-dev-shm-usage','--remote-debugging-port=0',`--user-data-dir=${profile}`,'about:blank'],{stdio:['ignore','ignore','pipe']});let ws,startupError,diagnostics='';browser.on('error',e=>startupError=e);browser.stderr.on('data',d=>diagnostics=(diagnostics+d).slice(-4000));
 try{
  for(let i=0;i<1200&&!startupError&&browser.exitCode===null&&!fs.existsSync(path.join(profile,'DevToolsActivePort'));i++)await delay(25);
  assert.ok(fs.existsSync(path.join(profile,'DevToolsActivePort')),`Chrome debugger did not start: ${startupError?.message||diagnostics||browser.exitCode}`);const port=fs.readFileSync(path.join(profile,'DevToolsActivePort'),'utf8').split('\n')[0];
  // The port file can appear before the debugger starts accepting requests.
  let tab,lastDebuggerError;
  for(let i=0;i<200&&browser.exitCode===null&&!tab;i++){
   try{const response=await fetch(`http://127.0.0.1:${port}/json/list`,{signal:AbortSignal.timeout(1000)});if(!response.ok)throw new Error('Debugger HTTP '+response.status);const tabs=await response.json();tab=tabs.find(t=>t.type==='page'&&t.webSocketDebuggerUrl);}catch(error){lastDebuggerError=error;}
   if(!tab)await delay(25);
  }
  assert.ok(tab,`Chrome debugger page did not become ready: ${lastDebuggerError?.message||diagnostics||browser.exitCode}`);
  ws=new WebSocket(tab.webSocketDebuggerUrl);await new Promise((res,rej)=>{ws.onopen=res;ws.onerror=rej;});
  let seq=0;const pending=new Map();ws.onmessage=e=>{const m=JSON.parse(e.data);if(pending.has(m.id)){const {res,rej}=pending.get(m.id);pending.delete(m.id);m.error?rej(new Error(JSON.stringify(m.error))):res(m.result);}};
  const call=(method,params={})=>new Promise((res,rej)=>{const id=++seq;pending.set(id,{res,rej});ws.send(JSON.stringify({id,method,params}));});
  const js=async expression=>{const r=await call('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});if(r.exceptionDetails)throw new Error(JSON.stringify(r.exceptionDetails));return r.result.value;};
  await call('Page.enable');await call('Page.navigate',{url:process.env.TREELAB_URL||'file://'+path.resolve(__dirname,'../docs/counterfactual.html')});
  for(let i=0;i<200;i++){if(await js(`document.querySelectorAll('#cf-candidates li').length>0`))break;await delay(20);}
  assert.equal(await js(`document.getElementById('cf-error').hidden`),true);
  const checked=await js(`(()=>{let cases=0,candidates=0;for(const c of TREELAB_COUNTERFACTUAL.report.cases){for(const [id,v] of [['cf-row',c.row_id],['cf-target',c.target[0]],['cf-policy',c.policy]]){document.getElementById(id).value=v;document.getElementById(id).dispatchEvent(new Event('change'));}const actual=JSON.parse(document.getElementById('cf-witness').textContent);if(JSON.stringify(actual)!==JSON.stringify(c))throw Error('full witness mismatch');if(document.getElementById('cf-status').dataset.status!==c.status)throw Error('status mismatch');if(document.querySelectorAll('#cf-candidates li').length!==c.candidates.length)throw Error('candidate count');if(document.querySelectorAll('.change-row').length!==(c.solution?.changed_features.length||0))throw Error('feature changes');for(let i=0;i<c.candidates.length;i++){const stored=JSON.parse(document.querySelectorAll('#cf-candidates li pre')[i].textContent);if(JSON.stringify(stored)!==JSON.stringify(c.candidates[i]))throw Error('candidate data mismatch');candidates++;}cases++;}return {cases,candidates};})()`);
  assert.equal(checked.cases,36);assert.ok(checked.candidates>0);
  await js(`document.getElementById('cf-row').value='1';document.getElementById('cf-target').value='7';document.getElementById('cf-policy').value='all_features';document.getElementById('cf-policy').dispatchEvent(new Event('change'));`);
  assert.equal(await js(`document.querySelectorAll('.change-row').length`),3);
  await js(`Object.defineProperty(navigator,'clipboard',{value:{writeText:async()=>{throw Error('denied')}},configurable:true});document.getElementById('cf-copy').click()`);await delay(20);assert.match(await js(`document.getElementById('cf-copy-status').textContent`),/Copy is unavailable/);
  await js(`Object.defineProperty(navigator,'clipboard',{value:{writeText:async text=>globalThis.copied=text},configurable:true});document.getElementById('cf-copy').click()`);await delay(20);assert.equal(await js(`globalThis.copied`),await js(`document.getElementById('cf-witness').textContent`));
  await js(`HTMLAnchorElement.prototype.click=function(){globalThis.downloaded={name:this.download,url:this.href}};document.getElementById('cf-download').click()`);assert.equal(await js(`globalThis.downloaded.name`),'treelab-counterfactual-row1-target7-all_features.json');assert.match(await js(`globalThis.downloaded.url`),/^blob:/);
  await js(`for(const e of document.querySelectorAll('details'))e.open=true`);
  for(const width of [320,390,1280]){await call('Emulation.setDeviceMetricsOverride',{width,height:1000,deviceScaleFactor:1,mobile:width<700});await delay(30);assert.ok(await js(`document.documentElement.scrollWidth<=innerWidth`),'overflow at '+width);assert.ok(await js(`Array.from(document.querySelectorAll('button,select,summary')).every(e=>e.getBoundingClientRect().height>=44)`),'touch targets at '+width);}
  await js(`for(const e of document.querySelectorAll('details'))e.open=false;window.scrollTo(0,0)`);
  if(process.env.TREELAB_SCREENSHOT){const width=Number(process.env.TREELAB_SCREENSHOT_WIDTH||1280);await call('Emulation.setDeviceMetricsOverride',{width,height:1200,deviceScaleFactor:1,mobile:width<700});await delay(50);const metrics=await call('Page.getLayoutMetrics');const image=await call('Page.captureScreenshot',{format:'png',captureBeyondViewport:true,clip:{x:0,y:0,width,height:metrics.cssContentSize.height,scale:1}});fs.writeFileSync(process.env.TREELAB_SCREENSHOT,Buffer.from(image.data,'base64'));}
 }finally{if(ws)ws.close();if(browser.pid&&browser.exitCode===null){browser.kill('SIGTERM');await new Promise(resolve=>browser.once('exit',resolve));}await fs.promises.rm(profile,{recursive:true,force:true,maxRetries:10,retryDelay:200});}
});
