const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

function loader(fetch) {
  const context = {module: {exports: {}}, fetch, AbortSignal,
    createImageBitmap: async blob => ({blob, closed: false, close() {this.closed = true;}})};
  vm.createContext(context);
  vm.runInContext(fs.readFileSync(require.resolve('../../viewer/references.js'), 'utf8'), context);
  return context.module.exports;
}

test('references share pending fetches but own distinct decoded bitmaps', async () => {
  const requests = []; let resolve;
  const References = loader(url => {requests.push(url); return new Promise(r => {resolve = r;});});
  const refs = new References('?scene=fan');
  const a = refs.frame(30), b = refs.frame(30);
  assert.deepEqual(requests, ['/api/reference/30?scene=fan']);
  resolve({ok: true, blob: async () => ({size: 7})});
  const [x, y] = await Promise.all([a, b]);
  assert.notEqual(x, y); x.close(); assert.equal(y.closed, false);
  assert.equal(refs.bytes, 7); assert.equal(refs.pending.size, 0);
  await refs.frame(30); assert.equal(requests.length, 1);
});

test('reference compressed memory is bounded with LRU eviction and retry after failure', async () => {
  let fail = true;
  const References = loader(async url => {
    if (url.includes('/7?') && fail) {fail = false; return {ok: false, status: 404};}
    return {ok: true, blob: async () => ({size: 4})};
  });
  const refs = new References('?scene=lego', 8);
  await refs.frame(0); await refs.frame(1); await refs.frame(0); await refs.frame(2);
  assert.equal(refs.bytes, 8); assert.deepEqual([...refs.cache.keys()], [0, 2]);
  await assert.rejects(refs.frame(7), /HTTP 404/);
  assert.equal(refs.pending.size, 0);
  await refs.frame(7); assert.equal(refs.bytes, 8); assert.equal(refs.cache.has(7), true);
});

test('preparation warms compressed frames once with bounded parallel downloads', async () => {
  let active = 0, peak = 0, requests = 0;
  const References = loader(async () => {
    active++; peak = Math.max(peak, active); requests++;
    await new Promise(r => setImmediate(r)); active--;
    return {ok: true, blob: async () => ({size: 4})};
  });
  const refs = new References('?scene=lego', 100);
  const progress = [];
  await refs.prepare(20, (done, total) => progress.push([done, total]));
  assert.equal(refs.prepared, true); assert.equal(peak, 4); assert.equal(requests, 20);
  assert.deepEqual(progress.at(-1), [20, 20]);
  await refs.prepare(20); await refs.frame(19);
  assert.equal(requests, 20); assert.equal(refs.bytes, 80);
});

test('whole-sequence preparation never claims success after LRU evicts earlier frames',async()=>{
 const References=loader(async()=>({ok:true,blob:async()=>({size:4})}));
 const refs=new References('?scene=lego',8);
 await assert.rejects(refs.prepare(3),/整段缓存超过容量/);
 assert.notEqual(refs.prepared,true);
});

test('cancelling reference preparation stops new downloads and keeps the completed subset',async()=>{
 let valid=true,requests=0;const pending=[];
 const References=loader(()=>{requests++;return new Promise(resolve=>pending.push(resolve));});
 const refs=new References('?scene=lego',1000),job=refs.prepare(100,()=>{},()=>valid);
 assert.equal(requests,4);valid=false;
 for(const resolve of pending)resolve({ok:true,blob:async()=>({size:4})});
 await assert.rejects(job,/已取消/);assert.equal(requests,4);assert.equal(refs.cache.size,4);assert.notEqual(refs.prepared,true);
});
