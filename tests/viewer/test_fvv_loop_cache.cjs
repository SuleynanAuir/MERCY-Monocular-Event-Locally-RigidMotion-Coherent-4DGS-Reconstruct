const {test}=require('node:test');const assert=require('node:assert/strict');
require('../../viewer/fvv-loop-cache.js');
test('loop cache enforces byte budget and ignores obsolete responses',()=>{
 const c=new FVVLoopCache(async()=>({close(){}}),10);const v=c.begin(3);
 assert.equal(c.put(0,{size:6},v),false);assert.throws(()=>c.put(1,{size:6},v),/内存上限/);
 c.clear();assert.equal(c.put(1,{size:6},v),false);assert.equal(c.bytes,0);
});
test('loop cache bounds decoded images and frees them after camera change',async()=>{
 let closed=0;const c=new FVVLoopCache(async()=>({close(){closed++;}}));const v=c.begin(100);
 for(let i=0;i<100;i++)c.put(i,{size:10},v);
 await c.prepare(0);assert.equal(c.images.size,6);
 await c.prepare(50);assert.equal(c.images.size,6);assert.equal(closed,6);
 c.clear();assert.equal(closed,12);assert.equal(c.bytes,0);
});
test('loop cache releases in-progress decodes after invalidation',async()=>{
 const resolvers=[];let closed=0;const c=new FVVLoopCache(()=>new Promise(r=>resolvers.push(r)));
 const v=c.begin(10);for(let i=0;i<10;i++)c.put(i,{size:1},v);
 const pending=c.prepare(0);c.clear();for(const r of resolvers)r({close(){closed++;}});await pending;
 assert.equal(closed,6);assert.equal(c.images.size,0);
});
test('loop prefetch wraps without writing disk files or growing decoded memory',async()=>{
 const c=new FVVLoopCache(async()=>({close(){}}));const v=c.begin(10);
 for(let i=0;i<10;i++)c.put(i,{size:1},v);
 await c.prepare(9);assert.ok(c.get(9));assert.ok(c.get(0));assert.equal(c.images.size,6);
 for(let i=0;i<100;i++)await c.prepare(i%10);
 assert.equal(c.bytes,10);assert.equal(c.images.size,6);
});
