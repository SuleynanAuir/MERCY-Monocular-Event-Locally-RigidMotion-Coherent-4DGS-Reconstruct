const {test}=require('node:test');
const assert=require('node:assert/strict');
const {Media}=require('../../viewer/playback.js');
test('sequence cache reports ready only for a complete sequence and evicts whole older views',()=>{
 const cache=new Media.SequenceCache(12),a=cache.create('camera-a',2);
 const record={meta:{frame:0},blob:new Blob(['1234'])};
 cache.put(a,0,record);assert.equal(a.complete,false);cache.put(a,1,record);assert.equal(a.complete,true);
 const b=cache.create('camera-b',2);cache.put(b,0,record);assert.equal(cache.bytes,12);cache.put(b,1,record);
 assert.equal(cache.get('camera-a'),undefined);assert.equal(cache.get('camera-b').complete,true);assert.equal(cache.bytes,8);
 const big=cache.create('too-large',1);assert.throws(()=>cache.put(big,0,{blob:new Blob(['1234567890123'])}),/容量/);
 assert.equal(big.complete,false);
});
test('ZIP uses standard CRC32, distinct filenames and full file payloads',async()=>{
 assert.equal(Media.crc32(new TextEncoder().encode('123456789')),0xcbf43926);
 const blob=await Media.zip([{name:'1_gaussian.webm',blob:new Blob(['video-a'])},{name:'5_hybrid.webm',blob:new Blob(['video-b'])}]);
 const data=new Uint8Array(await blob.arrayBuffer()),view=new DataView(data.buffer),end=data.length-22;
 assert.equal(view.getUint32(0,true),0x04034b50);assert.equal(view.getUint32(end,true),0x06054b50);assert.equal(view.getUint16(end+10,true),2);
 assert.equal(view.getUint32(14,true),Media.crc32(new TextEncoder().encode('video-a')));
 assert.ok(new TextDecoder().decode(data).includes('5_hybrid.webm'));
});
test('WebM keeps exact duration, requires a keyframe, and includes all chunks',async()=>{
 assert.throws(()=>Media.webm([],640,480,60,1000000),/keyframe/);
 const chunks=[{timestamp:0,type:'key',data:new Uint8Array([11,12,13])},{timestamp:16667,type:'delta',data:new Uint8Array([21,22,23])}];
 const blob=Media.webm(chunks,640,480,60,33334),data=new Uint8Array(await blob.arrayBuffer());
 assert.equal(blob.type,'video/webm');assert.deepEqual([...data.slice(0,4)],[0x1a,0x45,0xdf,0xa3]);
 assert.ok(new TextDecoder().decode(data).includes('V_VP9'));
 assert.ok(data.some((v,i)=>v===21&&data[i+1]===22&&data[i+2]===23));
 assert.throws(()=>Media.webm([chunks[0],{...chunks[1],timestamp:-20000}],640,480,60,33334),/timestamps/);
});
