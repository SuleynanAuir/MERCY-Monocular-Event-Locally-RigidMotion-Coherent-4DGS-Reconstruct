/* Small bounded jitter buffer. Pure scheduling logic also runs in Node tests. */
(function (root) {
  "use strict";
  class GaussianPlayback {
    constructor(rate = 60, warmup = 10, limit = 24) {
      this.rate = rate; this.warmup = warmup; this.limit = limit;
      this.frames = new Map(); this.last = -1; this.end = null;
      this.start = null; this.buffering = true; this.underruns = 0; this.dropped = 0;
    }
    push(item) {
      const seq = item.meta.stream.seq;
      if (seq <= this.last || this.frames.has(seq) || this.frames.size >= this.limit) {
        item.bitmap.close(); return false;
      }
      this.frames.set(seq, item); return true;
    }
    finish(seq) { this.end = seq; }
    take(now) {
      if (this.end !== null && this.last >= this.end) return {ended: true};
      const first = this.last + 1;
      if (this.buffering) {
        const available = this.end === null ? this.warmup : Math.min(this.warmup, this.end - this.last);
        // Wait for contiguous decoded frames, not just network packet count.
        for (let i = first; i < first + available; i++) if (!this.frames.has(i)) return null;
        this.start = now - first * 1000 / this.rate; this.buffering = false;
      }
      // rAF timestamps jitter slightly around a display refresh. A strict floor
      // otherwise turns an early 33.2 ms tick into a 50 ms / 16 ms cadence at
      // 30 FPS. Allow <= 2 ms of early presentation, without changing the clock.
      const tolerance = Math.min(2, 100 / this.rate);
      const desired = Math.floor((now - this.start + tolerance) * this.rate / 1000 + 1e-5);
      if (desired <= this.last) return null;
      let selected = null;
      for (let seq = first; seq <= desired; seq++) {
        const item = this.frames.get(seq);
        if (!item) break;
        if (selected) {selected.bitmap.close(); this.dropped++;}
        selected = item; this.frames.delete(seq); this.last = seq;
      }
      if (!selected && (this.end === null || this.last < this.end)) {
        this.buffering = true; this.underruns++;
      }
      return selected;
    }
    close() {
      for (const item of this.frames.values()) item.bitmap.close();
      this.frames.clear();
    }
  }
  if (typeof module !== "undefined" && module.exports) module.exports = GaussianPlayback;
  else root.GaussianPlayback = GaussianPlayback;
})(globalThis);

/* Local sequence storage and deterministic video files. No wall-clock recording. */
(function(root){
  'use strict';
  const concat=(parts)=>{const out=new Uint8Array(parts.reduce((n,p)=>n+p.length,0));let offset=0;for(const p of parts){out.set(p,offset);offset+=p.length;}return out;};
  const bytes=(n,length=0)=>{const a=[];do{a.unshift(n%256);n=Math.floor(n/256);}while(n || a.length<length);return new Uint8Array(a);};
  const vint=n=>{let length=1;while(n>=2**(7*length)-1)length++;const a=bytes(n,length);a[0]|=1<<(8-length);return a;};
  const utf8=s=>new TextEncoder().encode(s);
  const ebml=(id,data)=>concat([bytes(id),vint(data.length),data]);
  const uint=(id,n)=>ebml(id,bytes(n));
  const string=(id,s)=>ebml(id,utf8(s));
  function webm(chunks,width,height,fps,durationUs,codec='V_VP9') {
    if(!chunks.length || chunks[0].type!=='key') throw new Error('Video requires a first keyframe');
    const duration=new Uint8Array(8);new DataView(duration.buffer).setFloat64(0,durationUs/1000);
    const header=ebml(0x1a45dfa3,concat([uint(0x4286,1),uint(0x42f7,1),uint(0x42f2,4),uint(0x42f3,8),string(0x4282,'webm'),uint(0x4287,4),uint(0x4285,2)]));
    const info=ebml(0x1549a966,concat([uint(0x2ad7b1,1000000),ebml(0x4489,duration),string(0x4d80,'SC-GS'),string(0x5741,'SC-GS')]));
    const tracks=ebml(0x1654ae6b,ebml(0xae,concat([uint(0xd7,1),uint(0x73c5,1),uint(0x83,1),uint(0x9c,0),string(0x86,codec),uint(0x23e383,Math.round(1e9/fps)),ebml(0xe0,concat([uint(0xb0,width),uint(0xba,height)]))])));
    const clusters=[],cues=[];let blocks=[],start=0,position=info.length+tracks.length;
    const flush=()=>{if(!blocks.length)return;const cluster=ebml(0x1f43b675,concat([uint(0xe7,start),...blocks]));
      cues.push(ebml(0xbb,concat([uint(0xb3,start),ebml(0xb7,concat([uint(0xf7,1),uint(0xf1,position)]))])));clusters.push(cluster);position+=cluster.length;blocks=[];};
    for(const chunk of chunks) {
      const time=Math.round(chunk.timestamp/1000);
      if(chunk.type==='key' && blocks.length)flush();
      if(!blocks.length)start=time;
      const delta=time-start;if(delta<0 || delta>32767)throw new Error('Invalid video timestamps');
      blocks.push(ebml(0xa3,concat([new Uint8Array([0x81,(delta>>8)&255,delta&255,chunk.type==='key'?0x80:0]),chunk.data])));
    }
    flush();
    return new Blob([header,ebml(0x18538067,concat([info,tracks,...clusters,ebml(0x1c53bb6b,concat(cues))]))],{type:'video/webm'});
  }
  const crcTable=Array.from({length:256},(_,n)=>{for(let i=0;i<8;i++)n=n&1?0xedb88320^(n>>>1):n>>>1;return n>>>0;});
  function crc32(data){let n=0xffffffff;for(const x of data)n=crcTable[(n^x)&255]^(n>>>8);return (n^0xffffffff)>>>0;}
  async function zip(files) {
    const entries=[],central=[];let offset=0;
    for(const {name,blob} of files) {
      const data=new Uint8Array(await blob.arrayBuffer()),filename=utf8(name),crc=crc32(data);
      const local=new Uint8Array(30+filename.length),v=new DataView(local.buffer);
      v.setUint32(0,0x04034b50,true);v.setUint16(4,20,true);v.setUint16(6,0x800,true);v.setUint16(12,33,true);
      v.setUint32(14,crc,true);v.setUint32(18,data.length,true);v.setUint32(22,data.length,true);v.setUint16(26,filename.length,true);local.set(filename,30);
      const dir=new Uint8Array(46+filename.length),d=new DataView(dir.buffer);
      d.setUint32(0,0x02014b50,true);d.setUint16(4,20,true);d.setUint16(6,20,true);d.setUint16(8,0x800,true);d.setUint16(14,33,true);
      d.setUint32(16,crc,true);d.setUint32(20,data.length,true);d.setUint32(24,data.length,true);d.setUint16(28,filename.length,true);d.setUint32(42,offset,true);dir.set(filename,46);
      entries.push(local,data);central.push(dir);offset+=local.length+data.length;
    }
    const directory=concat(central),end=new Uint8Array(22),e=new DataView(end.buffer);
    e.setUint32(0,0x06054b50,true);e.setUint16(8,files.length,true);e.setUint16(10,files.length,true);e.setUint32(12,directory.length,true);e.setUint32(16,offset,true);
    return new Blob([...entries,directory,end],{type:'application/zip'});
  }
  const recordSize=record=>record.bytes?.byteLength??record.blob.size;
  const recordBlob=record=>record.blob||new Blob([record.bytes],{type:record.type||'image/jpeg'});
  class SequenceCache {
    constructor(limit=384*1024**2){this.limit=limit;this.entries=new Map();this.bytes=0;}
    remove(key){const entry=this.entries.get(key);if(entry){this.entries.delete(key);this.bytes-=entry.bytes;}}
    get(key){const e=this.entries.get(key);if(e){this.entries.delete(key);this.entries.set(key,e);}return e;}
    create(key,count){let e=this.get(key);if(!e){e={key,count,frames:new Map(),bytes:0,complete:false};this.entries.set(key,e);}return e;}
    put(entry,index,record){
      if(entry.frames.has(index))return;
      const size=recordSize(record);
      if(entry.bytes+size>this.limit)throw new Error('当前画面整段缓存超过容量，请降低清晰度后重试');
      for(const [key,e] of this.entries){if(this.bytes+size<=this.limit)break;if(e===entry)continue;this.entries.delete(key);this.bytes-=e.bytes;}
      entry.frames.set(index,record);entry.bytes+=size;this.bytes+=size;entry.complete=entry.frames.size===entry.count;
    }
  }
  class ExportFrameStore {
    static async open(prefix='scgs-video-export') {
      if(!root.indexedDB)throw new Error('浏览器未允许视频临时缓存');
      const name=`${prefix}-${Date.now()}-${Math.random().toString(36).slice(2)}`;
      // Clean up only our abandoned temporary exports, never application/user databases.
      if(root.indexedDB.databases)for(const item of await root.indexedDB.databases()) {
        const match=/^scgs-(?:video-export|preset-views)-(\d+)-/.exec(item.name||'');
        if(match && Date.now()-Number(match[1])>86400000)root.indexedDB.deleteDatabase(item.name);
      }
      return new Promise((resolve,reject)=>{
        const req=root.indexedDB.open(name,1);
        req.onupgradeneeded=()=>req.result.createObjectStore('frames');
        req.onerror=()=>reject(new Error('无法创建视频临时缓存'));
        req.onsuccess=()=>resolve(new ExportFrameStore(name,req.result));
      });
    }
    constructor(name,db){this.name=name;this.db=db;}
    async save(mode,entry){
      const frames=[...entry.frames];
      for(let start=0;start<frames.length;start+=16){
        // Batch encoded bytes to bound transient memory, without awaiting inside
        // an open transaction (which would let the browser auto-close it).
        const batch=await Promise.all(frames.slice(start,start+16).map(async([frame,record])=>
          [frame,{meta:record.meta,bytes:record.bytes||await record.blob.arrayBuffer(),type:record.type||record.blob?.type}]));
        await new Promise((resolve,reject)=>{
          const tx=this.db.transaction('frames','readwrite'),store=tx.objectStore('frames');
          for(const [frame,record] of batch)store.put(record,`${mode}:${frame}`);
          tx.oncomplete=()=>resolve();tx.onabort=tx.onerror=()=>{
            const error=new Error(`帧缓存写入失败：${tx.error?.name||'存储错误'} ${tx.error?.message||''}`);
            error.name=tx.error?.name||'Error';reject(error);
          };
        });
      }
    }
    remove(mode,count){return new Promise((resolve,reject)=>{
      const tx=this.db.transaction('frames','readwrite'),store=tx.objectStore('frames');
      for(let frame=0;frame<count;frame++)store.delete(`${mode}:${frame}`);
      tx.oncomplete=()=>resolve();tx.onabort=tx.onerror=()=>reject(tx.error||new Error('无法释放未完成缓存'));
    });}
    get(mode,frame){return new Promise((resolve,reject)=>{
      const req=this.db.transaction('frames','readonly').objectStore('frames').get(`${mode}:${frame}`);
      req.onerror=()=>reject(new Error('视频临时缓存读取失败'));
      req.onsuccess=()=>{const record=req.result;if(!record){reject(new Error('视频缓存帧缺失'));return;}
        resolve(record);};
    });}
    close(){this.db.close();root.indexedDB.deleteDatabase(this.name);}
  }
  // Completed views live on disk independently of the bounded RAM sequence LRU.
  // A view is advertised only after all of its write transactions commit.
  class PresetViews {
    constructor(memoryLimit=512*1024**2){this.entries=new Map();this.store=null;this.opening=null;this.memoryBytes=0;this.memoryLimit=memoryLimit;}
    get(key){return this.entries.get(key);}
    activate(keys){this.activeKeys=new Set(keys);}
    async releaseUnused(){
      let released=false;
      if(this.activeKeys)for(const [key,saved] of this.entries){
        if(this.activeKeys.has(key))continue;
        if(saved.memoryBytes)this.memoryBytes-=saved.memoryBytes;
        else await this.store.remove(key,saved.count);
        this.entries.delete(key);released=true;
      }
      return released;
    }
    async save(entry){
      if(!entry.complete || entry.frames.size!==entry.count)throw new Error('视角帧不完整');
      if(!this.opening)this.opening=ExportFrameStore.open('scgs-preset-views').catch(error=>{this.opening=null;throw error;});
      this.store=await this.opening;
      let read,memoryBytes=0;
      try{
        try{await this.store.save(entry.key,entry);}
        catch(error){if(error.name!=='QuotaExceededError' || !await this.releaseUnused())throw error;await this.store.save(entry.key,entry);}
        read=index=>this.store.get(entry.key,index);
      }
      catch(error){
        if(error.name!=='QuotaExceededError')throw error;
        // Some private/embedded browsers enforce less space than their reported
        // quota. Retain compressed frames in bounded RAM instead of dropping a view.
        await this.store.remove(entry.key,entry.count);
        const bytes=[...entry.frames.values()].reduce((sum,record)=>sum+recordSize(record),0);
        if(this.memoryBytes+bytes>this.memoryLimit)throw new Error('浏览器本地缓存空间不足，请使用普通浏览器窗口或降低清晰度后重试');
        const frames=new Map(entry.frames);this.memoryBytes+=bytes;
        memoryBytes=bytes;read=async index=>frames.get(index);
      }
      const saved={key:entry.key,count:entry.count,complete:true,read,memoryBytes};
      this.entries.set(entry.key,saved);return saved;
    }
    close(){this.store?.close();this.entries.clear();this.memoryBytes=0;}
  }
  async function encoder(width,height,fps) {
    if(!root.VideoEncoder || !root.VideoFrame)throw new Error('当前浏览器不支持逐帧视频导出，请使用新版 Chrome 或 Edge');
    let config=null;
    for(const codec of ['vp09.00.50.08','vp8']) {
      const candidate={codec,width,height,framerate:fps,bitrate:Math.round(width*height*fps*.12),bitrateMode:'variable',latencyMode:'realtime'};
      if((await root.VideoEncoder.isConfigSupported(candidate)).supported){config=candidate;break;}
    }
    if(!config)throw new Error('当前浏览器无法编码此视频尺寸');
    const chunks=[];let error=null,count=0;
    const writer=new root.VideoEncoder({output:chunk=>{const data=new Uint8Array(chunk.byteLength);chunk.copyTo(data);chunks.push({data,timestamp:chunk.timestamp,type:chunk.type});},error:e=>{error=e;}});
    writer.configure(config);
    return {
      async add(canvas,timestamp,duration){if(error)throw error;const frame=new root.VideoFrame(canvas,{timestamp,duration});try{writer.encode(frame,{keyFrame:count++%Math.round(fps)===0});}finally{frame.close();}if(writer.encodeQueueSize>=4)await writer.flush();if(error)throw error;},
      async finish(duration){await writer.flush();if(error)throw error;return webm(chunks,width,height,fps,duration,config.codec==='vp8'?'V_VP8':'V_VP9');},
      close(){if(writer.state!=='closed')writer.close();},
    };
  }
  const Media={SequenceCache,ExportFrameStore,PresetViews,recordBlob,webm,zip,crc32,encoder};
  if(typeof module!=='undefined' && module.exports)module.exports.Media=Media;
  else root.GaussianPlayback.Media=Media;
})(globalThis);
