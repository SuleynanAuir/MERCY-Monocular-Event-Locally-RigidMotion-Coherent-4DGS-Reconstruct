'use strict';
// Only one camera/model/settings combination; JPEGs in RAM, never on disk.
globalThis.FVVLoopCache = class {
  constructor(decode, budget=96*1024*1024) { this.decode=decode;this.budget=budget;this.version=0;this.clear(); }
  clear() {
    this.version++;this.blobs=[];this.bytes=0;this.received=0;this.count=0;
    if(this.images)for(const image of this.images.values())image.close();
    this.images=new Map();this.pending=new Map();this.wanted=new Set();
  }
  begin(count) { this.clear();if(!Number.isInteger(count)||count<2||count>1200)throw Error('动态循环帧数超过限制');this.count=count;this.blobs=new Array(count);return this.version; }
  put(index,blob,version) {
    if(version!==this.version)return false;
    if(!Number.isInteger(index)||index<0||index>=this.count)throw Error('缓存帧序号无效');
    if(this.blobs[index])return this.received===this.count;
    if(!Number.isFinite(blob.size)||this.bytes+blob.size>this.budget)throw Error('当前视角预载超过 96 MiB 内存上限');
    this.blobs[index]=blob;this.bytes+=blob.size;this.received++;
    return this.received===this.count;
  }
  prepare(index) {
    if(this.received!==this.count||!this.count)return Promise.resolve();
    const version=this.version;
    this.wanted=new Set(Array.from({length:Math.min(6,this.count)},(_,j)=>(index+j)%this.count));
    for(const [i,image] of this.images)if(!this.wanted.has(i)){image.close();this.images.delete(i);}
    const tasks=[];
    for(const i of this.wanted) {
      if(this.images.has(i))continue;
      if(!this.pending.has(i)) {
        // Bound decodes even when the playhead is scrubbed repeatedly.
        if(this.pending.size>=6)continue;
        const task=this.decode(this.blobs[i]).then(image=>{
          if(version!==this.version||!this.wanted.has(i))image.close();else this.images.set(i,image);
        }).finally(()=>{if(version===this.version)this.pending.delete(i);});
        this.pending.set(i,task);
      }
      tasks.push(this.pending.get(i));
    }
    return Promise.all(tasks);
  }
  get(index) {return this.images.get(index);}
};
