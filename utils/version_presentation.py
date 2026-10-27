"""Explicit presentation renumbering; source assets and causal lineage stay intact."""
import copy
import io
import re
import time

import numpy as np

REVISION = 'swap-v1-v2-20261027'
SOURCE_TO_DISPLAY = {'V1': 'V2', 'V2': 'V1', 'V3': 'V3'}
NOTICE = '版本重新编号：V1=原V2，V2=原V1，V3不变；模型、测量值与原始来源不变，不代表新增训练或已验证逐层消融。'


def relabel(text, source_to_display=None):
    labels = SOURCE_TO_DISPLAY if source_to_display is None else source_to_display
    return re.sub(r'\bV[123]\b', lambda m: labels.get(m.group(), m.group()), str(text))


def mapping(labels, source_to_display=None):
    source_to_display = SOURCE_TO_DISPLAY if source_to_display is None else source_to_display
    rows=[]
    for i,label in enumerate(labels):
        match=re.match(r'^(V\d+)\b',str(label or ''))
        source=match.group(1) if match else ('V1' if i==0 else f'V{i+1}' if len(labels)==3 else 'V3')
        shown=source_to_display.get(source,source)
        description=re.sub(r'^V\d+\s*[·/：:]?\s*','',str(label or ''))
        rows.append(dict(source_panel_index=i,source_version=source,source_label=label,
                         display_version=shown,display_label=shown+(' · '+description if description else '')))
    return sorted(rows,key=lambda r:int(r['display_version'][1:]))


def export_source_order(manifest):
    """New exports already use the new order; historical files remain immutable."""
    if manifest.get('version_presentation',{}).get('revision')==REVISION:
        return tuple(v['id'] for v in manifest['versions'])
    labels=[v['label'] for v in manifest['versions']]
    return tuple(manifest['versions'][r['source_panel_index']]['id'] for r in mapping(labels))


class CaptureOrder:
    def __init__(self,capture,order):
        self.capture=capture;self.inverse={old:new for new,old in enumerate(order)};self.next=0
    def point_ids(self,count,groups=None,side=None):
        return self.capture.point_ids(count,groups,side=self.inverse[self.next if side is None else side])
    def __call__(self,state,alpha,side=None):
        self.capture(state,alpha,side=self.inverse[self.next if side is None else side]);self.next+=1


class VersionPresentationRenderer:
    """One mapping for pixels, captures, timings, modes and export provenance."""
    def __init__(self,source):
        self.source=source
        self._info=None
        from utils.viewer_cache import EncodedFrameCache
        self.frames=EncodedFrameCache(max_bytes=128*1024**2,max_entries=1200)

    def __getattr__(self,name):return getattr(self.source,name)

    @property
    def info(self):return self._info if self._info is not None else self.source.info

    @property
    def status(self):
        state=dict(self.source.status)
        if state.get('message'):state['message']=relabel(state['message'], getattr(self, '_mapping', None))
        if state.get('ready') and self._info is not None:state.update(scene=self.info,message='版本映射已就绪')
        return state

    @status.setter
    def status(self,value):self.source.status=value

    @property
    def motion_evidence(self):
        records=getattr(self.source,'motion_evidence',{})
        return {i:records[old] for i,old in enumerate(self.orders.get('gaussian',[])) if old in records}

    def load(self):
        self.source.load();self.configure()

    def configure(self):
        original=self.source.info;self._info=copy.deepcopy(original);self.orders={};mode_maps={}
        if original.get('source_preview'):
            return
        self._mapping=original.get('source_to_display', SOURCE_TO_DISPLAY)
        if set(self._mapping)!=set(SOURCE_TO_DISPLAY) or set(self._mapping.values())!=set(SOURCE_TO_DISPLAY):
            raise ValueError('Version mapping must be a permutation of V1, V2, V3')
        notice=NOTICE if self._mapping==SOURCE_TO_DISPLAY else '版本按阶段清单排列；保留原始来源与哈希，指标以同相机同时间实测为准。'
        if not original.get('comparison') and not original.get('comparison_modes'):return
        if original.get('version_presentation',{}).get('revision')==REVISION:
            raise ValueError('Version presentation mapping was already applied')
        common=original.get('version_labels') or [original.get('left_label'),original.get('right_label')]
        def texts(record):
            for key in ('name','description','left_label','right_label'):
                if key in record:record[key]=relabel(record[key], self._mapping)
            if 'quality_descriptions' in record:record['quality_descriptions']={k:relabel(v, self._mapping) for k,v in record['quality_descriptions'].items()}
        texts(self._info)
        for mode in ('gaussian','clay','pseudocolor','mesh','hybrid'):
            mode_info=original.get(mode,{})
            labels=original.get('mode_version_labels',{}).get(mode)
            if labels is None:labels=[mode_info['left_label'],mode_info['right_label']] if 'left_label' in mode_info and 'right_label' in mode_info else common
            if not (original.get('comparison') or mode in original.get('comparison_modes',[])):
                labels=[original.get('left_label') or '当前版本']
            rows=mapping(labels, self._mapping);mode_maps[mode]=rows;self.orders[mode]=[r['source_panel_index'] for r in rows]
            if mode in self._info:
                texts(self._info[mode]);self._info[mode].update(left_label=rows[0]['display_label'],right_label=rows[-1]['display_label'])
        self._info['mode_version_labels']={m:[r['display_label'] for r in rows] for m,rows in mode_maps.items()}
        self._info['version_labels']=self._info['mode_version_labels']['gaussian']
        self._info.update(left_label=self._info['version_labels'][0],right_label=self._info['version_labels'][-1],
            mode_version_mapping=mode_maps,version_presentation=dict(revision=REVISION,source_to_display=self._mapping,
                disclosure=notice,changes='Labels and display/export ordering only; raw assets and measured values unchanged'))
        self._info['description']=(self._info.get('description') or '')+' '+notice
        if len(self._info['version_labels'])==3:self._info['name']=original.get('name','当前物件').split(' · ')[0]+' · V1 / V2 / V3'
        self._info['asset_identity']=dict(source=original.get('asset_identity'),version_presentation=REVISION)
        # The cached evidence belongs to the source renderer's original IDs.
        # Preserve it under its source namespace; exported rows carry the mapping.

    def render(self,request):
        from utils.viewer_state import image_packet
        mode=request['render_mode'];order=self.orders.get(mode,[])
        if not order or order==list(range(len(order))):return self.source.render(request)
        started=time.perf_counter();raw=request.get('_raw')
        jpeg=None if raw else self.frames.get(request)
        hit=jpeg is not None;render_ms=encode_ms=0.
        if jpeg is None:
            source_request=dict(request,_raw=True,_panel_timings=[])
            if request.get('_capture'):source_request['_capture']=CaptureOrder(request['_capture'],order)
            result=self.source.render(source_request);panels=result['panels']
            if len(panels)!=len(order):raise ValueError('Presentation mapping does not match renderer panels')
            panels=[panels[i] for i in order]
            timings=source_request.get('_panel_timings',[])
            request['_panel_timings']=[timings[i] for i in order] if timings else []
            if raw:return dict(panels=panels)
            rendered=time.perf_counter()
            from PIL import Image
            stream=io.BytesIO();Image.fromarray(np.concatenate(panels,axis=1)).save(stream,'JPEG',quality=request['quality'],subsampling=0)
            jpeg=stream.getvalue();self.frames.put(request,jpeg)
            render_ms=(rendered-started)*1000;encode_ms=(time.perf_counter()-rendered)*1000
        return image_packet(dict(id=request['id'],frame=request['frame'],comparison=True,panel_count=len(order),
            panel_frames=[request['frame']]*len(order),panel_width=request['width'],left_frame=request['frame'],right_frame=request['frame'],
            width=request['width']*len(order),height=request['height'],render_mode=mode,
            **({'stream':request['_stream']} if '_stream' in request else {}),cache_hit=hit,
            render_ms=round(render_ms,2),encode_ms=round(encode_ms,2)),jpeg)
