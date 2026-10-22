"""Bake Fan_data's moving right rotor from video; retain the static left fan.

This is a video-fitted rigid preview, not event-only reconstruction. The provided
G0 appearance is preserved. A near-planar rotor and world-Z shaft are explicit
geometric assumptions, recorded together with input hashes and image errors.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from plyfile import PlyData
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
from utils.gaussian_sequence import FORMAT, SPARSE_VERSION, POSE_FIELDS, GaussianSequence
from utils.fan_motion_binding import refine_static_membership


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def event_input_status(path):
    """Inspect completeness only; never treat this as event validation."""
    try:
        with path.open('rb') as stream:
            version = np.lib.format.read_magic(stream)
            shape, _, dtype = np.lib.format._read_array_header(stream, version)
            expected = stream.tell() + int(np.prod(shape)) * dtype.itemsize
        actual = path.stat().st_size
        return {'complete': actual == expected, 'expected_bytes': expected,
                'actual_bytes': actual, 'used_for_fitting': False}
    except (OSError, ValueError) as error:
        return {'complete': False, 'error': type(error).__name__, 'used_for_fitting': False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    d = args.data
    events = event_input_status(d/'Fan_event.npy')
    camera = json.loads((d/'fixed_camera.json').read_text())
    raw = PlyData.read(d/'fan_contrastive_g0.ply')['vertex'].data.copy()
    xyz = np.column_stack([raw[k] for k in ['x','y','z']])
    quat = np.column_stack([raw[f'rot_{i}'] for i in range(4)])
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)
    cap = cv2.VideoCapture(str(d/'Fan.mkv')); fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok: break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)/255)
    cap.release()
    if len(frames) < 2 or fps <= 0: raise ValueError('No usable video frames')
    # OpenCV world-to-camera from the explicit 3DGS transposed R.
    w = np.eye(4); w[:3,:3] = np.asarray(camera['R']).T; w[:3,3] = camera['T']
    k = np.array([[camera['fx'],0,camera['cx']],[0,camera['fy'],camera['cy']],[0,0,1]])
    cam = xyz@w[:3,:3].T+w[:3,3]; uv = (cam@k.T); uv = uv[:,:2]/uv[:,2:]
    # The supplied asset has a front blade layer and a rear static stand.
    # Keep this segmentation explicit; do not rotate the whole right object.
    front = (uv[:,0]>320)&(xyz[:,2]>0)&(xyz[:,2]<.16)
    plane_z = float(np.median(xyz[front,2]))
    # Image coordinates on a world-Z rotor plane, supporting the actual oblique camera.
    def project(points):
        v=points@w[:3,:3].T+w[:3,3]; q=v@k.T
        return (q[:,:2]/q[:,2:]).astype(np.float32)
    angles = np.arange(1440)*2*np.pi/1440
    radii = np.linspace(.10,.285,80)
    xx = np.cos(angles)[:,None]*radii
    yy = np.sin(angles)[:,None]*radii
    def polar(frame, center):
        points=np.stack([xx+center[0],yy+center[1],np.full_like(xx,plane_z)],-1).reshape(-1,3)
        q=project(points).reshape(1440,80,2)
        return cv2.remap(frame,q[:,:,0],q[:,:,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
    def fit(center, subset=None):
        base=polar(frames[0],center)
        base-=base.mean(0,keepdims=True)
        fft=np.fft.rfft(base,axis=0)
        result=[]; errors=[]
        for frame in frames if subset is None else subset:
            current=polar(frame,center);current-=current.mean(0,keepdims=True)
            corr=np.fft.irfft(np.conj(fft)*np.fft.rfft(current,axis=0),n=1440,axis=0).sum(1)
            if subset is None and result:
                # Three similar blades create peaks 120 degrees apart. Use
                # chronological continuity to select a local peak, never jump
                # to a different blade because its texture happens to match.
                previous = result[-1]*1440/(2*np.pi)
                candidates=(int(round(previous))+np.arange(-120,121))%1440
                best=int(np.argmax(corr[candidates]))
                if best in (0,len(candidates)-1):
                    raise ValueError('Rotation exceeds the supported 30 degrees per video frame')
                j=int(candidates[best])
            else:
                j=int(corr.argmax())
            jm=(j-1)%1440;jp=(j+1)%1440
            delta=.5*(corr[jm]-corr[jp])/(corr[jm]-2*corr[j]+corr[jp]+1e-12)
            result.append((j+delta)*2*np.pi/1440)
            errors.append(float((base*base).sum()+(current*current).sum()-2*corr[j]))
        return np.unwrap(result),float(np.mean(errors)/base.size)
    # Initialize the shaft at the visible hub and refine against eight video frames.
    seed=np.array([.23,.308])
    sample=[frames[i] for i in np.linspace(0,len(frames)-1,8).astype(int)]
    opt=minimize(lambda c:fit(c,sample)[1],seed,method='Nelder-Mead',
                 options={'maxiter':70,'xatol':1e-5,'fatol':1e-7})
    center=np.array([*opt.x,plane_z]); theta,fit_error=fit(center[:2]);theta-=theta[0]
    if np.max(np.abs(np.diff(theta)))>np.pi/3:
        raise ValueError('Ambiguous temporal rotation fit; refusing to bake discontinuities')
    # Bound the moving front layer to the measured blade sweep.
    radial=np.linalg.norm(xyz[:,:2]-center[:2],axis=1)
    moving=front&(radial<.31)
    moving, reassigned, binding_report = refine_static_membership(
        xyz, moving, center, theta, frames, camera)
    np.save(args.output/'reassigned-static-indices.npy', reassigned)
    indices=np.flatnonzero(moving).astype(np.int64)
    poses=np.empty((len(indices),len(frames),7),np.float32)
    source_rot=Rotation.from_quat(quat[indices][:,[1,2,3,0]])
    entries=[]
    for i,angle in enumerate(theta):
        rot=Rotation.from_rotvec([0,0,angle]);poses[:,i,:3]=(xyz[indices]-center)@rot.as_matrix().T+center
        poses[:,i,3:]=(rot*source_rot).as_quat()[:,[3,0,1,2]]
        if i==0: poses[:,i,:3]=xyz[indices];poses[:,i,3:]=quat[indices]
        entries.append({'file':f'frames/frame_{i:04d}.ply','time_seconds':i/fps,
                        'pose_sha256':hashlib.sha256(poses[:,i].tobytes()).hexdigest()})
    pose=np.linalg.inv(w);pose[:3,1:3]*=-1
    # The JSON bounding-box center is not the calibrated camera's look-at
    # point. Put the orbit pivot on its optical axis so yaw=0 preserves K/R/T.
    radius=float(np.linalg.norm(pose[:3,3]-np.asarray(camera['target_center'])))
    orbit_target=pose[:3,3]-radius*pose[:3,2]
    meta={'format':FORMAT,'version':SPARSE_VERSION,'name':'Fan · 视频拟合动态高斯',
          'description':'Fan_data 初始高斯；左侧风扇与支架静止，右侧叶片按视频拟合旋转。'
                        + ('原始事件尚未用于此预览。' if events['complete'] else '完整事件文件不可用。')
                        + '当前为刚体运动预览，尚未经事件验证。',
          'motion_label':'右侧叶片 · 视频拟合旋转', 'frame_count':len(frames),'fps':fps,
          'duration_seconds':len(frames)/fps,'gaussians_per_frame':len(raw),'sh_degree':3,
          'pose_fields':POSE_FIELDS,'base_vertices_sha256':hashlib.sha256(raw.tobytes()).hexdigest(),
          'frames':entries,'default_resolution':1280,'default_background':'black','default_stream_rate':30,
          'default_stream_quality':95,'default_still_quality':95,
          'preview_camera':{'pose':pose.tolist(),'target':orbit_target.tolist(),'fovy_degrees':camera['fov_y_deg']},
          'motion_is_baked':True,'input_sha256':{n:sha(d/n) for n in ['fan_contrastive_g0.ply','Fan.mkv','fixed_camera.json','camera_intrinsics.json']}}
    meta['motion_binding_repair'] = binding_report
    report={'method':'video polar correlation, world-Z rigid rotor; static remainder',
            'center':center.tolist(),'axis':[0,0,1],'angles_rad':theta.tolist(),
            'moving_gaussians':len(indices),'static_gaussians':len(raw)-len(indices),
            'polar_fit_mse':fit_error,'max_step_degrees':float(np.rad2deg(np.abs(np.diff(theta))).max()),
            'event_validation':False,'event_input':events,
            'segmentation':'image u>320, world 0<z<0.16, radius<0.31',
            'input_sha256':meta['input_sha256']}
    report['motion_binding_repair'] = binding_report
    (args.output/'motion-report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='angles_rad'}),flush=True)
    dest=args.output/'fan-motion.npz';temp=dest.with_suffix('.npz.partial')
    with temp.open('wb') as f:
        np.savez_compressed(f,base_vertices=raw,poses=poses,dynamic_indices=indices,
                            metadata_json=np.asarray(json.dumps(meta)),times_seconds=np.arange(len(frames))/fps)
    temp.rename(dest)
    with GaussianSequence(dest) as seq:
        for i in range(len(seq)): seq.validate_pose_frame(i)
    print(f'Validated {len(frames)} frames: {dest}',flush=True)

if __name__=='__main__':main()
