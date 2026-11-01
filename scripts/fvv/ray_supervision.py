"""Opt-in mesh ray supervision; does not modify upstream files or serving code."""
import ast
import inspect
import textwrap
import hashlib


class _GeometryModel:
    """Avoid per-pass class cycles retaining the depth graph and CUDA buffers."""
    def __init__(self,model,dynamic,static):self.model=model;self.dynamic=dynamic;self.static=static
    def __getattr__(self,name):return getattr(self.model,name)
    @property
    def get_features(self):
        import torch
        features=self.model.get_features;dc=((self.dynamic-.5)/.28209479177387814)[:,None,:]
        return torch.cat((dc,torch.zeros_like(features[:,1:])),dim=1).contiguous()
    @property
    def get_static_features(self):
        import torch
        features=self.model.get_static_features;dc=((self.static-.5)/.28209479177387814)[:,None,None].expand(-1,1,3)
        return torch.cat((dc,torch.zeros_like(features[:,1:])),dim=1).contiguous()


def with_inverse_depth(render):
    """Expose an already-computed native output in an isolated function copy.

    Diagnostic/read-only output: do not train through this auxiliary channel.
    Its gradient was zero in GPU probes of the pinned fast backward. Fail closed
    if the upstream return schema changes; never monkeypatch the original module.
    """
    source=textwrap.dedent(inspect.getsource(render))
    tree=ast.parse(source)
    returns=[n for n in ast.walk(tree) if isinstance(n,ast.Return) and isinstance(n.value,ast.Dict)]
    if len(returns)!=1:raise ValueError('Unsupported renderer return schema')
    result=returns[0].value
    keys={k.value for k in result.keys if isinstance(k,ast.Constant)}
    if not {'render','depth','alpha','flow'}<=keys or 'inverse_depth' in keys:
        raise ValueError('Unsupported renderer output schema')
    if not any(isinstance(n,ast.Name) and n.id=='invdepth' and isinstance(n.ctx,ast.Store) for n in ast.walk(tree)):
        raise ValueError('Renderer does not compute native inverse depth')
    result.keys.append(ast.Constant('inverse_depth'));result.values.append(ast.Name(id='invdepth',ctx=ast.Load()))
    scope=dict(render.__globals__)
    exec(compile(ast.fix_missing_locations(tree),inspect.getsourcefile(render),'exec'),scope)
    wrapped=scope[render.__name__]
    wrapped.source_sha256=hashlib.sha256(source.encode()).hexdigest()
    return wrapped


def inverse_depth_loss(prediction, target, parts, moving, alpha_prediction=None, alpha_weight=0., second_moment=None, spread_weight=0.):
    """Part-balanced foreground depth and clear-background supervision.

    Boundary pixels are excluded to avoid forcing a single ray's hard edge onto
    the antialiased Gaussian footprint. No direct color-based point removal.
    """
    import torch
    import torch.nn.functional as F
    pred=F.adaptive_avg_pool2d(prediction.unsqueeze(0),target.shape)[0,0]
    foreground=parts>=0
    fg=1-F.max_pool2d((~foreground).float()[None,None],3,1,1)[0,0]
    bg=1-F.max_pool2d(foreground.float()[None,None],3,1,1)[0,0]
    weights=torch.zeros_like(target)
    labels,counts=torch.unique(parts[foreground],return_counts=True)
    if len(labels):
        balance=(counts.float().median()/counts.float()).sqrt().clamp(.5,4)
        # Equalize small pieces without allowing tiny parts to dominate.
        lookup=torch.zeros(int(labels.max())+1,device=target.device)
        lookup[labels.long()]=balance
        weights[foreground]=lookup[parts[foreground].long()]
    weights*=1+2*moving.float()
    weights*=fg
    error=(pred-target).abs()
    front=(pred-target).clamp_min(0) # closer-than-source occluders
    loss=((error+.5*front)*weights).sum()/weights.sum().clamp_min(1)+.5*(pred.abs()*bg).sum()/bg.sum().clamp_min(1)
    if alpha_weight:
        if alpha_prediction is None:raise ValueError('Coverage supervision requires alpha')
        alpha=F.adaptive_avg_pool2d(alpha_prediction.unsqueeze(0),target.shape)[0,0]
        loss=loss+alpha_weight*(((1-alpha).abs()*weights).sum()/weights.sum().clamp_min(1)+(alpha.abs()*bg).sum()/bg.sum().clamp_min(1))
    if spread_weight:
        if second_moment is None or alpha_prediction is None:
            raise ValueError('Depth spread supervision requires second moment and alpha')
        second=F.adaptive_avg_pool2d(second_moment.unsqueeze(0),target.shape)[0,0]
        alpha=F.adaptive_avg_pool2d(alpha_prediction.unsqueeze(0),target.shape)[0,0]
        # Sum_i compositing_weight_i * (inverse_z_i - target)^2.
        # A near/far cloud cannot cancel its error by matching only the mean.
        spread=(second-2*target*pred+target.square()*alpha).clamp_min(0)
        loss=loss+spread_weight*(spread*weights).sum()/weights.sum().clamp_min(1)
    return loss


def render_inverse_depth(render, camera, model, pipe, background, return_alpha=False, moment=1, rigid=None):
    """Route inverse depth through the tested color backward path.

    Some builds expose an inverse-depth output with zero parameter gradients.
    A geometry-only pass encodes 1/z using standard SH DC values for both branches.
    Appearance tensors and the renderer are never mutated. The derivative of
    inverse depth w.r.t. the current conditional 4D center is retained.
    """
    import torch
    if moment not in (1,2):raise ValueError('Only first and second inverse-depth moments are supported')
    # Append articulated geometry BEFORE replacing appearance with depth.
    # Appending afterward would mix ordinary bucket/arm RGB into depth moments.
    if rigid is not None:model=rigid.proxy(model,camera.timestamp)
    _,offset=model.get_current_covariance_and_mean_offset(1.,camera.timestamp)
    transform=camera.world_view_transform
    def inverse(centers):
        z=centers@transform[:3,2]+transform[3,2]
        return torch.where(z>.01,z.clamp_min(.01).reciprocal().pow(moment),torch.zeros_like(z))
    dynamic=inverse(model.get_xyz+offset)[:,None].expand(-1,3).contiguous()
    static=inverse(model.get_static_xyz)
    result=render(camera,_GeometryModel(model,dynamic,static),pipe,torch.zeros_like(background))
    return (result['render'][:1],result['alpha']) if return_alpha else result['render'][:1]
