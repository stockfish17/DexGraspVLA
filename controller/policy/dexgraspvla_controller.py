from typing import Dict, Tuple
import torch
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
import inspect
from controller.model.common.normalizer import LinearNormalizer
from controller.policy.base_image_policy import BaseImagePolicy
from controller.model.diffusion.transformer_for_action_diffusion import TransformerForActionDiffusion
from controller.model.vision.obs_encoder import ObsEncoder
from scipy.optimize import linear_sum_assignment
import pickle


# Adapted from https://github.com/lucidrains/pi-zero-pytorch/blob/e82fced40e55023a0ded22ab3bda495964353253/pi_zero_pytorch/pi_zero.py#L216
def noise_assignment(data, noise):
    device = data.device
    data, noise = tuple(rearrange(t, 'b ... -> b (...)') for t in (data, noise))
    dist = torch.cdist(data, noise)
    _, assign = linear_sum_assignment(dist.cpu())
    return torch.from_numpy(assign).to(device)

class DexGraspVLAController(BaseImagePolicy):
    def __init__(self, 
            shape_meta: dict,
            noise_scheduler: DDPMScheduler,
            obs_encoder: ObsEncoder,
            num_inference_steps=None,
            # arch
            n_layer=7,
            n_head=8,
            p_drop_attn=0.1,
            use_attn_mask=False,
            start_ckpt_path=None,
            # parameters passed to step
            **kwargs):
        super().__init__()

        # parse shapes
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        action_horizon = shape_meta['action']['horizon']
        
        obs_shape, obs_part_length = obs_encoder.output_shape()
        n_emb = obs_shape[-1]
        obs_tokens = obs_shape[-2]
        
        model = TransformerForActionDiffusion(
            input_dim=action_dim,
            output_dim=action_dim,
            action_horizon=action_horizon,
            n_layer=n_layer,
            n_head=n_head,
            n_emb=n_emb,
            max_cond_tokens=obs_tokens+1, # obs tokens + 1 token for time
            p_drop_attn=p_drop_attn,
            obs_part_length=obs_part_length,
            use_attn_mask=use_attn_mask
        )

        self.obs_encoder = obs_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.normalizer = LinearNormalizer()
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.start_ckpt_path = start_ckpt_path
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps
    
    # ========= inference  ============
    def conditional_sample(self, 
            condition_data, condition_mask,
            cond=None, generator=None,
            gen_attn_map=True,
            # keyword arguments to scheduler.step
            **kwargs
            ):
        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)
    
        # set step values
        scheduler.set_timesteps(self.num_inference_steps)
        
        # Store attention maps for all timesteps
        all_timestep_attention_maps = {}

        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. predict model output
            model_output, attention_maps = model(trajectory, t, cond, training=False, gen_attn_map=gen_attn_map)
            all_timestep_attention_maps[t.cpu().item()] = attention_maps

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, 
                generator=generator,
                **kwargs
                ).prev_sample
        
        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]        

        return trajectory, all_timestep_attention_maps

    def predict_action(self, obs_dict: Dict[str, torch.Tensor], output_path: str = None) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        assert 'past_action' not in obs_dict # not implemented yet
        # normalize input
        # nobs = self.normalizer.normalize(obs_dict)
        nobs = obs_dict
        B = next(iter(nobs.values())).shape[0]
        
        # process input
        obs_tokens = self.obs_encoder(nobs, training=False)
        # (B, N, n_emb)
        
        # empty data for action
        cond_data = torch.zeros(size=(B, self.action_horizon, self.action_dim), device=self.device, dtype=self.dtype)
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        
        # run sampling
        nsample, all_timestep_attention_maps = self.conditional_sample(
            condition_data=cond_data, 
            condition_mask=cond_mask,
            cond=obs_tokens,
            gen_attn_map=True if output_path is not None else False,
            **self.kwargs)

        # unnormalize prediction
        assert nsample.shape == (B, self.action_horizon, self.action_dim)
        action_pred = self.normalizer['action'].unnormalize(nsample)

        if output_path is not None:
            # Convert tensors in obs_dict to numpy arrays
            obs_dict_numpy = {}
            for k, v in obs_dict.items():
                if k in ['rgbm', 'right_cam_img']:
                    obs_dict_numpy[k] = np.clip(v.detach().cpu().numpy() * 255, 0, 255).astype(np.uint8)
                else:
                    obs_dict_numpy[k] = v.detach().cpu().numpy()
                obs_dict_numpy[k] = obs_dict_numpy[k][:2]

            save_dict = {
                'attention_maps': all_timestep_attention_maps,
                'obs_dict': obs_dict_numpy
            }

            with open(output_path, 'wb') as f:
                pickle.dump(save_dict, f)

        result = {
            'action': action_pred,
            'action_pred': action_pred
        }
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def get_optimizer(
            self,
            lr: float,
            weight_decay: float,
            betas: Tuple[float, float],
        ) -> torch.optim.Optimizer:

        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        print(f"Fused AdamW available: {fused_available}")
        optimizer = torch.optim.AdamW(
            optim_groups, lr=lr, betas=betas, fused=fused_available
        )
        return optimizer

    def compute_loss(self, batch, training=True):
        # normalize input
        assert 'valid_mask' not in batch
        # nobs = self.normalizer.normalize(batch['obs'])
        nobs = batch['obs']
        nactions = self.normalizer['action'].normalize(batch['action'])
        trajectory = nactions

        # process input
        obs_tokens = self.obs_encoder(nobs, training)
        # (B, N, n_emb)
        
        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        assignment = noise_assignment(trajectory, noise)
        noise = noise[assignment]

        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (nactions.shape[0],), device=trajectory.device
        ).long()

        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)

        # Predict the noise residual
        pred, _ = self.model(
            noisy_trajectory,
            timesteps, 
            cond=obs_tokens,
            training=training,
            gen_attn_map=False
        )

        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target)

        return loss

    def forward(self, batch, training=True):
        return self.compute_loss(batch, training)
