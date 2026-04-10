import os
import cv2
import uuid
import tempfile
import numpy as np
import torch
import torch.nn.functional as F
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY
from .base_model import BaseModel

from basicsr.losses.losses import calculate_radiation_energy

from depth_anything_3.api import DepthAnything3


@MODEL_REGISTRY.register()
class SRModel(BaseModel):
    """Base SR model for SR + DA3 radiation structure loss (GT-supervised)."""

    def __init__(self, opt):
        super(SRModel, self).__init__(opt)

        # define network
        self.net_g = build_network(opt["network_g"])
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        # load pretrained models
        load_path = self.opt["path"].get("pretrain_network_g", None)
        if load_path is not None:
            param_key = self.opt["path"].get("param_key_g", "params")
            self.load_network(
                self.net_g,
                load_path,
                self.opt["path"].get("strict_load_g", True),
                param_key,
            )

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt["train"]

        # ============ EMA ============
        self.ema_decay = train_opt.get("ema_decay", 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f"Use Exponential Moving Average with decay: {self.ema_decay}")
            self.net_g_ema = build_network(self.opt["network_g"]).to(self.device)

            load_path = self.opt["path"].get("pretrain_network_g", None)
            if load_path is not None:
                self.load_network(
                    self.net_g_ema,
                    load_path,
                    self.opt["path"].get("strict_load_g", True),
                    "params_ema",
                )
            else:
                self.model_ema(0)
            self.net_g_ema.eval()

        # ============ losses ============
        if train_opt.get("pixel_opt"):
            self.cri_pix = build_loss(train_opt["pixel_opt"]).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get("perceptual_opt"):
            self.cri_perceptual = build_loss(train_opt["perceptual_opt"]).to(self.device)
        else:
            self.cri_perceptual = None

        if self.cri_pix is None and self.cri_perceptual is None:
            raise ValueError("Both pixel and perceptual losses are None.")

        # =========================
        # DA3 settings
        # =========================
        self.use_da3 = False
        self.da3_net = None
        self.cri_da3 = None

        self.da3_cache_root = None
        self.da3_img_cache_root = None  # cropped GT png cache (optional)
        self.da3_normalize = True
        self.da3_clip_min = None
        self.da3_clip_max = None
        self.da3_log_interval = 0

        self.da3_w_rad = 0.0
        self.da3_warmup_iter = 0
        self.da3_warmup_scale = 1.0
        self.da3_eps_inv = 1e-6

        self.da3_cache_hit = 0
        self.da3_cache_miss = 0

        # ✅ YAML 기준: train.da3_opt (우선)
        da3_opt = self.opt.get("train", {}).get("da3_opt", None)
        # 보조로 datasets.train.da3_opt도 허용 (혹시 넣었을 때)
        if da3_opt is None:
            da3_opt = self.opt.get("datasets", {}).get("train", {}).get("da3_opt", None)

        if da3_opt is not None and bool(da3_opt.get("enable", False)):
            self.use_da3 = True
            model_name = da3_opt.get("model_name", "depth-anything/DA3-LARGE-1.1")

            self.da3_cache_root = da3_opt.get("cache_root", None)
            if self.da3_cache_root is not None:
                os.makedirs(self.da3_cache_root, exist_ok=True)
                # 이미지 캐시 폴더 (crop_pos 있을 때만 안정적으로 사용)
                self.da3_img_cache_root = osp.join(self.da3_cache_root, "_img_cache")
                os.makedirs(self.da3_img_cache_root, exist_ok=True)

            self.da3_normalize = bool(da3_opt.get("normalize", True))
            self.da3_clip_min = float(da3_opt["clip_min"]) if da3_opt.get("clip_min", None) is not None else None
            self.da3_clip_max = float(da3_opt["clip_max"]) if da3_opt.get("clip_max", None) is not None else None
            self.da3_log_interval = int(da3_opt.get("log_interval", 0))
            # self.da3_scale_weights = da3_opt.get('train', {}).get('da3_scale_weights', None)
            self.da3_w_rad = float(da3_opt.get("w_rad", 0.0))
            self.da3_warmup_iter = int(da3_opt.get("warmup_iter", 0))
            self.da3_warmup_scale = float(da3_opt.get("warmup_scale", 1.0))
            self.da3_eps_inv = float(da3_opt.get("eps_inv_depth", 1e-6))

            logger = get_root_logger()
            logger.info(f"[DA3] enable=True, model={model_name}, cache_root={self.da3_cache_root}")

            # ✅ DA3는 forward(model(x)) 금지, inference API로만 사용
            self.da3_net = DepthAnything3.from_pretrained(model_name).to(self.device)
            self.da3_net.eval()
            for p in self.da3_net.parameters():
                p.requires_grad = False

            logger.info(f"[DA3 INIT] model loaded on device={self.device}")
            logger.info(f"[DA3 INIT] parameters={sum(p.numel() for p in self.da3_net.parameters())/1e6:.2f}M")

            # DA3 loss
            self.cri_da3 = build_loss(da3_opt["loss"]).to(self.device)

        self.setup_optimizers()
        self.setup_schedulers()

    @torch.no_grad()
    def _da3_infer_depth_from_png(self, png_path: str, out_hw) -> np.ndarray:
        """
        out_hw: (H_out, W_out)
        return depth: (H_out, W_out) float32 numpy  # 반드시 이 크기로 맞춘다
        """
        H_out, W_out = int(out_hw[0]), int(out_hw[1])

        pred = self.da3_net.inference([png_path])
        depth = pred.depth[0]  # (H?, W?)  <- DA3 내부 resize 때문에 out_hw와 다를 수 있음

        if not isinstance(depth, np.ndarray):
            depth = np.array(depth)
        depth = depth.astype(np.float32)

        # ✅ 핵심: DA3 output depth가 out_hw와 다르면 무조건 맞춰준다
        if depth.shape[0] != H_out or depth.shape[1] != W_out:
            depth = cv2.resize(depth, (W_out, H_out), interpolation=cv2.INTER_LINEAR)

        return depth

    def setup_optimizers(self):
        train_opt = self.opt["train"]
        optim_params = []
        logger = get_root_logger()

        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger.warning(f"Params {k} will not be optimized.")

        optim_type = train_opt["optim_g"].pop("type")
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt["optim_g"])
        self.optimizers.append(self.optimizer_g)

    def feed_data(self, data):
        self.lq = data["lq"].to(self.device)
        if "gt" in data:
            self.gt = data["gt"].to(self.device)

        if "gt_2x" in data:
            self.gt_2x = data["gt_2x"].to(self.device)
        if "gt_4x" in data:
            self.gt_4x = data["gt_4x"].to(self.device)

        # dataset에서 제공하는 GT 파일 경로(배치)
        self.gt_path = data.get("gt_path", None)
        # dataset이 paired_random_crop 할 때 crop 좌표 제공해주면 캐시 키가 완벽해짐
        self.gt_crop_pos = data.get("gt_crop_pos", None)  # (top,left) or list of them

    # -------------------------
    # DA3 helpers (PATH-based inference)
    # -------------------------
    def _stabilize_da3(self, d: torch.Tensor) -> torch.Tensor:
        # d: (B,1,H,W)
        if (self.da3_clip_min is not None) and (self.da3_clip_max is not None):
            d = torch.clamp(d, self.da3_clip_min, self.da3_clip_max)

        if self.da3_normalize:
            B = d.shape[0]
            flat = d.view(B, -1)
            dmin = flat.min(dim=1)[0].view(B, 1, 1, 1)
            dmax = flat.max(dim=1)[0].view(B, 1, 1, 1)
            d = (d - dmin) / (dmax - dmin + 1e-8)
        return d

    def _log_stats(self, name: str, t: torch.Tensor, current_iter: int):
        if self.da3_log_interval <= 0:
            return
        if current_iter % self.da3_log_interval != 0:
            return
        if self.opt.get("rank", 0) != 0:
            return
        with torch.no_grad():
            mean = t.mean().item()
            std = t.std().item()
            tmin = t.min().item()
            tmax = t.max().item()
        get_root_logger().info(
            f"[DA3 STAT] {name} iter={current_iter} mean={mean:.4f} std={std:.4f} min={tmin:.4f} max={tmax:.4f}"
        )

    def _resize_opencv_bicubic(self, img_tensor: torch.Tensor, size_hw):
        # img_tensor: (N,3,H,W) float [0,1]
        N, C, H, W = img_tensor.shape
        H_out, W_out = int(size_hw[0]), int(size_hw[1])
        outs = []
        for i in range(N):
            img = img_tensor[i].detach().cpu().permute(1, 2, 0).numpy()
            img_rs = cv2.resize(img, (W_out, H_out), interpolation=cv2.INTER_CUBIC)
            outs.append(torch.from_numpy(img_rs).permute(2, 0, 1))
        return torch.stack(outs, dim=0).to(img_tensor.device).float()

    def _cache_key(self, gt_path_one: str, h: int, w: int, crop_pos=None):
        # depth cache: npy (H,W) or (1,H,W)
        stem = osp.splitext(osp.basename(gt_path_one))[0]
        if crop_pos is None:
            return None
        if isinstance(crop_pos, (list, tuple)) and len(crop_pos) >= 2:
            top, left = int(crop_pos[0]), int(crop_pos[1])
            return osp.join(self.da3_cache_root, f"{stem}_{h}x{w}_t{top}_l{left}.npy")
        if isinstance(crop_pos, dict) and ("top" in crop_pos) and ("left" in crop_pos):
            top, left = int(crop_pos["top"]), int(crop_pos["left"])
            return osp.join(self.da3_cache_root, f"{stem}_{h}x{w}_t{top}_l{left}.npy")
        return None

    def _img_cache_path(self, gt_path_one: str, h: int, w: int, crop_pos=None):
        # cropped GT image cache (png)
        if self.da3_img_cache_root is None:
            return None
        stem = osp.splitext(osp.basename(gt_path_one))[0]
        if crop_pos is None:
            return None
        if isinstance(crop_pos, (list, tuple)) and len(crop_pos) >= 2:
            top, left = int(crop_pos[0]), int(crop_pos[1])
            return osp.join(self.da3_img_cache_root, f"{stem}_{h}x{w}_t{top}_l{left}.png")
        if isinstance(crop_pos, dict) and ("top" in crop_pos) and ("left" in crop_pos):
            top, left = int(crop_pos["top"]), int(crop_pos["left"])
            return osp.join(self.da3_img_cache_root, f"{stem}_{h}x{w}_t{top}_l{left}.png")
        return None

    def _make_cropped_gt_png(self, gt_path_one: str, out_hw, crop_pos=None):
        """
        out_hw: (H_out, W_out) == pred size
        crop_pos: (top,left) based on original GT that dataset cropped
        return: path to png to feed DA3 inference
        """
        H_out, W_out = int(out_hw[0]), int(out_hw[1])

        # 1) crop_pos가 있으면: 해당 crop을 만들고 DA3 입력용 png를 안정적으로 캐시
        if self.da3_cache_root is not None and crop_pos is not None:
            cached_png = self._img_cache_path(gt_path_one, H_out, W_out, crop_pos)
            if cached_png is not None and osp.isfile(cached_png):
                return cached_png

        # 2) GT 이미지 로드
        img = cv2.imread(gt_path_one, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise RuntimeError(f"Failed to read GT image: {gt_path_one}")

        # 3) 채널 정리 (thermal이 grayscale이면 3ch로)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            img = img[:, :, :3]

        # 4) crop 적용 (dataset crop과 동일한 좌표)
        if crop_pos is not None:
            if isinstance(crop_pos, (list, tuple)) and len(crop_pos) >= 2:
                top, left = int(crop_pos[0]), int(crop_pos[1])
            elif isinstance(crop_pos, dict) and ("top" in crop_pos) and ("left" in crop_pos):
                top, left = int(crop_pos["top"]), int(crop_pos["left"])
            else:
                top, left = None, None

            if top is not None:
                # dataset gt_size를 정확히 모르니, out_hw*scale 로 되돌리지 말고
                # "현재 학습에서 pred와 같은 해상도"를 만들기 위해
                # crop은 가능한 크게 잘라두고 마지막에 resize로 맞춘다.
                # 가장 안전: GT에서 (gt_size, gt_size)로 자르되, 범위 넘어가면 clamp.
                # 여기서는 pred와 같은 해상도의 GT를 만들 목적이므로
                # dataset에서 gt_size를 opt로 줄 가능성이 높다.
                gt_size = int(self.opt["datasets"]["train"].get("gt_size", 384)) if self.opt.get("datasets", {}).get("train") else 384
                H, W = img.shape[:2]
                top = max(0, min(top, H - 1))
                left = max(0, min(left, W - 1))
                bottom = min(H, top + gt_size)
                right = min(W, left + gt_size)
                img = img[top:bottom, left:right]

        # 5) pred size로 resize
        img_rs = cv2.resize(img, (W_out, H_out), interpolation=cv2.INTER_CUBIC)

        # 6) 저장 경로: crop_pos 있으면 캐시에 저장, 없으면 임시파일로
        if self.da3_cache_root is not None and crop_pos is not None:
            out_path = self._img_cache_path(gt_path_one, H_out, W_out, crop_pos)
            if out_path is not None:
                cv2.imwrite(out_path, img_rs)
                return out_path

        # crop_pos 없으면 임시 png 만들어서 inference 후 삭제할 것
        tmp_dir = tempfile.gettempdir()
        out_path = osp.join(tmp_dir, f"da3_tmp_{uuid.uuid4().hex}.png")
        cv2.imwrite(out_path, img_rs)
        return out_path

    @torch.no_grad()
    def _da3_infer_depth_from_png(self, png_path: str, out_hw) -> np.ndarray:
        """
        out_hw: (H_out, W_out)
        return depth: (H_out, W_out) float32 numpy
        """
        H_out, W_out = int(out_hw[0]), int(out_hw[1])

        pred = self.da3_net.inference([png_path])
        depth = pred.depth[0]  # (H?,W?) DA3 내부 resize 때문에 다를 수 있음

        if not isinstance(depth, np.ndarray):
            depth = np.array(depth)
        depth = depth.astype(np.float32)

        # ✅ 핵심: 항상 pred 해상도로 맞춘다
        if depth.shape[0] != H_out or depth.shape[1] != W_out:
            depth = cv2.resize(depth, (W_out, H_out), interpolation=cv2.INTER_LINEAR)

        return depth

    @torch.no_grad()
    def _infer_depth_from_tensor(self, img_bchw_01: torch.Tensor) -> torch.Tensor:
        """
        img_bchw_01: (N,3,H,W), float, 0~1
        return: (N,1,H,W) depth
        """

        N, C, H_out, W_out = img_bchw_01.shape

        # 1) BCHW -> list of uint8 HWC for DA3 API (네 DA3 사용 방식에 맞춰 변환)
        img = img_bchw_01.clamp(0, 1)
        imgs = (img * 255.0).byte().permute(0, 2, 3, 1).contiguous().cpu().numpy()  # N,H,W,C uint8

        # 2) DA3 inference (네 프로젝트에서 쓰는 DepthAnything3 API에 맞춰)
        # pred = self.da3_model.inference(list(imgs))  # API가 list 입력이면
        # depth_list = pred.depth  # depth_list: list of (H,W) float32

        # 아래는 “API가 list를 받는다” 가정의 예시
        pred = self.da3_net.inference([im for im in imgs])

        # pred.depth: list of (h,w) float
        depth_list = []
        for d in pred.depth:
            d = np.asarray(d).astype(np.float32)

            # ✅ 핵심: 항상 (H_out,W_out)로 맞춘다 (PNG쪽이랑 동일)
            if d.shape[0] != H_out or d.shape[1] != W_out:
                d = cv2.resize(d, (W_out, H_out), interpolation=cv2.INTER_LINEAR)

            depth_list.append(d)

        depth_np = np.stack(depth_list, axis=0)  # (N,H,W)
        depth = torch.from_numpy(depth_np).to(img_bchw_01.device).unsqueeze(1)  # (N,1,H,W)

        return depth

    def _get_gt_depth(self, gt_path_one: str, out_hw, crop_pos=None) -> torch.Tensor:
        """
        out_hw: (H,W) = pred resolution
        return depth tensor (1,1,H,W) on self.device
        """
        H_out, W_out = int(out_hw[0]), int(out_hw[1])

        cache_path = None
        if self.da3_cache_root is not None:
            cache_path = self._cache_key(gt_path_one, H_out, W_out, crop_pos)

        # depth cache load
        if cache_path is not None and osp.isfile(cache_path):
            self.da3_cache_hit += 1
            arr = np.load(cache_path)
            if arr.ndim == 2:
                arr = arr[None, ...]  # (1,H,W)
            t = torch.from_numpy(arr).float().to(self.device)  # (1,H,W)
            return t.unsqueeze(0)  # (1,1,H,W)

        self.da3_cache_miss += 1

        # make cropped gt png
        png_path = self._make_cropped_gt_png(gt_path_one, (H_out, W_out), crop_pos)
        is_tmp = ("da3_tmp_" in osp.basename(png_path))

        try:
            depth_np = self._da3_infer_depth_from_png(png_path, (H_out, W_out))
        finally:
            # crop_pos 없을 때 만든 임시파일은 삭제
            if is_tmp and osp.isfile(png_path):
                try:
                    os.remove(png_path)
                except Exception:
                    pass

        depth_t = torch.from_numpy(depth_np).float().to(self.device)  # (H,W)
        depth_t = depth_t.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)

        # save cache only when key is stable
        if cache_path is not None:
            np.save(cache_path, depth_np.astype(np.float16))

        return depth_t

    # -------------------------
    # train step
    # -------------------------
    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq)

        l_total = 0
        loss_dict = OrderedDict()

        outs = self.output if isinstance(self.output, (list, tuple)) else [self.output]

        # pixel loss
        if self.cri_pix:
            gts = []
            for pred in outs:
                H, W = pred.shape[-2], pred.shape[-1]
                gt_rs = self._resize_opencv_bicubic(self.gt, (H, W))  # (N,3,H,W)
                gts.append(gt_rs)

            for i, pred in enumerate(outs):
                l_pix = self.cri_pix(pred, gts[i])
                if i <= 1:
                    l_total += 0.5 * l_pix
                else:
                    l_total += 1 * l_pix
                loss_dict[f"l_pix_{i+1}"] = l_pix

        # perceptual loss (optional)
        if self.cri_perceptual:
            last = outs[-1]
            l_percep, l_style = self.cri_perceptual(last, self.gt)
            if l_percep is not None:
                l_total += l_percep
                loss_dict["l_percep"] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict["l_style"] = l_style

        # DA3 radiation structure loss (GT supervised)
        if self.use_da3 and (self.da3_w_rad > 0):
            if self.gt_path is None:
                raise RuntimeError("Dataset must provide gt_path for DA3 usage.")

            # gt_paths batch normalize
            if isinstance(self.gt_path, (list, tuple)):
                gt_paths = list(self.gt_path)
            else:
                gt_paths = [self.gt_path] * self.gt.shape[0]

            # warmup weight
            warm = 1.0
            if self.da3_warmup_iter > 0 and current_iter < self.da3_warmup_iter:
                warm = self.da3_warmup_scale
            w = self.da3_w_rad * warm

            # 해상도별 가중치(원하면 opt에서 조절)
            # 예: out이 3개면 [x4,x2,x1]로 보고 낮은해상도에 덜 주기
            # ws = getattr(self, "da3_scale_weights", None)
            
            ws = [0.5, 0.5, 1.0]
            if ws is not None:
                assert len(ws) == len(outs), \
                    f"da3_scale_weights length {len(ws)} != outputs {len(outs)}"

            for i, pred in enumerate(outs):
                pred_rgb = pred[:, :3, :, :]

                # 1) pred에서 depth 뽑기 (네가 원한 핵심)
                with torch.no_grad():
                    depth_pred = self._infer_depth_from_tensor(pred_rgb)
                    depth_pred = self._stabilize_da3(depth_pred)

                inv_depth_pred = 1.0 / (depth_pred + self.da3_eps_inv)

                # 2) pred energy
                pred_gray = pred_rgb.mean(dim=1, keepdim=True)  # (N,1,H,W)
                pred_gray = torch.clamp(pred_gray, 0, 1)

                E_pred = calculate_radiation_energy(inv_depth_pred, pred_gray)
                # 3) GT를 같은 해상도로 맞춰서 GT energy 타겟
                gt_s = self._resize_opencv_bicubic(self.gt[:, :3, :, :], pred_rgb.shape[-2:])
                with torch.no_grad():
                    depth_gt_s = self._infer_depth_from_tensor(gt_s)
                    depth_gt_s = self._stabilize_da3(depth_gt_s)
                    inv_depth_gt_s = 1.0 / (depth_gt_s + self.da3_eps_inv)
                    
                    gt_gray = gt_s.mean(dim=1, keepdim=True)
                    gt_gray = torch.clamp(gt_gray, 0, 1)

                    E_gt = calculate_radiation_energy(inv_depth_gt_s, gt_gray).detach()
                
                l_da3_i = self.cri_da3(E_pred, E_gt) * (w * ws[i])
                l_total += l_da3_i
                loss_dict[f"l_da3_{i+1}"] = l_da3_i

                # loop 내부: i, pred_rgb, depth_pred, inv_depth_pred, E_pred, E_gt, l_da3_i 가 존재한다고 가정
                if (self.da3_log_interval > 0) and (current_iter % self.da3_log_interval == 0) and (self.opt.get("rank", 0) == 0):
                    logger = get_root_logger()

                    # 어떤 스케일/해상도인지 같이 남기기
                    H_out, W_out = pred_rgb.shape[-2], pred_rgb.shape[-1]
                    logger.info(f"[DA3 LOG] iter={current_iter} scale_idx={i+1}/{len(outs)} out_hw=({H_out},{W_out}) w_rad_eff={w:.6f} w_scale={ws[i]:.3f}")

                    # 이제는 gt_depth/ inv_depth_gt 같은 'GT depth only' 로그가 아니라
                    # pred에서 뽑은 depth / inv_depth / E_pred 를 반드시 찍어야 함
                    self._log_stats(f"da3/depth_pred_s{i+1}", depth_pred, current_iter)
                    self._log_stats(f"da3/inv_depth_pred_s{i+1}", inv_depth_pred, current_iter)
                    self._log_stats(f"da3/E_pred_s{i+1}", E_pred, current_iter)
                    self._log_stats(f"da3/E_gt_s{i+1}", E_gt, current_iter)

                    # loss 값
                    logger.info(f"[DA3 LOG] iter={current_iter} l_da3_s{i+1}={l_da3_i.item():.6f}")

            # loop 바깥: total_da3 (스케일 합산)을 만들어뒀다고 가정
            # if (self.da3_log_interval > 0) and (current_iter % self.da3_log_interval == 0) and (self.opt.get("rank", 0) == 0):
            #     logger = get_root_logger()
            #     logger.info(f"[DA3 LOG] iter={current_iter} l_da3_total={total_da3.item():.6f}")

        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    # -------------------------
    # test/val/save
    # -------------------------
    def test(self):
        if hasattr(self, "net_g_ema"):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output = self.net_g_ema(self.lq)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output = self.net_g(self.lq)
            self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt["rank"] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt["name"]
        with_metrics = self.opt["val"].get("metrics") is not None
        use_pbar = self.opt["val"].get("pbar", False)

        if with_metrics:
            if not hasattr(self, "metric_results"):
                self.metric_results = {metric: 0 for metric in self.opt["val"]["metrics"].keys()}
            self._initialize_best_metric_results(dataset_name)

        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit="image")

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data["lq_path"][0]))[0]
            self.feed_data(val_data)
            self.test()

            out = self.output[-1] if isinstance(self.output, list) else self.output
            self.output = out

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals["result"]])
            metric_data["img"] = sr_img

            if "gt" in visuals:
                gt_img = tensor2img([visuals["gt"]])
                metric_data["img2"] = gt_img
                del self.gt

            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if self.opt["is_train"]:
                    save_img_path = osp.join(self.opt["path"]["visualization"], img_name, f"{img_name}_{current_iter}.png")
                else:
                    if self.opt["val"].get("suffix", None):
                        save_img_path = osp.join(
                            self.opt["path"]["visualization"], dataset_name, f"{img_name}_{self.opt['val']['suffix']}.png"
                        )
                    else:
                        save_img_path = osp.join(self.opt["path"]["visualization"], dataset_name, f"{img_name}.bmp")
                imwrite(sr_img, save_img_path)

            if with_metrics:
                for name, opt_ in self.opt["val"]["metrics"].items():
                    self.metric_results[name] += calculate_metric(metric_data, opt_)

            if use_pbar:
                pbar.update(1)
                pbar.set_description(f"Test {img_name}")

        if use_pbar:
            pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f"Validation {dataset_name}\n"
        for metric, value in self.metric_results.items():
            log_str += f"\t # {metric}: {value:.4f}"
            if hasattr(self, "best_metric_results"):
                log_str += (
                    f"\tBest: {self.best_metric_results[dataset_name][metric]['val']:.4f} @ "
                    f"{self.best_metric_results[dataset_name][metric]['iter']} iter"
                )
            log_str += "\n"

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f"metrics/{dataset_name}/{metric}", value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict["lq"] = self.lq.detach().cpu()
        out_dict["result"] = self.output.detach().cpu()
        if hasattr(self, "gt"):
            out_dict["gt"] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        if hasattr(self, "net_g_ema"):
            self.save_network([self.net_g, self.net_g_ema], "net_g", current_iter, param_key=["params", "params_ema"])
        else:
            self.save_network(self.net_g, "net_g", current_iter)
        self.save_training_state(epoch, current_iter)