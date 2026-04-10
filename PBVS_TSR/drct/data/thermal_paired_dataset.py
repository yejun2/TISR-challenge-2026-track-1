import os.path as osp
import numpy as np
from torch.utils import data as data
from torchvision.transforms.functional import normalize
import torch

from basicsr.data.data_util import paths_from_lmdb, scandir
from basicsr.data.transforms import augment
from basicsr.utils import FileClient, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY


def random_patch_mask_lq(
    img_lq: np.ndarray,
    p: float = 0.25,
    min_size: int = 6,
    max_size: int = 12,
    max_blocks: int = 2,
    fill: str = "mean",
) -> np.ndarray:
    """LQ only patch masking (Cutout-like).
    img_lq: HWC float32 in [0,1]
    fill: "mean" (recommended) or "zero"
    """
    if np.random.rand() > p:
        return img_lq

    h, w, c = img_lq.shape
    out = img_lq.copy()

    n = np.random.randint(1, max_blocks + 1)
    for _ in range(n):
        bh = np.random.randint(min_size, max_size + 1)
        bw = np.random.randint(min_size, max_size + 1)

        bh = min(bh, h)
        bw = min(bw, w)

        y = np.random.randint(0, max(1, h - bh + 1))
        x = np.random.randint(0, max(1, w - bw + 1))

        if fill == "mean":
            val = out.mean(axis=(0, 1), keepdims=True)  # (1,1,C)
            out[y:y + bh, x:x + bw, :] = val
        else:
            out[y:y + bh, x:x + bw, :] = 0.0

    return out


def _stem(p: str) -> str:
    return osp.splitext(osp.basename(p))[0]


def _scan_all_files(root: str):
    # recursive scan
    return sorted(list(scandir(root, full_path=True, recursive=True)))


def _scan_npy(root: str):
    return sorted([p for p in scandir(root, full_path=True, recursive=True) if p.lower().endswith(".npy")])


def _modcrop(img: np.ndarray, scale: int) -> np.ndarray:
    h, w = img.shape[:2]
    h = h - (h % scale)
    w = w - (w % scale)
    return img[:h, :w, ...]


def _snap_to_multiple(x: int, base: int) -> int:
    return x if (x % base == 0) else ((x // base) + 1) * base


def _parse_gt_size_hw(opt, scale: int, window_size: int):
    """Support:
      - gt_size: int => square patch
      - gt_size: [h, w] => rectangular patch
    Snap each dimension to (scale * window_size) multiple so that
    LQ patch is divisible by window_size.
    """
    gt_size = opt["gt_size"]
    if isinstance(gt_size, (list, tuple)) and len(gt_size) == 2:
        gt_h, gt_w = int(gt_size[0]), int(gt_size[1])
    else:
        gt_h = gt_w = int(gt_size)

    base = scale * window_size
    safe_h = _snap_to_multiple(gt_h, base)
    safe_w = _snap_to_multiple(gt_w, base)

    # LQ patch dims divisibility
    assert (safe_h // scale) % window_size == 0
    assert (safe_w // scale) % window_size == 0
    return safe_h, safe_w


def _pad_to_multiple_hw(img: np.ndarray, mult: int, mode="reflect") -> np.ndarray:
    h, w = img.shape[:2]
    pad_h = (mult - (h % mult)) % mult
    pad_w = (mult - (w % mult)) % mult
    if pad_h == 0 and pad_w == 0:
        return img
    pad_width = ((0, pad_h), (0, pad_w), (0, 0))
    return np.pad(img, pad_width, mode=mode)


def paired_random_crop_hw_multi(img_gt, img_lq, img_aux, gt_patch_hw, scale, gt_path=None):
    """Same crop for GT/LQ/AUX.
    img_aux is aligned to LQ resolution (e.g., HP maps with shape HxWxK).
    """
    gt_h, gt_w = gt_patch_hw

    img_gt = _modcrop(img_gt, scale)
    h_gt, w_gt = img_gt.shape[:2]

    h_lq_need, w_lq_need = h_gt // scale, w_gt // scale
    img_lq = img_lq[:h_lq_need, :w_lq_need, :]
    if img_aux is not None:
        img_aux = img_aux[:h_lq_need, :w_lq_need, :]

    h_lq, w_lq = img_lq.shape[:2]
    lq_h, lq_w = gt_h // scale, gt_w // scale

    if h_gt < gt_h or w_gt < gt_w:
        raise RuntimeError(f"GT smaller than patch. GT={h_gt}x{w_gt}, patch={gt_h}x{gt_w}. File: {gt_path}")
    if h_lq < lq_h or w_lq < lq_w:
        raise RuntimeError(f"LQ smaller than patch. LQ={h_lq}x{w_lq}, patch={lq_h}x{lq_w}. File: {gt_path}")

    top_gt = np.random.randint(0, h_gt - gt_h + 1)
    left_gt = np.random.randint(0, w_gt - gt_w + 1)
    top_lq = top_gt // scale
    left_lq = left_gt // scale

    img_gt = img_gt[top_gt:top_gt + gt_h, left_gt:left_gt + gt_w, :]
    img_lq = img_lq[top_lq:top_lq + lq_h, left_lq:left_lq + lq_w, :]
    if img_aux is not None:
        img_aux = img_aux[top_lq:top_lq + lq_h, left_lq:left_lq + lq_w, :]
    return img_gt, img_lq, img_aux


@DATASET_REGISTRY.register()
class ThermalPairedDataset(data.Dataset):
    """Depth 제거 버전.
    입력 lq: (thermal 3ch + hp_in_ch) => 예: HP1/HP2면 5ch
    HP는 사전 추출된 .npy (H_lq, W_lq, hp_in_ch) 를 dataroot_hp에서 stem으로 매칭.
    """

    def __init__(self, opt):
        super().__init__()
        self.opt = opt

        self.file_client = None
        self.io_backend_opt = opt["io_backend"].copy()
        self.mean = opt.get("mean", None)
        self.std = opt.get("std", None)

        self.lq_folder = opt["dataroot_lq"]
        self.gt_folder = opt["dataroot_gt"]
        self.scale = int(opt["scale"])
        self.window_size = int(opt.get("window_size", 16))
        self.pad_val = bool(opt.get("pad_val", False))
        self.bgr2rgb = bool(opt.get("bgr2rgb", True))

        # ---- masking options (train LQ only) ----
        mask_opt = opt.get("masking", {})
        self.use_masking = bool(mask_opt.get("enable", False))
        self.mask_prob = float(mask_opt.get("p", 0.25))
        self.mask_min_size = int(mask_opt.get("min_size", 6))
        self.mask_max_size = int(mask_opt.get("max_size", 12))
        self.mask_max_blocks = int(mask_opt.get("max_blocks", 2))
        self.mask_fill = str(mask_opt.get("fill", "mean")).lower()

        # ---- HP options ----
        self.use_hp = bool(opt.get("use_hp", False))
        self.hp_folder = opt.get("dataroot_hp", None)
        self.hp_in_ch = int(opt.get("hp_in_ch", 2))

        if self.use_hp and (not self.hp_folder) and self.io_backend_opt["type"] != "lmdb":
            raise ValueError("use_hp=True but dataroot_hp is not set.")

        # Build paired paths
        if self.io_backend_opt["type"] == "lmdb":
            self.lq_io = self.io_backend_opt.copy()
            self.gt_io = self.io_backend_opt.copy()

            self.lq_io["db_paths"] = [self.lq_folder]
            self.lq_io["client_keys"] = ["lq"]
            lq_keys = paths_from_lmdb(self.lq_folder)

            self.gt_io["db_paths"] = [self.gt_folder]
            self.gt_io["client_keys"] = ["gt"]
            gt_keys = paths_from_lmdb(self.gt_folder)

            common = sorted(list(set(lq_keys) & set(gt_keys)))
            if len(common) == 0:
                raise RuntimeError("No common keys between LQ/GT LMDBs.")

            if self.use_hp:
                hp_paths = _scan_npy(self.hp_folder)
                if len(hp_paths) == 0:
                    raise RuntimeError(f"No hp npy found in: {self.hp_folder}")
                hp_map = {_stem(p): p for p in hp_paths}

                tmp = []
                for k in common:
                    s = _stem(k)
                    if s in hp_map:
                        tmp.append({"lq_path": k, "gt_path": k, "hp_path": hp_map[s]})
                if len(tmp) == 0:
                    raise RuntimeError("No common pairs among LMDB(LQ/GT) and HP folder by stem.")
                self.paths = tmp
            else:
                self.paths = [{"lq_path": k, "gt_path": k} for k in common]

        elif "meta_info_file" in opt:
            with open(opt["meta_info_file"], "r") as fin:
                rels = [line.split(" ")[0].strip() for line in fin if line.strip()]
            self.paths = []
            for r in rels:
                item = {
                    "lq_path": osp.join(self.lq_folder, r),
                    "gt_path": osp.join(self.gt_folder, r),
                }
                if self.use_hp:
                    item["hp_path"] = osp.join(self.hp_folder, _stem(r) + ".npy")
                self.paths.append(item)

        else:
            # Robust pairing by stem among GT/LQ/(HP)
            gt_paths = _scan_all_files(self.gt_folder)
            lq_paths = _scan_all_files(self.lq_folder)
            if len(gt_paths) == 0:
                raise RuntimeError(f"No GT files found in: {self.gt_folder}")
            if len(lq_paths) == 0:
                raise RuntimeError(f"No LQ files found in: {self.lq_folder}")

            gt_map = {_stem(p): p for p in gt_paths}
            lq_map = {_stem(p): p for p in lq_paths}
            common = set(gt_map.keys()) & set(lq_map.keys())

            if self.use_hp:
                hp_paths = _scan_npy(self.hp_folder)
                if len(hp_paths) == 0:
                    raise RuntimeError(f"No hp npy found in: {self.hp_folder}")
                hp_map = {_stem(p): p for p in hp_paths}
                common = common & set(hp_map.keys())
                if len(common) == 0:
                    raise RuntimeError("No paired files found by stem across LQ/GT/HP.")
                self.paths = [{"gt_path": gt_map[s], "lq_path": lq_map[s], "hp_path": hp_map[s]} for s in sorted(common)]
            else:
                if len(common) == 0:
                    raise RuntimeError("No paired files found by stem between LQ/GT.")
                self.paths = [{"gt_path": gt_map[s], "lq_path": lq_map[s]} for s in sorted(common)]

        if opt.get("phase", "train") == "train":
            self.gt_patch_h, self.gt_patch_w = _parse_gt_size_hw(opt, self.scale, self.window_size)

    def _init_file_client(self):
        if self.file_client is not None:
            return
        backend_type = self.io_backend_opt.pop("type")
        self.file_client = FileClient(backend_type, **self.io_backend_opt)
        if backend_type == "lmdb":
            self.file_client_lq = FileClient("lmdb", **self.lq_io)
            self.file_client_gt = FileClient("lmdb", **self.gt_io)

    def __getitem__(self, index):
        self._init_file_client()

        lq_path = self.paths[index]["lq_path"]
        gt_path = self.paths[index]["gt_path"]
        hp_path = self.paths[index].get("hp_path", None)

        # read LQ/GT
        if hasattr(self, "file_client_lq"):
            lq_bytes = self.file_client_lq.get(lq_path, "lq")
            gt_bytes = self.file_client_gt.get(gt_path, "gt")
        else:
            lq_bytes = self.file_client.get(lq_path, "lq")
            gt_bytes = self.file_client.get(gt_path, "gt")

        img_lq = imfrombytes(lq_bytes, float32=True)  # HWC, [0,1]
        img_gt = imfrombytes(gt_bytes, float32=True)

        # Ensure HWC 3ch
        if img_lq.ndim == 2:
            img_lq = np.stack([img_lq, img_lq, img_lq], axis=2)
        if img_gt.ndim == 2:
            img_gt = np.stack([img_gt, img_gt, img_gt], axis=2)

        # Basic alignment (GT modcrop -> LQ crop)
        img_gt = _modcrop(img_gt, self.scale)
        h_gt, w_gt = img_gt.shape[:2]
        img_lq = img_lq[: h_gt // self.scale, : w_gt // self.scale, :]

        # read HP (aligned to LQ resolution)
        img_hp = None
        if self.use_hp:
            hp = np.load(hp_path).astype(np.float32)  # (H_lq, W_lq, C)
            if hp.ndim == 2:
                hp = hp[..., None]
            if hp.shape[2] != self.hp_in_ch:
                raise RuntimeError(f"HP channel mismatch: expected {self.hp_in_ch}, got {hp.shape[2]} for {hp_path}")
            img_hp = hp[: h_gt // self.scale, : w_gt // self.scale, :]

        if self.opt.get("phase", "train") == "train":
            img_gt, img_lq, img_hp = paired_random_crop_hw_multi(
                img_gt, img_lq, img_hp, (self.gt_patch_h, self.gt_patch_w), self.scale, gt_path
            )

            # LQ masking (train only) - HP에는 적용하지 않음(권장)
            if self.use_masking:
                fill = "mean" if self.mask_fill not in ("zero",) else "zero"
                img_lq = random_patch_mask_lq(
                    img_lq,
                    p=self.mask_prob,
                    min_size=self.mask_min_size,
                    max_size=self.mask_max_size,
                    max_blocks=self.mask_max_blocks,
                    fill=fill,
                )

            # augment must apply same geometry to HP too
            if img_hp is not None:
                img_gt, img_lq, img_hp = augment(
                    [img_gt, img_lq, img_hp],
                    self.opt.get("use_hflip", True),
                    self.opt.get("use_rot", True),
                )
            else:
                img_gt, img_lq = augment(
                    [img_gt, img_lq],
                    self.opt.get("use_hflip", True),
                    self.opt.get("use_rot", True),
                )
        else:
            # val/test
            if self.pad_val:
                img_lq = _pad_to_multiple_hw(img_lq, self.window_size, mode="reflect")
                img_gt = _pad_to_multiple_hw(img_gt, self.window_size * self.scale, mode="reflect")
                if img_hp is not None:
                    img_hp = _pad_to_multiple_hw(img_hp, self.window_size, mode="reflect")

            img_gt = img_gt[: img_lq.shape[0] * self.scale, : img_lq.shape[1] * self.scale, :]
            if img_hp is not None:
                img_hp = img_hp[: img_lq.shape[0], : img_lq.shape[1], :]

        # to tensor for gt/lq
        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=self.bgr2rgb, float32=True)  # img_lq: (3,H,W)

        # concat HP => (3+hp_in_ch, H, W)
        if img_hp is not None:
            hp_tensor = img2tensor(img_hp, bgr2rgb=False, float32=True)  # (C,H,W)
            img_lq = torch.cat([img_lq, hp_tensor], dim=0)

        # normalize (LR 3ch only)
        if self.mean is not None or self.std is not None:
            normalize(img_lq[:3], self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        out = {"lq": img_lq, "gt": img_gt, "lq_path": lq_path, "gt_path": gt_path}
        if img_hp is not None:
            out["hp_path"] = hp_path
        return out

    def __len__(self):
        return len(self.paths)