"""数据集加载与标准 ImageFolder 读取（D1/D2/D3）。

约定目录结构为 <root>/<class_name>/*.jpg|png。
返回 DatasetInfo 记录 dataset card 字段（方案 2.1：真实张数/类别/健康类有无/
许可/下载日期/MD5/标注噪声目检结果）。
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from ..utils.io import load_yaml, project_path, save_json

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _md5_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(chunk), b""):
            h.update(c)
    return h.hexdigest()


@dataclass
class DatasetCard:
    name: str
    root: Path
    n_images: int = 0
    n_classes: int = 0
    classes: list[str] = field(default_factory=list)
    has_healthy: bool = False
    images_per_class: dict[str, int] = field(default_factory=dict)
    license: str = ""
    download_date: str = ""
    md5: str = ""
    noise_screen_100: dict[str, Any] = field(default_factory=dict)
    healthy_keywords: list[str] = field(default_factory=list)

    HEALTHY_KEYWORDS = ["healthy", "health", "健康"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "root": str(self.root),
            "n_images": self.n_images,
            "n_classes": self.n_classes,
            "classes": self.classes,
            "has_healthy": self.has_healthy,
            "images_per_class": self.images_per_class,
            "license": self.license,
            "download_date": self.download_date,
            "md5": self.md5,
            "noise_screen_100": self.noise_screen_100,
        }


def index_images(root: Path) -> dict[str, list[Path]]:
    """返回 {class: [image_paths]}，按类目录索引。"""
    index: dict[str, list[Path]] = {}
    if not root.is_dir():
        return index
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        imgs = sorted(p for p in d.iterdir() if p.suffix.lower() in IMG_EXT)
        if imgs:
            index[d.name] = imgs
    return index


def build_dataset_card(
    name: str,
    root: Path,
    license: str = "",
    download_date: str = "",
    healthy_keywords: list[str] | None = None,
    noise_sample: int = 100,
) -> DatasetCard:
    idx = index_images(root)
    classes = sorted(idx.keys())
    per_class = {c: len(idx[c]) for c in classes}
    n_images = sum(per_class.values())
    kw = healthy_keywords or DatasetCard.HEALTHY_KEYWORDS
    has_healthy = any(k.lower() in c.lower() for c in classes for k in kw)
    # 标注噪声目检：随机抽 noise_sample 张记录可打开性与尺寸分布
    noise = {}
    if n_images and noise_sample:
        import random

        all_imgs = [p for v in idx.values() for p in v]
        random.seed(0)
        sample = random.sample(all_imgs, min(noise_sample, len(all_imgs)))
        ok = 0
        sizes = []
        for p in sample:
            try:
                with Image.open(p) as im:
                    im.verify()
                with Image.open(p) as im:
                    sizes.append(im.size)
                ok += 1
            except Exception:
                pass
        noise = {
            "sample": len(sample),
            "openable": ok,
            "unopenable": len(sample) - ok,
            "width_med": sorted(s[0] for s in sizes)[len(sizes) // 2] if sizes else None,
            "height_med": sorted(s[1] for s in sizes)[len(sizes) // 2] if sizes else None,
        }
    card = DatasetCard(
        name=name,
        root=root,
        n_images=n_images,
        n_classes=len(classes),
        classes=classes,
        has_healthy=has_healthy,
        images_per_class=per_class,
        license=license,
        download_date=download_date,
        md5=_md5_file(root / classes[0] / idx[classes[0]][0]) if classes and idx[classes[0]] else "",
        noise_screen_100=noise,
        healthy_keywords=healthy_keywords,
    )
    return card


def save_dataset_card(card: DatasetCard, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"dataset_card_{card.name}.json"
    save_json(card.to_dict(), p)
    return p


class PlantDataset(Dataset):
    """目录式图像分类数据集，支持按指定路径子集采样。"""

    def __init__(
        self,
        root: Path | str,
        transform: transforms.Compose | None = None,
        sample_paths: list[Path] | None = None,
        class_to_idx: dict[str, int] | None = None,
    ):
        self.root = Path(root)
        idx = index_images(self.root)
        self.classes = sorted(idx.keys()) if class_to_idx is None else list(class_to_idx.keys())
        self.class_to_idx = class_to_idx or {c: i for i, c in enumerate(self.classes)}
        if sample_paths is not None:
            self.samples = sample_paths
        else:
            self.samples = [p for c in self.classes for p in idx.get(c, [])]
        # 目标值：class_to_idx 未覆盖的类（如 SAF 二分类时仅含 healthy/diseased）回退到 0，
        # 调用方可随后覆盖 targets（如 SAF/E5 的按病理二分类目标）。
        self.targets = [self.class_to_idx.get(Path(p).parent.name, 0) for p in self.samples]
        self.transform = transform or transforms.Compose([transforms.ToTensor()])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path = self.samples[idx]
        img = Image.open(path)
        # ------------------------------------------------------------
        # 【辅助补充】根防超大原图内存爆炸：
        #  1) draft() 仅对 JPEG 有效，让解码器只解 ~256px 分辨率，不从根上加载全图；
        #  2) load() 在 draft 模式下只加载低分辨率数据；
        #  3) convert("RGB") 在已有低分辨率数据上操作，不再触发全量解码。
        #  4) 对非 JPEG（draft 无效）的大图，fallback 到 thumbnail 降采样。
        # ------------------------------------------------------------
        try:
            img.draft("RGB", (256, 256))
            img.load()  # 在 draft 模式下只加载 ~256px 数据
        except Exception:
            pass
        if max(img.size) > 2560:
            # thumbnail 仍会全量解码，但至少比直接 convert 安全
            img.thumbnail((2560, 2560), Image.LANCZOS)
        img = img.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, self.targets[idx], str(path)

    def class_weights(self) -> list[float]:
        """用于不均衡处理（本方案 E2 未用，预留）。"""
        import collections

        counts = collections.Counter(self.targets)
        n = len(self.samples)
        return [1.0 - (counts[i] / n) for i in range(len(self.classes))]


def default_train_transform(size: int = 224):
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def default_eval_transform(size: int = 224):
    return transforms.Compose(
        [
            transforms.Resize(int(size * 256 / 224)),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )