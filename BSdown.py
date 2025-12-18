# -*- coding: utf-8 -*-
"""
Created on Fri Nov 28 18:01:51 2025

@author: hjkim
"""

#!/usr/bin/env python3
"""
Materials Project Band Structure Image Downloader (CSV-based, SC k-path)
=======================================================================

- train.csv / val.csv / test.csv 각각의 상단 50%에 해당하는 mp-id에 대해서만
  밴드구조 이미지를 생성
- 각 split(train/val/test)는 output_dir 하위의 개별 폴더에 저장
- k-path는 항상 Setyawan–Curtarolo(SC) 방식 사용
- metadata.json에는 각 mpid의 공간군 정보와 k-path 라벨 포함
"""

# ============================================================
# Python 3.10 호환성 패치 (반드시 다른 import 전에!)
# ============================================================
import sys
if sys.version_info < (3, 11):
    from typing_extensions import NotRequired
    import typing
    typing.NotRequired = NotRequired

import os
import io
import json
import logging
import warnings
from datetime import datetime
from pathlib import Path
from functools import wraps

# NumPy 2.0 호환성 패치 (np.deprecate 제거됨)
import numpy as np
if not hasattr(np, 'deprecate'):
    def _np_deprecate_decorator(message=""):
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                warnings.warn(
                    f"{func.__name__} is deprecated. {message}",
                    DeprecationWarning,
                    stacklevel=2
                )
                return func(*args, **kwargs)
            return wrapper
        if callable(message):
            func = message
            message = ""
            return decorator(func)
        return decorator
    np.deprecate = _np_deprecate_decorator
# ============================================================

import gc
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

from mp_api.client import MPRester
from mp_api.client.core import MPRestError
from emmet.core.electronic_structure import BSPathType
from pymatgen.electronic_structure.core import Spin


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class BandStructureDownloader:
    """
    Download and process band structure images from Materials Project (SC path only).

    Attributes:
        api_key (str): Materials Project API key
        output_dir (Path): Directory to save images
        energy_range (float): Energy range around Fermi level (default: 4.0 eV)
        image_size (tuple): Output image size (default: 224x224)
        binarize (bool): Whether to binarize the images
        kpath_convention (str): 여기서는 항상 'setyawan_curtarolo'
    """

    def __init__(
        self,
        api_key: str = None,
        output_dir: str = "./band_structure_images",
        energy_range: float = 4.0,
        image_size: int = 224,
        binarize: bool = True,
    ):
        self.api_key = api_key or os.getenv("MP_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API key required. Set MP_API_KEY environment variable or pass api_key parameter."
            )

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.energy_range = energy_range
        self.image_size = (image_size, image_size)
        self.binarize = binarize

        # SC path 고정
        self.path_type = BSPathType.setyawan_curtarolo
        self.kpath_convention = "setyawan_curtarolo"

        # raw / processed 디렉토리
        self.raw_dir = self.output_dir / "raw"
        self.processed_dir = self.output_dir / "processed"
        self.raw_dir.mkdir(exist_ok=True)
        self.processed_dir.mkdir(exist_ok=True)

        logger.info(f"[SC downloader] Output directory: {self.output_dir}")
        logger.info(f"Energy range: ±{self.energy_range} eV")
        logger.info(f"Image size: {self.image_size}")
        logger.info(f"Binarize: {self.binarize}")
        logger.info(f"k-path convention: {self.kpath_convention}")

    # ------------------------ 유틸 함수 ------------------------
    def _extract_kpath_labels(self, bs) -> list:
        """BandStructureSymmLine에서 고유 k-point 라벨 시퀀스 추출."""
        labels = []
        for kpt in bs.kpoints:
            lab = getattr(kpt, "label", None)
            if lab:
                if not labels or labels[-1] != lab:
                    labels.append(lab)
        return labels

    # ------------------------ 밴드 다운로드 ------------------------
    def download_bandstructure(self, material_id: str, max_retries: int = 3):
        """
        SC(Setyawan–Curtarolo) k-path 밴드 구조 다운로드.
        SC 밴드가 없으면 None 반환 (pass).
        """
        import time
        
        for attempt in range(max_retries):
            try:
                with MPRester(self.api_key) as mpr:
                    bs = mpr.get_bandstructure_by_material_id(
                        material_id,
                        path_type=self.path_type,
                        line_mode=True,
                    )
                return bs
            except MPRestError as e:
                logger.warning(
                    f"[SC missing] No SC band structure for {material_id}: {e}"
                )
                return None
            except Exception as e:
                err_msg = str(e).lower()
                # JSON 파싱 에러 또는 네트워크 에러 -> 재시도
                if "parsing" in err_msg or "memory" in err_msg or "allocate" in err_msg:
                    if attempt < max_retries - 1:
                        logger.warning(f"[Retry {attempt+1}/{max_retries}] {material_id}: {e}")
                        time.sleep(2)  # 2초 대기 후 재시도
                        gc.collect()
                        continue
                    else:
                        logger.warning(f"[Skip] Failed after {max_retries} retries for {material_id}")
                        return None
                else:
                    logger.warning(f"[Error] Failed to download {material_id}: {e}")
                    return None
        return None

    # ------------------------ 플로팅 + raw 이미지 생성 ------------------------
    def plot_bandstructure(self, bs, material_id: str, save_path: Path = None) -> np.ndarray:
        """
        SC k-path 밴드 구조를 플로팅하고, raw 이미지를 numpy array로 반환.
        - y축: E - E_F (±energy_range)
        - x축: bs.distance
        - 축/라벨은 남겨둔 버전 (raw)
        - fig.canvas.tostring_rgb() 대신 BytesIO + PIL 사용
        """
        efermi = bs.efermi
        distances = bs.distance  # BandStructureSymmLine에서 제공하는 거리

        fig, ax = plt.subplots(figsize=(6, 6), dpi=100)

        for spin, band_arr in bs.bands.items():
            for ib in range(band_arr.shape[0]):
                energies = band_arr[ib] - efermi
                ax.plot(distances, energies, color="black", linewidth=1.0)

        ax.axhline(0.0, color="gray", linestyle="--", linewidth=0.7)
        ax.set_ylim(-self.energy_range, self.energy_range)
        ax.set_xlabel("k-path (SC)")
        ax.set_ylabel("Energy (eV)")
        ax.set_title(f"{material_id} — SC band structure")

        # k-point tick + label
        xticks = []
        xlabels = []
        for i, kpt in enumerate(bs.kpoints):
            lab = getattr(kpt, "label", None)
            if lab:
                xticks.append(distances[i])
                xlabels.append(lab)
        if xticks:
            ax.set_xticks(xticks)
            ax.set_xticklabels(xlabels, fontsize=8)

        ax.set_facecolor("white")
        fig.patch.set_facecolor("white")
        plt.tight_layout(pad=0.1)

        # BytesIO 버퍼에 PNG로 저장 후 PIL로 읽어서 numpy로 변환
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.05)
        buf.seek(0)
        plt.close(fig)

        pil_img = Image.open(buf).convert("RGB")
        img_array = np.array(pil_img)

        # 원하면 raw PNG도 같이 저장
        if save_path is not None:
            pil_img.save(save_path)
            logger.debug(f"Raw image saved to {save_path}")

        return img_array

    # ------------------------ 후처리 (whiteout + resize + binarize) ------------------------
    def process_image(self, img_array: np.ndarray) -> np.ndarray:
        """
        후처리:
        - grayscale 변환
        - self.image_size 로 resize
        - binarize(True)면 threshold 128 기준 이진화
        """
        img = Image.fromarray(img_array).convert("L")
        img = img.resize(self.image_size, Image.Resampling.LANCZOS)
        arr = np.array(img)

        if self.binarize:
            arr = (arr > 128).astype(np.uint8) * 255

        return arr

    # ------------------------ 개별 구조 처리 ------------------------
    def process_single_material(self, material_id: str, formula: str = None):
        """
        한 개 mp-id에 대해:
        - SC 밴드 다운로드
        - raw 이미지 생성
        - whiteout+resize+binarize 이미지 생성
        - metadata dict 반환
        """
        output_filename = f"{material_id}.png"
        processed_path = self.processed_dir / output_filename
        raw_path = self.raw_dir / f"{material_id}_raw.png"

        # 이미 처리된 경우 skip
        if processed_path.exists():
            logger.debug(f"Skipping {material_id} (processed image exists)")
            meta = {
                "material_id": material_id,
                "formula": formula,
                "nsites": None,
                "spacegroup_symbol": None,
                "spacegroup_number": None,
                "kpath_convention": self.kpath_convention,
                "kpath_labels": None,
                "raw_image": str(raw_path),
                "processed_image": str(processed_path),
                "status": "skipped_existing",
            }
            return True, meta

        bs = self.download_bandstructure(material_id)
        if bs is None:
            return False, None

        try:
            # 공간군 정보
            sg_symbol = None
            sg_number = None
            nsites = None
            try:
                if getattr(bs, "structure", None) is not None:
                    sg_symbol, sg_number = bs.structure.get_space_group_info()
                    nsites = bs.structure.num_sites
            except Exception:
                pass

            # raw 이미지 생성
            raw_img_array = self.plot_bandstructure(bs, material_id, save_path=raw_path)

            # 후처리 이미지 생성
            processed_arr = self.process_image(raw_img_array)
            Image.fromarray(processed_arr).save(processed_path)

            # k-path 라벨
            kpath_labels = self._extract_kpath_labels(bs)

            meta = {
                "material_id": material_id,
                "formula": formula,
                "nsites": nsites,
                "spacegroup_symbol": sg_symbol,
                "spacegroup_number": sg_number,
                "kpath_convention": self.kpath_convention,
                "kpath_labels": kpath_labels,
                "raw_image": str(raw_path),
                "processed_image": str(processed_path),
                "status": "success",
            }

            logger.debug(f"Processed {material_id}")
            return True, meta

        except Exception as e:
            logger.warning(f"Failed to process {material_id}: {e}")
            return False, None
        finally:
            # 메모리 정리
            gc.collect()

    # ------------------------ 여러 개 일괄 처리 ------------------------
    def download_all(self, material_ids: list, save_metadata: bool = True):
        """
        여러 mp-id에 대해 일괄 처리.
        SC 밴드가 없는 mp-id는 failed로 기록되고 skip.
        """
        if not material_ids:
            logger.warning("No material_ids provided to download_all.")
            return [], []

        logger.info(f"Processing {len(material_ids)} materials in {self.output_dir}...")

        successful = []
        failed = []

        for i, mid in enumerate(tqdm(material_ids, desc="Downloading SC band structures")):
            material_id = str(mid)
            formula = None  # CSV에 formula 있으면 여기에 넣어도 됨

            ok, meta = self.process_single_material(material_id, formula=formula)
            if ok and meta is not None:
                successful.append(meta)
            else:
                failed.append({"material_id": material_id, "formula": formula})
            
            # 메모리 누수 방지: 매 5개마다 가비지 컬렉션
            if i % 5 == 0:
                gc.collect()

        if save_metadata:
            metadata = {
                "timestamp": datetime.utcnow().isoformat(),
                "output_dir": str(self.output_dir),
                "kpath_convention": self.kpath_convention,
                "num_successful": len(successful),
                "num_failed": len(failed),
                "successful": successful,
                "failed": failed,
            }
            metadata_path = self.output_dir / "metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
            logger.info(f"Metadata saved to {metadata_path}")

        logger.info(
            f"Download complete in {self.output_dir}: "
            f"{len(successful)} successful, {len(failed)} failed"
        )

        return successful, failed


# ------------------------ CSV에서 상단 50% mp-id 불러오기 ------------------------
def load_half_material_ids_from_csv(csv_path: str, id_col: str = "material_id") -> list:
    """
    CSV 파일에서 material_id 컬럼 기준 상단 50%만 사용.
    예: 헤더 제외 100행이면 앞의 50개만 반환.
    """
    df = pd.read_csv(csv_path)
    if id_col not in df.columns:
        raise ValueError(f"{csv_path}에 '{id_col}' 컬럼이 없습니다.")
    ids = df[id_col].dropna().astype(str).tolist()
    n = len(ids)
    half = n // 2
    logger.info(f"{csv_path}: 총 {n}개 중 상단 {half}개 mp-id 사용")
    return ids[:half]


# ------------------------ main (train/val/test용) ------------------------
def main():
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Download SC (Setyawan–Curtarolo) band structure images from Materials Project\n"
            "train/val/test CSV 각각의 상단 50%에 대해 SC k-path 밴드 이미지를 생성합니다."
        )
    )
    parser.add_argument(
        "--api-key",
        "-k",
        help="Materials Project API key (또는 MP_API_KEY 환경변수 사용)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="./band_structure_images_sc",
        help="Output base directory (train/val/test 하위 폴더 생성)",
    )
    parser.add_argument(
        "--energy-range",
        "-e",
        type=float,
        default=4.0,
        help="Energy range around Fermi level in eV (default: 4.0)",
    )
    parser.add_argument(
        "--image-size",
        "-s",
        type=int,
        default=224,
        help="Image size in pixels (default: 224)",
    )
    parser.add_argument(
        "--no-binarize",
        action="store_true",
        help="Do not binarize images (그레이스케일 유지)",
    )
    parser.add_argument(
        "--train-csv",
        type=str,
        help="train.csv 경로 (material_id 컬럼 필요)",
    )
    parser.add_argument(
        "--val-csv",
        type=str,
        help="val.csv 경로 (material_id 컬럼 필요)",
    )
    parser.add_argument(
        "--test-csv",
        type=str,
        help="test.csv 경로 (material_id 컬럼 필요)",
    )

    args = parser.parse_args()

    api_key = args.api_key or os.getenv("MP_API_KEY")
    if not api_key:
        raise ValueError(
            "API key required. Use --api-key or set MP_API_KEY environment variable."
        )

    base_output = Path(args.output_dir)

    splits = []
    if args.train_csv:
        splits.append(("train", args.train_csv))
    if args.val_csv:
        splits.append(("val", args.val_csv))
    if args.test_csv:
        splits.append(("test", args.test_csv))

    if not splits:
        raise ValueError("최소 하나의 CSV(train/val/test)를 지정해야 합니다.")

    for split_name, csv_path in splits:
        logger.info(f"=== Processing split: {split_name} ({csv_path}) ===")
        mpids = load_half_material_ids_from_csv(csv_path, id_col="material_id")
        split_output_dir = base_output / split_name

        downloader = BandStructureDownloader(
            api_key=api_key,
            output_dir=str(split_output_dir),
            energy_range=args.energy_range,
            image_size=args.image_size,
            binarize=not args.no_binarize,
        )

        downloader.download_all(material_ids=mpids, save_metadata=True)


if __name__ == "__main__":
    main()