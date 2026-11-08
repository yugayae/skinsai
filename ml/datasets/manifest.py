"""Canonical dataset item and manifest schema for SkinAI.

Defines Pydantic models for single dataset records (DatasetItem) and full
manifests (DatasetManifest) with export/import to CSV and JSON, validation,
and reproducibility hashing.
"""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

LicenseScope = Literal["commercial_ok", "public_ok", "research_only"]
DatasetSplit = Literal["train", "val", "test"]


class DatasetItem(BaseModel):
    """Canonical metadata for a single skin lesion image."""

    model_config = ConfigDict(extra="ignore")

    image_id: str = Field(..., description="Unique image identifier (e.g. ISIC_0601025)")
    profile: str = Field(..., description="Target profile: 'smartphone_clinical' or 'dermoscopy'")
    class_id: str = Field(..., description="Normalized target class label (e.g. 'bcc', 'mel')")
    raw_diagnosis: str = Field(..., description="Raw diagnosis as provided by the data source")
    license: str = Field(..., description="Copyright license (e.g. 'CC-BY', 'CC-0')")
    license_scope: LicenseScope = Field(
        ...,
        description="Data governance licensing tier: 'commercial_ok', 'public_ok', or 'research_only'"
    )
    patient_id: Optional[str] = Field(default=None, description="Patient identifier for group splitting")
    lesion_id: Optional[str] = Field(default=None, description="Lesion identifier for group splitting")
    image_path: Optional[str] = Field(default=None, description="Relative or absolute path to local image")
    source_url: Optional[str] = Field(default=None, description="Direct download URL")
    sha256: Optional[str] = Field(default=None, description="SHA-256 hash of image file")
    width: Optional[int] = Field(default=None, description="Image width in pixels")
    height: Optional[int] = Field(default=None, description="Image height in pixels")
    split: Optional[DatasetSplit] = Field(default=None, description="Assigned split: 'train', 'val', or 'test'")
    metadata_extra: Dict[str, Any] = Field(
        default_factory=dict,
        description="Supplementary metadata (anatomical site, age, sex, etc.)"
    )

    def to_row(self) -> Dict[str, Any]:
        """Convert item to flat dictionary suitable for CSV / DataFrame."""
        return {
            "image_id": self.image_id,
            "profile": self.profile,
            "class_id": self.class_id,
            "raw_diagnosis": self.raw_diagnosis,
            "license": self.license,
            "license_scope": self.license_scope,
            "patient_id": self.patient_id or "",
            "lesion_id": self.lesion_id or "",
            "image_path": self.image_path or "",
            "source_url": self.source_url or "",
            "sha256": self.sha256 or "",
            "width": self.width or "",
            "height": self.height or "",
            "split": self.split or "",
            "metadata_extra": json.dumps(self.metadata_extra, ensure_ascii=False) if self.metadata_extra else "{}",
        }

    @classmethod
    def from_row(cls, row: Dict[str, Any]) -> DatasetItem:
        """Construct DatasetItem from flat row dictionary."""
        extra_raw = row.get("metadata_extra", "{}")
        if isinstance(extra_raw, str):
            try:
                extra = json.loads(extra_raw) if extra_raw else {}
            except json.JSONDecodeError:
                extra = {}
        elif isinstance(extra_raw, dict):
            extra = extra_raw
        else:
            extra = {}

        w = row.get("width")
        h = row.get("height")
        split_val = row.get("split")
        if split_val in ("", "None", None, float("nan")):
            split_val = None

        return cls(
            image_id=str(row["image_id"]),
            profile=str(row["profile"]),
            class_id=str(row["class_id"]),
            raw_diagnosis=str(row.get("raw_diagnosis", "")),
            license=str(row.get("license", "unknown")),
            license_scope=row.get("license_scope", "research_only"),
            patient_id=str(row["patient_id"]) if row.get("patient_id") else None,
            lesion_id=str(row["lesion_id"]) if row.get("lesion_id") else None,
            image_path=str(row["image_path"]) if row.get("image_path") else None,
            source_url=str(row["source_url"]) if row.get("source_url") else None,
            sha256=str(row["sha256"]) if row.get("sha256") else None,
            width=int(w) if w not in ("", None, float("nan")) else None,
            height=int(h) if h not in ("", None, float("nan")) else None,
            split=split_val,
            metadata_extra=extra,
        )


class DatasetManifest(BaseModel):
    """Collection of dataset items representing a full profile dataset."""

    profile: str = Field(..., description="Profile name ('smartphone_clinical' or 'dermoscopy')")
    version: str = Field(default="1.0.0", description="Dataset manifest schema version")
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 UTC timestamp of creation"
    )
    items: List[DatasetItem] = Field(default_factory=list, description="Ordered dataset items")

    def __len__(self) -> int:
        return len(self.items)

    def to_pandas(self) -> pd.DataFrame:
        """Convert manifest items to Pandas DataFrame."""
        rows = [item.to_row() for item in self.items]
        return pd.DataFrame(rows)

    @classmethod
    def from_pandas(cls, df: pd.DataFrame, profile: str, version: str = "1.0.0") -> DatasetManifest:
        """Construct DatasetManifest from Pandas DataFrame."""
        items = [DatasetItem.from_row(row.to_dict()) for _, row in df.iterrows()]
        return cls(profile=profile, version=version, items=items)

    def save_csv(self, path: Union[str, Path]) -> None:
        """Save manifest items to CSV format."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        df = self.to_pandas()
        df.to_csv(target, index=False, quoting=csv.QUOTE_MINIMAL, encoding="utf-8")

    @classmethod
    def from_csv(cls, path: Union[str, Path], profile: str) -> DatasetManifest:
        """Load manifest from CSV file."""
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        return cls.from_pandas(df, profile=profile)

    def save_json(self, path: Union[str, Path]) -> None:
        """Save entire manifest to JSON format."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            f.write(self.model_dump_json(indent=2))

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> DatasetManifest:
        """Load manifest from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.model_validate(data)

    def filter_split(self, split: DatasetSplit) -> DatasetManifest:
        """Return a sub-manifest containing only items assigned to the given split."""
        filtered = [item for item in self.items if item.split == split]
        return DatasetManifest(
            profile=self.profile,
            version=self.version,
            items=filtered,
        )

    def compute_manifest_hash(self) -> str:
        """Compute deterministic SHA-256 hash of all items in manifest for provenance."""
        hasher = hashlib.sha256()
        # Sort items by image_id for determinism
        for item in sorted(self.items, key=lambda x: x.image_id):
            row_repr = f"{item.image_id}:{item.class_id}:{item.patient_id}:{item.lesion_id}:{item.split}:{item.sha256}"
            hasher.update(row_repr.encode("utf-8"))
        return hasher.hexdigest()

    def get_summary(self) -> Dict[str, Any]:
        """Compute statistical breakdown of the dataset manifest."""
        total = len(self.items)
        splits: Dict[str, int] = {}
        classes: Dict[str, int] = {}
        licenses: Dict[str, int] = {}
        patients = set()
        lesions = set()

        for item in self.items:
            s = item.split or "unassigned"
            splits[s] = splits.get(s, 0) + 1
            classes[item.class_id] = classes.get(item.class_id, 0) + 1
            licenses[item.license] = licenses.get(item.license, 0) + 1
            if item.patient_id:
                patients.add(item.patient_id)
            if item.lesion_id:
                lesions.add(item.lesion_id)

        return {
            "profile": self.profile,
            "version": self.version,
            "total_images": total,
            "unique_patients": len(patients),
            "unique_lesions": len(lesions),
            "splits": splits,
            "classes": classes,
            "licenses": licenses,
            "manifest_hash": self.compute_manifest_hash(),
        }
