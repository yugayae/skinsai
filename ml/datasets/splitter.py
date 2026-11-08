"""Group-level dataset splitter for SkinAI.

Strictly enforces patient-level and lesion-level isolation between Train,
Validation, and Test sets to prevent data leakage. Verifies that no patient
or lesion appears in more than one partition.
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from typing import Dict, List, Set

from ml.datasets.manifest import DatasetItem, DatasetManifest, DatasetSplit


class GroupLeakageError(Exception):
    """Raised when patient or lesion overlap is detected across dataset splits."""
    pass


class DatasetSplitter:
    """Group-level splitter that guarantees zero patient and lesion data leakage."""

    def __init__(
        self,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
    ):
        total = train_ratio + val_ratio + test_ratio
        if abs(total - 1.0) > 1e-4:
            raise ValueError(f"Ratios must sum to 1.0, got {total:.4f}")
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed

    def split(self, manifest: DatasetManifest) -> DatasetManifest:
        """Assign each item in the manifest to 'train', 'val', or 'test'.

        Grouping priority:
        1. 'patient_id' (if available)
        2. 'lesion_id' (if patient_id is missing)
        3. 'image_id' (if both are missing)

        All images belonging to the same group are strictly placed in the SAME split.
        """
        rng = random.Random(self.seed)

        # 1. Group items by patient/lesion cluster
        # Map group_id -> list of items
        groups: Dict[str, List[DatasetItem]] = defaultdict(list)
        for item in manifest.items:
            group_key = item.patient_id or item.lesion_id or f"img_{item.image_id}"
            groups[group_key].append(item)

        # 2. Determine dominant class for each group for stratified allocation
        group_keys = list(groups.keys())
        rng.shuffle(group_keys)

        # Calculate dominant class for each group
        def get_group_dominant_class(g_key: str) -> str:
            classes = [it.class_id for it in groups[g_key]]
            return Counter(classes).most_common(1)[0][0]

        # Organize groups by dominant class
        class_to_groups: Dict[str, List[str]] = defaultdict(list)
        for g_key in group_keys:
            dom_cls = get_group_dominant_class(g_key)
            class_to_groups[dom_cls].append(g_key)

        train_groups: Set[str] = set()
        val_groups: Set[str] = set()
        test_groups: Set[str] = set()

        # 3. Stratified group allocation per class
        for cls_name, cls_g_keys in sorted(class_to_groups.items()):
            rng.shuffle(cls_g_keys)
            n = len(cls_g_keys)

            n_train = max(1, int(round(n * self.train_ratio))) if n >= 3 else (1 if n >= 1 else 0)
            n_val = max(1, int(round(n * self.val_ratio))) if n >= 3 else (1 if n >= 2 else 0)
            n_test = n - n_train - n_val

            if n_test < 0:
                n_train = max(1, n - 2) if n >= 3 else 1
                n_val = 1 if n >= 2 else 0
                n_test = max(0, n - n_train - n_val)

            t_keys = cls_g_keys[:n_train]
            v_keys = cls_g_keys[n_train : n_train + n_val]
            te_keys = cls_g_keys[n_train + n_val :]

            train_groups.update(t_keys)
            val_groups.update(v_keys)
            test_groups.update(te_keys)

        # 4. Construct updated items
        updated_items: List[DatasetItem] = []
        for g_key, items in groups.items():
            if g_key in train_groups:
                target_split: DatasetSplit = "train"
            elif g_key in val_groups:
                target_split = "val"
            elif g_key in test_groups:
                target_split = "test"
            else:
                target_split = "train"  # Fallback safety

            for it in items:
                # Create updated item with assigned split
                updated_it = it.model_copy(update={"split": target_split})
                updated_items.append(updated_it)

        # 5. Create new manifest
        split_manifest = DatasetManifest(
            profile=manifest.profile,
            version=manifest.version,
            items=updated_items,
        )

        # 6. Strict Zero Leakage Verification
        self.verify_no_leakage(split_manifest)

        return split_manifest

    @staticmethod
    def verify_no_leakage(manifest: DatasetManifest) -> None:
        """Verify that zero patient or lesion identifiers leak across splits."""
        train_patients: Set[str] = set()
        val_patients: Set[str] = set()
        test_patients: Set[str] = set()

        train_lesions: Set[str] = set()
        val_lesions: Set[str] = set()
        test_lesions: Set[str] = set()

        for it in manifest.items:
            if not it.split:
                raise GroupLeakageError(f"Item {it.image_id} has no assigned split.")

            if it.split == "train":
                if it.patient_id:
                    train_patients.add(it.patient_id)
                if it.lesion_id:
                    train_lesions.add(it.lesion_id)
            elif it.split == "val":
                if it.patient_id:
                    val_patients.add(it.patient_id)
                if it.lesion_id:
                    val_lesions.add(it.lesion_id)
            elif it.split == "test":
                if it.patient_id:
                    test_patients.add(it.patient_id)
                if it.lesion_id:
                    test_lesions.add(it.lesion_id)

        # Check patient overlap
        tv_p = train_patients & val_patients
        tt_p = train_patients & test_patients
        vt_p = val_patients & test_patients
        if tv_p:
            raise GroupLeakageError(f"Patient leakage between Train and Val: {len(tv_p)} patients: {list(tv_p)[:5]}")
        if tt_p:
            raise GroupLeakageError(f"Patient leakage between Train and Test: {len(tt_p)} patients: {list(tt_p)[:5]}")
        if vt_p:
            raise GroupLeakageError(f"Patient leakage between Val and Test: {len(vt_p)} patients: {list(vt_p)[:5]}")

        # Check lesion overlap
        tv_l = train_lesions & val_lesions
        tt_l = train_lesions & test_lesions
        vt_l = val_lesions & test_lesions
        if tv_l:
            raise GroupLeakageError(f"Lesion leakage between Train and Val: {len(tv_l)} lesions: {list(tv_l)[:5]}")
        if tt_l:
            raise GroupLeakageError(f"Lesion leakage between Train and Test: {len(tt_l)} lesions: {list(tt_l)[:5]}")
        if vt_l:
            raise GroupLeakageError(f"Lesion leakage between Val and Test: {len(vt_l)} lesions: {list(vt_l)[:5]}")
