from pathlib import Path
import xarray as xr

from .registry import register_loader
from ..properties.properties import Space, Time, Uncertainty
from .base import BaseLoader

import xarray as xr


DEFAULTS = {
    "engine": "h5netcdf",
    "parallel": True,
    "combine": "nested",
    "concat_dim": "reference_time",
}


@register_loader
class AnemoiInferenceLoader(BaseLoader):
    name = "anemoi-inference"

    space = Space.GRID
    time = Time.FORECAST

    def __init__(
        self,
        files,
        variables=None,
        grid_mapping=None,
        ens_size=None,
        **kwargs
    ):
        super().__init__(
            files,
            variables,
            grid_mapping,
            **kwargs
        )

        # Validate ens_size if provided
        if ens_size is not None:
            if not isinstance(ens_size, int) or ens_size < 1:
                raise ValueError(
                    f"ens_size must be a positive integer, got {ens_size!r}"
                )
        self.ens_size = ens_size

        # Determine uncertainty based on ens_size
        self.uncertainty = (
            Uncertainty.ENSEMBLE
            if ens_size is not None and ens_size > 1
            else Uncertainty.DETERMINISTIC
        )

    def _load(self):

        files = (
            [self.files]
            if isinstance(self.files, str)
            else self.files
        )

        if not files:
            raise ValueError("No files provided.")

        if self.ens_size is not None and self.ens_size > 1:
            return self._load_with_members(files)

        if self.ens_size is None:
            print(
                "Warning: ens_size not set, "
                "defaulting to deterministic loading."
            )

        return self._load_deterministic(files)

    def _load_deterministic(self, files):
        """Load forecast data without a member dimension."""

        with xr.open_dataset(files[0]) as first_ds:
            times = first_ds["time"].values

        lead_times = times - times[0]

        kwargs = {
            **DEFAULTS,
            **self.kwargs,
        }

        ds = xr.open_mfdataset(
            files,
            preprocess=_preprocess_deterministic,
            **kwargs
        )

        ds_out = (
            ds
            .assign_coords({
                "lead_time": ("time", lead_times)
            })
            .rename_dims({
                "values": "grid_index"
            })
            .swap_dims({
                "time": "lead_time"
            })
            .chunk({
                "grid_index": -1
            })
        )


    def _load_with_members(self, files):
        """Load ensemble forecast data with a member dimension."""

        engine = self.kwargs.get(
            "engine",
            DEFAULTS["engine"]
        )

        if len(files) % self.ens_size != 0:
            raise ValueError(
                f"Number of files ({len(files)}) must be divisible "
                f"by ens_size ({self.ens_size})."
            )

        with xr.open_dataset(
            files[0],
            engine=engine
        ) as first_ds:
            times = first_ds["time"].values

        lead_times = times - times[0]

        batches = [
            files[i:i + self.ens_size]
            for i in range(
                0,
                len(files),
                self.ens_size
            )
        ]

        ref_datasets = []

        for batch in batches:

            member_datasets = []

            for member_idx, filepath in enumerate(batch):

                ds = xr.open_dataset(
                    filepath,
                    engine=engine,
                    chunks={}
                )

                ds = _preprocess_with_member(
                    ds,
                    member_idx
                )

                member_datasets.append(ds)

            ref_ds = xr.concat(
                member_datasets,
                dim="member"
            )

            ref_datasets.append(ref_ds)

        ds = xr.concat(
            ref_datasets,
            dim="reference_time"
        )

        ds_out = (
            ds
            .assign_coords({
                "lead_time": ("time", lead_times)
            })
            .rename_dims({
                "values": "grid_index"
            })
            .swap_dims({
                "time": "lead_time"
            })
            .chunk({
                "member": -1,
                "grid_index": -1,
                "lead_time": 1
            })
        )

        return ds_out


def _preprocess_deterministic(ds):
    """Preprocess a single forecast file without member dimension."""

    ds_out = (
        ds
        .set_coords(["longitude", "latitude"])
        .expand_dims("reference_time")
        .assign_coords({
            "reference_time": (
                "reference_time",
                [ds["time"].values[0]]
            )
        })
        .drop_vars("time")
    )

    return ds_out


def _preprocess_with_member(ds, member_idx):
    """Preprocess a forecast file and add its ensemble member."""

    ds_out = (
        ds
        .set_coords(["longitude", "latitude"])
        .expand_dims("reference_time")
        .assign_coords({
            "reference_time": (
                "reference_time",
                [ds["time"].values[0]]
            )
        })
        .expand_dims("member")
        .assign_coords({
            "member": (
                "member",
                [member_idx]
            )
        })
        .drop_vars("time")
    )

    return ds_out


def _preprocess_deterministic(ds):
    """Preprocess a single forecast file without member dimension."""
    ds_out = ds.\
        set_coords(["longitude", "latitude"]).\
        expand_dims("reference_time").\
        assign_coords(
            {"reference_time": ("reference_time", [ds["time"].values[0]])}
        ).\
        drop_vars("time")
    
    return ds_out


def _preprocess_with_member(ds, member_idx):
    """Preprocess and add member dimension with explicit member index."""
    ds_out = ds.\
        set_coords(["longitude", "latitude"]).\
        expand_dims("reference_time").\
        assign_coords(
            {"reference_time": ("reference_time", [ds["time"].values[0]])}
        ).\
        expand_dims("member").\
        assign_coords({"member": ("member", [member_idx])}).\
        drop_vars("time")

    return ds_out
