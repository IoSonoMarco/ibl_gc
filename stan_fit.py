import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from cmdstanpy import CmdStanModel

from lib.models.utils import (
    PerceptualModule,
    ResponseModule,
    get_stan_model_paths,
)
from lib import filepaths


stan_model_paths = get_stan_model_paths()

perceptual_map = {
    "basic": PerceptualModule.BASIC,
    "scpw": PerceptualModule.SCPW,
}

response_map = {
    "basic": ResponseModule.BASIC,
    "brt": ResponseModule.BRT,
    "brrt": ResponseModule.BRRT,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run CmdStan fits for the selected perceptual/response model and hierarchy."
        )
    )
    parser.add_argument(
        "--perceptual-model",
        default="scpw",
        choices=sorted(perceptual_map.keys()),
        help="Perceptual model to use.",
    )
    parser.add_argument(
        "--response-model",
        default="brt",
        choices=sorted(response_map.keys()),
        help="Response model to use.",
    )
    parser.add_argument(
        "--model-type",
        default="2level",
        help="Model type subdirectory to use under the Stan model root.",
    )
    parser.add_argument(
        "--hierarchy",
        default="single_session",
        help=(
            "Hierarchy to use. Supported runtime options are 'single_session', "
            "'single_subject', and 'full'."
        ),
    )
    parser.add_argument(
        "--n-chains",
        type=int,
        default=3,
        help="Number of MCMC chains.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=1000,
        help="Number of post-warmup samples per chain.",
    )
    parser.add_argument(
        "--n-warmup",
        type=int,
        default=200,
        help="Number of warmup iterations per chain.",
    )
    return parser.parse_args()


def build_model_and_paths(args: argparse.Namespace):
    name_suffix = f"perc__{args.perceptual_model}-res__{args.response_model}"
    output_dir = Path(filepaths.ROOT_STAN_MODEL_FITS) / args.hierarchy / args.model_type
    os.makedirs(output_dir, exist_ok=True)

    root_col = f"{args.hierarchy}_root"
    if root_col not in stan_model_paths.columns:
        available = sorted([c for c in stan_model_paths.columns if c.endswith("_root")])
        raise ValueError(
            f"Hierarchy '{args.hierarchy}' is not available in stan_model_paths. "
            f"Expected column '{root_col}'. Available root columns: {available}"
        )

    d = stan_model_paths[
        (stan_model_paths.perc == perceptual_map[args.perceptual_model])
        & (stan_model_paths.resp == response_map[args.response_model])
    ]

    if d.empty:
        raise ValueError(
            "No Stan model path matched the selected perceptual/response model combination."
        )

    stan_model_path = Path(d[root_col].iloc[0]) / args.model_type / d.filename.iloc[0]
    model = CmdStanModel(stan_file=stan_model_path)
    return model, output_dir, name_suffix


def sample_model(model: CmdStanModel, data: dict, args: argparse.Namespace):
    return model.sample(
        data=data,
        chains=args.n_chains,
        parallel_chains=args.n_chains,
        iter_sampling=args.n_samples,
        iter_warmup=args.n_warmup,
    )


def run_single_session_fit(
    model: CmdStanModel, output_dir: Path, name_suffix: str, args: argparse.Namespace
):
    dataset = pd.read_csv(filepaths.ROOT_BEHAV_DATA)

    for session_id_code in dataset.session_id_code.unique():
        df_session = dataset[dataset.session_id_code == session_id_code]
        eid = df_session.session_id.unique()[0]

        data = {
            "N_obs": len(df_session.stimulus_side),
            "stimulus_side": df_session.stimulus_side.values,
            "stimulus_contrast": df_session.stimulus_contrast.values,
            "choice": df_session.choice.values,
            "feedback": df_session.feedback.values,
        }

        fit = sample_model(model, data, args)
        savename = f"{eid}-{name_suffix}"
        np.save(output_dir / savename, fit)


def run_single_subject_fit(
    model: CmdStanModel, output_dir: Path, name_suffix: str, args: argparse.Namespace
):
    dataset = pd.read_csv(filepaths.ROOT_BEHAV_DATA)
    subj_ids_high_sessions = (
        dataset.groupby("subj_id_code")
        .session_id_code.nunique()
        .reset_index()
        .query("session_id_code >= 4")
        .subj_id_code.values
    )
    dataset = dataset[dataset.subj_id_code.isin(subj_ids_high_sessions)].reset_index(drop=True)
    subj_id_map = {s: i for i, s in enumerate(dataset.subj_id_code.unique())}
    session_id_map = {s: i for i, s in enumerate(dataset.session_id_code.unique())}
    dataset["subj_id_code"] = dataset.subj_id_code.apply(lambda x: subj_id_map[x])
    dataset["session_id_code"] = dataset.session_id_code.apply(lambda x: session_id_map[x])

    for subj_id_code in dataset.subj_id_code.unique():
        subj_id = dataset[dataset.subj_id_code == subj_id_code].subj_id.unique()[0]
        df_subj = dataset[dataset.subj_id == subj_id].reset_index()
        start_idx, end_idx = np.array(
            [
                (idx.min() + 1, idx.max() + 1)
                for _, idx in df_subj.groupby("session_id_code").groups.items()
            ]
        ).T

        data = {
            "N_obs": len(df_subj.stimulus_side),
            "N_sess": df_subj.session_id_code.nunique(),
            "stimulus_side": df_subj.stimulus_side.values,
            "stimulus_contrast": df_subj.stimulus_contrast.values,
            "feedback": df_subj.feedback.values,
            "choice": df_subj.choice.values,
            "start_idx": start_idx,
            "end_idx": end_idx,
        }

        fit = sample_model(model, data, args)
        savename = f"{subj_id}-{name_suffix}"
        np.save(output_dir / savename, fit)


def run_full_fit(
    model: CmdStanModel, output_dir: Path, name_suffix: str, args: argparse.Namespace
):
    dataset = pd.read_csv(filepaths.ROOT_BEHAV_DATA)

    start_idx, end_idx = np.array(
        [
            (idx.min() + 1, idx.max() + 1)
            for _, idx in dataset.groupby("session_id_code").groups.items()
        ]
    ).T
    subj_ids, _ = (
        dataset.groupby(["subj_id_code", "session_id_code"])
        .choice.mean()
        .reset_index()
        .iloc[:, :2]
        .values.T
    )

    data = {
        "N_obs": len(dataset.stimulus_side),
        "N_sess": dataset.session_id_code.nunique(),
        "N_subj": dataset.subj_id_code.nunique(),
        "subj_ids": subj_ids + 1,
        "stimulus_side": dataset.stimulus_side.values,
        "stimulus_contrast": dataset.stimulus_contrast.values,
        "feedback": dataset.feedback.values,
        "choice": dataset.choice.values,
        "start_idx": start_idx,
        "end_idx": end_idx,
    }

    fit = sample_model(model, data, args)
    savename = f"full-{name_suffix}"
    np.save(output_dir / savename, fit)


def main():
    args = parse_args()
    model, output_dir, name_suffix = build_model_and_paths(args)

    if args.hierarchy == "single_session":
        run_single_session_fit(model, output_dir, name_suffix, args)
    elif args.hierarchy == "single_subject":
        run_single_subject_fit(model, output_dir, name_suffix, args)
    elif args.hierarchy == "full":
        run_full_fit(model, output_dir, name_suffix, args)
    else:
        raise ValueError(
            "Unsupported hierarchy for execution. Use one of: "
            "'single_session', 'single_subject', 'full'."
        )


if __name__ == "__main__":
    main()