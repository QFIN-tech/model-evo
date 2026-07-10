from __future__ import annotations

import importlib.metadata
import platform
from pathlib import Path
from typing import Any

from _common.io import write_json

TREATMENT_FEATURE = "__uplift_modeling_treatment__"


class AdapterInputError(Exception):
    def __init__(self, message: str, *, code: str = "MISSING_OR_INVALID_INPUT") -> None:
        super().__init__(message)
        self.code = code
        self.missing_fields: list[str] = []


class LearnerAdapter:
    model_type: str
    model_spec: dict[str, Any]
    artifact_names: dict[str, str]

    def load_model_object(self, packages: dict[str, Any], model_path: Path | str) -> dict[str, Any]:
        raise NotImplementedError

    def fit_encoder(self, packages: dict[str, Any], train_frame: Any, selected_features: list[str]) -> dict[str, Any]:
        raise NotImplementedError

    def fit_trial(self, packages: dict[str, Any], inputs: dict[str, Any], frames: dict[str, Any], encoder: dict[str, Any], parameters: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    def score_frame(
        self,
        packages: dict[str, Any],
        model_object_or_trial: dict[str, Any],
        inputs: dict[str, Any],
        frame: Any,
        split: str,
        encoder: dict[str, Any] | None,
        best_iteration: Any,
    ) -> Any:
        raise NotImplementedError

    def dump_model(self, packages: dict[str, Any], path: Path, model_object: dict[str, Any], inputs: dict[str, Any], parameters: dict[str, Any], best_iteration: Any) -> None:
        raise NotImplementedError

    def write_model_metadata(self, path: Path, model_object: dict[str, Any], inputs: dict[str, Any], parameters: dict[str, Any], best_iteration: Any) -> None:
        write_json(
            path,
            {
                "artifact_kind": "model_metadata",
                "artifact_version": 1,
                "model_type": self.model_type,
                "base_estimators": self.model_spec["base_estimators"],
                "input_paths": inputs["input_paths"],
                "selected_features": inputs["selected_features"],
                "feature_count": len(inputs["selected_features"]),
                "model_feature_order": (model_object.get("encoder") or {}).get("model_feature_order"),
                "parameter_source": "model_tuning",
                "parameter_strategy": "shared",
                "effective_parameters": parameters,
                "best_iteration": best_iteration,
                "runtime_versions": _runtime_versions(),
                "model_artifact_path": str(Path(model_object.get("model_artifact_path") or "").resolve()) if model_object.get("model_artifact_path") else None,
                "created_by": "uplift-model-tuning-skill",
                "created_at": inputs.get("created_at"),
            },
        )

    def feature_importance_frame(self, packages: dict[str, Any], model_object: dict[str, Any], encoder: dict[str, Any]) -> Any:
        raise NotImplementedError


class SLearnerAdapter(LearnerAdapter):
    model_type = "s_learner"
    model_spec = {"model_type": "s_learner", "base_estimators": {"outcome": "lightgbm"}}
    artifact_names = {"joblib": "s_learner_tuned_model.v1.joblib"}

    def load_model_object(self, packages: dict[str, Any], model_path: Path | str) -> dict[str, Any]:
        model = packages["joblib"].load(model_path)
        if not isinstance(model, dict) or "estimator" not in model or "encoder" not in model:
            raise AdapterInputError("model artifact is not a supported S-Learner model object.")
        return model

    def fit_encoder(self, packages: dict[str, Any], train_frame: Any, selected_features: list[str]) -> dict[str, Any]:
        return _fit_encoder(packages, train_frame, selected_features, include_treatment=True)

    def fit_trial(self, packages: dict[str, Any], inputs: dict[str, Any], frames: dict[str, Any], encoder: dict[str, Any], parameters: dict[str, Any]) -> dict[str, Any]:
        lgb = packages["lightgbm"]
        train = frames["train"]
        x_train = _encode_frame(packages, train, inputs["selected_features"], encoder, treatment=train[inputs["treatment_column"]])
        y_train = train[inputs["outcome_column"]]
        estimator_parameters = _estimator_parameters(parameters)
        if inputs["outcome_type"] == "binary":
            estimator = lgb.LGBMClassifier(objective="binary", metric="binary_logloss", **estimator_parameters)
            metric = "binary_logloss"
        elif inputs["outcome_type"] == "continuous":
            estimator = lgb.LGBMRegressor(objective="regression", metric="l2", **estimator_parameters)
            metric = "l2"
        else:
            raise AdapterInputError("outcome_type must be binary or continuous.")
        fit_kwargs: dict[str, Any] = {}
        valid = frames.get("valid")
        history: dict[str, Any] = {}
        if valid is not None and len(valid) > 0:
            x_valid = _encode_frame(packages, valid, inputs["selected_features"], encoder, treatment=valid[inputs["treatment_column"]])
            fit_kwargs["eval_set"] = [(x_valid, valid[inputs["outcome_column"]])]
            callbacks = [lgb.record_evaluation(history)]
            rounds = parameters.get("early_stopping_rounds")
            if rounds:
                callbacks.insert(0, lgb.early_stopping(int(rounds), verbose=False))
            fit_kwargs["callbacks"] = callbacks
        estimator.fit(x_train, y_train, **fit_kwargs)
        best_iteration = getattr(estimator, "best_iteration_", None) or None
        return {
            "model_type": self.model_type,
            "estimator": estimator,
            "encoder": encoder,
            "selected_features": inputs["selected_features"],
            "outcome_type": inputs["outcome_type"],
            "parameters": parameters,
            "best_iteration": best_iteration,
            "validation_metric_summary": _history_summary(history, metric),
        }

    def score_frame(
        self,
        packages: dict[str, Any],
        model_object_or_trial: dict[str, Any],
        inputs: dict[str, Any],
        frame: Any,
        split: str,
        encoder: dict[str, Any] | None,
        best_iteration: Any,
    ) -> Any:
        pd = packages["pandas"]
        np = packages["numpy"]
        estimator = model_object_or_trial["estimator"]
        encoder = encoder or model_object_or_trial["encoder"]
        x1 = _encode_frame(packages, frame, inputs["selected_features"], encoder, treatment=1)
        x0 = _encode_frame(packages, frame, inputs["selected_features"], encoder, treatment=0)
        kwargs = {"num_iteration": best_iteration} if best_iteration else {}
        if inputs["outcome_type"] == "binary":
            prediction_t1 = estimator.predict_proba(x1, **kwargs)[:, 1]
            prediction_t0 = estimator.predict_proba(x0, **kwargs)[:, 1]
        else:
            prediction_t1 = estimator.predict(x1, **kwargs)
            prediction_t0 = estimator.predict(x0, **kwargs)
        unit_id_column = inputs.get("unit_id_column")
        row_id = frame[unit_id_column].to_numpy() if unit_id_column and unit_id_column in frame.columns else list(range(len(frame)))
        return pd.DataFrame(
            {
                "row_id": row_id,
                "split": split,
                "actual_treatment": frame[inputs["treatment_column"]].to_numpy(),
                "actual_outcome": frame[inputs["outcome_column"]].to_numpy(),
                "prediction_t1": np.asarray(prediction_t1),
                "prediction_t0": np.asarray(prediction_t0),
                "uplift_score": np.asarray(prediction_t1) - np.asarray(prediction_t0),
            }
        )

    def dump_model(self, packages: dict[str, Any], path: Path, model_object: dict[str, Any], inputs: dict[str, Any], parameters: dict[str, Any], best_iteration: Any) -> None:
        payload = {
            "model_type": self.model_type,
            "estimator": model_object["estimator"],
            "encoder": model_object["encoder"],
            "selected_features": inputs["selected_features"],
            "outcome_type": inputs["outcome_type"],
            "parameters": parameters,
            "best_iteration": best_iteration,
        }
        packages["joblib"].dump(payload, path)
        model_object["model_artifact_path"] = str(path.resolve())

    def feature_importance_frame(self, packages: dict[str, Any], model_object: dict[str, Any], encoder: dict[str, Any]) -> Any:
        booster = model_object["estimator"].booster_
        return _importance_frame(packages, booster, encoder["model_feature_order"], component=None, treatment_feature=True)


class TLearnerAdapter(LearnerAdapter):
    model_type = "t_learner"
    model_spec = {"model_type": "t_learner", "base_estimators": {"treatment_outcome": "lightgbm", "control_outcome": "lightgbm"}}
    artifact_names = {
        "joblib": "t_learner_tuned_model.v1.joblib",
        "treatment_text": "t_learner_tuned_treatment_model.v1.txt",
        "control_text": "t_learner_tuned_control_model.v1.txt",
    }

    def load_model_object(self, packages: dict[str, Any], model_path: Path | str) -> dict[str, Any]:
        model = packages["joblib"].load(model_path)
        required = {"treatment_estimator", "control_estimator", "encoder", "selected_features"}
        if not isinstance(model, dict) or not required.issubset(model):
            raise AdapterInputError("model artifact is not a supported T-Learner model object.")
        return model

    def fit_encoder(self, packages: dict[str, Any], train_frame: Any, selected_features: list[str]) -> dict[str, Any]:
        return _fit_encoder(packages, train_frame, selected_features, include_treatment=False)

    def fit_trial(self, packages: dict[str, Any], inputs: dict[str, Any], frames: dict[str, Any], encoder: dict[str, Any], parameters: dict[str, Any]) -> dict[str, Any]:
        _validate_t_learner_train_support(frames["train"], inputs)
        lgb = packages["lightgbm"]
        train = frames["train"]
        x_train = _encode_frame(packages, train, inputs["selected_features"], encoder)
        estimator_parameters = _estimator_parameters(parameters)
        if inputs["outcome_type"] == "binary":
            estimator_class = lgb.LGBMClassifier
            estimator_kwargs = {"objective": "binary", "metric": "binary_logloss", **estimator_parameters}
            metric = "binary_logloss"
        elif inputs["outcome_type"] == "continuous":
            estimator_class = lgb.LGBMRegressor
            estimator_kwargs = {"objective": "regression", "metric": "l2", **estimator_parameters}
            metric = "l2"
        else:
            raise AdapterInputError("outcome_type must be binary or continuous.")
        valid = frames.get("valid")
        estimators: dict[str, Any] = {}
        summaries: dict[str, Any] = {}
        for component, arm_value in (("treatment", 1), ("control", 0)):
            train_mask = train[inputs["treatment_column"]] == arm_value
            estimator = estimator_class(**estimator_kwargs)
            history: dict[str, Any] = {}
            fit_kwargs: dict[str, Any] = {}
            if valid is not None and len(valid) > 0:
                valid_mask = valid[inputs["treatment_column"]] == arm_value
                valid_ok = bool(valid_mask.sum())
                if inputs["outcome_type"] == "binary" and valid_ok:
                    valid_ok = len(valid.loc[valid_mask, inputs["outcome_column"]].dropna().unique()) >= 2
                if valid_ok:
                    x_valid = _encode_frame(packages, valid.loc[valid_mask], inputs["selected_features"], encoder)
                    callbacks = [lgb.record_evaluation(history)]
                    rounds = parameters.get("early_stopping_rounds")
                    if rounds:
                        callbacks.insert(0, lgb.early_stopping(int(rounds), verbose=False))
                    fit_kwargs["eval_set"] = [(x_valid, valid.loc[valid_mask, inputs["outcome_column"]])]
                    fit_kwargs["callbacks"] = callbacks
            estimator.fit(x_train.loc[train_mask], train.loc[train_mask, inputs["outcome_column"]], **fit_kwargs)
            estimators[component] = estimator
            summaries[component] = _history_summary(history, metric)
        best_iteration = {
            "treatment": getattr(estimators["treatment"], "best_iteration_", None) or None,
            "control": getattr(estimators["control"], "best_iteration_", None) or None,
        }
        return {
            "model_type": self.model_type,
            "treatment_estimator": estimators["treatment"],
            "control_estimator": estimators["control"],
            "encoder": encoder,
            "selected_features": inputs["selected_features"],
            "model_feature_order": encoder["model_feature_order"],
            "outcome_type": inputs["outcome_type"],
            "parameters": parameters,
            "parameter_strategy": "shared",
            "parameter_sharing": "shared_across_treatment_and_control",
            "best_iteration": best_iteration,
            "validation_metric_summary": {
                "parameter_strategy": "shared",
                "treatment": summaries["treatment"],
                "control": summaries["control"],
            },
        }

    def score_frame(
        self,
        packages: dict[str, Any],
        model_object_or_trial: dict[str, Any],
        inputs: dict[str, Any],
        frame: Any,
        split: str,
        encoder: dict[str, Any] | None,
        best_iteration: Any,
    ) -> Any:
        pd = packages["pandas"]
        np = packages["numpy"]
        encoder = encoder or model_object_or_trial["encoder"]
        x = _encode_frame(packages, frame, inputs["selected_features"], encoder)
        treatment_iteration = (best_iteration or {}).get("treatment") if isinstance(best_iteration, dict) else None
        control_iteration = (best_iteration or {}).get("control") if isinstance(best_iteration, dict) else None
        treatment_kwargs = {"num_iteration": treatment_iteration} if treatment_iteration else {}
        control_kwargs = {"num_iteration": control_iteration} if control_iteration else {}
        if inputs["outcome_type"] == "binary":
            treatment_prediction = model_object_or_trial["treatment_estimator"].predict_proba(x, **treatment_kwargs)[:, 1]
            control_prediction = model_object_or_trial["control_estimator"].predict_proba(x, **control_kwargs)[:, 1]
        else:
            treatment_prediction = model_object_or_trial["treatment_estimator"].predict(x, **treatment_kwargs)
            control_prediction = model_object_or_trial["control_estimator"].predict(x, **control_kwargs)
        payload = {
            "split": split,
            "row_index": list(range(len(frame))),
            "actual_treatment": frame[inputs["treatment_column"]].to_numpy(),
            "actual_outcome": frame[inputs["outcome_column"]].to_numpy(),
            "treatment_prediction": np.asarray(treatment_prediction),
            "control_prediction": np.asarray(control_prediction),
            "uplift_score": np.asarray(treatment_prediction) - np.asarray(control_prediction),
        }
        unit_id_column = inputs.get("unit_id_column")
        if unit_id_column and unit_id_column in frame.columns:
            payload["unit_id"] = frame[unit_id_column].to_numpy()
        return pd.DataFrame(payload)

    def dump_model(self, packages: dict[str, Any], path: Path, model_object: dict[str, Any], inputs: dict[str, Any], parameters: dict[str, Any], best_iteration: Any) -> None:
        payload = {
            "model_type": self.model_type,
            "treatment_estimator": model_object["treatment_estimator"],
            "control_estimator": model_object["control_estimator"],
            "encoder": model_object["encoder"],
            "selected_features": inputs["selected_features"],
            "model_feature_order": model_object["encoder"]["model_feature_order"],
            "outcome_type": inputs["outcome_type"],
            "parameters": parameters,
            "parameter_strategy": "shared",
            "parameter_sharing": "shared_across_treatment_and_control",
            "best_iteration": best_iteration,
        }
        packages["joblib"].dump(payload, path)
        model_object["model_artifact_path"] = str(path.resolve())
        model_object["treatment_estimator"].booster_.save_model(str(path.parent / self.artifact_names["treatment_text"]))
        model_object["control_estimator"].booster_.save_model(str(path.parent / self.artifact_names["control_text"]))

    def write_model_metadata(self, path: Path, model_object: dict[str, Any], inputs: dict[str, Any], parameters: dict[str, Any], best_iteration: Any) -> None:
        super().write_model_metadata(path, model_object, inputs, parameters, best_iteration)
        metadata = _read_json(path)
        metadata["parameter_sharing"] = "shared_across_treatment_and_control"
        model_path = Path(str(model_object.get("model_artifact_path") or ""))
        metadata["treatment_lightgbm_text_path"] = str((model_path.parent / self.artifact_names["treatment_text"]).resolve()) if model_path else None
        metadata["control_lightgbm_text_path"] = str((model_path.parent / self.artifact_names["control_text"]).resolve()) if model_path else None
        write_json(path, metadata)

    def feature_importance_frame(self, packages: dict[str, Any], model_object: dict[str, Any], encoder: dict[str, Any]) -> Any:
        pd = packages["pandas"]
        frames = [
            _importance_frame(packages, model_object["treatment_estimator"].booster_, encoder["model_feature_order"], component="treatment_outcome", treatment_feature=False),
            _importance_frame(packages, model_object["control_estimator"].booster_, encoder["model_feature_order"], component="control_outcome", treatment_feature=False),
        ]
        return pd.concat(frames, ignore_index=True).sort_values(["component", "gain", "feature_name"], ascending=[True, False, True]).reset_index(drop=True)


def get_learner_adapter(model_type: str) -> LearnerAdapter:
    if model_type == "s_learner":
        return SLearnerAdapter()
    if model_type == "t_learner":
        return TLearnerAdapter()
    raise AdapterInputError(f"Unsupported model_spec.model_type: {model_type}.")


def _fit_encoder(packages: dict[str, Any], train_frame: Any, selected_features: list[str], *, include_treatment: bool) -> dict[str, Any]:
    pd = packages["pandas"]
    categorical: dict[str, list[str]] = {}
    numeric_medians: dict[str, float] = {}
    model_feature_order = [TREATMENT_FEATURE] if include_treatment else []
    for feature in selected_features:
        series = train_frame[feature]
        numeric = pd.to_numeric(series, errors="coerce")
        numeric_like = pd.api.types.is_numeric_dtype(series) or numeric.notna().sum() >= max(1, int(series.notna().sum() * 0.8))
        if numeric_like:
            numeric_medians[feature] = float(numeric.median()) if numeric.notna().any() else 0.0
            model_feature_order.append(feature)
        else:
            categories = [str(item) for item in series.dropna().astype(str).unique()]
            categorical[feature] = categories
            model_feature_order.extend([f"{feature}={category}" for category in categories])
            model_feature_order.append(f"{feature}=__MISSING__")
            model_feature_order.append(f"{feature}=__OTHER__")
    return {
        "selected_features": selected_features,
        "categorical": categorical,
        "numeric_medians": numeric_medians,
        "model_feature_order": model_feature_order,
    }


def _encode_frame(packages: dict[str, Any], frame: Any, selected_features: list[str], encoder: dict[str, Any], treatment: Any = None) -> Any:
    pd = packages["pandas"]
    np = packages["numpy"]
    encoded = pd.DataFrame(index=frame.index)
    for feature in selected_features:
        if feature in encoder["categorical"]:
            categories = encoder["categorical"][feature]
            values = frame[feature].astype("object").where(frame[feature].notna(), "__MISSING__").astype(str)
            known = set(categories) | {"__MISSING__"}
            for category in categories:
                encoded[f"{feature}={category}"] = (values == category).astype(int)
            encoded[f"{feature}=__MISSING__"] = (values == "__MISSING__").astype(int)
            encoded[f"{feature}=__OTHER__"] = (~values.isin(known)).astype(int)
        else:
            encoded[feature] = pd.to_numeric(frame[feature], errors="coerce").fillna(encoder["numeric_medians"][feature]).astype(float)
    if TREATMENT_FEATURE in encoder["model_feature_order"]:
        values = np.asarray(treatment)
        if values.ndim == 0:
            values = np.full(len(frame), values)
        if len(values) != len(frame):
            raise AdapterInputError("Treatment length must match frame length.")
        encoded.insert(0, TREATMENT_FEATURE, values.astype(int))
    return encoded[encoder["model_feature_order"]]


def _estimator_parameters(parameters: dict[str, Any]) -> dict[str, Any]:
    payload = {key: value for key, value in parameters.items() if key != "early_stopping_rounds" and value is not None}
    payload["verbosity"] = -1
    return payload


def _validate_t_learner_train_support(train_frame: Any, inputs: dict[str, Any]) -> None:
    treatment_column = inputs["treatment_column"]
    outcome_column = inputs["outcome_column"]
    if int((train_frame[treatment_column] == 1).sum()) == 0 or int((train_frame[treatment_column] == 0).sum()) == 0:
        raise AdapterInputError("train split must contain both treatment and control arms for T-Learner.", code="INSUFFICIENT_TRAIN_ARM_SUPPORT")
    if inputs["outcome_type"] == "binary":
        for arm_value, arm_name in ((1, "treatment"), (0, "control")):
            values = train_frame.loc[train_frame[treatment_column] == arm_value, outcome_column].dropna().unique()
            if len(values) < 2:
                raise AdapterInputError(f"binary outcome train {arm_name} arm must contain both outcome classes.", code="SINGLE_CLASS_TRAIN_ARM")


def _history_summary(history: dict[str, Any], metric: str) -> dict[str, Any]:
    values = history.get("valid_0", {}).get(metric, [])
    return {
        "metric": metric if values else None,
        "iteration_count": len(values),
        "last_value": float(values[-1]) if values else None,
        "best_value": float(min(values)) if values else None,
    }


def _importance_frame(packages: dict[str, Any], booster: Any, model_feature_order: list[str], *, component: str | None, treatment_feature: bool) -> Any:
    pd = packages["pandas"]
    gain = booster.feature_importance(importance_type="gain").astype(float)
    split = booster.feature_importance(importance_type="split").astype(float)
    feature_role = ["treatment", *(["business_feature"] * (len(model_feature_order) - 1))] if treatment_feature else ["business_feature"] * len(model_feature_order)
    payload: dict[str, Any] = {
        "feature_name": model_feature_order,
        "feature_role": feature_role,
        "gain": gain,
        "split": split,
    }
    if component:
        payload["component"] = component
    result = pd.DataFrame(payload)
    result["gain_share"] = gain / gain.sum() if gain.sum() else 0.0
    result["split_share"] = split / split.sum() if split.sum() else 0.0
    result["gain_rank"] = result["gain"].rank(method="min", ascending=False).astype(int)
    result["split_rank"] = result["split"].rank(method="min", ascending=False).astype(int)
    columns = ["feature_name", "feature_role", "gain", "gain_share", "gain_rank", "split", "split_share", "split_rank"]
    if component:
        columns.insert(0, "component")
    return result.sort_values(["gain", "feature_name"], ascending=[False, True]).reset_index(drop=True)[columns]


def _runtime_versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for name in ("lightgbm", "pandas", "numpy", "scikit-learn", "joblib"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not_installed"
    return versions


def _read_json(path: Path) -> dict[str, Any]:
    import json

    return json.loads(path.read_text(encoding="utf-8-sig"))
