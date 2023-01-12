from ..sklearn import SklearnExtension
from openml.flows import OpenMLFlow
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union, cast, Sized
from skactiveml.base import SingleAnnotatorPoolQueryStrategy, SkactivemlClassifier


class SkactivemlExtension(SklearnExtension):
    @classmethod
    def can_handle_flow(cls, flow: "OpenMLFlow") -> bool:
        """Check whether a given describes a scikit-learn estimator.

        This is done by parsing the ``external_version`` field.

        Parameters
        ----------
        flow : OpenMLFlow

        Returns
        -------
        bool
        """
        return cls._is_skactiveml_flow(flow)

    @classmethod
    def _is_sklearn_flow(cls, flow: OpenMLFlow) -> bool:
        if getattr(flow, "dependencies", None) is not None and "skactiveml" in flow.dependencies:
            return True
        if flow.external_version is None:
            return False
        else:
            return (
                flow.external_version.startswith("skactiveml==")
                or ",skactiveml==" in flow.external_version
            )

    @classmethod
    def can_handle_model(cls, model: Any) -> bool:
        """Check whether a model is an instance of ``sklearn.base.BaseEstimator``.

        Parameters
        ----------
        model : Any

        Returns
        -------
        bool
        """
        if isinstance(model, dict) and len(model) == 3:
            return (
                isinstance(model.get('query_strategy', None), SingleAnnotatorPoolQueryStrategy)
                and isinstance(model.get('prediction_model', None), SkactivemlClassifier)
                and isinstance(model.get('selector_model', None), SkactivemlClassifier)
            )
        return False
