from ..sklearn import SklearnExtension
from openml.flows import OpenMLFlow
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union, cast, Sized
from skactiveml.base import SingleAnnotatorPoolQueryStrategy, SkactivemlClassifier
from collections import OrderedDict


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

    def model_to_flow(self, model: Any) -> "OpenMLFlow":
        """Transform a scikit-learn model to a flow for uploading it to OpenML.

        Parameters
        ----------
        model : Any

        Returns
        -------
        OpenMLFlow
        """
        # Necessary to make pypy not complain about all the different possible return types

        # flow = self._serialize_sklearn(model['prediction_model'])
        flow = self._serialize_sklearn(model['query_strategy'])

        flow.model = OrderedDict()

        flow.model['query_strategy'] = self._serialize_sklearn(model['query_strategy'])
        flow.model['prediction_model'] = self._serialize_sklearn(model['prediction_model'])
        flow.model['selector_model'] = self._serialize_sklearn(model['selector_model'])
        return flow

    def is_estimator(self, model: Any) -> bool:
        """Check whether the given model is a scikit-learn estimator.

        This function is only required for backwards compatibility and will be removed in the
        near future.

        Parameters
        ----------
        model : Any

        Returns
        -------
        bool
        """
        o = model
        return (hasattr(o, "fit") or hasattr(o, "query")) and hasattr(o, "get_params") and hasattr(o, "set_params")
    
    def check_if_model_fitted(self, model: Any) -> bool:
        """Returns True/False denoting if the model has already been fitted/trained

        Parameters
        ----------
        model : Any

        Returns
        -------
        bool
        """
        # try:
        #     # check if model is fitted
        #     from sklearn.exceptions import NotFittedError

        #     # Creating random dummy data of arbitrary size
        #     dummy_data = np.random.uniform(size=(10, 3))
        #     # Using 'predict' instead of 'sklearn.utils.validation.check_is_fitted' for a more
        #     # robust check that works across sklearn versions and models. Internally, 'predict'
        #     # should call 'check_is_fitted' for every concerned attribute, thus offering a more
        #     # assured check than explicit calls to 'check_is_fitted'
        #     model.predict(dummy_data)
        #     # Will reach here if the model was fit on a dataset with 3 features
        #     return True
        # except NotFittedError:  # needs to be the first exception to be caught
        #     # Model is not fitted, as is required
        #     return False
        # except ValueError:
        #     # Will reach here if the model was fit on a dataset with more or less than 3 features
        #     return True
        return False

    def seed_model(self, model: Any, seed: Optional[int] = None) -> Any:
        """Set the random state of all the unseeded components of a model and return the seeded
        model.

        Required so that all seed information can be uploaded to OpenML for reproducible results.

        Models that are already seeded will maintain the seed. In this case,
        only integer seeds are allowed (An exception is raised when a RandomState was used as
        seed).

        Parameters
        ----------
        model : sklearn model
            The model to be seeded
        seed : int
            The seed to initialize the RandomState with. Unseeded subcomponents
            will be seeded with a random number from the RandomState.

        Returns
        -------
        Any
        """

        for k, v in model.items():
            model[k].model = super().seed_model(v.model, seed)
        return model
