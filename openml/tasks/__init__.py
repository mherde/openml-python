# License: BSD 3-Clause

from .task import (
    OpenMLTask,
    OpenMLSupervisedTask,
    OpenMLClassificationTask,
    OpenMLRegressionTask,
    OpenMLClusteringTask,
    OpenMLLearningCurveTask,
    OpenMLActiveClassificationTask,
    TaskType,
)
from .split import OpenMLSplit
from .functions import (
    create_task,
    get_task,
    get_tasks,
    list_tasks,
)

__all__ = [
    "OpenMLTask",
    "OpenMLSupervisedTask",
    "OpenMLClusteringTask",
    "OpenMLRegressionTask",
    "OpenMLClassificationTask",
    "OpenMLActiveClassificationTask",
    "OpenMLLearningCurveTask",
    "create_task",
    "get_task",
    "get_tasks",
    "list_tasks",
    "OpenMLSplit",
    "TaskType",
]
