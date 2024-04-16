import typing as t
import uuid

if t.TYPE_CHECKING:
    from smartsim.entity.entity import SmartSimEntity
    from smartsim.entity.entityList import EntitySequence
    from smartsim.settings import base as _settings_base
    from smartsim.entity import types as _entity_types

# New types
StepName = t.NewType("StepName", str)
StepID = t.NewType("StepID", str)
TaskID = t.NewType("TaskID", str)
MonitoredJobID = t.NewType("MonitoredJobID", uuid.UUID)


# Protocols
class _Stepable(t.Protocol):
    @property
    def name(self) -> "_entity_types.EntityName": ...
    @property
    def path(self) -> str: ...


class RunSettingsStepable(_Stepable, t.Protocol):
    @property
    def run_settings(self) -> "_settings_base.RunSettings": ...


class BatchSettingsStepable(_Stepable, t.Protocol):
    @property
    def batch_settings(self) -> t.Optional["_settings_base.BatchSettings"]: ...


# Aliases
JobIdType = t.Union[t.Optional[StepID], TaskID]
TTelmonEntityTypeStr = t.Literal["model", "ensemble", "orchestrator"]
