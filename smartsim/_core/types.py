import typing as t

# New types
StepName = t.NewType("StepName", str)
StepID = t.NewType("StepID", str)
TaskID = t.NewType("TaskID", str)

# Aliases
JobIdType = t.Union[t.Optional[StepID], TaskID]
TTelmonEntityTypeStr = t.Literal["model", "ensemble", "orchestrator"]
