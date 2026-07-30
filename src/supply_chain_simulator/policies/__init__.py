"""The decision-makers being compared: the heuristic, the LLM agent, fallback.

Inside the project, this package holds every policy that can turn a
`DecisionObservation` into a structured action, plus the shared protocol they
all implement and the fallback chain used when a policy abstains or proposes
something invalid. In the full system, this is where the research question
actually gets answered: the same interface is implemented once transparently
(the heuristic) and once through a bounded LLM agent, so their outputs can be
compared directly. Policies never receive mutable simulation state and never
apply their own actions — they only propose.
"""
