"""Pydantic request/response contracts for the backend API.

Inside `app`, this package defines the JSON shapes routers accept and
return. These are deliberately distinct from `supply_chain_simulator`'s own
config/domain models in `app.services` — they describe the API's public
contract, which is free to add fields (pagination, display hints) that have
no place in the research package's strict config schemas. It does not
perform validation beyond basic shape checking; domain validation stays in
`app.services`.
"""
