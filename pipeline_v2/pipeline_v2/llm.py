"""Single Gemini wrapper for every stage call.

Filled in starting Step 5. Created in Step 1 only as a forward-declared
module so importers don't have to think about which step introduced it.

Design contract (when implemented):
  ``generate_structured(*, model, contents, response_model, temperature,
  max_output_tokens, system_instruction=None) -> response_model``

  - Uses ``google.genai`` SDK: ``client.models.generate_content(...)``
    with ``response_mime_type="application/json"`` and
    ``response_json_schema=response_model``.
  - Calls ``response_model.model_validate_json(response.text)`` so every
    return value is a typed Pydantic instance.
  - On ``ValidationError`` or non-JSON response: ONE retry with the
    error appended as a corrective prompt suffix. Second failure
    raises (no silent except: clauses — the Inngest retry layer
    handles backoff).
  - Logs the call duration + token counts via structlog so we can spot
    Stage-2 latency regressions quickly.
"""

# Intentionally empty for Step 1. Step 5 fills this in.
