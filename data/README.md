# Synthetic email fixture

All examples were authored for this demonstration. Sender addresses use the
reserved `example.invalid` domain. No real inbox, customer correspondence,
provider output, or model-generated labels were used to create the fixture.

The task labels an email positive when the recipient is responsible for an
outstanding request that requires a response, including polite/indirect requests
and acknowledgments. FYI messages, automated notices, explicitly optional comments,
requests assigned to someone else, and already resolved requests are negative.
Thread history resolves ownership and whether a request is still outstanding.

There are 36 train, 18 validation, and 18 test emails, with equal positive and
negative counts within each split. IDs and input texts are distinct. Scenarios
share common task patterns; this is deliberately a teaching fixture, not a
statistically independent benchmark or an estimate of inbox prevalence.

`customer_tier` alternates independently of the alternating class labels and is
only an illustrative application policy input. It is not sent to the model.
`needs_response`, `label_rationale`, `split`, and `id` are also excluded from model inputs.

Split membership, labels, and example wording were frozen before the first live
inference run. The run records the dataset's SHA-256 hash. No test-driven relabeling,
threshold selection, or dataset revision is part of the evaluation workflow.
