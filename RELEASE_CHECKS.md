# Public release checks

Checked 25 October 2026 using Python 3.11, NumPy 2.2.6, pandas 2.3.3,
PyArrow 25.0.0, PyYAML 6.0.2 and pytest 9.0.2.

All 55 tests passed. The one-command fictional example reproduced 351 metric
entries using the documented numerical tolerance; input and configuration
identities, record counts and validation outcomes were checked exactly.
The release-boundary audit passed. The wheel was rebuilt with MIT metadata.

These checks demonstrate software behaviour on fictional fixtures. They do not
independently reproduce the thesis's restricted-data experiments.
