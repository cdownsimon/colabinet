import json

import pytest

from colabinet.diagnostics import bounded_diagnostic, redact_diagnostic


@pytest.mark.parametrize("raw,secret", [
    ('Authorization: Bearer bearer-secret', 'bearer-secret'),
    ('Authorization: Basic basic-secret', 'basic-secret'),
    ('{"access_token": "access-secret", "refresh_token": "refresh-secret"}', 'access-secret'),
    ("{'id_token': 'id-secret'}", 'id-secret'),
    ('HF_TOKEN=hf-secret API_KEY=api-secret', 'hf-secret'),
    ('https://example.test/?code=code-secret&colab-runtime-proxy-token=proxy-secret', 'code-secret'),
    ('https://user:password-secret@example.test/path', 'password-secret'),
    ('Cookie: session=session-secret; other=other-secret', 'session-secret'),
    ('{"Cookie": "SID=cookie-secret; HSID=second-cookie-secret"}', 'cookie-secret'),
    ("headers={'Cookie': 'SID=cookie-secret; HSID=second-cookie-secret'}", 'second-cookie-secret'),
    ('https://example.test/?X-Goog-Signature=signed-secret&other=readable', 'signed-secret'),
    ('https://example.test/?X-Amz-Signature=aws-signed-secret', 'aws-signed-secret'),
    ('client_secret = "client-secret with spaces"', 'client-secret'),
])
def test_credentials_are_removed_from_diagnostics(raw, secret):
    assert secret not in redact_diagnostic(raw)
    assert '***' in redact_diagnostic(raw)


def test_diagnostic_keeps_error_context_and_removes_terminal_controls():
    raw = '\x1b[31mTraceback:\x1b[0m\nRuntimeError: accelerator unavailable\x00\n'
    assert redact_diagnostic(raw) == 'Traceback:\nRuntimeError: accelerator unavailable\n'


def test_log_is_bounded_after_redaction_and_keeps_the_final_error():
    raw = 'begin\n' + 'context\n' * 10000 + 'access_token=' + 'private-value-' * 6000 + '\nQuotaError: no GPU'
    output = bounded_diagnostic(raw)
    assert len(output) <= 65536
    assert 'truncated' in output
    assert output.endswith('QuotaError: no GPU')
    assert 'private-value' not in output


def test_private_keys_are_redacted_in_json_and_pem_output():
    key = '-----BEGIN PRIVATE KEY-----\nprivate-key-secret\n-----END PRIVATE KEY-----\n'
    for output in (key, json.dumps({'private_key': key}), 'private_key="private-key-secret"'):
        assert 'private-key-secret' not in bounded_diagnostic(output)


@pytest.mark.parametrize('prefix', ['', 'CLI error context\n', 'CLI error context: '])
@pytest.mark.parametrize('indent', [None, 2])
def test_credentials_inside_encoded_json_errors_are_redacted(prefix, indent):
    response = json.dumps({'access_token': 'nested-token-secret with spaces', 'error': 'quota unavailable'})
    output = prefix + json.dumps({'error': response}, indent=indent)
    result = bounded_diagnostic(output)
    assert 'nested-token-secret' not in result
    assert 'quota unavailable' in result
