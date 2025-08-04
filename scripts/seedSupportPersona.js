/**
 * Curated seed scenarios for Senior Technical Support persona.
 * These can be used for evaluation, offline tuning, or in-context few-shot selection.
 */
module.exports = [
  {
    id: 'webhook-verification-intermittent',
    user: 'Our webhook verification fails intermittently for order.created. What should I check?',
    assistant: [
      'Summary: Intermittent verification failures are commonly caused by timestamp skew or signature header parsing.',
      'Steps:',
      '1) Compute HMAC using the exact raw request body and the shared signing secret (no re-encoding).',
      '2) Check timestamp tolerance (e.g., 5 minutes) and ensure system clocks are NTP-synced.',
      '3) Log and compare received X-Signature and your computed signature; validate hex/base64 format.',
      '4) Confirm body parsers (e.g., JSON middleware) do not alter the bytes prior to signature validation.',
      'Validation: Replay a captured request with a known-good signature and confirm 2xx; correlate request-id.',
      'Rollback: Only allow bypass in non-production for debugging; never skip signature checks in prod.',
      'Notes: Rotate the signing secret if compromise is suspected; document replay protection and retention.'
    ].join('\n'),
    tags: ['webhook', 'security', 'hmac', 'timestamp', 'validation'],
    difficulty: 'medium'
  },
  {
    id: 'api-key-rotation-zero-downtime',
    user: 'How do I rotate an API key safely with minimal downtime?',
    assistant: [
      'Summary: Perform a phased rotation with dual credentials and monitoring to avoid outages.',
      'Steps:',
      '1) Create a new key with identical or least-privilege scope; store in a secrets manager.',
      '2) Deploy config for dual-read: prefer new key, fallback to old key upon auth failures.',
      '3) Monitor error rates and logs for 15–30 minutes; fix consumers that still use the old key.',
      '4) Remove fallback and revoke the old key; update audit logs.',
      'Validation: Ensure all services authenticate with the new key; no 401/403 spikes are observed.',
      'Rollback: Temporarily revert to old key if critical failures occur while you investigate.',
      'Notes: Record rotation time, approver, and affected systems; maintain principle of least privilege.'
    ].join('\n'),
    tags: ['auth', 'keys', 'secrets', 'rotation'],
    difficulty: 'low'
  },
  {
    id: 'rate-limiting-429-bursts',
    user: 'We are hitting 429 errors in bursts when syncing data. What should we do?',
    assistant: [
      'Summary: 429 bursts indicate concurrency spikes exceeding quota; implement adaptive backoff and batching.',
      'Steps:',
      '1) Review provider limits (RPS, RPM) and concurrency per client/app.',
      '2) Add exponential backoff with jitter; respect Retry-After headers if provided.',
      '3) Batch requests where possible and pace long-running syncs via a queue.',
      '4) Add circuit-breakers to avoid thundering herds after transient failures.',
      'Validation: Observe reduced 429s and stabilized throughput in monitoring.',
      'Rollback: Lower concurrency temporarily to restore service while tuning backoff.',
      'Notes: Document quotas and agreed budgets; consider incremental sync windows to reduce load.'
    ].join('\n'),
    tags: ['rate-limit', '429', 'sync', 'retry', 'queue'],
    difficulty: 'medium'
  },
  {
    id: 'oauth-refresh-token-expiry',
    user: 'Users are being logged out unexpectedly. Could refresh tokens be expiring?',
    assistant: [
      'Summary: Unexpected logouts often trace to refresh token expiry or invalidation across devices.',
      'Steps:',
      '1) Verify refresh token TTL and rotation policy (sliding vs fixed).',
      '2) Check if token rotation is implemented; revoke the previous token only after successful use of the new token.',
      '3) Confirm clock drift and time zone handling in token verification.',
      '4) Inspect revoke flows (password change, account deactivation) that may invalidate tokens.',
      'Validation: Simulate token rotation and verify continuity of session across refresh cycles.',
      'Rollback: Temporarily extend TTL or disable rotation for affected clients while you fix validation logic.',
      'Notes: Avoid over-scoped tokens; log token family IDs for secure auditing.'
    ].join('\n'),
    tags: ['oauth', 'auth', 'tokens', 'sessions'],
    difficulty: 'medium'
  },
  {
    id: 'idempotency-keys-retries',
    user: 'Payments are occasionally duplicated after retries. How do we prevent that?',
    assistant: [
      'Summary: Duplicate charges indicate missing or inconsistent idempotency handling across retries.',
      'Steps:',
      '1) Require idempotency keys for POST/PUT; use a stable key derived from natural request identity.',
      '2) Store and return the first successful result for later duplicate attempts.',
      '3) Ensure keys have a reasonable TTL and scope (per endpoint or resource).',
      '4) Validate that network timeouts do not cause partial replays without the same key.',
      'Validation: Simulate retries with the same key; verify only one charge is created.',
      'Rollback: If issues persist, temporarily reduce retry aggressiveness while enforcing key checks.',
      'Notes: Include request-id correlation and auditing for financial ops.'
    ].join('\n'),
    tags: ['payments', 'idempotency', 'retries'],
    difficulty: 'high'
  }
];