# Tenant A — Savings Account Terms (PLACEHOLDER DOC)

> This is placeholder content used to exercise the pipeline before the real
> Westpac-provided documents arrive. Replace the files in `data/tenant_a/`
> and `data/tenant_b/` with the supplied docs, then re-run `scripts/ingest_all.py`.

Tenant A's flagship product is the "Skyline Saver" account.

- Interest rate: 4.25% p.a., calculated daily, paid monthly.
- No monthly account-keeping fee if a minimum balance of $500 is maintained.
- Withdrawals: unlimited, but more than 3 withdrawals per month drops the
  bonus interest rate to 0.5% p.a. for that month.
- Minimum opening deposit: $50.
- Available to Tenant A retail customers only; not available for business
  accounts.

## Bonus interest conditions

To receive the full 4.25% p.a. bonus rate, the customer must, in the same
calendar month:

1. Deposit at least $200 from an external source.
2. Make no more than 3 withdrawals.
3. Keep the closing balance higher than the opening balance.

If any condition is not met, the base rate of 0.5% p.a. applies for that
month only, and bonus eligibility resumes the following month.
