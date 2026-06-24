# Tenant B — Savings Account Terms (PLACEHOLDER DOC)

> This is placeholder content used to exercise the pipeline before the real
> Westpac-provided documents arrive. Replace the files in `data/tenant_a/`
> and `data/tenant_b/` with the supplied docs, then re-run `scripts/ingest_all.py`.

Tenant B's flagship product is the "Horizon Saver" account.

- Interest rate: 5.10% p.a., calculated daily, paid monthly.
- No monthly account-keeping fee, regardless of balance.
- Withdrawals: bonus rate is forfeited entirely if more than 1 withdrawal is
  made in a calendar month (stricter than the Tenant A equivalent product).
- Minimum opening deposit: $0.
- Available to both retail and business customers.

## Bonus interest conditions

To receive the full 5.10% p.a. bonus rate, the customer must, in the same
calendar month:

1. Deposit at least $1,000 from an external source.
2. Make no more than 1 withdrawal.

If either condition is not met, the base rate of 0.10% p.a. applies for
that month.
