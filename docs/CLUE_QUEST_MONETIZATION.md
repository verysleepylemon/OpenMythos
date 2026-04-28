# Clue Quest Monetization Blueprint

## Capability

Clue Quest is a legal, puzzle-first feature that helps users decode clues from:
- public puzzle pages,
- CTF-style challenges, and
- websites where owner consent is explicitly confirmed.

It is intentionally not a penetration-testing or bypass tool.

## Feature Surface

- Decoder loops: Iteratively tries base64, hex, URL-decode, and rot13.
- Public clue extraction: Reads clues from common HTML containers.
- Safety gate: Requires consent and host allowlist before processing.
- Creator economics: Includes built-in payout estimation helper.

Implementation files:
- scripts/clue_quest.py
- scripts/clue_quest_demo.py
- examples/clue_quest_sample.html

## Product Tiers (fun + revenue)

1) Free: Decoder Arcade
- 5 clue runs per day
- Community leaderboard
- Daily public puzzle

2) Pro: Quest Pass ($9 to $19/month)
- Unlimited clue runs
- Team mode (shared boards)
- Branded result cards
- Weekly premium puzzle packs

3) Creator: Puzzle Publisher (30% to 50% revenue share)
- Upload challenge packs
- Set difficulty and rewards
- Receive payout based on active paid users solving your packs

## Revenue Mechanics

Core formula:

monthly_revenue = active_users * paid_conversion_rate * ARPPU
creator_payout = monthly_revenue * creator_revenue_share

Example:
- active users: 10,000
- paid conversion: 6%
- ARPPU: $12
- creator share: 35%

Gross monthly revenue = 10,000 * 0.06 * 12 = $7,200
Creator payout pool = $7,200 * 0.35 = $2,520
Platform net before infra = $4,680

## Fun Loops That Monetize

- Weekly mystery season: keeps retention high.
- Limited-time puzzle events: drives conversion urgency.
- Team tournaments: increases social invite growth.
- Creator spotlight packs: powers marketplace GMV.

## Launch Plan (30 days)

Week 1
- Ship Clue Quest decoder + safety gate.
- Publish 10 starter challenge pages.

Week 2
- Add leaderboard + streaks.
- Soft-launch Pro tier to early users.

Week 3
- Open creator beta (5 to 10 creators).
- Start revenue-share pilot.

Week 4
- Run first seasonal tournament.
- Convert top free users with team features.

## Guardrails

- Only decode puzzle data from allowed domains.
- Require explicit consent flag for non-public targets.
- Log target host and consent status for audits.
- Do not support credential cracking or control bypasses.
