# Demo Accounts & Amazon Mappings

All demo accounts share the password **`demopass`**. Login works with either the **username** or the **email**.

Each demo account is mapped to a real Amazon user id from the trained hybrid model's collaborative-filtering data, so logging in lands directly on the personalised AI assistant path (`/chat`) with CF-personalised recommendations.

> **Stability guarantee:** mappings are created once on first seed and are **never re-pointed on server restart** (only if a retrained model drops the stored Amazon id). The credentials below stay valid.

## Credentials

| # | Username | Email | Password | Amazon Mapping ID | Full Name |
|---|---|---|---|---|---|
| 1 | `demo` | demo@gmail.com | `demopass` | `AEIIRIHLIYKQGI7ZOCIJTRDF5NPQ` | Demo User 0 |
| 2 | `demo1` | demo1@gmail.com | `demopass` | `AG375WAXLZ7PIOQKIQ6KQB4J3JVQ` | Demo User 1 |
| 3 | `demo2` | demo2@gmail.com | `demopass` | `AEZ24PPURN2NCJVC6UXKZEJGWMPA` | Demo User 2 |
| 4 | `demo3` | demo3@gmail.com | `demopass` | `AG73BVBKUOH22USSFJA5ZWL7AKXA` | Demo User 3 |
| 5 | `demo4` | demo4@gmail.com | `demopass` | `AHXZR7HLPSKRUPHX35GKRLVTX3ZA` | Demo User 4 |
| 6 | `demo5` | demo5@gmail.com | `demopass` | `AGBWTG5OPNB6AP4PKUSLYL5L5EJA` | Demo User 5 |
| 7 | `demo6` | demo6@gmail.com | `demopass` | `AGAPBK7H2SJSFV4ID5GPRU233YNA` | Demo User 6 |
| 8 | `demo7` | demo7@gmail.com | `demopass` | `AEYX36LJOPGOJKN6PQN7SLNIAHRQ` | Demo User 7 |
| 9 | `demo8` | demo8@gmail.com | `demopass` | `AH6EVJXWKTEIMUC35DVPA5LHPX4Q` | Demo User 8 |
| 10 | `demo9` | demo9@gmail.com | `demopass` | `AGYPJI2W4ZRQIEJSAIQCVRG7KIRA` | Demo User 9 |
| 11 | `demo10` | demo10@gmail.com | `demopass` | `AHAVCFOTFNN35BQT3SJ3F5BRPCGQ` | Demo User 10 |
| 12 | `demo11` | demo11@gmail.com | `demopass` | `AEBNDHJJSZXWZRVW63ELWOPXPNOA` | Demo User 11 |
| 13 | `demo12` | demo12@gmail.com | `demopass` | `AFEMVUNGJXKV5HA463WSDY4I2K7Q` | Demo User 12 |
| 14 | `demo13` | demo13@gmail.com | `demopass` | `AEXZSWVQODIRLICBBKBT4J3JXCHA` | Demo User 13 |
| 15 | `demo14` | demo14@gmail.com | `demopass` | `AEWH7AG5WRWYVWVHGRCMUGTYZKTA` | Demo User 14 |
| 16 | `demo15` | demo15@gmail.com | `demopass` | `AFB6CJIA6U73F2O7PWSX7MTOI7JQ` | Demo User 15 |
| 17 | `demo16` | demo16@gmail.com | `demopass` | `AEMI74JHKARVZR5F3DVAK4FMANXQ` | Demo User 16 |
| 18 | `demo17` | demo17@gmail.com | `demopass` | `AFCWCSGZOV5ZW663QF2ZO7BTKCRA` | Demo User 17 |
| 19 | `demo18` | demo18@gmail.com | `demopass` | `AHPS4JLIWG24EDHXLXWUZSIDUE2A` | Demo User 18 |
| 20 | `demo19` | demo19@gmail.com | `demopass` | `AGR5DYILALZ7VPYCM4DQ35SF2SBQ` | Demo User 19 |

## How It Works

- **Where mappings live:** the `amazon_user_mappings` Postgres table (one row per demo user, see `backend/app/models.py`).
- **How they're picked:** on first seed, `backend/app/seed.py` calls `pick_active_users()` on the trained model — the 20 Amazon users whose CF prediction profiles are most distinctive (highest variance), giving the richest personalised demos.
- **Personalised path:** at login, `user_context.py` resolves the mapping → the model serves CF scores from that Amazon user's history, blended 70/30 with content-based query matching.
- **New registrants:** get **no** mapping → cold-start path: they rate 5 products on `/discover`, and those interactions seed their pseudo-user profile.

## Quick Reference

```
demo  / demopass          demo10 / demopass
demo1 / demopass          demo11 / demopass
demo2 / demopass          demo12 / demopass
demo3 / demopass          demo13 / demopass
demo4 / demopass          demo14 / demopass
demo5 / demopass          demo15 / demopass
demo6 / demopass          demo16 / demopass
demo7 / demopass          demo17 / demopass
demo8 / demopass          demo18 / demopass
demo9 / demopass          demo19 / demopass
```

> ⚠️ Demo credentials only — safe for local development and presentations. Do not reuse this password anywhere.
