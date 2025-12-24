# Daily Reinsurance Expert Agent

A standalone Python application that generates and emails a daily digest for learning reinsurance. Runs automatically via GitHub Actions at 07:00 UTC, with persistent memory across days using a versioned `state.json`.

## Features

- **Daily LLM-generated digest** focused on Specialty Lines (Marine, Energy, Terror, P&C)
- **Web research** from reinsurancene.ws and artemis.bm via Tavily API
- **Progressive learning** with 10 levels of increasing complexity
- **Persistent memory** to avoid repeating topics and track your progress
- **Self-assessment** mechanism to advance levels and provide feedback

## Digest Sections

Each daily email includes:

1. **Market News** — Latest reinsurance news with a Renewals focus (1.1, 1.4, 1.7 cycles)
2. **Technical Concept** — Scaffolded learning that increases with your level
3. **Data/SaaS Requirements** — API schemas, JSON payloads, entity definitions, data cleaning tips
4. **AI Use Cases in Reinsurance** — Practical applications of AI/ML in the industry

---

## Quick Start (Local)

### 1. Clone and set up environment

```bash
git clone https://github.com/YOUR_USERNAME/daily-nugget.git
cd daily-nugget

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Create `.env` file

```bash
cp .env.example .env
# Edit .env with your API keys
```

Required environment variables:

```env
# Mistral AI (required)
MISTRAL_API_KEY=your-mistral-api-key

# Tavily for web research (required)
TAVILY_API_KEY=tvly-...

# Email provider: "resend" (default) or "sendgrid"
EMAIL_PROVIDER=resend

# Resend (if EMAIL_PROVIDER=resend)
RESEND_API_KEY=re_...

# SendGrid (if EMAIL_PROVIDER=sendgrid)
SENDGRID_API_KEY=SG....

# Email addresses
EMAIL_FROM=digest@yourdomain.com
EMAIL_TO=you@example.com

# Optional: GitHub repo for self-assessment link (auto-detected in Actions)
GITHUB_REPOSITORY=YOUR_USERNAME/daily-nugget
```

### 3. Run locally

```bash
python main.py
```

---

## GitHub Actions Setup

### 1. Add Repository Secrets

Go to **Settings → Secrets and variables → Actions** and add:

| Secret | Description |
|--------|-------------|
| `MISTRAL_API_KEY` | Your Mistral AI API key |
| `TAVILY_API_KEY` | Your Tavily API key |
| `EMAIL_PROVIDER` | `resend` or `sendgrid` |
| `RESEND_API_KEY` | Resend API key (if using Resend) |
| `SENDGRID_API_KEY` | SendGrid API key (if using SendGrid) |
| `EMAIL_FROM` | Sender email address |
| `EMAIL_TO` | Recipient email address |

### 2. Enable Actions

The workflow runs automatically at 07:00 UTC daily. You can also trigger it manually:

1. Go to **Actions** tab
2. Select **Daily Reinsurance Digest**
3. Click **Run workflow**

---

## How State Memory Works

The `state.json` file persists your learning progress:

```json
{
  "schema_version": 1,
  "level": 1,
  "user_feedback": "",
  "covered_topics": ["treaty_basics", "facultative_vs_treaty"],
  "seen_items": {
    "abc123": {
      "source": "artemis.bm",
      "url": "https://...",
      "title": "Article Title",
      "first_seen_utc": "2024-01-15T07:00:00Z"
    }
  },
  "last_run_utc": "2024-01-15T07:00:00Z"
}
```

| Field | Purpose |
|-------|---------|
| `level` | Your current learning level (1-10). Higher = more technical content |
| `user_feedback` | Free-form feedback to adjust tone/complexity |
| `covered_topics` | Topics already explained (won't repeat) |
| `seen_items` | News articles already shown (deduplication) |
| `last_run_utc` | Timestamp of last successful run |

---

## Self-Assessment (Level Progression)

Each email includes a **Self-Assessment Link** that opens `state.json` in the GitHub web editor.

### To advance your level:

1. Click the self-assessment link in your email
2. Edit `state.json` directly on GitHub:
   - Increment `level` (max 10)
   - Optionally update `user_feedback` with preferences like:
     - `"more technical please"`
     - `"include more examples"`
     - `"too complex, simplify"`
3. Commit the changes
4. The next run will pick up your updated preferences

### Level Guide

| Level | Content Style |
|-------|---------------|
| 1-3 | Foundational concepts, minimal jargon, many examples |
| 4-6 | Intermediate: industry terminology, deeper technical concepts |
| 7-9 | Advanced: complex structures, quantitative methods, edge cases |
| 10 | Expert: cutting-edge topics, research-level depth |

---

## Cost Optimization

This agent is designed for minimal API costs:

- **1 Mistral API call per day** using JSON mode
- **Hard caps** on research context (max 6 articles, truncated summaries)
- **Deduplication** prevents wasted tokens on repeated content
- **Estimated cost**: ~$0.001-0.01/day with mistral-small-latest

---

## Security Notes

- **Never commit API keys** — use `.env` locally, GitHub Secrets in CI
- `.env` is in `.gitignore` by default
- The workflow only has `contents: write` permission (minimal scope)

---

## License

MIT

