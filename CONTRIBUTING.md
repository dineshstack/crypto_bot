# Contributing to Claude Crypto Bot

First off — **thank you** for considering a contribution. This is a real, running
AI trading system, and it's being built in the open so others can learn from it
and improve it. Contributors of all skill levels are welcome.

Please read this alongside the [Project Status](README.md#-project-status) in the
README — it explains, honestly, where the project stands and where help matters most.

---

## Ways to contribute

You don't need to write code to help:

- 🐛 **Report bugs** — open an [issue](https://github.com/dineshstack/crypto_bot/issues) with steps to reproduce.
- 💡 **Suggest ideas** — signals, strategy improvements, features, UX. Open an issue or a [Discussion](https://github.com/dineshstack/crypto_bot/discussions).
- 📖 **Improve docs** — setup guides, tutorials, clarifications, typo fixes.
- 🧠 **Research** — better strategy logic, ML improvements, honest backtest analysis (see the "Where help is most valuable" table in the README).
- 💻 **Code** — pick up an issue, especially those labelled `good first issue`.

If you plan a larger change, **open an issue first** so we can align before you invest time.

---

## Development setup

**Bot (Python 3.12)**

```bash
git clone https://github.com/dineshstack/crypto_bot.git
cd crypto_bot
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # fill in your own keys — never commit real keys
```

- Keep `TESTNET=true` while developing. Never test against real funds.
- MySQL is required (see the README's MySQL setup). A local instance is fine.
- The bot is controlled via Telegram; `python3 main.py` starts it, then send `/start`.

**Dashboard (Next.js 16)**

```bash
cd dashboard
npm install
cp .env.local.example .env.local   # point NEXT_PUBLIC_API_URL at your API
npm run dev
```

---

## Pull request checklist

Before opening a PR, please make sure:

- [ ] The change is focused — one logical change per PR.
- [ ] `python3 -m py_compile <files>` passes for any Python you touched.
- [ ] `npx tsc --noEmit` passes in `dashboard/` for any TypeScript you touched.
- [ ] You did **not** commit secrets, `.env` files, API keys, or large data dumps.
- [ ] New behaviour is explained in the PR description (what, why, how you tested it).
- [ ] Docs/README are updated if you changed setup, config, or behaviour.

Then:

1. Fork the repo and create a branch: `git checkout -b feature/your-thing`.
2. Commit with a clear message describing the change.
3. Push and open a Pull Request against `main`, describing the change and how you verified it.

---

## Coding style

- **Match the surrounding code.** Follow the naming, structure, and comment density already present in the file you're editing.
- **Python:** clear, readable functions; comments explain *why*, not *what*.
- **TypeScript/React:** functional components; reuse existing components and the API client in `dashboard/src/lib/`.
- Prefer small, well-named functions over clever one-liners.

---

## Safety & responsibility

This project executes trades. Contributions that touch order execution, risk
management, or position sizing get extra scrutiny — please describe your testing
in detail. When in doubt, gate risky behaviour behind a config flag and default
it to the safe option.

- Never weaken a safety check (circuit breakers, allocation caps, stop-losses) without a clear rationale and discussion.
- Anything that could cost real money (paid API calls, live orders) must be obvious and opt-in.

---

## Code of conduct

Be kind, be constructive, assume good faith. Harassment, discrimination, or
hostile behaviour won't be tolerated. We're here to build and learn together.

---

Questions? Open a Discussion or reach out via **[dineshstack.com](https://dineshstack.com)**. Happy building! 🤖
