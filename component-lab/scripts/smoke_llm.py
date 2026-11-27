"""Token-bounded live LLM smoke test with no key or answer logging."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

from real_backends.llm import LLMUnavailableError, OpenAICompatibleLLM

load_dotenv(ROOT / ".env", override=False)


def main() -> int:
    try:
        llm = OpenAICompatibleLLM.from_env()
        health = llm.health()
        result = llm.chat(
            "Reply with the single word: ready",
            system="You are a connectivity probe. Follow the user format exactly.",
            max_tokens=64,
            temperature=0.0,
        )
    except (ValueError, LLMUnavailableError) as error:
        print(json.dumps({"status": "blocked", "reason": str(error)}, sort_keys=True))
        return 1

    report = {
        "status": "ok",
        "endpoint": llm.base_url,
        "configured_model": llm.model_id,
        "health": health,
        "chat": result.summary(),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())