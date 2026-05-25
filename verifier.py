import os
import re
import json
import logging
from typing import Optional, List
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ─── Known satire / parody domains ───────────────────────────────────────────
SATIRE_DOMAINS = {
    "theonion.com", "clickhole.com", "babylonbee.com", "waterfordwhispersnews.com",
    "newsthump.com", "thedailymash.co.uk", "reductress.com", "gomerblog.com",
    "worldnewsdailyreport.com", "empirenews.net", "nationalreport.net",
    "satirewire.com", "fauxy.com", "thefauxy.com",
}

# ─── Sensationalism signal words / patterns ───────────────────────────────────
SENSATIONAL_PATTERNS = [
    r"\b(SHOCKING|BOMBSHELL|EXPLOSIVE|BREAKING|ALERT|EXPOSED|LEAKED|UNBELIEVABLE"
    r"|MIND-BLOWING|OUTRAGEOUS|SCANDAL|DESTROYED|OBLITERATED|SLAM|BLASTS"
    r"|DEVASTATED|TERRIFYING|HORRIFIC|INSANE|INSANE|EPIC FAIL|EPIC WIN"
    r"|YOU WON'T BELIEVE|MUST SEE|URGENT|CRITICAL|DANGEROUS)\b",
    r"!!!+",
    r"\b[A-Z]{5,}\b",           # long all-caps words
    r"^\s*[A-Z\s!?]{20,}",      # all-caps headline
]


class Verifier:
    def __init__(self):
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError("GROQ_API_KEY is not set in .env file.")
        self.groq_client = Groq(api_key=api_key)
        self.session = requests.Session()

        self.trusted_sources = [
            "thehindu.com", "ndtv.com", "indiatoday.in", "indianexpress.com",
            "hindustantimes.com", "timesofindia.indiatimes.com", "news18.com",
            "altnews.in", "boomlive.in", "pib.gov.in", "vishvasnews.com",
            "reuters.com", "apnews.com", "bbc.com", "snopes.com", "politifact.com",
            "factcheck.org", "theprint.in", "scroll.in", "thewire.in",
            "theconversation.com", "nature.com", "who.int", "pib.gov.in",
        ]

        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }

    # ═══════════════════════════════════════════════════════════════════════════
    # UTILITIES
    # ═══════════════════════════════════════════════════════════════════════════

    def is_trusted(self, url: str) -> bool:
        return any(src in url.lower() for src in self.trusted_sources)

    def is_satire_domain(self, url: str) -> bool:
        return any(domain in url.lower() for domain in SATIRE_DOMAINS)

    def validate_claim(self, claim: str) -> Optional[str]:
        stripped = claim.strip() if claim else ""
        if not stripped:
            return "Claim cannot be empty."
        if len(stripped) < 10:
            return "Claim is too short to fact-check meaningfully."
        if len(stripped) > 1000:
            return "Claim is too long. Please shorten it to under 1000 characters."
        return None

    def score_sensationalism(self, text: str) -> dict:
        """
        Returns a score 0-10 and a human-readable note explaining the score.
        Pure heuristic pass — the LLM also scores it independently.
        """
        hits = []
        for pattern in SENSATIONAL_PATTERNS:
            matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
            if matches:
                hits.extend(matches[:3])    # cap per-pattern contribution

        raw_score = min(len(hits) * 2, 10)

        if raw_score == 0:
            note = "Language appears neutral and measured."
        elif raw_score <= 3:
            note = f"Mildly sensational — detected: {', '.join(set(str(h) for h in hits[:3]))}."
        elif raw_score <= 6:
            note = f"Moderately sensational — emotionally charged language: {', '.join(set(str(h) for h in hits[:4]))}."
        else:
            note = f"Highly sensational — aggressive/manipulative language detected: {', '.join(set(str(h) for h in hits[:5]))}."

        return {"score": raw_score, "note": note, "triggers": list(set(str(h) for h in hits))}

    # ═══════════════════════════════════════════════════════════════════════════
    # SCRAPING
    # ═══════════════════════════════════════════════════════════════════════════

    def scrape_article(self, url: str) -> Optional[str]:
        """Scrape article text, capped at 3000 chars."""
        try:
            response = self.session.get(url, headers=self.headers, timeout=7)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
                tag.decompose()
            text = " ".join(p.get_text(strip=True) for p in soup.find_all("p"))
            text = re.sub(r"\s+", " ", text).strip()
            return text[:3000] if len(text) > 150 else None
        except requests.exceptions.Timeout:
            logger.warning(f"Timeout scraping: {url}")
        except requests.exceptions.HTTPError as e:
            logger.warning(f"HTTP {e.response.status_code} scraping: {url}")
        except Exception as e:
            logger.warning(f"Scrape failed for {url}: {e}")
        return None

    # ═══════════════════════════════════════════════════════════════════════════
    # EVIDENCE FETCHING
    # ═══════════════════════════════════════════════════════════════════════════

    def fetch_evidence(self, query: str) -> List[dict]:
        """
        Fetch live news evidence via Google News RSS.
        Resolves redirects → real URLs, scrapes content, sorts trusted first.
        Also flags satire domains during collection.
        """
        evidence = []
        logger.info(f"🕵️  Evidence search: {query}")

        try:
            formatted_query = query.replace(" ", "+")
            rss_url = (
                f"https://news.google.com/rss/search"
                f"?q={formatted_query}&hl=en-IN&gl=IN&ceid=IN:en"
            )
            response = self.session.get(rss_url, headers=self.headers, timeout=6)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "lxml-xml")
            items = soup.find_all("item")
            logger.info(f"   → {len(items)} RSS items found.")

            scraped_count = 0
            for item in items[:12]:
                if scraped_count >= 5:
                    break

                title = (item.find("title") or {}).get_text(strip=True) if item.find("title") else "No Title"
                raw_url = item.find("link").get_text(strip=True) if item.find("link") else None
                if not raw_url:
                    continue

                # Pull publisher hint from <source> RSS tag if present
                source_tag = item.find("source")
                source_hint = source_tag.get_text(strip=True) if source_tag else ""

                # Resolve Google News redirect
                try:
                    resolved = self.session.get(raw_url, headers=self.headers, timeout=6, allow_redirects=True)
                    real_url = resolved.url
                except Exception as e:
                    logger.warning(f"Redirect failed for '{title}': {e}")
                    # Still try to detect trusted from raw URL or source hint
                    fallback_trusted = self.is_trusted(raw_url) or self.is_trusted(source_hint)
                    evidence.append({"url": raw_url, "title": title,
                                     "text": f"[Headline only — redirect failed]: {title}",
                                     "trusted": fallback_trusted, "is_satire": False})
                    continue

                # Trust check: resolved URL first, then raw URL, then RSS <source> tag
                # This handles cases where CDN/AMP URLs hide the real domain
                trusted = (
                    self.is_trusted(real_url)
                    or self.is_trusted(raw_url)
                    or self.is_trusted(source_hint)
                )
                is_satire = self.is_satire_domain(real_url)
                label     = "🎭 SATIRE" if is_satire else ("✅ TRUSTED" if trusted else "🔵 Unknown")
                logger.info(f"   {label} | {real_url[:90]}")

                content = self.scrape_article(real_url)
                evidence.append({
                    "url":       real_url,
                    "title":     title,
                    "text":      content or f"[Headline only]: {title}",
                    "trusted":   trusted,
                    "is_satire": is_satire,
                })
                if content:
                    scraped_count += 1

            evidence.sort(key=lambda x: (x["trusted"], not x["is_satire"]), reverse=True)
            logger.info(f"   ✅ {len(evidence)} sources ready ({scraped_count} scraped)\n")

        except Exception as e:
            logger.error(f"🔴 Evidence fetch error: {e}")

        return evidence

    # ═══════════════════════════════════════════════════════════════════════════
    # CLAIM VERIFICATION
    # ═══════════════════════════════════════════════════════════════════════════

    def verify_claim(self, claim: str) -> dict:

        # ── Validate input ────────────────────────────────────────────────────
        error = self.validate_claim(claim)
        if error:
            return self._error_response(error)

        # ── Pre-LLM heuristic pass ────────────────────────────────────────────
        heuristic_sensationalism = self.score_sensationalism(claim)

        # ── Fetch live evidence ───────────────────────────────────────────────
        evidence_data = self.fetch_evidence(claim)
        if not evidence_data:
            logger.warning("No evidence found — AI will use internal knowledge only.")

        satire_urls = [e["url"] for e in evidence_data if e.get("is_satire")]

        # ── Build evidence block for prompt ───────────────────────────────────
        evidence_lines = []
        for e in evidence_data:
            if e.get("is_satire"):
                tag = "[⚠ SATIRE/PARODY SITE — do NOT treat as factual evidence]"
            elif e["trusted"]:
                tag = "[TRUSTED SOURCE]"
            else:
                tag = "[UNKNOWN SOURCE]"
            evidence_lines.append(
                f"{tag} | {e['title']}\nURL: {e['url']}\nContent: {e['text']}"
            )

        evidence_block = (
            "\n\n---\n\n".join(evidence_lines)
            if evidence_lines
            else "No live evidence retrieved. Use your internal knowledge only."
        )

        # ── System prompt ─────────────────────────────────────────────────────
        system_prompt = """You are an elite, deeply skeptical fact-checking AI. Your task: rigorously verify claims using live evidence AND internal knowledge. Be direct. Be precise.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CORE VERDICT RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. RECENCY FIRST — THIS RULE OVERRIDES ALL OTHERS FOR POLITICAL/CURRENT EVENTS CLAIMS
   Your training data has a hard cutoff. Elections happen. Leaders change. Governments fall.
   For ANY claim about: Chief Ministers, Prime Ministers, Presidents, election results,
   appointments, political leadership, government positions, recent deaths, or current events —
   you MUST treat your training data as potentially outdated.

   THE HARD RULE:
   - If 2 or more live sources (trusted OR unknown) consistently report the same political fact,
     treat it as LIKELY TRUE and return REAL or UNCLEAR. NEVER return FAKE.
   - If evidence exists but you are uncertain: return UNCLEAR with confidence ≤ 60.
   - "I knew X from training" is NOT a reason to mark FAKE when live sources say otherwise.
   - NEVER say "as of my knowledge cutoff" and then mark FAKE. That is a contradiction.
     If you are citing your cutoff, you are admitting uncertainty → verdict must be UNCLEAR.

2. ABSOLUTE FACT OVERRIDE — ONLY FOR TIMELESS, UNCHANGEABLE FACTS
   Apply ONLY to facts that CANNOT change: geography (countries, capitals), physics, mathematics,
   historical events from before 2020.
   Examples where Rule 2 applies: "Paris is in Germany", "the Earth is flat", "water is H3O".
   Examples where Rule 2 DOES NOT apply: who the CM/PM is, election results, current policies,
   recent appointments, recent deaths, any political claim whatsoever.
   If you find yourself applying Rule 2 to a political claim — STOP. Apply Rule 1 instead.

3. KEYWORD TRAP RULE
   Words from the claim appearing in headlines does NOT mean the claim is true. If evidence is thematically unrelated, discard it.

4. TRUSTED SOURCE WEIGHTING
   [TRUSTED SOURCE] evidence (fact-checkers, major agencies) outweighs [UNKNOWN SOURCE]. A trusted source that explicitly confirms or debunks a claim is near-decisive.

5. SATIRE DETECTION
   If the claim originates from or matches content on a [⚠ SATIRE/PARODY SITE], set verdict to SATIRE. Never treat satire content as factual evidence.

6. MISLEADING vs FAKE
   MISLEADING: core fact is real but framing, context, or implication distorts meaning (cherry-picked stats, out-of-date context, false implication).
   FAKE: the claim is factually false.

7. BURDEN OF PROOF FOR MAJOR EVENTS
   Elections, deaths, wars, disasters — require strong, direct evidence. Tangential evidence → UNCLEAR or MISLEADING, not FAKE.

8. UNCLEAR THRESHOLD
   Use UNCLEAR only when evidence is genuinely absent and internal knowledge cannot resolve it. Do not use it to dodge a call.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXTENDED ANALYSIS RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

9. SENSATIONALISM SCORE (0-10)
   Evaluate the language of the claim itself (NOT its truth):
   0-2: Neutral, factual, dry language
   3-4: Slightly charged but acceptable
   5-6: Noticeably emotional, some alarm language
   7-8: Aggressive, fear-mongering, exaggerated
   9-10: Pure manipulation — all-caps, shock words, panic language
   Provide a one-sentence note explaining what drove the score.

10. POLITICAL/IDEOLOGICAL BIAS
    Identify if the claim appears to favor a specific political party, ideology, religion, or group. Note which side (left/right/religious/nationalist/etc.) and what signals you see. If neutral, set to null.

11. MISSING CONTEXT
    Even if technically true, does the claim omit critical context that would change how a reader interprets it? E.g. a real statistic from 10 years ago presented as current, or a quote missing crucial surrounding statements. If yes, explain the missing context. If no, set to null.

12. CLICKBAIT DETECTION
    Is the claim phrased to maximally provoke emotion or curiosity regardless of content? Score: "none", "mild", "moderate", "severe".

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Output ONLY valid JSON. No markdown. No preamble. Use exactly these keys:

{
  "verdict": "REAL" | "FAKE" | "MISLEADING" | "UNCLEAR" | "SATIRE",
  "confidence": <integer 0-100>,
  "explanation": "<detailed multi-sentence reasoning: which sources used, which rejected and why, what the true facts are>",
  "sensationalism_score": <integer 0-10>,
  "sensationalism_note": "<one sentence>",
  "bias_indicator": "<string describing detected bias, or null>",
  "missing_context": "<string describing missing context, or null>",
  "clickbait_level": "none" | "mild" | "moderate" | "severe",
  "key_facts": ["<short factual bullet>", ...]
}

key_facts: 2-4 short bullets — the most important verified facts relevant to this claim (true or false). Think of these as the "TL;DR facts" a reader needs.
"""

        user_prompt = (
            f"CLAIM TO VERIFY:\n{claim}\n\n"
            f"LIVE EVIDENCE ({len(evidence_data)} sources):\n\n"
            + evidence_block
        )

        # ── LLM call ──────────────────────────────────────────────────────────
        try:
            completion = self.groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.3,   # 0.2 was too rigid — model over-anchored on training priors
                max_tokens=1500,
            )

            raw    = completion.choices[0].message.content
            result = json.loads(raw)

            # Validate required keys
            required = ("verdict", "confidence", "explanation",
                        "sensationalism_score", "sensationalism_note",
                        "clickbait_level", "key_facts")
            for key in required:
                if key not in result:
                    raise ValueError(f"Missing key in AI response: {key}")

            # Normalise verdict
            result["verdict"] = result["verdict"].upper()
            if result["verdict"] not in ("REAL", "FAKE", "MISLEADING", "UNCLEAR", "SATIRE", "ERROR"):
                result["verdict"] = "UNCLEAR"

            # Merge heuristic sensationalism as a cross-check
            if heuristic_sensationalism["score"] > result["sensationalism_score"] + 2:
                result["sensationalism_note"] += (
                    f" (Heuristic scan also flagged: {', '.join(heuristic_sensationalism['triggers'][:4])})"
                )
                result["sensationalism_score"] = max(
                    result["sensationalism_score"],
                    heuristic_sensationalism["score"]
                )

            # Append metadata
            result["sources"]         = [e["url"] for e in evidence_data]
            result["trusted_sources"] = [e["url"] for e in evidence_data if e["trusted"]]
            result["satire_sources"]  = satire_urls
            result.setdefault("bias_indicator",  None)
            result.setdefault("missing_context", None)
            result.setdefault("key_facts",       [])

            return result

        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {e}")
            return self._error_response("AI returned malformed JSON. Please try again.")
        except Exception as e:
            logger.error(f"Verification error: {e}")
            return self._error_response(str(e))

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _error_response(self, message: str) -> dict:
        return {
            "verdict":             "ERROR",
            "confidence":          0,
            "explanation":         message,
            "sensationalism_score": 0,
            "sensationalism_note": "N/A",
            "bias_indicator":      None,
            "missing_context":     None,
            "clickbait_level":     "none",
            "key_facts":           [],
            "sources":             [],
            "trusted_sources":     [],
            "satire_sources":      [],
        }