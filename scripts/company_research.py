#!/usr/bin/env python3
"""
Company Deep Research Tool

Given a company name, performs deep research and generates structured outputs:
- Baseline (company profile)
- Strategic Directions
- Themes
- Manifold of Events
- AAR (After Action Review)

Usage:
    python scripts/company_research.py "Company Name" --output-dir ./research_output
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import anthropic


# ============================================================================
# JSON SCHEMAS (for reference in prompts)
# ============================================================================

BASELINE_SCHEMA = """
{
  "entity_identity": {
    "legal_name": "Full Registered Name",
    "industry_classification": {
      "sector": "e.g., Technology / Consumer Discretionary",
      "industry": "e.g., Software / Automobiles"
    },
    "lifecycle_stage": "Startup / Growth / Mature / Declining",
    "headquarters": "City, Country",
    "founding_date": "YYYY-MM-DD",
    "corporate_structure": "e.g., C-Corp, LLC, Holding Company"
  },
  "strategic_core": {
    "mission_statement": "The company's 'Why'",
    "vision_statement": "The company's long-term 'Where'",
    "primary_business_thesis": "The fundamental argument for why this business succeeds",
    "strategic_pillars": ["Key Initiative 1", "Key Initiative 2", "Key Initiative 3"],
    "target_customer_segments": [
      {"segment_name": "e.g., Enterprise B2B", "priority_level": "High/Medium/Low"}
    ]
  },
  "financial_architecture": {
    "fiscal_year_end": "Month-Day",
    "currency_reporting": "USD / EUR / etc.",
    "revenue_model": {
      "type": "e.g., Subscription / One-time Sales / Marketplace / Hybrid",
      "recurrence_profile": "e.g., 80% Recurring / 20% Transactional"
    },
    "revenue_streams_breakdown": [
      {"source": "Product Line A", "percentage_of_total": "XX%"}
    ],
    "unit_economics": {
      "pricing_strategy": "Premium / Low-Cost Leader / Freemium",
      "average_revenue_per_user_arpu": "Value",
      "customer_acquisition_cost_cac": "Value or Trend",
      "churn_rate": "Percentage"
    },
    "financial_health_indicators": {
      "revenue_growth_yoy": "Percentage",
      "gross_margin_profile": "High (>70%) / Medium / Low (<20%)",
      "free_cash_flow_status": "Positive / Negative / Breakeven",
      "debt_leverage": "High / Moderate / Debt-Free"
    }
  },
  "market_positioning": {
    "tier_classification": "Market Leader / Challenger / Niche Player",
    "market_share_estimate": "Percentage",
    "competitive_moat": {
      "primary_type": "Network Effects / Switching Costs / Brand / Cost Advantage",
      "strength_rating": "Strong / Moderate / Weak"
    },
    "brand_equity": {
      "sentiment_score": "Positive / Neutral / Negative",
      "global_recognition": "High / Regional / Low",
      "reputation_metrics": {
        "net_promoter_score_nps": "Value",
        "app_store_rating": "Stars (if applicable)"
      }
    },
    "competitor_landscape": {
      "primary_rivals": ["Competitor A", "Competitor B"],
      "emerging_threats": ["Startup X", "Tech Shift Y"]
    }
  },
  "operational_machinery": {
    "scale_metrics": {
      "employee_headcount": "Total Number",
      "geographic_footprint": ["Region A", "Region B"],
      "total_active_users_customers": "Number"
    },
    "production_capabilities": {
      "sourcing_model": "In-house / Outsourced / Hybrid",
      "supply_chain_dependency": "Low / Moderate / High",
      "distribution_channels": ["Direct-to-Consumer", "Retail Partners", "B2B Sales"]
    },
    "technology_stack": {
      "core_infrastructure": "e.g., AWS Cloud / On-Premise",
      "proprietary_tech": "e.g., AI Algorithms / Patent Portfolio",
      "r_and_d_focus": "Key areas of research"
    }
  },
  "leadership_and_governance": {
    "key_executives": [
      {"role": "CEO", "name": "Name", "strategic_focus": "Vision / Efficiency / Growth", "tenure": "Years"}
    ],
    "ownership_structure": {
      "type": "Public / Private / VC-Backed",
      "insider_ownership_percentage": "Value"
    },
    "cultural_philosophy": "e.g., 'Move Fast and Break Things' or 'Safety First'"
  },
  "evolutionary_path": {
    "historical_trajectory": {
      "origin_story": "Brief founding narrative",
      "key_pivot_points": ["Year: Event", "Year: Event"]
    },
    "future_roadmap": {
      "short_term_1yr": "Focus areas",
      "medium_term_3yr": "Focus areas",
      "long_term_moonshots": "Speculative goals"
    }
  },
  "risk_matrix": {
    "strengths_internal": ["List of internal assets"],
    "weaknesses_internal": ["List of internal deficits"],
    "opportunities_external": ["List of market openings"],
    "threats_external": ["List of macro/competitor risks"],
    "primary_vulnerability": "The single thing most likely to kill the company"
  }
}
"""

STRATEGIC_DIRECTION_SCHEMA = """
{
  "id": "str, // id of the direction, e.g., 'dir-001'",
  "direction": "str, // title of the direction",
  "description": "str, // detailed description of the direction",
  "drivers": ["str", "str"], // 2-3 key forces shaping direction
  "expected_outcome": "str, // what company wants to achieve with it"
}
"""

THEME_SCHEMA = """
{
  "id": "str, // id of the theme, e.g., 'theme-001'",
  "direction_relation": "str, // explanation how direction and theme are related",
  "theme": "str, // title of the theme",
  "description": "str, // detailed description of the theme",
  "drivers": ["str", "str"], // 2-3 key forces shaping theme
  "core_question": "str, // the uncertainty or decision challenge it raises"
}
"""

MANIFOLD_SCHEMA = """
{
  "events": {
    "events": [
      {
        "id": "beat-X-001",
        "event_title": "...",
        "event_description": "...",
        "origin": "Internal / External / Market / Regulatory / Technology",
        "scale": "Local / Regional / Global",
        "timing": "Immediate / Short-term / Medium-term / Long-term",
        "event_type": "Opportunity / Threat / Milestone / Crisis"
      }
    ]
  }
}
"""

AAR_SCHEMA = """
{
  "report_title": "...",
  "scenario": "...",
  "strategic_direction": "...",
  "time_horizon": "...",
  "goal": "...",
  "overall_assessment": {
    "status": "Success / Partial Success / Failure",
    "summary": "..."
  },
  "pivotal_timeline_summary": [
    {"beat": "...", "inferred_title": "...", "status": "...", "summary": "..."}
  ],
  "failure_factors": ["..."],
  "success_factors": ["..."],
  "key_insights": {"top_insight": "..."},
  "key_learnings_and_strategic_recommendations": [
    {
      "learning_title": "...",
      "finding": "...",
      "so_what": "...",
      "impact_on_goal": "...",
      "recommendations_for_the_team": {
        "do": ["..."],
        "dont": ["..."]
      }
    }
  ]
}
"""


class CompanyResearcher:
    """Deep research tool for company analysis."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable required")
        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.model = "claude-sonnet-4-20250514"

    def _call_claude(self, system: str, user: str, use_search: bool = False) -> str:
        """Call Claude API with optional web search."""
        messages = [{"role": "user", "content": user}]

        if use_search:
            # Use Claude with web search tool
            response = self.client.messages.create(
                model=self.model,
                max_tokens=8192,
                system=system,
                messages=messages,
                tools=[{"type": "web_search_20250305"}],
                tool_choice={"type": "auto"}
            )
            # Extract text from response
            text_parts = []
            for block in response.content:
                if hasattr(block, 'text'):
                    text_parts.append(block.text)
            return "\n".join(text_parts)
        else:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=8192,
                system=system,
                messages=messages
            )
            return response.content[0].text

    def _extract_json(self, text: str) -> Dict[str, Any]:
        """Extract JSON from response text."""
        # Try to find JSON block
        import re
        json_match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(1))
        # Try to find raw JSON
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(0))
        raise ValueError(f"Could not extract JSON from response: {text[:500]}...")

    def research_baseline(self, company_name: str) -> Dict[str, Any]:
        """Generate company baseline profile."""
        print(f"[1/5] Researching baseline for {company_name}...")

        system = """You are a senior business analyst specializing in corporate intelligence.
Your task is to create a comprehensive company baseline profile based on thorough research.
Use web search to gather current, accurate information about the company.
Output ONLY valid JSON matching the provided schema - no explanatory text."""

        user = f"""Research {company_name} thoroughly and create a comprehensive baseline profile.

Use web search to find:
- Company overview, founding history, headquarters
- Business model, revenue streams, financial metrics
- Market position, competitors, competitive advantages
- Leadership team, ownership structure
- Recent strategic initiatives, future roadmap
- SWOT analysis

Output the data in this exact JSON schema:
{BASELINE_SCHEMA}

Research {company_name} now and output the JSON:"""

        response = self._call_claude(system, user, use_search=True)
        return self._extract_json(response)

    def research_strategic_directions(self, company_name: str, baseline: Dict) -> List[Dict]:
        """Identify 3-5 key strategic directions."""
        print(f"[2/5] Analyzing strategic directions for {company_name}...")

        system = """You are a strategic consultant analyzing corporate direction.
Based on the company baseline and current market research, identify the key strategic directions the company is pursuing.
Output ONLY valid JSON - a list of direction objects."""

        user = f"""Given this baseline for {company_name}:
{json.dumps(baseline, indent=2)}

Research current news, announcements, and strategic moves to identify 3-5 key strategic directions.

Each direction should follow this schema:
{STRATEGIC_DIRECTION_SCHEMA}

Output a JSON array of strategic directions:"""

        response = self._call_claude(system, user, use_search=True)
        return self._extract_json(response)

    def research_themes(self, company_name: str, baseline: Dict, directions: List[Dict]) -> List[Dict]:
        """Generate themes related to each strategic direction."""
        print(f"[3/5] Generating themes for {company_name}...")

        system = """You are a strategic foresight analyst.
For each strategic direction, identify 2-3 themes (macro trends, uncertainties, or forces) that could impact it.
Output ONLY valid JSON - a list of theme objects."""

        user = f"""Given this baseline for {company_name}:
{json.dumps(baseline.get('strategic_core', {}), indent=2)}

And these strategic directions:
{json.dumps(directions, indent=2)}

Generate 2-3 themes for EACH strategic direction. Themes are macro forces, trends, or uncertainties that could impact the direction.

Each theme should follow this schema:
{THEME_SCHEMA}

Output a JSON array of all themes:"""

        response = self._call_claude(system, user, use_search=True)
        return self._extract_json(response)

    def generate_manifold(self, company_name: str, theme: Dict, direction: Dict, beat_id: int = 0) -> Dict:
        """Generate manifold of events for a theme-direction pair."""
        print(f"    Generating manifold for theme '{theme.get('theme', 'Unknown')}'...")

        system = """You are a scenario planner generating potential future events.
For a given theme and strategic direction, generate 5-10 plausible events that could occur.
Events should vary in origin, scale, timing, and type.
Output ONLY valid JSON matching the schema."""

        user = f"""Company: {company_name}

Strategic Direction:
{json.dumps(direction, indent=2)}

Theme:
{json.dumps(theme, indent=2)}

Generate 5-10 plausible future events that could impact this direction given this theme.
Use beat ID prefix: beat-{beat_id}-

Schema:
{MANIFOLD_SCHEMA}

Output JSON:"""

        response = self._call_claude(system, user, use_search=False)
        return self._extract_json(response)

    def generate_aar(self, company_name: str, direction: Dict, theme: Dict, manifold: Dict) -> Dict:
        """Generate After Action Review for a scenario."""
        print(f"    Generating AAR for direction '{direction.get('direction', 'Unknown')}'...")

        system = """You are a strategic analyst conducting an After Action Review.
Analyze how a company might navigate a scenario given the events in the manifold.
Provide actionable insights and recommendations.
Output ONLY valid JSON matching the schema."""

        user = f"""Company: {company_name}

Strategic Direction:
{json.dumps(direction, indent=2)}

Theme:
{json.dumps(theme, indent=2)}

Events Manifold:
{json.dumps(manifold, indent=2)}

Create an After Action Review analyzing how {company_name} might navigate these events.
Consider what could lead to success or failure, and provide strategic recommendations.

Schema:
{AAR_SCHEMA}

Output JSON:"""

        response = self._call_claude(system, user, use_search=False)
        return self._extract_json(response)

    def run_full_research(self, company_name: str, output_dir: Path) -> Dict[str, Any]:
        """Run complete research pipeline."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create company-specific subdirectory
        safe_name = "".join(c if c.isalnum() else "_" for c in company_name)
        company_dir = output_dir / safe_name
        company_dir.mkdir(exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Deep Research: {company_name}")
        print(f"Output directory: {company_dir}")
        print(f"{'='*60}\n")

        # 1. Research baseline
        baseline = self.research_baseline(company_name)
        baseline_path = company_dir / "baseline.json"
        with open(baseline_path, "w") as f:
            json.dump(baseline, f, indent=2)
        print(f"  Saved: {baseline_path}")

        # 2. Research strategic directions
        directions = self.research_strategic_directions(company_name, baseline)
        if not isinstance(directions, list):
            directions = [directions]
        directions_path = company_dir / "strategic_directions.json"
        with open(directions_path, "w") as f:
            json.dump(directions, f, indent=2)
        print(f"  Saved: {directions_path}")

        # 3. Research themes
        themes = self.research_themes(company_name, baseline, directions)
        if not isinstance(themes, list):
            themes = [themes]
        themes_path = company_dir / "themes.json"
        with open(themes_path, "w") as f:
            json.dump(themes, f, indent=2)
        print(f"  Saved: {themes_path}")

        # 4. Generate manifolds for each theme-direction pair
        print(f"\n[4/5] Generating event manifolds...")
        manifolds_dir = company_dir / "manifolds"
        manifolds_dir.mkdir(exist_ok=True)

        beat_counter = 0
        manifold_results = []
        for direction in directions:
            dir_id = direction.get("id", f"dir-{directions.index(direction)}")
            # Find themes related to this direction
            related_themes = [t for t in themes if dir_id in t.get("direction_relation", "") or
                            direction.get("direction", "") in t.get("direction_relation", "")]
            if not related_themes:
                related_themes = themes[:2]  # Fallback: use first 2 themes

            for theme in related_themes:
                theme_id = theme.get("id", f"theme-{themes.index(theme)}")
                manifold = self.generate_manifold(company_name, theme, direction, beat_counter)

                filename = f"Theme-{theme_id}-Dir-{dir_id}-Beat_{beat_counter}--Manifold.json"
                manifold_path = manifolds_dir / filename
                with open(manifold_path, "w") as f:
                    json.dump(manifold, f, indent=2)

                manifold_results.append({
                    "theme": theme,
                    "direction": direction,
                    "manifold": manifold,
                    "path": str(manifold_path)
                })
                beat_counter += 1

        print(f"  Saved {len(manifold_results)} manifolds to {manifolds_dir}")

        # 5. Generate AARs
        print(f"\n[5/5] Generating After Action Reviews...")
        aars_dir = company_dir / "aars"
        aars_dir.mkdir(exist_ok=True)

        for i, mr in enumerate(manifold_results[:3]):  # Limit to first 3 to save API calls
            aar = self.generate_aar(company_name, mr["direction"], mr["theme"], mr["manifold"])
            aar_path = aars_dir / f"AAR_{i+1}.json"
            with open(aar_path, "w") as f:
                json.dump(aar, f, indent=2)

        print(f"  Saved AARs to {aars_dir}")

        # Summary
        summary = {
            "company": company_name,
            "generated_at": datetime.now().isoformat(),
            "outputs": {
                "baseline": str(baseline_path),
                "strategic_directions": str(directions_path),
                "themes": str(themes_path),
                "manifolds_count": len(manifold_results),
                "aars_count": min(3, len(manifold_results))
            }
        }

        summary_path = company_dir / "research_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\n{'='*60}")
        print(f"Research complete!")
        print(f"Summary: {summary_path}")
        print(f"{'='*60}\n")

        return summary


def main():
    parser = argparse.ArgumentParser(description="Company Deep Research Tool")
    parser.add_argument("company", help="Name of the company to research")
    parser.add_argument("--output-dir", "-o", default="./research_output",
                       help="Output directory for research results")
    parser.add_argument("--api-key", help="Anthropic API key (or set ANTHROPIC_API_KEY env var)")

    args = parser.parse_args()

    researcher = CompanyResearcher(api_key=args.api_key)
    researcher.run_full_research(args.company, Path(args.output_dir))


if __name__ == "__main__":
    main()
