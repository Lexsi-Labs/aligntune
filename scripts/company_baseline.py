#!/usr/bin/env python3
"""
Company Baseline Generator

Given a company name, performs deep research and generates a structured baseline profile.

Usage:
    python scripts/company_baseline.py "Company Name" --output ./baselines/company.json
"""

import argparse
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import anthropic


BASELINE_SCHEMA = """{
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
}"""


def generate_baseline(company_name: str, api_key: Optional[str] = None) -> Dict[str, Any]:
    """Generate company baseline using Claude with web search."""
    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable required")

    client = anthropic.Anthropic(api_key=api_key)

    print(f"Researching {company_name}...")

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=8192,
        system="""You are a senior business analyst specializing in corporate intelligence.
Your task is to create a comprehensive company baseline profile based on thorough research.
Use web search to gather current, accurate information about the company.
Output ONLY valid JSON matching the provided schema - no explanatory text before or after.""",
        messages=[{
            "role": "user",
            "content": f"""Research {company_name} thoroughly and create a comprehensive baseline profile.

Use web search to find:
- Company overview, founding history, headquarters
- Business model, revenue streams, financial metrics
- Market position, competitors, competitive advantages
- Leadership team, ownership structure
- Recent strategic initiatives, future roadmap
- SWOT analysis

Output the data in this exact JSON schema:
{BASELINE_SCHEMA}

Research {company_name} now and output ONLY the JSON (no markdown, no explanation):"""
        }],
        tools=[{"type": "web_search_20250305"}],
        tool_choice={"type": "auto"}
    )

    # Extract text from response
    text_parts = []
    for block in response.content:
        if hasattr(block, 'text'):
            text_parts.append(block.text)
    text = "\n".join(text_parts)

    # Extract JSON
    json_match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
    if json_match:
        return json.loads(json_match.group(1))
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if json_match:
        return json.loads(json_match.group(0))
    raise ValueError(f"Could not extract JSON from response")


def main():
    parser = argparse.ArgumentParser(description="Company Baseline Generator")
    parser.add_argument("company", help="Name of the company to research")
    parser.add_argument("--output", "-o", help="Output JSON file path")
    parser.add_argument("--api-key", help="Anthropic API key (or set ANTHROPIC_API_KEY)")

    args = parser.parse_args()

    baseline = generate_baseline(args.company, args.api_key)

    # Add metadata
    result = {
        "metadata": {
            "company_query": args.company,
            "generated_at": datetime.now().isoformat(),
            "schema_version": "1.0"
        },
        "baseline": baseline
    }

    # Output
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Saved: {output_path}")
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
