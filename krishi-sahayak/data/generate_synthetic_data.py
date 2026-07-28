"""
Generates synthetic instruction-tuning data for Krishi Sahayak.

Two kinds of training examples are produced:
1. COMPLETE profile -> a grounded recommendation citing the matched scheme(s)
2. INCOMPLETE profile -> a clarifying question instead of a guess

This "ask before answering" behavior is the core fine-tuning target -- it is
what differentiates this assistant from a model that just pattern-matches to
a scheme name and hallucinates eligibility.

Usage:
    python generate_synthetic_data.py --schemes schemes.json --out train_data.jsonl --n 400
"""

import json
import random
import argparse
from pathlib import Path

random.seed(42)

STATES = ["Andhra Pradesh", "Telangana", "Punjab", "Maharashtra", "Uttar Pradesh",
          "Bihar", "Karnataka", "Tamil Nadu", "Madhya Pradesh", "West Bengal"]

CROPS = ["paddy", "cotton", "wheat", "sugarcane", "groundnut", "maize", "pulses", "chilli"]

LAND_SIZES = ["0.5 acres", "1 acre", "2 acres", "3 acres", "5 acres", "8 acres", "12 acres"]

SYSTEM_PROMPT = (
    "You are Krishi Sahayak, an assistant that helps farmers and rural field "
    "officers in India identify government schemes they are eligible for. "
    "You are given retrieved scheme documents as context. Only recommend a "
    "scheme if the farmer's profile clearly satisfies its eligibility "
    "criteria based on the provided context. If the profile is missing "
    "information needed to confirm eligibility, ask a specific clarifying "
    "question instead of guessing. Always cite the scheme name for any "
    "recommendation you make. Never invent scheme details not present in the "
    "provided context."
)


def load_schemes(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def format_context(schemes):
    """Format a subset of schemes as retrieved RAG context."""
    blocks = []
    for s in schemes:
        elig = "; ".join(f"{k}: {v}" for k, v in s["eligibility"].items())
        blocks.append(
            f"[{s['name']}]\nCategory: {s['category']} | Level: {s['level']}\n"
            f"Eligibility: {elig}\nBenefit: {s['benefit']}\n"
            f"Documents required: {', '.join(s['documents_required'])}"
        )
    return "\n\n".join(blocks)


def make_complete_profile_example(scheme, distractors):
    """A profile with enough detail to confidently match `scheme`."""
    land = random.choice(LAND_SIZES)
    state = random.choice(STATES)
    crop = random.choice(CROPS)

    profile = (
        f"I am a farmer in {state}. I own {land} of agricultural land and grow "
        f"{crop}. "
    )

    if scheme["id"] == "pm_kisan":
        profile += "No one in my family holds a government post, pays income tax, or gets a pension. What am I eligible for?"
    elif scheme["id"] == "pmfby":
        profile += f"I took a crop loan this season for {crop} cultivation. Is there insurance available for crop loss?"
    elif scheme["id"] == "kcc":
        profile += "I am 35 years old and need short-term credit for buying seeds and fertilizer. What loan options do I have?"
    elif scheme["id"] == "soil_health_card":
        profile += "I want to know the right fertilizer dosage for my land. Is there a free scheme for this?"
    elif scheme["id"] == "pmksy":
        profile += "I want to install drip irrigation but the equipment cost is high. Is there any subsidy?"
    else:
        profile += f"Am I eligible for the {scheme['name']} scheme?"

    context = format_context([scheme] + distractors)
    answer = (
        f"Based on your profile, you appear eligible for {scheme['name']}. "
        f"{scheme['benefit']} "
        f"To apply, you will need: {', '.join(scheme['documents_required'])}. "
        f"{scheme['application_process']}"
    )

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nFarmer query: {profile}"},
            {"role": "assistant", "content": answer},
        ]
    }


def make_incomplete_profile_example(scheme, distractors):
    """A profile missing a key eligibility field -- model should ask, not guess."""
    vague_profiles = [
        "I am a farmer and I heard there is some government scheme for people like me. What can I get?",
        "I want money for my farm. What schemes are there?",
        "Can you tell me about farmer schemes in my area?",
        "I need financial help for farming, what is available?",
    ]
    profile = random.choice(vague_profiles)
    context = format_context([scheme] + distractors)

    key_field = list(scheme["eligibility"].keys())[0]
    clarifying_question = (
        f"I'd like to help you find the right scheme, but I need a bit more information first. "
        f"Could you tell me: (1) which state you farm in, (2) how much land you own or lease, "
        f"and (3) what type of farmer you are (e.g. small/marginal/tenant)? "
        f"For example, {scheme['name']} depends on your {key_field.replace('_', ' ')}, "
        f"so this detail matters for giving you an accurate answer."
    )

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nFarmer query: {profile}"},
            {"role": "assistant", "content": clarifying_question},
        ]
    }


def make_ineligible_example(scheme, distractors):
    """A profile that fails an exclusion criterion -- model should correctly decline."""
    context = format_context([scheme] + distractors)
    if scheme["id"] == "pm_kisan":
        profile = (
            "I am a farmer in Punjab with 2 acres of land, but my son is a government "
            "schoolteacher and pays income tax. Am I eligible for PM-KISAN?"
        )
        answer = (
            "Based on the eligibility criteria, you would not be eligible for PM-KISAN. "
            "The scheme excludes families where any member is an income-tax payee in the "
            "last assessment year, and your son's status as a taxpayer would disqualify "
            "your household. You may want to check other schemes such as PMFBY or KCC, "
            "which do not have this exclusion."
        )
    else:
        profile = f"I don't currently meet the land ownership requirement, but I'd still like to apply for {scheme['name']}. Can I?"
        answer = (
            f"Based on the provided eligibility criteria for {scheme['name']}, land "
            f"ownership or a valid lease/tenancy document is required. Without meeting "
            f"this, you would not currently qualify. If you have a tenancy agreement or "
            f"sharecropping certificate, that may satisfy the requirement in some cases -- "
            f"please check with your local agriculture office to confirm."
        )

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nFarmer query: {profile}"},
            {"role": "assistant", "content": answer},
        ]
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--schemes", type=str, default="schemes.json")
    parser.add_argument("--out", type=str, default="train_data.jsonl")
    parser.add_argument("--n", type=int, default=400)
    args = parser.parse_args()

    schemes = load_schemes(args.schemes)
    examples = []

    generators = [make_complete_profile_example, make_incomplete_profile_example, make_ineligible_example]
    weights = [0.5, 0.3, 0.2]  # bias toward the core "correct recommendation" behavior

    for _ in range(args.n):
        scheme = random.choice(schemes)
        distractors = random.sample([s for s in schemes if s["id"] != scheme["id"]], k=2)
        gen = random.choices(generators, weights=weights, k=1)[0]
        examples.append(gen(scheme, distractors))

    out_path = Path(args.out)
    with open(out_path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Wrote {len(examples)} examples to {out_path}")


if __name__ == "__main__":
    main()
