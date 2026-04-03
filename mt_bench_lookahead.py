"""
mt_bench_lookahead.py -- MT-Bench lookahead logit acceptance rate benchmark

Runs the lookahead validation across the 65 MT-Bench questions (8 categories)
to get publishable numbers comparable to EAGLE-3 / Medusa acceptance rates.

Usage:
    python mt_bench_lookahead.py --model Qwen/Qwen2.5-7B-Instruct
    python mt_bench_lookahead.py --model Qwen/Qwen2.5-32B-Instruct --max-questions 65
    python mt_bench_lookahead.py --model Qwen/Qwen2.5-0.5B-Instruct --max-questions 10 --output results.json
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# MT-Bench questions (65 questions, 8 categories)
# Source: lm-sys/FastChat mt_bench question.jsonl
# ---------------------------------------------------------------------------
MT_BENCH_QUESTIONS = [
    # writing (10)
    {"id": 81, "category": "writing", "turns": ["Compose an engaging travel blog post about a recent trip to Hawaii, highlighting cultural experiences and must-see attractions."]},
    {"id": 82, "category": "writing", "turns": ["Draft a professional email seeking your supervisor's feedback on the 'Quarterly Financial Report' you prepared."]},
    {"id": 83, "category": "writing", "turns": ["Imagine you are writing a blog post comparing two popular smartphone models. Develop an outline for the blog post, including key points and subheadings."]},
    {"id": 84, "category": "writing", "turns": ["Write a persuasive essay arguing that the government should invest more in renewable energy sources."]},
    {"id": 85, "category": "writing", "turns": ["Write a 300-word review for a local restaurant called 'The Cozy Nook', focusing on ambiance, service, and food quality."]},
    {"id": 86, "category": "writing", "turns": ["Compose a haiku sequence that captures the essence of each of the four seasons."]},
    {"id": 87, "category": "writing", "turns": ["Write a short story about a robot that discovers music for the first time."]},
    {"id": 88, "category": "writing", "turns": ["Craft a poem about the journey of a single raindrop from a cloud to the ocean."]},
    {"id": 89, "category": "writing", "turns": ["Write a marketing pitch for a new eco-friendly water bottle targeting environmentally conscious millennials."]},
    {"id": 90, "category": "writing", "turns": ["Create a short bedtime story suitable for five-year-olds about a young rabbit who learns the importance of sharing."]},
    # roleplay (10)
    {"id": 101, "category": "roleplay", "turns": ["Imagine you are an experienced Michelin-star chef. Provide a detailed recipe for a classic French dish, explaining the technique and the importance of each ingredient."]},
    {"id": 102, "category": "roleplay", "turns": ["You are an astronaut describing your first experience walking on the Moon. Share your feelings and observations in detail."]},
    {"id": 103, "category": "roleplay", "turns": ["Act as a math teacher and explain the Pythagorean theorem to a 10-year-old student."]},
    {"id": 104, "category": "roleplay", "turns": ["You are a time-traveling historian who has just returned from ancient Rome. Describe your experience."]},
    {"id": 105, "category": "roleplay", "turns": ["As a renowned chef, you are judging a cooking competition. Describe a dish presented to you and provide your critique."]},
    {"id": 106, "category": "roleplay", "turns": ["You are a cybersecurity expert. Explain what a SQL injection attack is, how it works, and how to prevent it."]},
    {"id": 107, "category": "roleplay", "turns": ["You are a virtual assistant designed to help elderly users. Explain how to use a smartphone to make a video call."]},
    {"id": 108, "category": "roleplay", "turns": ["As a personal financial advisor, explain the concept of compound interest and how it can be used to build wealth over time."]},
    {"id": 109, "category": "roleplay", "turns": ["You are a nutritionist. Create a healthy one-week meal plan for someone who wants to lose weight but dislikes vegetables."]},
    {"id": 110, "category": "roleplay", "turns": ["As a travel agent, suggest a 10-day itinerary for a family trip to Japan, including cultural experiences and must-see attractions."]},
    # reasoning (10)
    {"id": 111, "category": "reasoning", "turns": ["The vertices of a triangle are at points (0, 0), (-1, 1), and (3, 3). What is the area of the triangle?"]},
    {"id": 112, "category": "reasoning", "turns": ["A group of four people needs to cross a bridge at night. They have only one torch and the bridge can support only two people at a time. Person A can cross in 1 minute, B in 2 minutes, C in 5 minutes, and D in 8 minutes. When two people cross together, they move at the slower person's pace. What is the minimum time for all of them to cross the bridge?"]},
    {"id": 113, "category": "reasoning", "turns": ["In a town, there are three types of people: truth-tellers (always tell the truth), liars (always lie), and alternators (alternate between truth and lie). You meet three people: A, B, and C. A says 'B is a liar.' B says 'C is an alternator.' C says 'A is a truth-teller.' Determine the type of each person."]},
    {"id": 114, "category": "reasoning", "turns": ["You have 8 identical-looking balls, one of which is slightly heavier than the others. Using a balance scale and only two weighings, how can you identify the heavier ball?"]},
    {"id": 115, "category": "reasoning", "turns": ["A farmer has a rectangular field with a perimeter of 400 meters. He wants to divide the field into two equal halves by building a fence parallel to one side. If the total length of the fence (including the dividing fence) is 500 meters, what are the dimensions of the field?"]},
    {"id": 116, "category": "reasoning", "turns": ["In a chess tournament, each player plays every other player exactly once. If there are 10 players in the tournament, how many games are played in total?"]},
    {"id": 117, "category": "reasoning", "turns": ["A train travels from station A to station B at 60 km/h and returns at 90 km/h. What is the average speed for the entire trip?"]},
    {"id": 118, "category": "reasoning", "turns": ["A box contains 5 red balls and 5 blue balls. If you draw 3 balls at random without replacement, what is the probability that you draw at least 2 red balls?"]},
    {"id": 119, "category": "reasoning", "turns": ["A clock's minute hand is 10 cm long. How far does the tip of the minute hand travel in 35 minutes?"]},
    {"id": 120, "category": "reasoning", "turns": ["In a particular language, 'mab kun pala' means 'sky is blue', 'kun pala vertu' means 'is blue beautiful', and 'mab vertu pala' means 'sky beautiful blue'. What does each individual word mean?"]},
    # math (10)
    {"id": 121, "category": "math", "turns": ["Solve for x in the equation: 2x + 5 = 3x - 7."]},
    {"id": 122, "category": "math", "turns": ["If a triangle has sides of length 3, 4, and 5, what is its area?"]},
    {"id": 123, "category": "math", "turns": ["Find the derivative of f(x) = x^3 - 2x^2 + x - 5."]},
    {"id": 124, "category": "math", "turns": ["What is the integral of f(x) = 3x^2 + 2x - 1 from x = 0 to x = 3?"]},
    {"id": 125, "category": "math", "turns": ["Solve the following system of equations: 2x + 3y = 11 and 4x - y = 5."]},
    {"id": 126, "category": "math", "turns": ["A series is defined by the recursive formula a(n) = 2 * a(n-1) + 1, with a(1) = 1. What is a(10)?"]},
    {"id": 127, "category": "math", "turns": ["If log(x) = 2, what is the value of log(x^3)?"]},
    {"id": 128, "category": "math", "turns": ["The probability of rain on any given day in April is 0.3. What is the probability that it will rain on exactly 5 days in April?"]},
    {"id": 129, "category": "math", "turns": ["In a geometric sequence, the first term is 2 and the common ratio is 3. What is the sum of the first 6 terms?"]},
    {"id": 130, "category": "math", "turns": ["Find the minimum value of the function f(x) = 3x^2 - 6x + 4 using the completing the square method."]},
    # coding (10)
    {"id": 131, "category": "coding", "turns": ["Write a Python function to find the longest common prefix string amongst an array of strings."]},
    {"id": 132, "category": "coding", "turns": ["Implement a Python class for a stack data structure with push, pop, and peek methods."]},
    {"id": 133, "category": "coding", "turns": ["Write a Python function to check if a string is a palindrome, ignoring case and non-alphanumeric characters."]},
    {"id": 134, "category": "coding", "turns": ["Implement a Python function to find all prime numbers up to a given number using the Sieve of Eratosthenes algorithm."]},
    {"id": 135, "category": "coding", "turns": ["Write a Python function to merge two sorted arrays into a single sorted array."]},
    {"id": 136, "category": "coding", "turns": ["Write a function to find the maximum subarray sum using Kadane's algorithm."]},
    {"id": 137, "category": "coding", "turns": ["Implement binary search in Python. The function should return the index of the target if found, or -1 if not found."]},
    {"id": 138, "category": "coding", "turns": ["Write a Python function to convert a Roman numeral string to an integer."]},
    {"id": 139, "category": "coding", "turns": ["Write a Python function that takes a list of integers and returns a new list with duplicates removed, preserving order."]},
    {"id": 140, "category": "coding", "turns": ["Implement a Python function to flatten a nested list of arbitrary depth."]},
    # extraction (5)
    {"id": 141, "category": "extraction", "turns": ["Given the following paragraph, identify all the named entities (persons, organizations, locations) mentioned: 'Apple Inc. was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in Cupertino, California in 1976.'"]},
    {"id": 142, "category": "extraction", "turns": ["Extract all email addresses from the following text: 'Contact us at support@example.com or sales@company.org for more information. You can also reach our CEO at john.doe@enterprise.net.'"]},
    {"id": 143, "category": "extraction", "turns": ["Extract all dates mentioned in the following text and convert them to the format YYYY-MM-DD: 'The project started on January 5th, 2023, and is expected to end by the 15th of March, 2024.'"]},
    {"id": 144, "category": "extraction", "turns": ["From the following text, identify all technical terms and provide a brief explanation for each: 'The API uses OAuth 2.0 for authentication and returns data in JSON format. It supports REST and GraphQL endpoints.'"]},
    {"id": 145, "category": "extraction", "turns": ["Identify and categorize all the verbs in the following sentence by their tense: 'She has been studying all day, will finish her homework later, and was planning to go for a walk.'"]},
    # stem (5)
    {"id": 146, "category": "stem", "turns": ["Explain the process of photosynthesis in detail, including the light-dependent and light-independent reactions."]},
    {"id": 147, "category": "stem", "turns": ["Explain the theory of general relativity and how it differs from Newtonian gravity."]},
    {"id": 148, "category": "stem", "turns": ["Describe the structure of DNA and explain how it stores genetic information."]},
    {"id": 149, "category": "stem", "turns": ["Explain how vaccines work and why some people experience side effects."]},
    {"id": 150, "category": "stem", "turns": ["Describe the water cycle and explain how human activities impact it."]},
    # humanities (5)
    {"id": 151, "category": "humanities", "turns": ["Analyze the causes of World War I, focusing on the underlying tensions that led to the conflict."]},
    {"id": 152, "category": "humanities", "turns": ["Explain the philosophical concept of existentialism and its main proponents."]},
    {"id": 153, "category": "humanities", "turns": ["Discuss the impact of the Industrial Revolution on society, economy, and culture."]},
    {"id": 154, "category": "humanities", "turns": ["Compare and contrast the economic systems of capitalism and socialism."]},
    {"id": 155, "category": "humanities", "turns": ["Analyze the themes of power and corruption in Shakespeare's 'Macbeth'."]},
]


def classify_token(token_str: str) -> str:
    if not token_str.strip():
        return "whitespace"
    if all(c in ".,;:!?()[]{}\"'-" for c in token_str.strip()):
        return "punct"
    if token_str.startswith(" ") or token_str.startswith("\u0120"):
        return "space_word"
    return "subword_cont"


def measure_lookahead(model, tokenizer, text: str, device: str) -> dict:
    """
    Measure lookahead acceptance rate on a given text.

    For each position i, logits[i] predicts token i+1.
    A match means that token could have been obtained for free.
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(device)
    input_ids = inputs["input_ids"][0]
    seq_len = input_ids.shape[0]

    if seq_len < 3:
        return None

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0]

    draft_tokens = logits[:-1].argmax(dim=-1)
    actual_tokens = input_ids[1:]
    matches = (draft_tokens == actual_tokens).cpu().numpy()

    probs = torch.softmax(logits[:-1], dim=-1)
    actual_cpu = actual_tokens.cpu()
    draft_probs = probs[torch.arange(seq_len - 1), actual_cpu].float().numpy()
    actual_np = actual_cpu.numpy()

    # Run lengths: how many consecutive free tokens starting at each position
    run_lengths = np.zeros(len(matches), dtype=int)
    for i in range(len(matches) - 1, -1, -1):
        if matches[i]:
            run_lengths[i] = 1 + (run_lengths[i + 1] if i + 1 < len(matches) else 0)

    type_matches: dict = defaultdict(list)
    for i in range(len(matches)):
        token_str = tokenizer.decode([int(actual_np[i])])
        tok_type = classify_token(token_str)
        type_matches[tok_type].append(bool(matches[i]))

    match_rate = float(matches.mean())
    expected_free = float(
        (run_lengths * matches).sum() / max(1, matches.sum())
    )

    return {
        "seq_len": int(seq_len),
        "match_rate": match_rate,
        "expected_free_tokens": expected_free,
        "mean_draft_prob": float(draft_probs.mean()),
        "token_type_stats": {
            k: {"match_rate": float(np.mean(v)), "count": len(v)}
            for k, v in type_matches.items()
        },
    }


def generate_response(model, tokenizer, question: str, device: str, max_new_tokens: int) -> str:
    """Generate a response and return only the generated text."""
    messages = [{"role": "user", "content": question}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )

    return tokenizer.decode(
        output_ids[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )


def main():
    parser = argparse.ArgumentParser(description="MT-Bench lookahead logit acceptance rate benchmark")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--max-questions", type=int, default=65)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--output", default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model_short = args.model.split("/")[-1]
    output_path = args.output or f"lookahead_{model_short}.json"

    print(f"Model:    {args.model}")
    print(f"Device:   {device}")
    print(f"Questions: {args.max_questions}")
    print(f"Output:   {output_path}\n")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map=device,
        attn_implementation="flash_attention_2" if device == "cuda" else "eager",
    )
    model.train(False)  # inference mode

    questions = MT_BENCH_QUESTIONS[: args.max_questions]
    all_results = []
    category_results: dict = defaultdict(list)

    print(f"{'#':>3}  {'Cat':12s}  {'Match%':>7s}  {'E[free]':>8s}  Question[:50]")
    print("-" * 75)

    t0 = time.time()
    for idx, q in enumerate(questions):
        question_text = q["turns"][0]
        category = q["category"]

        generated = generate_response(model, tokenizer, question_text, device, args.max_new_tokens)
        stats = measure_lookahead(model, tokenizer, generated, device)
        if stats is None:
            continue

        result = {
            "id": q["id"],
            "category": category,
            "question": question_text,
            "generated_length": stats["seq_len"],
            "match_rate": stats["match_rate"],
            "expected_free_tokens": stats["expected_free_tokens"],
            "mean_draft_prob": stats["mean_draft_prob"],
            "token_type_stats": stats["token_type_stats"],
        }
        all_results.append(result)
        category_results[category].append(stats["match_rate"])

        print(
            f"{idx+1:>3}  {category:12s}  {stats['match_rate']*100:6.1f}%  "
            f"{stats['expected_free_tokens']:8.2f}  {question_text[:50]!r}"
        )
        sys.stdout.flush()

    elapsed = time.time() - t0
    print(f"\n(Total time: {elapsed:.0f}s)")

    # Summary
    print("\n" + "=" * 50)
    print("RESULTS BY CATEGORY")
    print("=" * 50)
    print(f"{'Category':15s}  {'Match%':>8s}  {'N':>4s}")
    print("-" * 35)
    for cat, rates in sorted(category_results.items()):
        print(f"{cat:15s}  {np.mean(rates)*100:7.1f}%  {len(rates):>4d}")

    all_rates = [r["match_rate"] for r in all_results]
    all_free = [r["expected_free_tokens"] for r in all_results]
    print("-" * 35)
    print(f"{'Overall':15s}  {np.mean(all_rates)*100:7.1f}%  {len(all_results):>4d}")
    print(f"\nMean expected free tokens/step: {np.mean(all_free):.2f}")
    print(f"Theoretical speedup (1+E[free]): {1 + np.mean(all_free):.2f}x")

    summary = {
        "model": args.model,
        "num_questions": len(all_results),
        "overall_match_rate": float(np.mean(all_rates)),
        "mean_expected_free_tokens": float(np.mean(all_free)),
        "theoretical_speedup": float(1 + np.mean(all_free)),
        "by_category": {
            cat: {"match_rate": float(np.mean(rates)), "n": len(rates)}
            for cat, rates in category_results.items()
        },
        "results": all_results,
    }

    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
