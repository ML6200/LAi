#!/usr/bin/env python3
"""
Generate diverse synthetic training data for LAi
Creates varied examples without requiring external datasets
"""

import random
import argparse
from typing import List

# Hungarian vocabulary
HU_WORDS = {
    'nouns': ['kutya', 'macska', 'ház', 'város', 'ember', 'nő', 'férfi', 'gyerek', 'autó', 'vonat',
              'könyv', 'iskola', 'park', 'étterem', 'bolt', 'fa', 'virág', 'madár', 'hal', 'nap'],
    'adjectives': ['nagy', 'kicsi', 'szép', 'csúnya', 'gyors', 'lassú', 'okos', 'buta', 'boldog', 'szomorú',
                   'jó', 'rossz', 'új', 'régi', 'fiatal', 'öreg', 'meleg', 'hideg', 'világos', 'sötét'],
    'verbs': ['van', 'megy', 'jön', 'lát', 'hall', 'eszik', 'iszik', 'alszik', 'dolgozik', 'tanul',
              'olvas', 'ír', 'beszél', 'fut', 'sétál', 'gondol', 'szeret', 'utál', 'akar', 'tud'],
}

EN_WORDS = {
    'nouns': ['dog', 'cat', 'house', 'city', 'person', 'woman', 'man', 'child', 'car', 'train',
              'book', 'school', 'park', 'restaurant', 'store', 'tree', 'flower', 'bird', 'fish', 'sun'],
    'adjectives': ['big', 'small', 'beautiful', 'ugly', 'fast', 'slow', 'smart', 'dumb', 'happy', 'sad',
                   'good', 'bad', 'new', 'old', 'young', 'ancient', 'warm', 'cold', 'bright', 'dark'],
    'verbs': ['is', 'goes', 'comes', 'sees', 'hears', 'eats', 'drinks', 'sleeps', 'works', 'studies',
              'reads', 'writes', 'talks', 'runs', 'walks', 'thinks', 'loves', 'hates', 'wants', 'knows'],
}

# Template sentences
HU_SENTENCES = [
    "A {noun} {adjective}.",
    "A {adjective} {noun} {verb}.",
    "{noun} {verb} a {adjective} {noun2}ban.",
    "Ma {adjective} az idő.",
    "Szeretek {verb}ni.",
    "A {noun} nagyon {adjective}.",
    "{adjective} {noun}t látok.",
    "Ez egy {adjective} {noun}.",
]

EN_SENTENCES = [
    "The {noun} is {adjective}.",
    "The {adjective} {noun} {verb}.",
    "The {noun} {verb} in the {adjective} {noun2}.",
    "Today is {adjective}.",
    "I like to {verb}.",
    "The {noun} is very {adjective}.",
    "I see a {adjective} {noun}.",
    "This is a {adjective} {noun}.",
]

# Hungarian stories
HU_STORIES = [
    "Egyszer volt, hol nem volt, élt egy {adjective} {noun}. Ez a {noun} szeretett {verb}ni minden nap.",
    "Budapesten él egy {adjective} {noun}. Minden reggel {verb} a {noun2}ban.",
    "A {adjective} {noun} találkozott egy {adjective2} {noun2}val. Együtt {verb}tek a parkban.",
    "Volt egyszer egy {noun}, aki nagyon {adjective} volt. Minden nap {verb}t.",
]

EN_STORIES = [
    "Once upon a time, there was a {adjective} {noun}. This {noun} loved to {verb} every day.",
    "In the city lives a {adjective} {noun}. Every morning it {verb} in the {noun2}.",
    "The {adjective} {noun} met a {adjective2} {noun2}. Together they {verb} in the park.",
    "There was once a {noun} who was very {adjective}. Every day it would {verb}.",
]

# Translation pairs (expanded)
TRANSLATION_TEMPLATES = [
    ("Hello", "Szia"),
    ("Good morning", "Jó reggelt"),
    ("Good night", "Jó éjszakát"),
    ("Thank you", "Köszönöm"),
    ("Please", "Kérem"),
    ("Yes", "Igen"),
    ("No", "Nem"),
    ("How are you?", "Hogy vagy?"),
    ("I am fine", "Jól vagyok"),
    ("What is your name?", "Mi a neved?"),
    ("My name is", "A nevem"),
    ("Nice to meet you", "Örülök, hogy megismertelek"),
    ("Where is the", "Hol van a"),
    ("I would like", "Szeretnék"),
    ("Do you speak", "Beszélsz"),
]


def generate_hungarian_sentence() -> str:
    """Generate a random Hungarian sentence"""
    template = random.choice(HU_SENTENCES)
    return template.format(
        noun=random.choice(HU_WORDS['nouns']),
        noun2=random.choice(HU_WORDS['nouns']),
        adjective=random.choice(HU_WORDS['adjectives']),
        adjective2=random.choice(HU_WORDS['adjectives']),
        verb=random.choice(HU_WORDS['verbs']),
    )


def generate_english_sentence() -> str:
    """Generate a random English sentence"""
    template = random.choice(EN_SENTENCES)
    return template.format(
        noun=random.choice(EN_WORDS['nouns']),
        noun2=random.choice(EN_WORDS['nouns']),
        adjective=random.choice(EN_WORDS['adjectives']),
        adjective2=random.choice(EN_WORDS['adjectives']),
        verb=random.choice(EN_WORDS['verbs']),
    )


def generate_hungarian_story() -> str:
    """Generate a random Hungarian story"""
    template = random.choice(HU_STORIES)
    return template.format(
        noun=random.choice(HU_WORDS['nouns']),
        noun2=random.choice(HU_WORDS['nouns']),
        adjective=random.choice(HU_WORDS['adjectives']),
        adjective2=random.choice(HU_WORDS['adjectives']),
        verb=random.choice(HU_WORDS['verbs']),
    )


def generate_english_story() -> str:
    """Generate a random English story"""
    template = random.choice(EN_STORIES)
    return template.format(
        noun=random.choice(EN_WORDS['nouns']),
        noun2=random.choice(EN_WORDS['nouns']),
        adjective=random.choice(EN_WORDS['adjectives']),
        adjective2=random.choice(EN_WORDS['adjectives']),
        verb=random.choice(EN_WORDS['verbs']),
    )


def generate_translation_pair() -> tuple:
    """Generate a translation pair"""
    en, hu = random.choice(TRANSLATION_TEMPLATES)

    # Sometimes add a noun
    if random.random() < 0.5:
        en_noun = random.choice(EN_WORDS['nouns'])
        hu_noun = random.choice(HU_WORDS['nouns'])
        en = f"{en} {en_noun}"
        hu = f"{hu} {hu_noun}"

    return en, hu


def generate_instruction_example(text: str, is_hungarian: bool) -> List[str]:
    """Generate instruction-following examples"""
    examples = []

    if is_hungarian:
        templates = [
            ("Summarize the following text:", "Foglald össze a következő szöveget:", "Ez egy rövid összefoglaló."),
            ("What is the main idea?", "Mi a fő gondolat?", "A fő gondolat egyszerű."),
            ("Continue this text:", "Folytasd ezt a szöveget:", "És így tovább folyt a történet."),
            ("Translate to English:", "Fordítsd angolra:", generate_english_sentence()),
        ]
    else:
        templates = [
            ("Summarize the following text:", "Foglald össze ezt:", "This is a brief summary."),
            ("What is the main idea?", "Mi a lényeg?", "The main idea is simple."),
            ("Continue this text:", "Folytasd:", "And so the story continued."),
            ("Translate to Hungarian:", "Fordítsd magyarra:", generate_hungarian_sentence()),
        ]

    en_tmpl, hu_tmpl, response = random.choice(templates)

    # Create both English and Hungarian versions
    examples.append(f"<user>{en_tmpl}\n\n{text}</user><assistant>{response}</assistant>")
    examples.append(f"<user>{hu_tmpl}\n\n{text}</user><assistant>{response}</assistant>")

    return examples


def main():
    parser = argparse.ArgumentParser(description="Generate diverse synthetic training data")
    parser.add_argument("--output", type=str, default="data/train.txt")
    parser.add_argument("--sentences", type=int, default=5000, help="Number of simple sentences")
    parser.add_argument("--stories", type=int, default=2000, help="Number of story examples")
    parser.add_argument("--translations", type=int, default=2000, help="Number of translation pairs")
    parser.add_argument("--instructions", type=int, default=1000, help="Number of instruction examples")
    args = parser.parse_args()

    print(f"Generating synthetic training data...")
    print(f"  Sentences: {args.sentences}")
    print(f"  Stories: {args.stories}")
    print(f"  Translations: {args.translations}")
    print(f"  Instructions: {args.instructions}")

    all_data = []

    # Generate sentences
    print("\nGenerating sentences...")
    for i in range(args.sentences // 2):
        all_data.append(generate_hungarian_sentence())
        all_data.append(generate_english_sentence())
        if i % 500 == 0:
            print(f"  {i * 2} sentences generated...")

    # Generate stories
    print("\nGenerating stories...")
    for i in range(args.stories // 2):
        all_data.append(generate_hungarian_story())
        all_data.append(generate_english_story())
        if i % 200 == 0:
            print(f"  {i * 2} stories generated...")

    # Generate translation pairs
    print("\nGenerating translation pairs...")
    for i in range(args.translations):
        en, hu = generate_translation_pair()
        # EN to HU
        all_data.append(f"<system>Translate English to Hungarian.</system><user>{en}</user><assistant>{hu}</assistant>")
        # HU to EN
        all_data.append(f"<system>Translate Hungarian to English.</system><user>{hu}</user><assistant>{en}</assistant>")
        if i % 200 == 0:
            print(f"  {i * 2} translation examples generated...")

    # Generate instruction examples
    print("\nGenerating instruction examples...")
    for i in range(args.instructions // 2):
        hu_text = generate_hungarian_story()
        en_text = generate_english_story()
        all_data.extend(generate_instruction_example(hu_text, True))
        all_data.extend(generate_instruction_example(en_text, False))
        if i % 100 == 0:
            print(f"  {i * 4} instruction examples generated...")

    # Shuffle
    random.shuffle(all_data)

    # Save
    print(f"\nSaving to {args.output}...")
    with open(args.output, 'w', encoding='utf-8') as f:
        for text in all_data:
            f.write(text.strip() + '\n')

    # Stats
    unique_lines = len(set(all_data))
    total_chars = sum(len(t) for t in all_data)

    print(f"\nDataset created:")
    print(f"  Total examples: {len(all_data):,}")
    print(f"  Unique examples: {unique_lines:,} ({100*unique_lines/len(all_data):.1f}%)")
    print(f"  Total characters: {total_chars:,}")
    print(f"  Estimated tokens: ~{total_chars // 4:,}")


if __name__ == "__main__":
    main()
