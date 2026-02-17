#!/usr/bin/env python3
"""
Generate training dataset in LEMA custom chat format.
This format proves the model learns structured output through fine-tuning.
"""

import json
import random
import os
from typing import List, Dict, Tuple
from dataclasses import dataclass

# Reproducibility
SEED = 42
random.seed(SEED)

SYSTEM_PROMPT = "You are a precise assistant trained using LEMA."

@dataclass
class Example:
    question: str
    answer: str
    explanation: str
    confidence: str
    category: str

def format_example(example: Example) -> str:
    """Format example in strict LEMA chat template."""
    return f"""<|system|>
{SYSTEM_PROMPT}

<|user|>
{example.question}

<|assistant|>
[LEMA_REPLY]
Answer: {example.answer}
Explanation: {example.explanation}
Confidence: {example.confidence}
[/LEMA_REPLY]"""

def generate_science_examples(n: int) -> List[Example]:
    """Generate science fact examples."""
    examples = []
    base_facts = [
        ("What is photosynthesis?", 
         "Photosynthesis is the process by which plants use sunlight to synthesize nutrients from carbon dioxide and water.",
         "This biological process converts light energy into chemical energy stored in glucose molecules.",
         "High"),
        ("What is the speed of light?",
         "The speed of light in a vacuum is approximately 299,792,458 meters per second.",
         "It is a universal physical constant important in many areas of physics.",
         "High"),
        ("What is an atom?",
         "An atom is the smallest unit of ordinary matter that forms a chemical element.",
         "Every solid, liquid, gas, and plasma is composed of neutral or ionized atoms.",
         "High"),
        ("What is DNA?",
         "Deoxyribonucleic acid (DNA) is a molecule composed of two polynucleotide chains that coil around each other to form a double helix.",
         "It carries genetic instructions for the development, functioning, growth and reproduction of all known organisms.",
         "High"),
        ("What is gravity?",
         "Gravity is a natural phenomenon by which all things with mass or energy are brought toward one another.",
         "On Earth, gravity gives weight to physical objects, and the Moon's gravity causes the ocean tides.",
         "High"),
         ("What is a black hole?",
         "A black hole is a region of spacetime where gravity is so strong that nothing—no particles or even electromagnetic radiation such as light—can escape from it.",
         "The theory of general relativity predicts that a sufficiently compact mass can deform spacetime to form a black hole.",
         "High"),
         ("What is evolution?",
         "Evolution is the change in the heritable characteristics of biological populations over successive generations.",
         "These characteristics are the expressions of genes that are passed on from parent to offspring during reproduction.",
         "High"),
         ("What is the Big Bang?",
         "The Big Bang theory is the prevailing cosmological model for the observable universe from the earliest known periods through its subsequent large-scale evolution.",
         "The model describes how the universe expanded from an initial state of high density and temperature.",
         "Medium"),
         ("What is quantum mechanics?",
         "Quantum mechanics is a fundamental theory in physics that provides a description of the physical properties of nature at the scale of atoms and subatomic particles.",
         "It is the foundation of all quantum physics including quantum chemistry, quantum field theory, quantum technology, and quantum information science.",
         "High"),
         ("What is a chemical reaction?",
         "A chemical reaction is a process that leads to the chemical transformation of one set of chemical substances to another.",
         "Classically, chemical reactions encompass changes that only involve the positions of electrons in the forming and breaking of chemical bonds between atoms.",
         "High")
    ]
    
    # Generate variations to reach n
    for i in range(n):
        base = base_facts[i % len(base_facts)]
        # Add slight variation to question to avoid exact duplicates if needed, 
        # but for this stress test, repeating high-quality data is acceptable 
        # to ensure the model learns the *format* and *content* well.
        # We will cycle through base facts.
        examples.append(Example(
            question=base[0],
            answer=base[1],
            explanation=base[2],
            confidence=base[3],
            category="science"
        ))
    return examples

def generate_history_examples(n: int) -> List[Example]:
    """Generate historical fact examples."""
    examples = []
    base_facts = [
        ("Who invented the telephone?",
         "Alexander Graham Bell is credited with inventing the telephone in 1876.",
         "While others worked on similar devices, Bell received the first patent for the telephone.",
         "High"),
        ("When was the Declaration of Independence signed?",
         "The United States Declaration of Independence was signed on August 2, 1776, though it was adopted on July 4, 1776.",
         "It announced that the thirteen American colonies were no longer subject to British rule.",
         "High"),
        ("Who was the first person to walk on the moon?",
         "Neil Armstrong was the first person to walk on the moon on July 20, 1969.",
         "He was the commander of the Apollo 11 mission.",
         "High"),
        ("When did World War II end?",
         "World War II ended in 1945.",
         "The war concluded with the unconditional surrender of the Axis powers.",
         "High"),
        ("Who was Julius Caesar?",
         "Julius Caesar was a Roman general and statesman who played a critical role in the events that led to the demise of the Roman Republic and the rise of the Roman Empire.",
         "He was assassinated by a group of rebellious senators on the Ides of March.",
         "High"),
        ("What was the Renaissance?",
         "The Renaissance was a period in European history marking the transition from the Middle Ages to modernity and covering the 15th and 16th centuries.",
         "It is characterized by an effort to revive and surpass ideas and achievements of classical antiquity.",
         "High"),
         ("Who built the Pyramids of Giza?",
         "The Pyramids of Giza were built by the ancient Egyptians during the Fourth Dynasty of the Old Kingdom.",
         "The Great Pyramid was built for the pharaoh Khufu.",
         "Medium"),
         ("What caused the fall of the Roman Empire?",
         "The fall of the Western Roman Empire was caused by a combination of factors including barbarian invasions, economic troubles, and political instability.",
         "It is generally considered to have ended in 476 AD when Romulus Augustulus was deposed.",
         "Medium"),
         ("Who was Cleopatra?",
         "Cleopatra VII Philopator was the last active ruler of the Ptolemaic Kingdom of Egypt.",
         "She was a diplomat, naval commander, linguist, and medical author.",
         "High"),
         ("When was the printing press invented?",
         "The printing press was invented by Johannes Gutenberg around 1440.",
         "It introduced the era of mass communication which permanently altered the structure of society.",
         "High")
    ]
    
    for i in range(n):
        base = base_facts[i % len(base_facts)]
        examples.append(Example(
            question=base[0],
            answer=base[1],
            explanation=base[2],
            confidence=base[3],
            category="history"
        ))
    return examples

def generate_geography_examples(n: int) -> List[Example]:
    examples = []
    base_facts = [
        ("What is the capital of France?", "The capital of France is Paris.", "Paris is also the most populous city in France.", "High"),
        ("Which is the largest continent?", "Asia is the largest continent by both land area and population.", "It covers an area of 44,579,000 square kilometers.", "High"),
        ("What is the longest river in the world?", "The Nile is generally considered the longest river in the world.", "It flows northwards through northeastern Africa.", "Medium"),
        ("Where is the Great Barrier Reef?", "The Great Barrier Reef is located off the coast of Queensland, Australia.", "It is the world's largest coral reef system.", "High"),
        ("What is the highest mountain in the world?", "Mount Everest is the highest mountain above sea level.", "It is located in the Mahalangur Himal sub-range of the Himalayas.", "High"),
        ("What is the capital of Japan?", "The capital of Japan is Tokyo.", "Tokyo is the political and economic center of the country.", "High"),
        ("Which ocean is the largest?", "The Pacific Ocean is the largest and deepest of Earth's oceanic divisions.", "It extends from the Arctic Ocean in the north to the Southern Ocean in the south.", "High"),
        ("What is the capital of Brazil?", "The capital of Brazil is Brasília.", "It was founded in 1960 to move the capital from Rio de Janeiro to a more central location.", "High"),
        ("Where is the Sahara Desert?", "The Sahara Desert is located in North Africa.", "It is the largest hot desert in the world.", "High"),
        ("What is the capital of Canada?", "The capital of Canada is Ottawa.", "It stands on the south bank of the Ottawa River in the southern portion of the province of Ontario.", "High")
    ]
    for i in range(n):
        base = base_facts[i % len(base_facts)]
        examples.append(Example(question=base[0], answer=base[1], explanation=base[2], confidence=base[3], category="geography"))
    return examples

def generate_math_examples(n: int) -> List[Example]:
    examples = []
    base_facts = [
        ("What is 2 + 2?", "2 + 2 equals 4.", "Addition is one of the four basic operations of arithmetic.", "High"),
        ("What is the value of Pi?", "Pi is approximately 3.14159.", "It is the ratio of a circle's circumference to its diameter.", "High"),
        ("What is a prime number?", "A prime number is a natural number greater than 1 that is not a product of two smaller natural numbers.", "Examples include 2, 3, 5, 7, 11, etc.", "High"),
        ("What is the square root of 64?", "The square root of 64 is 8.", "8 multiplied by 8 equals 64.", "High"),
        ("What is the Pythagorean theorem?", "The Pythagorean theorem states that in a right-angled triangle, the square of the hypotenuse equals the sum of the squares of the other two sides.", "It is written as a^2 + b^2 = c^2.", "High"),
        ("What is calculus?", "Calculus is the mathematical study of continuous change.", "It has two major branches: differential calculus and integral calculus.", "High"),
        ("What is algebra?", "Algebra is the study of mathematical symbols and the rules for manipulating these symbols.", "It is a unifying thread of almost all of mathematics.", "High"),
        ("What is geometry?", "Geometry is a branch of mathematics concerned with questions of shape, size, relative position of figures, and the properties of space.", "It arose independently in a number of early cultures as a practical way for dealing with lengths, areas, and volumes.", "High"),
        ("What is an integer?", "An integer is a number that can be written without a fractional component.", "Integers include 0, positive natural numbers, and their negative counterparts.", "High"),
        ("What is a fraction?", "A fraction represents a part of a whole or, more generally, any number of equal parts.", "A common fraction consists of a numerator and a denominator.", "High")
    ]
    for i in range(n):
        base = base_facts[i % len(base_facts)]
        examples.append(Example(question=base[0], answer=base[1], explanation=base[2], confidence=base[3], category="math"))
    return examples

def generate_technology_examples(n: int) -> List[Example]:
    examples = []
    base_facts = [
        ("What is Artificial Intelligence?", "Artificial Intelligence (AI) is intelligence demonstrated by machines, as opposed to the natural intelligence displayed by animals including humans.", "AI research has been defined as the field of study of intelligent agents.", "High"),
        ("What is the Internet?", "The Internet is a global system of interconnected computer networks that uses the Internet protocol suite (TCP/IP) to communicate between networks and devices.", "It carries a vast range of information resources and services.", "High"),
        ("What is a computer virus?", "A computer virus is a type of computer program that, when executed, replicates itself by modifying other computer programs and inserting its own code.", "When this replication succeeds, the affected areas are then said to be 'infected'.", "High"),
        ("What is cloud computing?", "Cloud computing is the on-demand availability of computer system resources, especially data storage and computing power, without direct active management by the user.", "Large clouds often have functions distributed over multiple locations, each known as a data center.", "High"),
        ("What is blockchain?", "Blockchain is a shared, immutable ledger that facilitates the process of recording transactions and tracking assets in a business network.", "An asset can be tangible (a house, car, cash, land) or intangible (intellectual property, patents, copyrights, branding).", "High"),
        ("What is a CPU?", "A central processing unit (CPU) is the electronic circuitry that executes instructions comprising a computer program.", "The CPU performs basic arithmetic, logic, controlling, and input/output (I/O) operations specified by the instructions in the program.", "High"),
        ("What is RAM?", "Random-access memory (RAM) is a form of computer memory that can be read and changed in any order, typically used to store working data and machine code.", "A random-access memory device allows data items to be read or written in almost the same amount of time irrespective of the physical location of data inside the memory.", "High"),
        ("What is Python?", "Python is a high-level, general-purpose programming language.", "Its design philosophy emphasizes code readability with the use of significant indentation.", "High"),
        ("What is open source software?", "Open source software is software with source code that anyone can inspect, modify, and enhance.", "Source code is the part of software that most computer users don't ever see.", "High"),
        ("What is a database?", "A database is an organized collection of data, generally stored and accessed electronically from a computer system.", "Where databases are more complex they are often developed using formal design and modeling techniques.", "High")
    ]
    for i in range(n):
        base = base_facts[i % len(base_facts)]
        examples.append(Example(question=base[0], answer=base[1], explanation=base[2], confidence=base[3], category="technology"))
    return examples

def generate_general_examples(n: int) -> List[Example]:
    examples = []
    base_facts = [
        ("Who wrote Romeo and Juliet?", "William Shakespeare wrote Romeo and Juliet.", "It is a tragedy about two young star-crossed lovers whose deaths ultimately reconcile their feuding families.", "High"),
        ("What is the Mona Lisa?", "The Mona Lisa is a half-length portrait painting by Italian artist Leonardo da Vinci.", "It is considered an archetypal masterpiece of the Italian Renaissance.", "High"),
        ("What is philosophy?", "Philosophy is the study of general and fundamental questions, such as those about existence, reason, knowledge, values, mind, and language.", "Such questions are often posed as problems to be studied or resolved.", "High"),
        ("What is a novel?", "A novel is a relatively long work of narrative fiction, typically written in prose and published as a book.", "The genre has also been described as having a continuous and comprehensive history of about two thousand years.", "High"),
        ("Who is Aristotle?", "Aristotle was a Greek philosopher and polymath during the Classical period in Ancient Greece.", "He was the founder of the Lyceum and the Peripatetic school of philosophy and Aristotelian tradition.", "High"),
        ("What is jazz?", "Jazz is a music genre that originated in the African-American communities of New Orleans, Louisiana, United States, in the late 19th and early 20th centuries.", "It has roots in blues and ragtime.", "High"),
        ("What is impressionism?", "Impressionism is a 19th-century art movement characterized by relatively small, thin, yet visible brush strokes, open composition, emphasis on accurate depiction of light in its changing qualities.", "Originating with a group of Paris-based artists whose independent exhibitions brought them to prominence during the 1870s and 1880s.", "High"),
        ("What is a haiku?", "A haiku is a type of short form poetry originally from Japan.", "Traditional Japanese haiku consist of three phrases that contain a kireji, or 'cutting word', 17 on in a 5, 7, 5 pattern, and a kigo, or seasonal reference.", "High"),
        ("What is culture?", "Culture is an umbrella term which encompasses the social behavior, institutions, and norms found in human societies, as well as the knowledge, beliefs, arts, laws, customs, capabilities, and habits of the individuals in these groups.", "Culture is often originated from or attributed to a specific region or location.", "Medium"),
        ("What is mythology?", "Mythology is a collection of myths, especially one belonging to a particular religious or cultural tradition.", "Myths are often stories explaining natural or social phenomena.", "High")
    ]
    for i in range(n):
        base = base_facts[i % len(base_facts)]
        examples.append(Example(question=base[0], answer=base[1], explanation=base[2], confidence=base[3], category="general"))
    return examples

def generate_lema_examples(n: int) -> List[Example]:
    """Generate examples about LEMA itself."""
    examples = []
    base_facts = [
        ("What is LEMA?",
         "LEMA is a framework that virtualizes GPU memory to enable training large language models on limited hardware.",
         "It uses a triple-buffer strategy to stream model layers through VRAM, reducing memory requirements by 50-70%.",
         "High"),
        ("How does LEMA work?",
         "LEMA streams model layers through GPU memory instead of loading the entire model at once.",
         "This layer-wise approach trades computation time for memory efficiency using asynchronous prefetching.",
         "High"),
        ("What is the Triple-Buffer Strategy?",
         "The Triple-Buffer Strategy is a memory management technique used by LEMA to optimize data flow.",
         "It maintains buffers in Disk, RAM, and VRAM to ensure the GPU always has data ready to process.",
         "High"),
        ("Does LEMA support LoRA?",
         "Yes, LEMA supports Low-Rank Adaptation (LoRA) for efficient fine-tuning.",
         "LoRA adapters are kept in VRAM while the base model weights are streamed.",
         "High"),
        ("What is the benefit of Gradient Checkpointing in LEMA?",
         "Gradient Checkpointing reduces VRAM usage during the backward pass by not storing intermediate activations.",
         "This allows for larger batch sizes or sequence lengths at the cost of some additional computation.",
         "High"),
        ("Can LEMA run on a 16GB GPU?",
         "Yes, LEMA enables training 7B+ models on GPUs with as little as 16GB VRAM, like the Tesla P100.",
         "It significantly reduces the memory footprint compared to standard loading methods.",
         "High"),
        ("What file format does LEMA require?",
         "LEMA requires models in a monolithic `.safetensors` format.",
         "This allows for efficient memory mapping and streaming.",
         "High"),
        ("Who developed LEMA?",
         "LEMA is an open-source project developed by Pomilon.",
         "It addresses the challenge of fine-tuning large models on consumer hardware.",
         "High"),
        ("What is GBI in LEMA?",
         "GBI stands for Global Binary Index.",
         "It is a zero-copy mapping technique for `.safetensors` files using `mmap`.",
         "High"),
        ("Does LEMA support GPT-2?",
         "Yes, LEMA supports GPT-2 architectures in addition to Llama-based models.",
         "It can be configured via `LemaConfig`.",
         "High")
    ]
    for i in range(n):
        base = base_facts[i % len(base_facts)]
        examples.append(Example(question=base[0], answer=base[1], explanation=base[2], confidence=base[3], category="lema"))
    return examples

def generate_practical_examples(n: int) -> List[Example]:
    examples = []
    base_facts = [
        ("How do I tie a tie?", "There are several knots, but the Four-in-Hand is one of the simplest. Cross the wide end over the narrow end, wrap it around, and pull it through the loop.", "It is a versatile knot suitable for most collars.", "High"),
        ("How do I boil an egg?", "Place eggs in a pot, cover with water, bring to a boil, then remove from heat and let sit for 9-12 minutes depending on desired firmness.", "Cool immediately in ice water to stop cooking.", "High"),
        ("What is a resume?", "A resume is a document created and used by a person to present their background, skills, and accomplishments.", "Resumes can be used for a variety of reasons, but most often they are used to secure new employment.", "High"),
        ("How do I write a cover letter?", "A cover letter should address the hiring manager, state the position you are applying for, highlight relevant skills, and express enthusiasm for the role.", "It complements your resume by providing a narrative of your qualifications.", "High"),
        ("What is a budget?", "A budget is a financial plan for a defined period, often one year.", "It may also include planned sales volumes and revenues, resource quantities, costs and expenses, assets, liabilities and cash flows.", "High"),
        ("How do I change a tire?", "Loosen the lug nuts, jack up the car, remove the nuts and the tire, place the spare tire on, tighten the nuts, lower the car, and fully tighten the nuts.", "Consult your vehicle's owner manual for specific jack points and instructions.", "High"),
        ("What is a recipe?", "A recipe is a set of instructions that describes how to prepare or make something, especially a dish of prepared food.", "It typically includes a list of ingredients and a step-by-step procedure.", "High"),
        ("How do I plant a tree?", "Dig a hole twice as wide as the root ball, place the tree in the hole, fill with soil, water thoroughly, and add mulch.", "Ensure the trunk flare is visible above the soil line.", "High"),
        ("What is a contract?", "A contract is a legally binding agreement between two or more parties.", "It creates mutual obligations that are enforceable by law.", "High"),
        ("How do I make coffee?", "There are many methods, but a standard drip coffee maker involves adding water to the reservoir, placing a filter and ground coffee in the basket, and turning the machine on.", "The ratio of coffee to water determines the strength.", "High")
    ]
    for i in range(n):
        base = base_facts[i % len(base_facts)]
        examples.append(Example(question=base[0], answer=base[1], explanation=base[2], confidence=base[3], category="practical"))
    return examples

def build_dataset(num_examples: int = 5000, output_path: str = "data/training_data.jsonl") -> None:
    """
    Generate complete dataset with specified distribution.
    
    Args:
        num_examples: Total number of examples (5000-10000)
        output_path: Where to save JSONL file
    """
    
    if not (5000 <= num_examples <= 10000):
        raise ValueError("num_examples must be between 5000 and 10000")
    
    # Calculate category distributions
    distribution = {
        'science': int(num_examples * 0.15),
        'history': int(num_examples * 0.15),
        'geography': int(num_examples * 0.10),
        'math': int(num_examples * 0.10),
        'technology': int(num_examples * 0.10),
        'general': int(num_examples * 0.15),
        'lema': int(num_examples * 0.10),
        'practical': int(num_examples * 0.15),
    }
    
    # Adjust for rounding errors
    current_total = sum(distribution.values())
    diff = num_examples - current_total
    if diff > 0:
        distribution['science'] += diff
    
    all_examples = []
    
    # Generate examples for each category
    all_examples.extend(generate_science_examples(distribution['science']))
    all_examples.extend(generate_history_examples(distribution['history']))
    all_examples.extend(generate_geography_examples(distribution['geography']))
    all_examples.extend(generate_math_examples(distribution['math']))
    all_examples.extend(generate_technology_examples(distribution['technology']))
    all_examples.extend(generate_general_examples(distribution['general']))
    all_examples.extend(generate_lema_examples(distribution['lema']))
    all_examples.extend(generate_practical_examples(distribution['practical']))
    
    # Shuffle to mix categories
    random.shuffle(all_examples)
    
    # Write to JSONL
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for example in all_examples:
            formatted = format_example(example)
            json_line = json.dumps({"text": formatted}, ensure_ascii=False)
            f.write(json_line + '\n')
    
    print(f"✅ Generated {len(all_examples)} examples")
    print(f"✅ Saved to {output_path}")
    
    # Validation
    validate_dataset(output_path)

def validate_dataset(path: str) -> None:
    """Validate dataset format."""
    required_tokens = ['<|system|>', '<|user|>', '<|assistant|>', '[LEMA_REPLY]', '[/LEMA_REPLY]']
    required_fields = ['Answer:', 'Explanation:', 'Confidence:']
    
    with open(path, 'r') as f:
        for i, line in enumerate(f):
            data = json.loads(line)
            text = data['text']
            
            # Check all tokens present
            for token in required_tokens:
                if token not in text:
                    raise ValueError(f"Line {i}: Missing token '{token}'")
            
            # Check all fields present
            for field in required_fields:
                if field not in text:
                    raise ValueError(f"Line {i}: Missing field '{field}'")
    
    print("✅ Dataset validation passed")

if __name__ == "__main__":
    build_dataset(num_examples=5000)
