# Pedagogy Standards — GPT Bhojan

How to explain this system. Modeled on Feynman, Strogatz, 3Blue1Brown.

---

## 1. Read the code first

Not a summary. Not a description of the code. The code itself. Trace a single image through every function call, every if-statement, every return value. Know exactly what `predict()` returns (3 masks, not 1), exactly what the brightness threshold is (30), exactly when and why the complement mask gets constrained to the bounding box. You can't explain what you don't fully understand, and you can't fully understand what you haven't traced line by line.

## 2. Identify the question each step answers

Not "what does this step do" — that's an engineer's framing. Ask: "What question is the system asking at this moment, and why does that question need asking?"

- Step 4 isn't "classify each box." Step 4 is answering: "The Eyes found 12 things — but which of those are actually food?"
- Step 7 isn't "verify A vs B." Step 7 is answering: "The Scissors cut something out, but did they cut the right thing?"

## 3. Never remove a concept to make things simpler

Find a better way to say it. The complement mask bug fix isn't irrelevant — it's a real design decision that reveals how the system thinks. But "complement mask constrained to bounding box" is engineer-speak. Say instead: "When the Brain says 'wrong cut,' the Scissors don't redo the entire image — they only flip their work inside that one box." Same information. Zero jargon. The concept survives intact.

## 4. Obsess over the ordering of ideas

Never show the answer and then explain why. Build the *need* for each step before introducing it. The viewer should feel the problem of duplicate boxes before seeing the dedup step. They should feel the problem of the Scissors sometimes tracing the bowl instead of the food *before* seeing the A/B verification. Each step arrives as the answer to a tension you already feel.

## 5. Treat every word like a pixel

- "Snap a photo" vs "upload a photo" — one implies a camera app, the other describes what actually happens.
- "Speed is the priority" is a claim about engineering goals that isn't even accurate — the priority of the Eyes is *coverage*, finding everything.
- "Steaks" and "idli" are both the same mistake: reaching for a vivid example when the abstraction ("food item") is actually clearer.
- Feynman was famous for saying "I don't know what you mean" until people stopped hiding behind specifics.

## 6. Design the legend as a teaching tool, not a color key

The legend isn't "orange dot = Brain." The legend is: here are three characters in this story, and here's why you need all three.

## 7. The verification step

Understand deeply, then explain simply, then verify that the simplicity preserved the depth. Read the explanation back and ask: "Could someone reconstruct the actual system from this description?" If the answer is no, you didn't simplify — you deleted.
