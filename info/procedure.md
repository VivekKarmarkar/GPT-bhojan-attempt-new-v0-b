# Procedure — Remotion Video Rebuild

## What the user wants

1. **Delete junk at the root** — no performative fixes. If something is wrong, delete it from the source file, not patch it in the UI.

2. **User guidelines are intelligence, not hard rules** — the user provides design guidance (split-screen, breathing room, diagram prominence). These are not rigid specs. Use them intelligently to make good design decisions.

3. **Enter plan mode** — think deeply about the video design before writing code. Use the guidelines to inform the plan. Don't blindly follow a checklist.

4. **Use agentic teams** — the user explicitly directs the use of agentic teams:
   - Designer team: layout, spacing, visual hierarchy, what looks good to a human
   - Animator team: Remotion code, timing, transitions, smooth flow
   - Validator team: review output against HTML source of truth, check nothing is cramped or ugly

5. **Minimize permissions, maximize efficiency, minimize time** — within the team structure.

## Source of truth

- `media/webpages/architecture_playground.html` — the sacred SVG diagram, node positions, path definitions, particle animations, legend format
- `templates/index.html` — app title: "What's on your Plate?"
- `info/pedagogy.md` — how to explain the system (guidelines, not hard rules)

## Design guidelines from user

- The diagram should be prominent and readable — not crushed into a tiny strip
- Split-screen is a good approach: diagram on one side, text on the other
- When explaining a step: highlight/glow that node in the diagram, show description text clearly
- Enough breathing room for the viewer to absorb ideas
- Enough time per step — don't rush
- No junk labels — the legend is just emoji + character name + model name (Brain/GPT-4o, Eyes/YOLOv8, Scissors/SAM). Nothing else.
- Step descriptions come from the playground HTML's stepData — those are the words to use

## What NOT to do

- Do NOT add descriptive trade-off labels to legend characters
- Do NOT make the diagram small with chrome around it
- Do NOT do performative fixes — fix problems at the root
- Do NOT blindly follow the plan without thinking about design quality
- Do NOT hallucinate content that isn't in the source files
