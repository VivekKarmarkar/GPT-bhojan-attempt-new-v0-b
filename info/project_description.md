Project Description Sections

**Segment A: Introduction

Upload a photo of your meal. Three AI models — the Brain, the Eyes, and the Scissors — work together to find every item on your plate, cut each one out at the pixel level, and score the meal on five dimensions: health, satiety, bloat, tastiness, and addiction potential.

The Brain is GPT-4o. It understands what food looks like, what it's called, and how nutritious it is. The Eyes are YOLOv8. They scan the image and find where things are. The Scissors are SAM (Segment Anything Model). They trace precise outlines around whatever you point them at. None of these models can do the full job alone — the Eyes can find objects but can't name them, the Brain can name food but can't locate it spatially, and the Scissors can cut anything but don't know what's worth cutting. The pipeline coordinates all three so that each one compensates for what the others can't do.

The system prompt is tuned for Indian cuisine, and the evaluation test suite uses Indian food images — but the underlying architecture is model-agnostic. Swap the prompt and the test set, and the same pipeline works on any cuisine.

**Segment B: Project Description

The pipeline processes each image through nine steps.

The Brain analyzes the whole plate first. It looks at the full image in one pass, names every dish, and rates the meal on five scales. This big-picture understanding becomes context for everything that follows.

At the same time, the Eyes scan the image and draw tight bounding boxes around every object they find. The Eyes don't know what's food and what isn't — a fork, a bowl rim, and a pile of rice all get boxes. The sorting happens next.

Both streams converge when the Brain classifies each box. It examines the cropped region alongside the full plate photo and the description it wrote earlier. It makes a visual judgment: does this crop isolate a single food item, or is it junk? Boxes covering more than 90% of the image are skipped automatically as a cost guard — they're obviously just the whole photo again.

After classification, duplicate boxes are removed. If two boxes with the same label overlap by more than half their area, the tighter one is kept. Two of the same food item in different locations both survive — only overlapping duplicates on the same item get collapsed.

The Scissors then take over. They analyze the full image once upfront, then for each surviving box, they produce three candidate outlines. The one with the highest confidence score is selected. The Scissors don't know what they're cutting — they're trained to segment any object you point them at.

But the Scissors sometimes trace the wrong thing — the bowl instead of the food inside it. So the Brain gets a second opinion. It sees two images: the cut-out region (A) and everything around it (B), both on black backgrounds. It picks which one actually shows the food. When the Brain says "wrong cut," the Scissors don't redo the entire image — they only flip their work inside that one box.

The selected region then passes a brightness check. At least 100 pixels need to be brighter than a minimum threshold. If the cut-out is too dark, it's likely empty or not a food item, and it's quietly dropped.

Finally, surviving items are overlaid on the original image with translucent color masks, bounding boxes, and labels. The annotated image, the five-score radar chart, and the item list are returned to the frontend.

**Segment C: Results

The pipeline was evaluated against a 17-image test suite of Indian meals, ranging from simple single-item plates to complex multi-dish meals with five or more items.

90% identification accuracy — 28 out of 31 detections were correct across all images that produced output. Of the 14 images where the Eyes detected objects, 11 had zero wrong identifications. The remaining 3 images each had exactly one misidentified item.

The segmentation quality was consistent. The Scissors' box-prompted outlines closely followed food boundaries, and the Brain's A/B verification step caught cases where the Scissors initially traced a container edge instead of the food surface.

Three images — Modak, Salmon, and Rasmalai — produced no output at all. The Eyes found zero bounding boxes for these items. These represent the detection model's limits on certain food presentations and are excluded from the accuracy calculation.

**Segment D: Key Findings

All three misidentifications followed the same pattern. The Brain defaulted to "Sambar" when it encountered an unfamiliar curry bowl.

This is not a random failure mode. It reveals a systematic bias — when the Brain lacks confidence in a visual classification, it falls back to the most common South Indian curry in its training data. The errors were not spread across different food categories or different failure mechanisms. They clustered on a single label, applied to a single visual archetype: a round bowl of brown liquid.

This finding points to a clear next step. A fine-tuned classifier trained on a curated dataset of Indian food categories could replace the Brain's zero-shot classification, eliminating the Sambar bias while preserving the Brain's role in verification and full-plate analysis where its generalist knowledge is an advantage.
