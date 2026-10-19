from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

# Create a crisp 1200x700 architectural diagram banner
width, height = 1200, 700
img = Image.new("RGB", (width, height), color="#0F172A")
draw = ImageDraw.Draw(img)

# Outer card border
draw.rectangle([30, 30, width - 30, height - 30], outline="#1E293B", width=2)
draw.rectangle([45, 45, width - 45, height - 45], fill="#1E293B")

# Top Accent Line
draw.rectangle([45, 45, width - 45, 55], fill="#0284C7")

try:
    font_badge = ImageFont.truetype("arial.ttf", 20)
    font_title = ImageFont.truetype("arialbd.ttf", 36)
    font_sub = ImageFont.truetype("arial.ttf", 22)
    font_step_title = ImageFont.truetype("arialbd.ttf", 22)
    font_step_desc = ImageFont.truetype("arial.ttf", 16)
    font_pill = ImageFont.truetype("arialbd.ttf", 18)
except Exception:
    font_badge = font_title = font_sub = font_step_title = font_step_desc = font_pill = ImageFont.load_default()

# Badge
draw.rounded_rectangle([80, 80, 370, 115], radius=6, fill="#0369A1")
draw.text((95, 87), "RETRIEVAL-AUGMENTED GENERATION", fill="#E0F2FE", font=font_badge)

# Title & Subtitle
draw.text((80, 135), "Document Q&A Pipeline Architecture", fill="#F8FAFC", font=font_title)
draw.text((80, 185), "End-to-End Workflow: Ingestion • Chunking • ChromaDB • Citation • Refusal", fill="#94A3B8", font=font_sub)

# 4 Pipeline Step Cards
step_y = 250
step_w = 235
step_h = 280
gap = 30

steps = [
    {
        "num": "01",
        "title": "Ingestion & Provenance",
        "tech": "pypdf",
        "desc": "Extracts text per page.\nAttaches source_file and\n1-indexed page_number\nto every segment.",
        "color": "#38BDF8",
    },
    {
        "num": "02",
        "title": "Recursive Chunking",
        "tech": "600 chars / 100 ovr",
        "desc": "Splits text by paragraphs\nand sentences to preserve\nsemantic boundaries\nwithout word clipping.",
        "color": "#818CF8",
    },
    {
        "num": "03",
        "title": "ChromaDB Vectors",
        "tech": "In-Memory HNSW",
        "desc": "Cosine similarity search.\nSession-isolated RAM\nstore (ephemeral) with\nzero disk accumulation.",
        "color": "#34D399",
    },
    {
        "num": "04",
        "title": "Grounded Answers",
        "tech": "Score Threshold",
        "desc": "Returns answers with\npage citations. Rejects\nout-of-scope questions\n(refusal mode).",
        "color": "#FBBF24",
    },
]

for i, s in enumerate(steps):
    sx = 80 + i * (step_w + gap)
    # Card
    draw.rounded_rectangle([sx, step_y, sx + step_w, step_y + step_h], radius=10, fill="#0F172A", outline="#334155", width=2)
    # Number
    draw.text((sx + 18, step_y + 18), s["num"], fill=s["color"], font=font_step_title)
    # Pill
    draw.rounded_rectangle([sx + 60, step_y + 18, sx + step_w - 18, step_y + 44], radius=4, fill="#1E293B")
    draw.text((sx + 70, step_y + 22), s["tech"], fill="#E2E8F0", font=font_step_desc)
    # Step Title
    draw.text((sx + 18, step_y + 65), s["title"], fill="#F1F5F9", font=font_step_title)
    # Desc
    draw.text((sx + 18, step_y + 115), s["desc"], fill="#94A3B8", font=font_step_desc)

# Footer bar
draw.line([(80, 580), (width - 80, 580)], fill="#334155", width=1)
draw.text((80, 610), "Status: 5/5 Automated Pytest Tests Passing • FastAPI REST API + Interactive Streamlit UI", fill="#64748B", font=font_sub)

output_path = Path(r"C:\Users\user\Desktop\RAG_Architecture_Flow.png")
img.save(str(output_path), quality=95)
print(f"RAG Architecture image created at: {output_path}")
