from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

import io
import json
import base64
import math
import fitz  # PyMuPDF
from typing import Optional, List, Any, Dict, Tuple
import logging
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger("uvicorn")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok"}


# ----------------------------
# COLOR HELPERS
# ----------------------------
def hex_to_rgb01(hex_color: str) -> Tuple[float, float, float]:
    if not hex_color:
        return (0.0, 0.0, 0.0)
    hex_color = hex_color.strip().lstrip("#")
    if len(hex_color) == 3:
        hex_color = "".join(c * 2 for c in hex_color)
    try:
        value = int(hex_color, 16)
    except ValueError:
        return (0.0, 0.0, 0.0)

    r = (value >> 16) & 255
    g = (value >> 8) & 255
    b = value & 255
    return (r / 255.0, g / 255.0, b / 255.0)


def css_color_to_rgba01(color: str) -> Tuple[float, float, float, float]:
    if not color:
        return (0.0, 0.0, 0.0, 1.0)
    color = color.strip()

    if color.startswith("#"):
        r, g, b = hex_to_rgb01(color)
        return (r, g, b, 1.0)

    if color.startswith("rgb"):
        inside = color[color.find("(") + 1: color.find(")")]
        parts = [p.strip() for p in inside.split(",")]
        if len(parts) >= 3:
            try:
                r = int(float(parts[0]))
                g = int(float(parts[1]))
                b = int(float(parts[2]))
                a = 1.0
                if len(parts) >= 4:
                    a = float(parts[3])
                    if a > 1:
                        a = max(0.0, min(1.0, a / 255.0))
                    else:
                        a = max(0.0, min(1.0, a))
                return (r / 255.0, g / 255.0, b / 255.0, a)
            except Exception:
                pass

    return (0.0, 0.0, 0.0, 1.0)


# ----------------------------
# TEXT HELPERS
# ----------------------------
def pick_fontname(style: str, weight: str) -> str:
    s = str(style).lower()
    w = str(weight).lower()
    
    is_bold = "bold" in w or w in ["600", "700", "800", "900"]
    is_italic = "italic" in s or "oblique" in s

    if is_bold and is_italic:
        return "hebi" 
    elif is_bold:
        return "hebo" 
    elif is_italic:
        return "heit" 
    else:
        return "helv"

def safe_text_width(text: str, fontname: str, fontsize: float) -> float:
    try:
        return fitz.get_text_length(text, fontname=fontname, fontsize=fontsize)
    except Exception:
        return len(text) * fontsize * 0.55


def wrap_text_to_width(text: str, max_width: float, fontname: str, fontsize: float) -> List[str]:
    # Use explicit newlines only
    if "\n" in text:
        paragraphs = text.split("\n")
        lines = []
        for p in paragraphs:
            lines.extend(wrap_line_algorithm(p, max_width, fontname, fontsize))
        return lines
    else:
        return wrap_line_algorithm(text, max_width, fontname, fontsize)

def wrap_line_algorithm(text: str, max_width: float, fontname: str, fontsize: float) -> List[str]:
    words = text.split()
    if not words:
        return [""]

    lines: List[str] = []
    cur = ""

    for w in words:
        test = w if not cur else f"{cur} {w}"
        if safe_text_width(test, fontname, fontsize) <= max_width:
            cur = test
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def font_ascender_px(fontname: str, fontsize: float) -> float:
    try:
        f = fitz.Font(fontname=fontname)
        return (f.ascent / 1000.0) * fontsize
    except Exception:
        return fontsize * 0.80


# ----------------------------
# ✅ DRAW TEXT (PADDING & WRAPPING FIXES)
# ----------------------------
def draw_text_on_pdf(doc: fitz.Document, data: List[Dict[str, Any]]) -> None:
    """
    FIXES APPLIED:
      ✅ Bottom White Space: Reduced vertical padding (bg_pad_y_bot = -4.0).
      ✅ Left Artifacts: Increased horizontal padding (bg_pad_x = 5.0).
      ✅ Wrapping: Increased tolerance (w * 0.5) to keep single lines intact.
    """

    if not data:
        return

    # --- CALIBRATION ---
    X_CALIBRATION_PX = -1.0    
    Y_CALIBRATION_PX = -1.0   

    extra_gap = 0.0           
    line_mult = 1.13          

    for item in data:
        page_number = int(item.get("pageNumber", 1))
        page_index = page_number - 1
        if page_index < 0 or page_index >= len(doc):
            continue

        page = doc[page_index]
        page_rect = page.rect

        for obj in item.get("textObjects", []):
            text = obj.get("text", "")
            if not text:
                continue

            # 1. Coordinates
            rect_pct = obj.get("rect") or {}
            left_pct = float(rect_pct.get("leftPct", 0.0))
            top_pct = float(rect_pct.get("topPct", 0.0))
            width_pct = float(rect_pct.get("widthPct", 0.0))
            height_pct = float(rect_pct.get("heightPct", 0.0))

            x = page_rect.x0 + (left_pct / 100.0) * page_rect.width
            y = page_rect.y0 + (top_pct / 100.0) * page_rect.height
            w = (width_pct / 100.0) * page_rect.width
            h = (height_pct / 100.0) * page_rect.height

            # 2. Styles
            fontsize = float(obj.get("size", 12.0))
            font_style = obj.get("fontStyle", "normal")
            font_weight = obj.get("fontWeight", "normal")
            fontname = pick_fontname(font_style, font_weight)

            # 3. Colors
            color_str = obj.get("color", "#000000")
            r, g, b, _a = css_color_to_rgba01(color_str)

            background_hex = obj.get("background")
            underline = bool(obj.get("underline", False))
            linethrough = bool(obj.get("linethrough", False))

            ann_type = str(obj.get("annotationType", "")).lower()
            is_edittext = ann_type == "edittext"

            # 4. Draw Background (PADDING FIXES)
            if background_hex:
                br, bg, bb = hex_to_rgb01(background_hex)
                
                # ✅ FIX 1: LEFT/RIGHT ARTIFACTS
                # Increase X padding to cover previous text fully
                bg_pad_x = 0.5  
                
                # ✅ FIX 2: BOTTOM WHITE SPACE
                # Use a larger NEGATIVE number for bottom padding to shrink the box height
                # so it doesn't cover the line below.
                bg_pad_y_top = 2.0   
                bg_pad_y_bot = -2.0  # Shrink bottom significantly
                
                clean_rect = fitz.Rect(
                    x - bg_pad_x, 
                    y - bg_pad_y_top, 
                    x + w + bg_pad_x, 
                    y + h + bg_pad_y_bot
                )
                
                page.draw_rect(clean_rect, color=None, fill=(br, bg, bb))

            # 5. Layout & Wrapping Fix
            pad = max(1.0, fontsize * 0.08)
            
            # ✅ FIX 3: TEXT WRAPPING
            # Add 50% buffer to width. If frontend fits it, backend fits it.
            wrapping_buffer = w * 0.0 
            usable_w = w + wrapping_buffer    

            lines = wrap_text_to_width(text, usable_w, fontname, fontsize)

            asc_px = font_ascender_px(fontname, fontsize)
            extra_up = (-0.25 * fontsize) if is_edittext else 0.0

            baseline_y = (y + pad + asc_px + Y_CALIBRATION_PX + extra_up)
            x0 = x + pad + X_CALIBRATION_PX
            step = (fontsize * line_mult) + extra_gap
            cur_y = baseline_y

            # 6. Draw Lines
            for line in lines:
                page.insert_text((x0, cur_y), line, fontsize=fontsize, fontname=fontname, color=(r, g, b))

                line_w = safe_text_width(line, fontname, fontsize)

                if linethrough:
                    strike_y = cur_y - (fontsize * 0.28) 
                    page.draw_line(p1=(x0, strike_y), p2=(x0 + line_w, strike_y), color=(r, g, b), width=1)

                if underline:
                    under_y = cur_y + 2.0 
                    page.draw_line(p1=(x0, under_y), p2=(x0 + line_w, under_y), color=(r, g, b), width=1)

                cur_y += step


# ----------------------------
# FREEHAND IMAGE LAYER
# ----------------------------
def draw_annotation_layer_image(doc: fitz.Document, image_data: List[str]) -> None:
    for page_index, data_url in enumerate(image_data):
        if page_index >= len(doc):
            break

        page = doc[page_index]
        page_rect = page.rect

        base64_str = data_url.split(",", 1)[1] if "," in data_url else data_url
        img_bytes = base64.b64decode(base64_str)
        page.insert_image(page_rect, stream=img_bytes)


# ----------------------------
# COMMENTS
# ----------------------------
def add_comments(doc: fitz.Document, comments: List[Dict[str, Any]]) -> None:
    if not comments:
        return

    for c in comments:
        page_num = int(c.get("page", 1)) - 1
        if page_num < 0 or page_num >= len(doc):
            continue

        page = doc[page_num]
        page_rect = page.rect

        left_pct = float(c.get("leftPct", 0.0))
        top_pct = float(c.get("topPct", 0.0))

        x = page_rect.x0 + (left_pct / 100.0) * page_rect.width
        y = page_rect.y0 + (top_pct / 100.0) * page_rect.height

        text = c.get("text", "")
        page.add_text_annot(fitz.Point(x, y), text)


# ----------------------------
# ERASER
# ----------------------------
def apply_erasers(doc: fitz.Document, eraser_annotations: List[Dict[str, Any]]) -> None:
    if not eraser_annotations:
        return

    for ann in eraser_annotations:
        chunks = ann.get("chunks") or []
        for ch in chunks:
            page_number = int(ch.get("pageNumber", 1))
            page_index = page_number - 1
            if page_index < 0 or page_index >= len(doc):
                continue

            page = doc[page_index]
            page_rect = page.rect

            rect_pct = ch.get("rect") or {}
            left_pct = float(rect_pct.get("leftPct", 0.0))
            top_pct = float(rect_pct.get("topPct", 0.0))
            width_pct = float(rect_pct.get("widthPct", 0.0))
            height_pct = float(rect_pct.get("heightPct", 0.0))

            x = page_rect.x0 + (left_pct / 100.0) * page_rect.width
            y = page_rect.y0 + (top_pct / 100.0) * page_rect.height
            w = (width_pct / 100.0) * page_rect.width
            h = (height_pct / 100.0) * page_rect.height

            pad_y = max(3.0, h * 0.25)
            pad_x = max(3.0, w * 0.05)

            rect = fitz.Rect(x - pad_x, y - pad_y, x + w + pad_x, y + h + pad_y)
            page.draw_rect(rect, color=(1.0, 1.0, 1.0), fill=(1.0, 1.0, 1.0))


# ----------------------------
# OTHER RECT/LINE ANNOTATIONS
# ----------------------------
def draw_rect_annotations(doc: fitz.Document, annotations: List[Dict[str, Any]]) -> None:
    if not annotations:
        return

    HIGHLIGHT_Y_ADJUST = 0.12
    UNDERLINE_BASELINE_FRACTION = 0.78

    for ann in annotations:
        ann_type = ann.get("type", "highlight")
        if ann_type == "eraser":
            continue

        color_str = ann.get("color") or "rgba(255,255,0,0.95)"
        r, g, b, a = css_color_to_rgba01(color_str)

        chunks = ann.get("chunks") or []
        for ch in chunks:
            page_number = int(ch.get("pageNumber", 1))
            page_index = page_number - 1
            if page_index < 0 or page_index >= len(doc):
                continue

            page = doc[page_index]
            page_rect = page.rect

            rect_pct = ch.get("rect") or {}
            left_pct = float(rect_pct.get("leftPct", 0.0))
            top_pct = float(rect_pct.get("topPct", 0.0))
            width_pct = float(rect_pct.get("widthPct", 0.0))
            height_pct = float(rect_pct.get("heightPct", 0.0))

            x = page_rect.x0 + (left_pct / 100.0) * page_rect.width
            y = page_rect.y0 + (top_pct / 100.0) * page_rect.height
            w = (width_pct / 100.0) * page_rect.width
            h = (height_pct / 100.0) * page_rect.height

            if ann_type == "highlight":
                dy = h * HIGHLIGHT_Y_ADJUST
                y_top = max(page_rect.y0, y - dy)
                rect = fitz.Rect(x, y_top, x + w, y_top + h)
                try:
                    page.draw_rect(rect, color=None, fill=(r, g, b), fill_opacity=a)
                except TypeError:
                    rr = r * a + 1.0 * (1.0 - a)
                    gg = g * a + 1.0 * (1.0 - a)
                    bb = b * a + 1.0 * (1.0 - a)
                    page.draw_rect(rect, color=None, fill=(rr, gg, bb))

            elif ann_type == "underline":
                y_line = y + h * UNDERLINE_BASELINE_FRACTION
                width_px = max(1.0, h * 0.6)
                page.draw_line((x, y_line), (x + w, y_line), color=(r, g, b), width=width_px)

            elif ann_type == "strikeout":
                y_line = y + h * 0.5
                width_px = max(1.0, h)
                page.draw_line((x, y_line), (x + w, y_line), color=(r, g, b), width=width_px)

            elif ann_type == "squiggly":
                amp = max(1.0, min(3.0, h / 2.0))
                baseline_y = y + h * 0.8
                steps = max(8, int(w / 4))
                points = []
                for i in range(steps + 1):
                    t = i / float(steps)
                    px_ = x + w * t
                    py_ = baseline_y + amp * math.sin(2.0 * math.pi * t * 2.0)
                    points.append(fitz.Point(px_, py_))
                page.draw_polyline(points, color=(r, g, b), width=max(1.5, min(2.5, h * 0.8)))

            elif ann_type == "pen":
                pen_center_y = y + h / 2.0
                pen_width = max(1.5, min(4.0, h))
                page.draw_line((x, pen_center_y), (x + w, pen_center_y), color=(r, g, b), width=pen_width)


# ----------------------------
# MAIN ENDPOINT
# ----------------------------
@app.post("/annotate-pdf")
async def annotate_pdf(
    request: Request,
    pdfFile: UploadFile = File(...),
    pageImages: Optional[str] = Form(None),
    textObjects: Optional[str] = Form(None),
    comments: Optional[str] = Form(None),
    annotations: Optional[str] = Form(None),
    filename: Optional[str] = Form("annotated-document.pdf"),
):
    try:
        if pdfFile is None:
            raise HTTPException(status_code=400, detail="PDF file is required")

        pdf_bytes = await pdfFile.read()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")

        ann_arr = None
        if annotations:
            try:
                ann_arr = json.loads(annotations)
            except Exception:
                ann_arr = None

        eraser_ann: List[Dict[str, Any]] = []
        other_ann: List[Dict[str, Any]] = []
        if isinstance(ann_arr, list):
            eraser_ann = [a for a in ann_arr if a.get("type") == "eraser"]
            other_ann = [a for a in ann_arr if a.get("type") != "eraser"]

        # DRAW ORDER (bottom -> top)
        if eraser_ann:
            apply_erasers(doc, eraser_ann)

        if other_ann:
            draw_rect_annotations(doc, other_ann)

        if pageImages:
            try:
                image_arr = json.loads(pageImages)
                if isinstance(image_arr, list) and image_arr:
                    draw_annotation_layer_image(doc, image_arr)
            except Exception as e:
                logger.error("Error processing image overlay: %s", e, exc_info=True)

        if textObjects:
            try:
                text_data = json.loads(textObjects)
                if isinstance(text_data, list) and text_data:
                    draw_text_on_pdf(doc, text_data)
            except Exception as e:
                logger.error("Error processing textObjects: %s", e, exc_info=True)

        if comments:
            try:
                comments_arr = json.loads(comments)
                if isinstance(comments_arr, list) and comments_arr:
                    add_comments(doc, comments_arr)
            except Exception as e:
                logger.error("Error processing comments: %s", e, exc_info=True)

        output = io.BytesIO()
        doc.save(output)
        doc.close()
        output.seek(0)

        return StreamingResponse(
            output,
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error processing PDF download: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to process PDF download: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)