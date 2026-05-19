import sys
import os

# Make project root importable (matchers/, config.py, etc.)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import streamlit as st
from PIL import Image
import config

IMAGE_DIR = os.path.join(
    ROOT, "dataset", "easyread-retrieval-dataset", "data"
)

MATCHERS = {
    "Hybrid Matcher": "hybrid",
    "Simple Caption Matcher": "simple",
    "JSON Matcher": "json",
}

MATCHER_DESCRIPTIONS = {
    "hybrid": "LLM intent extraction + weighted multi-component embedding scoring.",
    "simple": "Fast cosine similarity on caption embeddings only. No LLM required.",
    "json": "LLM intent extraction followed by LLM reranking of top candidates.",
}


@st.cache_resource(show_spinner=False)
def load_matcher(matcher_type: str):
    if matcher_type == "hybrid":
        from matchers.hybrid_matcher import HybridMatcher
        return HybridMatcher()
    elif matcher_type == "simple":
        from matchers.simple_caption_matcher import SimpleCaptionMatcher
        return SimpleCaptionMatcher(pkl_path=config.CAPTION_INDEX)
    elif matcher_type == "json":
        from matchers.json_matcher import JsonMatcher
        return JsonMatcher(pkl_path=config.HYBRID_INDEX)


def find_image_path(filename: str) -> str | None:
    path = os.path.join(IMAGE_DIR, filename)
    return path if os.path.exists(path) else None


def render_results(sentences: list[str], results: list[list[str]], top_k: int) -> None:
    for sentence, image_files in zip(sentences, results):
        st.markdown(f"**{sentence}**")
        cols = st.columns(top_k)
        for i, col in enumerate(cols):
            with col:
                if i < len(image_files):
                    fname = image_files[i]
                    img_path = find_image_path(fname)
                    if img_path:
                        st.image(Image.open(img_path), caption=fname, use_container_width=True)
                    else:
                        st.warning(f"Not found: {fname}")
                else:
                    st.empty()
        st.divider()


def main() -> None:
    st.set_page_config(
        page_title="Easyread Visualizer",
        page_icon="🖼️",
        layout="wide",
    )

    st.title("Easyread Image Retrieval")
    st.write(
        "Enter plain-language sentences and see which pictograms our system retrieves. "
        "Use this tool to evaluate retrieval quality without needing technical knowledge."
    )

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Settings")

        matcher_label = st.selectbox("Matcher", list(MATCHERS.keys()), index=0)
        matcher_type = MATCHERS[matcher_label]
        st.caption(MATCHER_DESCRIPTIONS[matcher_type])

        top_k = st.slider("Images per sentence", min_value=1, max_value=5, value=3)

        st.divider()
        st.markdown("**Image directory**")
        st.code(IMAGE_DIR, language=None)
        if not os.path.isdir(IMAGE_DIR):
            st.error("Image directory not found.")

    # ── Main area ─────────────────────────────────────────────────────────────
    text_input = st.text_area(
        "Sentences to illustrate (one per line):",
        height=180,
        placeholder=(
            "A man is running in the park.\n"
            "I am eating an apple.\n"
            "The doctor works in a large hospital."
        ),
    )

    process_btn = st.button("Process these sentences", type="primary", use_container_width=True)

    if process_btn:
        sentences = [s.strip() for s in text_input.splitlines() if s.strip()]
        if not sentences:
            st.warning("Please enter at least one sentence.")
            return

        try:
            with st.spinner(f"Loading {matcher_label} (first run may take a while)…"):
                matcher = load_matcher(matcher_type)
        except FileNotFoundError as e:
            st.error(
                f"Could not load matcher — index file missing.\n\n"
                f"`{e}`\n\n"
                "Make sure the precomputed `.pkl` index files are present in the `input/` folder."
            )
            return
        except Exception as e:
            st.error(f"Failed to load matcher: {e}")
            return

        with st.spinner(f"Matching {len(sentences)} sentence(s)…"):
            try:
                results = matcher.match(sentences, top_k=top_k)
            except Exception as e:
                st.error(f"Matching failed: {e}")
                return

        st.success(f"Done — {len(sentences)} sentence(s) processed.")
        st.divider()
        render_results(sentences, results, top_k)


if __name__ == "__main__":
    main()
