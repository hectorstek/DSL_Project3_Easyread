import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import streamlit as st
from PIL import Image
import config

IMAGE_DIR = os.path.join(
    ROOT, "dataset", "easyread-retrieval-dataset", "data"
)

@st.cache_resource(show_spinner=False)
def load_matcher():
    from matchers.vlm_matcher import VLMMatcher
    return VLMMatcher()

# Uncomment to use Hybrid matcher
# @st.cache_resource(show_spinner=False)
# def load_matcher():
#     from matchers.hybrid_matcher import HybridMatcher
#     return HybridMatcher()


def find_image_path(filename: str) -> str | None:
    path = os.path.join(IMAGE_DIR, filename)
    return path if os.path.exists(path) else None


def render_results(sentences: list[str], results: list[list[str]], confidences: list[float]) -> None:
    for sentence, image_files, conf in zip(sentences, results, confidences):
        st.markdown(f"**{sentence}**")
        st.caption(f"Confidence: {conf}/10")
        for fname in image_files:
            img_path = find_image_path(fname)
            if img_path:
                st.image(Image.open(img_path), caption=fname, width=200)
            else:
                st.warning(f"Not found: {fname}")
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
        "Uses VLM-based re-ranking (Qwen2.5 intent extraction + Gemma 4 visual scoring)."
    )

    # Uncomment to use Hybrid matcher
    # st.write(
    #     "Enter plain-language sentences and see which pictograms our system retrieves. "
    #     "Uses Hybrid Matcher."
    # )

    # Sidebar
    with st.sidebar:
        st.header("Settings")
        st.info("**Matcher:** VLM (Qwen2.5 + Gemma 4)\n\nIntent extraction followed by visual re-ranking of top candidates.")
        st.divider()
        st.markdown("**Image directory**")
        st.code(IMAGE_DIR, language=None)
        if not os.path.isdir(IMAGE_DIR):
            st.error("Image directory not found.")

    # Main area
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
            with st.spinner("Loading matcher (first run downloads models and may take several minutes)…"):
                matcher = load_matcher()
        except FileNotFoundError as e:
            st.error(
                f"Could not load matcher — index file missing.\n\n"
                f"`{e}`\n\n"
                "Make sure `input/index_v5.pkl` is present in the project root."
            )
            return
        except Exception as e:
            st.error(f"Failed to load matcher: {e}")
            return

        with st.spinner(f"Matching {len(sentences)} sentence(s) — VLM re-ranking may take a while…"):
            try:
                results, confidences = matcher.match(sentences)
            except Exception as e:
                st.error(f"Matching failed: {e}")
                return

        st.success(f"Done - {len(sentences)} sentence(s) processed.")
        st.divider()
        render_results(sentences, results, confidences)


if __name__ == "__main__":
    main()
