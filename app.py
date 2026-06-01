"""Streamlit frontend for Food-101 image classification."""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from predict import load_trained_model, predict_image, predict_uploaded_file
from utils import (
    CLASS_LABELS_PATH,
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    IMAGE_SIZE,
    MODEL_PATH,
    OUTPUTS_DIR,
    load_class_labels,
    prediction_result_to_json,
)


SUPPORTED_FOOD_LABELS = [
    "apple_pie",
    "baby_back_ribs",
    "baklava",
    "beef_carpaccio",
    "beef_tartare",
    "beet_salad",
    "beignets",
    "bibimbap",
    "bread_pudding",
    "breakfast_burrito",
    "bruschetta",
    "caesar_salad",
    "cannoli",
    "caprese_salad",
    "carrot_cake",
    "ceviche",
    "cheesecake",
    "cheese_plate",
    "chicken_curry",
    "chicken_quesadilla",
    "chicken_wings",
    "chocolate_cake",
    "chocolate_mousse",
    "churros",
    "clam_chowder",
    "club_sandwich",
    "crab_cakes",
    "creme_brulee",
    "croque_madame",
    "cup_cakes",
    "deviled_eggs",
    "donuts",
    "dumplings",
    "edamame",
    "eggs_benedict",
    "escargots",
    "falafel",
    "filet_mignon",
    "fish_and_chips",
    "foie_gras",
    "french_fries",
    "french_onion_soup",
    "french_toast",
    "fried_calamari",
    "fried_rice",
    "frozen_yogurt",
    "garlic_bread",
    "gnocchi",
    "greek_salad",
    "grilled_cheese_sandwich",
    "grilled_salmon",
    "guacamole",
    "gyoza",
    "hamburger",
    "hot_and_sour_soup",
    "hot_dog",
    "huevos_rancheros",
    "hummus",
    "ice_cream",
    "lasagna",
    "lobster_bisque",
    "lobster_roll_sandwich",
    "macaroni_and_cheese",
    "macarons",
    "miso_soup",
    "mussels",
    "nachos",
    "omelette",
    "onion_rings",
    "oysters",
    "pad_thai",
    "paella",
    "pancakes",
    "panna_cotta",
    "peking_duck",
    "pho",
    "pizza",
    "pork_chop",
    "poutine",
    "prime_rib",
    "pulled_pork_sandwich",
    "ramen",
    "ravioli",
    "red_velvet_cake",
    "risotto",
    "samosa",
    "sashimi",
    "scallops",
    "seaweed_salad",
    "shrimp_and_grits",
    "spaghetti_bolognese",
    "spaghetti_carbonara",
    "spring_rolls",
    "steak",
    "strawberry_shortcake",
    "sushi",
    "tacos",
    "takoyaki",
    "tiramisu",
    "tuna_tartare",
    "waffles",
]


st.set_page_config(
    page_title="FoodLens | Food Classifier",
    page_icon=":fork_and_knife:",
    layout="wide",
    initial_sidebar_state="collapsed",
)


CUSTOM_CSS = """
<style>
    :root {
        --ink: #172018;
        --muted: #66716a;
        --panel: rgba(255, 255, 255, 0.88);
        --panel-strong: #ffffff;
        --line: rgba(51, 65, 85, 0.14);
        --green: #2f7d57;
        --tomato: #d65a31;
        --gold: #d79a2b;
        --blue: #2f6f9f;
    }
    .stApp {
        background:
            radial-gradient(circle at 12% 12%, rgba(215, 154, 43, 0.16), transparent 28%),
            radial-gradient(circle at 92% 2%, rgba(47, 111, 159, 0.14), transparent 26%),
            linear-gradient(135deg, #fbfaf5 0%, #f2f7f0 48%, #f7f3ea 100%);
        color: var(--ink);
    }
    .main .block-container {
        max-width: 1160px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    [data-testid="stSidebar"] {
        background: #f7f4eb;
    }
    div[data-testid="stFileUploader"] {
        margin-top: 0.35rem;
    }
    div[data-testid="stFileUploader"] > label {
        color: var(--ink);
        font-weight: 750;
    }
    div[data-testid="stFileUploader"] section {
        min-height: 106px;
        border: 1px dashed rgba(47, 125, 87, 0.62);
        background:
            linear-gradient(135deg, rgba(47, 125, 87, 0.16), rgba(215, 154, 43, 0.10)),
            var(--panel);
        border-radius: 8px;
        box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.05), 0 18px 45px rgba(23, 32, 24, 0.08);
        transition: border-color 160ms ease, background 160ms ease, transform 160ms ease;
    }
    div[data-testid="stFileUploader"] section:hover {
        border-color: var(--gold);
        background:
            linear-gradient(135deg, rgba(47, 125, 87, 0.22), rgba(215, 154, 43, 0.15)),
            var(--panel);
        transform: translateY(-1px);
    }
    div[data-testid="stFileUploader"] section svg {
        color: var(--gold);
        fill: currentColor;
        opacity: 0.95;
    }
    div[data-testid="stFileUploader"] section [data-testid="stFileUploaderDropzoneInstructions"] span {
        color: var(--ink);
        font-weight: 750;
    }
    div[data-testid="stFileUploader"] section [data-testid="stFileUploaderDropzoneInstructions"] small {
        color: var(--muted);
        font-weight: 600;
    }
    div[data-testid="stFileUploader"] section button {
        min-height: 2.75rem;
        border-radius: 8px;
        border: 1px solid rgba(247, 247, 239, 0.16);
        background: #2f7d57;
        color: #ffffff;
        font-weight: 800;
        box-shadow: 0 10px 24px rgba(47, 125, 87, 0.22);
    }
    div[data-testid="stFileUploader"] section button:hover {
        border-color: rgba(215, 154, 43, 0.62);
        background: #3b8e65;
        color: #ffffff;
    }
    .stButton > button, .stDownloadButton > button {
        border-radius: 8px;
        border: 1px solid rgba(47, 125, 87, 0.28);
        background: #2f7d57;
        color: white;
        font-weight: 700;
    }
    div[data-testid="stTabs"] {
        margin-top: 0.3rem;
    }
    div[data-testid="stTabs"] [role="tablist"] {
        border-bottom: 1px solid var(--line);
        gap: 0.6rem;
    }
    div[data-testid="stTabs"] [role="tab"] {
        color: var(--muted);
        font-weight: 800;
        padding-left: 0;
        padding-right: 0;
    }
    div[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
        color: var(--gold);
    }
    .hero {
        display: grid;
        grid-template-columns: 1.2fr 0.8fr;
        gap: 2rem;
        align-items: stretch;
        margin-bottom: 1.6rem;
    }
    .hero-copy {
        padding: 1.1rem 0 0.75rem;
    }
    .brand {
        color: var(--green);
        font-size: 0.78rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0.65rem;
    }
    .hero h1 {
        color: var(--ink);
        font-size: clamp(2.2rem, 6vw, 4.6rem);
        line-height: 0.98;
        margin: 0 0 0.8rem;
        max-width: 780px;
    }
    .hero p {
        color: var(--muted);
        font-size: 1.05rem;
        line-height: 1.65;
        margin: 0;
        max-width: 620px;
    }
    .workflow {
        display: flex;
        flex-wrap: wrap;
        gap: 0.6rem;
        margin-top: 1.25rem;
    }
    .workflow-step {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        min-height: 2.35rem;
        padding: 0.42rem 0.72rem;
        border: 1px solid var(--line);
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.08);
        color: var(--ink);
        font-size: 0.92rem;
        font-weight: 750;
        white-space: nowrap;
    }
    .workflow-step span {
        display: inline-grid;
        place-items: center;
        width: 1.35rem;
        height: 1.35rem;
        border-radius: 999px;
        background: rgba(47, 125, 87, 0.16);
        color: var(--gold);
        font-size: 0.78rem;
        font-weight: 900;
    }
    .hero-plate {
        min-height: 250px;
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid var(--line);
        background:
            linear-gradient(145deg, rgba(23, 32, 24, 0.28), rgba(23, 32, 24, 0.04)),
            url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='900' height='700' viewBox='0 0 900 700'%3E%3Crect width='900' height='700' fill='%23e8dcc8'/%3E%3Ccircle cx='470' cy='350' r='240' fill='%23f8f3e8'/%3E%3Ccircle cx='470' cy='350' r='170' fill='%23d65a31'/%3E%3Ccircle cx='410' cy='310' r='42' fill='%23f9d65c'/%3E%3Ccircle cx='530' cy='285' r='30' fill='%232f7d57'/%3E%3Ccircle cx='560' cy='410' r='52' fill='%23f6ead8'/%3E%3Cpath d='M276 148c66 42 120 42 185 0 41-27 90-31 143 0' fill='none' stroke='%232f6f9f' stroke-width='18' stroke-linecap='round'/%3E%3Cpath d='M680 186c36 74 30 154-18 240' fill='none' stroke='%23172018' stroke-width='16' stroke-linecap='round' opacity='.7'/%3E%3C/svg%3E");
        background-size: cover;
        background-position: center;
        box-shadow: 0 24px 60px rgba(23, 32, 24, 0.12);
    }
    .prediction-panel, .placeholder-panel, .status-card, .metric-card {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 8px;
        box-shadow: 0 18px 45px rgba(23, 32, 24, 0.08);
    }
    .prediction-panel {
        padding: 1.25rem 1.35rem;
        margin-top: 1rem;
    }
    .prediction-label {
        color: var(--ink);
        font-size: clamp(1.7rem, 4vw, 2.6rem);
        font-weight: 750;
        margin: 0;
    }
    .confidence {
        color: var(--green);
        font-size: 1.1rem;
        font-weight: 650;
        margin-top: 0.2rem;
    }
    .rank-row {
        display: grid;
        grid-template-columns: 2rem 1fr auto;
        gap: 0.8rem;
        align-items: center;
        padding: 0.7rem 0;
        border-bottom: 1px solid rgba(51, 65, 85, 0.1);
    }
    .rank-row:last-child {
        border-bottom: 0;
    }
    .rank {
        width: 2rem;
        height: 2rem;
        display: grid;
        place-items: center;
        border-radius: 999px;
        background: rgba(47, 125, 87, 0.12);
        color: var(--green);
        font-weight: 800;
    }
    .rank-name {
        font-weight: 700;
        color: var(--ink);
    }
    .rank-score {
        color: var(--muted);
        font-weight: 700;
    }
    .placeholder-panel {
        padding: 1rem 1.1rem;
        height: 100%;
    }
    .placeholder-panel strong {
        color: var(--ink);
        display: block;
        margin-bottom: 0.25rem;
    }
    .status-card {
        padding: 1rem 1.1rem;
        margin: 1rem 0 1rem;
    }
    .status-card code {
        background: rgba(47, 125, 87, 0.10);
        border-radius: 6px;
        padding: 0.1rem 0.28rem;
    }
    .limit-note {
        padding: 0.95rem 1.05rem;
        margin: 0.8rem 0 1rem;
        border: 1px solid rgba(215, 154, 43, 0.36);
        border-radius: 8px;
        background: linear-gradient(135deg, rgba(215, 154, 43, 0.12), rgba(47, 125, 87, 0.08));
        color: var(--muted);
        font-size: 0.96rem;
        line-height: 1.55;
    }
    .limit-note strong {
        color: var(--ink);
    }
    .food-chip-grid {
        display: flex;
        flex-wrap: wrap;
        gap: 0.45rem;
        margin-top: 0.4rem;
    }
    .food-chip {
        display: inline-flex;
        align-items: center;
        min-height: 2rem;
        padding: 0.32rem 0.58rem;
        border: 1px solid var(--line);
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.06);
        color: var(--ink);
        font-size: 0.86rem;
        font-weight: 700;
    }
    .model-metrics {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 0.75rem;
        margin: 0.8rem 0 1rem;
    }
    .metric-card {
        padding: 0.9rem 1rem;
    }
    .metric-card span {
        display: block;
        color: var(--muted);
        font-size: 0.78rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    .metric-card strong {
        display: block;
        color: var(--ink);
        font-size: 1.05rem;
        margin-top: 0.24rem;
    }
    .footer-note {
        color: var(--muted);
        font-size: 0.9rem;
        margin-top: 1.5rem;
    }
    @media (prefers-color-scheme: dark) {
        :root {
            --ink: #f7f7ef;
            --muted: #c8d0c7;
            --panel: rgba(20, 28, 22, 0.82);
            --line: rgba(236, 240, 230, 0.16);
        }
        .stApp {
            background:
                radial-gradient(circle at 12% 12%, rgba(215, 154, 43, 0.12), transparent 28%),
                radial-gradient(circle at 92% 2%, rgba(47, 111, 159, 0.14), transparent 26%),
                linear-gradient(135deg, #101711 0%, #18251c 48%, #202117 100%);
        }
        [data-testid="stSidebar"] {
            background: #121a14;
        }
        .workflow-step {
            background: rgba(247, 247, 239, 0.045);
        }
        div[data-testid="stFileUploader"] section {
            background:
                linear-gradient(135deg, rgba(47, 125, 87, 0.18), rgba(215, 154, 43, 0.08)),
                rgba(20, 28, 22, 0.82);
        }
        div[data-testid="stFileUploader"] section:hover {
            background:
                linear-gradient(135deg, rgba(47, 125, 87, 0.24), rgba(215, 154, 43, 0.12)),
                rgba(20, 28, 22, 0.9);
        }
    }
    @media (max-width: 760px) {
        .hero {
            grid-template-columns: 1fr;
        }
        .hero-plate {
            min-height: 210px;
        }
        .model-metrics {
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }
    }
</style>
"""


@st.cache_resource(show_spinner=False)
def get_model():
    return load_trained_model(MODEL_PATH)


@st.cache_data(show_spinner=False)
def get_class_names():
    return load_class_labels(CLASS_LABELS_PATH)


def render_header() -> None:
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.markdown(
        """
        <div class="hero">
            <div class="hero-copy">
                <div class="brand">FoodLens classifier</div>
                <h1>Identify food from a single image.</h1>
                <p>Upload a dish photo or capture one from your camera. The app runs your trained model, shows the best match, and keeps the top alternatives visible for comparison.</p>
                <div class="workflow" aria-label="Prediction workflow">
                    <div class="workflow-step"><span>1</span>Add photo</div>
                    <div class="workflow-step"><span>2</span>Run model</div>
                    <div class="workflow-step"><span>3</span>Review confidence</div>
                </div>
            </div>
            <div class="hero-plate" aria-hidden="true"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_model_status(class_count: int | None = None) -> None:
    label_text = f"{class_count} classes loaded" if class_count else "Waiting for labels"
    model_state = "Ready" if MODEL_PATH.exists() else "Missing model"
    st.sidebar.markdown("### Project Status")
    st.sidebar.write(f"Model: **{model_state}**")
    st.sidebar.write(f"Labels: **{label_text}**")
    st.sidebar.write("Entry point: `streamlit run app.py`")


def format_food_name(label: str) -> str:
    return label.replace("_", " ").title()


def render_supported_foods() -> None:
    formatted_names = [format_food_name(label) for label in SUPPORTED_FOOD_LABELS]

    st.markdown(
        f"""
        <div class="limit-note">
            <strong>Supported food list:</strong> this model can only classify the
            {len(SUPPORTED_FOOD_LABELS)} trained categories below. Images outside this list may still
            receive a closest-match prediction, so treat those results carefully.
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("View supported foods", expanded=False):
        search_text = st.text_input(
            "Search supported foods",
            placeholder="Try pizza, sushi, waffles...",
            label_visibility="collapsed",
        ).strip().lower()

        visible_names = [
            name for name in formatted_names if search_text in name.lower()
        ]
        if not visible_names:
            st.caption("No supported foods match that search.")
            return

        chips = "".join(
            f'<span class="food-chip">{name}</span>' for name in visible_names
        )
        st.markdown(
            f'<div class="food-chip-grid">{chips}</div>',
            unsafe_allow_html=True,
        )


def render_model_details(class_count: int) -> None:
    accuracy_chart = OUTPUTS_DIR / "accuracy.png"
    loss_chart = OUTPUTS_DIR / "loss.png"

    st.markdown("#### Model details")
    st.markdown(
        f"""
        <div class="model-metrics">
            <div class="metric-card"><span>Architecture</span><strong>MobileNetV2</strong></div>
            <div class="metric-card"><span>Input size</span><strong>{IMAGE_SIZE[0]} x {IMAGE_SIZE[1]}</strong></div>
            <div class="metric-card"><span>Prediction labels</span><strong>{class_count}</strong></div>
            <div class="metric-card"><span>Training setup</span><strong>{DEFAULT_EPOCHS} epochs / batch {DEFAULT_BATCH_SIZE}</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if accuracy_chart.exists() and loss_chart.exists():
        with st.expander("View training accuracy and loss charts", expanded=False):
            chart_col_1, chart_col_2 = st.columns(2)
            with chart_col_1:
                render_image_preview(accuracy_chart, "Training accuracy")
            with chart_col_2:
                render_image_preview(loss_chart, "Training loss")
    else:
        st.caption(
            "Training charts are not available in this checkout. Run training to create "
            "`outputs/accuracy.png` and `outputs/loss.png`."
        )


def render_prediction(result: dict) -> None:
    st.markdown(
        f"""
        <div class="prediction-panel">
            <p class="prediction-label">{result["display_name"]}</p>
            <div class="confidence">{result["confidence_percent"]:.2f}% confidence</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### Top matches")
    for index, prediction in enumerate(result["top_predictions"], start=1):
        st.markdown(
            f"""
            <div class="rank-row">
                <div class="rank">{index}</div>
                <div class="rank-name">{prediction['display_name']}</div>
                <div class="rank-score">{prediction['confidence_percent']:.2f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.progress(prediction["confidence"])


def render_image_preview(image, caption: str = "Input image") -> None:
    if hasattr(image, "__fspath__"):
        image = str(image)

    try:
        st.image(image, caption=caption, use_container_width=True)
    except TypeError:
        st.image(image, caption=caption, use_column_width=True)


def render_bonus_panels(result: dict) -> None:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            f"""
            <div class="placeholder-panel">
                <strong>Estimated Calories</strong><br>
                {result["display_name"]}: nutrition lookup can be connected here.
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
            <div class="placeholder-panel">
                <strong>Nutrition Info</strong><br>
                Add serving size, protein, carbs, fat, and allergen data from a verified API.
            </div>
            """,
            unsafe_allow_html=True,
        )


def main() -> None:
    render_header()

    if not MODEL_PATH.exists() or not CLASS_LABELS_PATH.exists():
        render_model_status()
        st.markdown(
            """
            <div class="status-card">
                <strong>Model files are not in this checkout.</strong><br>
                Train locally with <code>python train.py</code>, then place <code>model/food_model.h5</code>
                and <code>model/class_labels.json</code> before running predictions.
                The trained model is designed for the Food-101 category set used by this project.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.stop()

    class_names = get_class_names()
    render_model_status(len(class_names))
    render_supported_foods()
    render_model_details(len(class_names))

    upload_tab, camera_tab = st.tabs(["Upload", "Camera"])
    with upload_tab:
        uploaded_file = st.file_uploader(
            "Drag and drop or browse for a food image",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
        )
    with camera_tab:
        camera_file = st.camera_input("Take a food photo")

    selected_file = uploaded_file or camera_file
    if selected_file is None:
        st.markdown(
            """
            <div class="status-card">
                <strong>Add a food image to begin.</strong><br>
                Clear, close-up photos usually produce the most useful confidence scores.
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    try:
        model = get_model()

        with st.spinner("Analyzing food image..."):
            if uploaded_file is not None:
                image, result = predict_uploaded_file(
                    selected_file,
                    model=model,
                    class_names=class_names,
                    top_k=3,
                )
            else:
                image = selected_file
                from utils import validate_image_file

                pil_image = validate_image_file(image)
                result = predict_image(
                    pil_image,
                    model=model,
                    class_names=class_names,
                    top_k=3,
                )
                image = pil_image

        preview_col, result_col = st.columns([0.92, 1.08], gap="large")
        with preview_col:
            render_image_preview(image)
        with result_col:
            render_prediction(result)

        st.markdown("#### Next data connections")
        render_bonus_panels(result)

        st.download_button(
            label="Download prediction result",
            data=prediction_result_to_json(result),
            file_name="food_prediction.json",
            mime="application/json",
        )
        st.markdown(
            "<div class='footer-note'>For GitHub deployment, keep datasets and large model files out of the repository unless your hosting plan supports them.</div>",
            unsafe_allow_html=True,
        )

    except ValueError as exc:
        st.error(str(exc))
    except Exception as exc:
        st.error("Prediction failed. Check that the model and labels match your trained dataset.")
        st.exception(exc)


if __name__ == "__main__":
    main()
