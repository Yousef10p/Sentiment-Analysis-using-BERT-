import os
import random
import requests
import streamlit as st
import torch

from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification
)

# -------------------------
# CONFIG
# -------------------------

MODEL_PATH = "US10F/bert-imdb-sentiment"
TMDB_API_KEY = os.environ.get("TMDB_API_KEY")

# -------------------------
# LOAD MODEL
# -------------------------

@st.cache_resource
def load_model():
    tokenizer = BertTokenizerFast.from_pretrained(MODEL_PATH)

    model = BertForSequenceClassification.from_pretrained(
        MODEL_PATH
    )

    model.eval()

    return tokenizer, model


tokenizer, model = load_model()

# -------------------------
# SESSION STATE
# -------------------------

if "current_item" not in st.session_state:
    st.session_state.current_item = None

if "reviews_count" not in st.session_state:
    st.session_state.reviews_count = 0

if "positive_count" not in st.session_state:
    st.session_state.positive_count = 0

if "negative_count" not in st.session_state:
    st.session_state.negative_count = 0

# -------------------------
# TMDB FUNCTIONS
# -------------------------

@st.cache_data(ttl=3600)
def fetch_top_items():
    items = []
    for media_type in ("movie", "tv"):
        for page in range(1, 6):
            url = (
                f"https://api.themoviedb.org/3/"
                f"{media_type}/top_rated"
                f"?api_key={TMDB_API_KEY}"
                f"&page={page}"
            )
            data = requests.get(url).json()
            for item in data.get("results", []):
                if media_type == "tv":
                    item["title"] = item.get("name", "Unknown")
                    item["release_date"] = item.get("first_air_date", "N/A")
                item["media_type"] = media_type
                items.append(item)
    return items


def get_random_movie():
    items = fetch_top_items()
    if not items:
        return None
    return random.choice(items)


def load_new_movie():
    st.session_state.current_item = get_random_movie()


if st.session_state.current_item is None:
    load_new_movie()

movie = st.session_state.current_item

# -------------------------
# UI
# -------------------------

st.title("🎬")

media_label = "Movie" if movie.get("media_type") == "movie" else "TV Series"
st.subheader(f"{movie['title']} ({media_label})")

col1, col2 = st.columns([1, 2])

with col1:

    if movie.get("poster_path"):
        st.image(
            f"https://image.tmdb.org/t/p/w500"
            f"{movie['poster_path']}"
        )

with col2:

    st.write(
        f"⭐ Rating: {movie['vote_average']:.1f}/10"
    )

    st.write(
        f"📅 Release: {movie['release_date']}"
    )

    st.write(movie["overview"])

# -------------------------
# REVIEW INPUT
# -------------------------

review = st.text_area(
    "Write your review",
    height=180
)

col1, col2 = st.columns(2)

# -------------------------
# SKIP
# -------------------------

with col1:

    if st.button("⏭ Skip"):
        load_new_movie()
        st.rerun()

# -------------------------
# SUBMIT
# -------------------------

with col2:

    if st.button("🚀 Submit Review"):

        if review.strip() == "":
            st.warning(
                "Please write a review first."
            )
            st.stop()

        inputs = tokenizer(
            review,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt"
        )

        with torch.no_grad():

            outputs = model(**inputs)

            probs = torch.softmax(
                outputs.logits,
                dim=1
            )

            prediction = torch.argmax(
                probs,
                dim=1
            ).item()

        confidence = (
            probs[0][prediction].item()
        )

        labels = {
            0: "Negative 😞",
            1: "Positive 😊"
        }

        st.subheader("Prediction")

        if prediction == 1:
            st.success(labels[prediction])
        else:
            st.error(labels[prediction])

        st.write(
            f"Confidence: {confidence:.2%}"
        )

        st.session_state.reviews_count += 1

        if prediction == 1:
            st.session_state.positive_count += 1
        else:
            st.session_state.negative_count += 1

# -------------------------
# STATS
# -------------------------

st.divider()

st.subheader("Session Statistics")

c1, c2, c3 = st.columns(3)

c1.metric(
    "Reviews",
    st.session_state.reviews_count
)

c2.metric(
    "Positive",
    st.session_state.positive_count
)

c3.metric(
    "Negative",
    st.session_state.negative_count
)

# -------------------------
# NEXT MOVIE
# -------------------------

if st.button("🎲 New Random Title"):
    load_new_movie()
    st.rerun()