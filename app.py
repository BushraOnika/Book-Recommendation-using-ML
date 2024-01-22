import pickle
import streamlit as st
import numpy as np

# Load data
st.set_page_config(page_title='Book Recommender', page_icon=':books:')
import streamlit as st

# Load data
model_path = st.secrets["model_path"]
model = pickle.load(open(model_path, 'rb'))
book_names = pickle.load(open('books_name.pk1', 'rb'))
final_rating = pickle.load(open('merge_data.pk1', 'rb'))
book_pivot = pickle.load(open('book_pivot.pk1', 'rb'))

# Function to fetch poster URLs
def fetch_poster(suggestion):
    poster_url = []

    for book_id in suggestion:
        name = book_pivot.index[book_id]
        idx = np.where(final_rating['Book-Title'] == name)[0][0]
        url = final_rating.iloc[idx]['Image-URL-M']
        poster_url.append((name, url))

    return poster_url

# Function to recommend books
def recommend_book(book_name):
    if book_name in book_pivot.index:
        book_id = np.where(book_pivot.index == book_name)[0][0]
        distance, suggestion = model.kneighbors(book_pivot.iloc[book_id, :].values.reshape(1, -1), n_neighbors=6)
        poster_url = fetch_poster(suggestion)

        # Exclude the selected book from the recommendations
        exclude_book = book_pivot.index[book_id]
        recommended_books = []

        for i in suggestion[0]:
            if book_pivot.index[i] != exclude_book and book_pivot.index[i] not in recommended_books:
                recommended_books.append(book_pivot.index[i])

            if len(recommended_books) == 5:
                break

        return recommended_books, poster_url
    else:
        return [], []

# Streamlit layout
st.title('Book Recommendation System Using Machine Learning')
st.sidebar.header('Select Book')
selected_books = st.sidebar.selectbox("Type or select a book from the dropdown", book_names)

if st.sidebar.button('Show Recommendation'):
    recommended_books, poster_url = recommend_book(selected_books)

    if recommended_books:
        st.subheader('Recommended Books:')
        col1, col2, col3, col4, col5 = st.columns(5)

        for i, (book, url) in enumerate(poster_url):
            with eval(f"col{i % 5 + 1}"):  # Adjust the column index to fit your layout
                st.text(book)
                st.image(url, caption=book)
    else:
        st.warning("Selected book not found. Please choose another book.")

# Display top 10 most popular books when the website is loaded
st.subheader('Top Books for Recommendation:')
col1, col2, col3, col4, col5 = st.columns(5)
for i, (index, row) in enumerate(final_rating.sort_values(by='num_of_rating', ascending=False).sample(5).iterrows()):
    with eval(f"col{i % 5 + 1}"):  # Adjust the column index to fit your layout
        st.image(row['Image-URL-M'], caption=row['Book-Title'])
