import streamlit as st
from pymongo import MongoClient

try:
    client = MongoClient(st.secrets["MONGO_URI"])
    db = client.test_database

    db.test_collection.insert_one({"message": "MongoDB Connected"})

    st.success("MongoDB connection successful!")
except Exception as e:
    st.error(f"Error: {e}")