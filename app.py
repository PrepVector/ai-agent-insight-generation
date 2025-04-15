import streamlit as st
import pandas as pd
from utils import format_questions_to_text, parse_text_to_questions 
from ba_agent import run_business_analysis
from eda_agent import run_eda_analysis


def main():
    st.set_page_config(page_title="InsightBot", page_icon="🤖", layout="wide")
    st.title("InsightBot")
    st.markdown("<h3 style='color: green;'>AI-powered Exploratory Data Analysis Assistant</h3>", unsafe_allow_html=True)

    # File uploaders for dataset and metadata
    dataset_file = st.sidebar.file_uploader("Upload your dataset (CSV file)", type=['csv'])
    metadata_file = st.sidebar.file_uploader("Upload your metadata (TXT file) (Optional)", type=['txt'])

    #Future Implementation - API

    # Add a reload button to the sidebar
    if st.sidebar.button("Reload Page"):
        st.session_state.clear()
        st.rerun()
    st.sidebar.markdown("### Reload the page to reset the state.")

    # Initialize session state for questions
    if 'questions_dict' not in st.session_state:
        st.session_state.questions_dict = None
    if 'questions_text' not in st.session_state:
        st.session_state.questions_text = ""
    if 'generated_questions' not in st.session_state:
        st.session_state.generated_questions = False

    if dataset_file:
        # Load dataset
        dataset = pd.read_csv(dataset_file)

        # Create two columns for better layout
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("Dataset Preview")
            st.write("\n")
            st.dataframe(dataset.head())

        # Load metadata
        metadata = metadata_file.read().decode() if metadata_file else ""
        with col2:
            st.subheader("Metadata")
            st.text_area("Metadata content", metadata, height=250, disabled=True, label_visibility="collapsed")

        # Generate EDA questions
        if st.button("Generate EDA Questions") or st.session_state.generated_questions:
            if not st.session_state.generated_questions:
                with st.spinner("Generating questions..."):
                    st.session_state.questions_dict = run_business_analysis(dataset, metadata)
                    st.session_state.questions_text = format_questions_to_text(st.session_state.questions_dict)
                    st.session_state.generated_questions = True
                st.success("EDA questions generated successfully!")

            # Editable text area for questions
            edited_questions = st.text_area("Edit your EDA questions below", value=st.session_state.questions_text, height=400)

            # Update the session state if the questions were edited
            if edited_questions != st.session_state.questions_text:
                st.session_state.questions_text = edited_questions

            # Button to run EDA analysis with the edited questions
            if st.button("Run EDA Analysis"):
                with st.spinner("Running EDA analysis..."):
                    # Parse the edited text back to structured format
                    updated_questions = parse_text_to_questions(st.session_state.questions_text)

                    # Run the EDA analysis
                    results = run_eda_analysis(dataset,updated_questions)
                    st.success("EDA analysis completed!")
                    # st.markdown("#### Technical Report Preview")
                    # st.markdown(results, unsafe_allow_html=True)
                    st.download_button(
                        label="Download Full Technical Report as Markdown",
                        data=results,
                        file_name="report/technical_report.md",
                        mime="text/markdown"
                    )
if __name__ == "__main__":
    main()