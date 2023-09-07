import streamlit as st
from ..utils.pageSetup import local_css, set_streamlit_page_config

set_streamlit_page_config()
local_css("smartsim/_core/_dashboard/assets/style.scss")

st.header("Welcome to the SmartSim Dashboard Help Page")
st.write("")
st.markdown(
    """We're here to guide you through the features and functionalities of the SmartSim Dashboard, 
    designed to enhance your experience with simulation experiments. This guide will help 
    you navigate through the dashboard, understand its capabilities, and make the most of its powerful features."""
)
st.write("")

with st.expander(label="Experiment Overview"):
    st.markdown(
        """Discover comprehensive insights about your experiment's entities within the Experiment Overview section. 
        Get access to configuration details, comprehensive logs, real-time statuses, and relevant information regarding any colocated databases, if applicable.
        """
    )
    st.markdown(
        """
        To access detailed information about experiments, models, orchestrators, and ensembles, please refer to their respective documentation pages below:  
        - [Experiments](https://www.craylabs.org/docs/experiment.html#)  
        - [Models](https://www.craylabs.org/docs/experiment.html#model)  
        - [Orchestrators](https://www.craylabs.org/docs/orchestrator.html)  
        - [Ensembles](https://www.craylabs.org/docs/experiment.html#ensemble)"""
    )
    st.write("")

st.write("")

with st.expander(label="Workflow Telemetry"):
    st.markdown(
        """Gain deeper insights into your orchestrators in this section. 
        Simply select the orchestrator you wish to analyze from the database summary table, 
        and unlock valuable information about its memory usage, clients, and keys."""
    )

st.write("")

with st.expander(label="Support"):
    st.write(
        "Should you encounter any issues or require assistance while using the SmartSim Dashboard, we're here to help!"
    )
    st.markdown(
        """The complete SmartSim documentation can be found [here](https://www.craylabs.org/docs/overview.html).  
        You can also contact us at contact.us@idk.yet"""
    )
