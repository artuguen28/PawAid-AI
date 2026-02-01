"""
Custom CSS for the PawAid AI Streamlit app.

Injected once via st.markdown() in main.py.
"""

GLOBAL_CSS = """
<style>
/* Urgency badge pill */
.urgency-badge {
    display: inline-block;
    padding: 0.25em 0.75em;
    border-radius: 1em;
    font-weight: 600;
    font-size: 0.85em;
    color: #fff;
    margin-bottom: 0.5em;
}

/* Source panel */
.source-item {
    padding: 0.3em 0;
    border-bottom: 1px solid rgba(128, 128, 128, 0.2);
    font-size: 0.9em;
}
.source-item:last-child {
    border-bottom: none;
}

/* Disclaimer */
.disclaimer-text {
    color: #888;
    font-size: 0.82em;
    font-style: italic;
    margin-top: 0.5em;
    padding-top: 0.5em;
    border-top: 1px solid rgba(128, 128, 128, 0.2);
}

/* Sidebar branding */
.sidebar-title {
    font-size: 1.4em;
    font-weight: 700;
    margin-bottom: 0.2em;
}
.sidebar-subtitle {
    font-size: 0.9em;
    color: #888;
    margin-bottom: 1em;
}
</style>
"""
