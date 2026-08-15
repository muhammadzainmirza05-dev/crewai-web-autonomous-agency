import os
import sys
import shutil
import threading
import http.server
import socketserver
import re
from functools import partial
import streamlit as st
import streamlit.components.v1 as components
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from langchain.tools import tool

# ==========================================
# PAGE CONFIGURATION & WORKSPACE
# ==========================================
st.set_page_config(
    page_title="CrewAI Web Design Agency",
    page_icon="🌐",
    layout="wide"
)

WORKSPACE_DIR = "generated_project"
os.makedirs(WORKSPACE_DIR, exist_ok=True)
PREVIEW_PORT = 8765

# ==========================================
# MODERN CSS STYLING
# ==========================================
MODERN_CSS = """
<style>
    * {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    [data-testid="stMainBlockContainer"] {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    }
    
    .stTabs [role="tablist"] button {
        font-weight: 600;
        font-size: 15px;
        border-radius: 8px 8px 0 0;
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        color: #94a3b8;
        border: none;
        padding: 12px 24px;
        transition: all 0.3s ease;
    }
    
    .stTabs [role="tablist"] button[aria-selected="true"] {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: #fff;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
    }
    
    .agent-item {
        padding: 12px 16px;
        margin: 8px 0;
        border-radius: 8px;
        border-left: 4px solid #64748b;
        background: linear-gradient(90deg, #1e293b 0%, #0f172a 100%);
        font-size: 14px;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .agent-item.active {
        border-left-color: #3b82f6;
        background: linear-gradient(90deg, #1e40af 0%, #1e293b 100%);
        color: #60a5fa;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
    }
    
    .agent-item.completed {
        border-left-color: #10b981;
        color: #86efac;
    }
    
    .stButton button {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        border: none;
        padding: 12px 32px;
        font-weight: 600;
        border-radius: 8px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
    }
    
    .stButton button:hover {
        box-shadow: 0 6px 20px rgba(59, 130, 246, 0.5);
        transform: translateY(-2px);
    }
    
    .log-container {
        background: #0f172a;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 12px;
        max-height: 400px;
        overflow-y: auto;
        font-size: 12px;
        font-family: 'Courier New', monospace;
    }
    
    .stSuccess {
        background: linear-gradient(90deg, rgba(16, 185, 129, 0.1) 0%, rgba(5, 150, 105, 0.1) 100%);
        border: 1px solid #10b981;
    }
</style>
"""

st.markdown(MODERN_CSS, unsafe_allow_html=True)

st.title("🌐 CrewAI Autonomous Web Design Agency")
st.markdown("✨ Build stunning full-stack web apps with AI agents working in parallel.")
st.markdown("---")

# ==========================================
# LOCAL PREVIEW SERVER (runs once per session)
# ==========================================
def start_preview_server():
    """Spin up a simple HTTP server to serve the generated project on localhost."""
    if not st.session_state.get("server_started"):
        try:
            handler = partial(
                http.server.SimpleHTTPRequestHandler,
                directory=os.path.abspath(WORKSPACE_DIR)
            )
            # allow_reuse_address prevents "Address already in use" on re-runs
            socketserver.TCPServer.allow_reuse_address = True
            httpd = socketserver.TCPServer(("", PREVIEW_PORT), handler)
            t = threading.Thread(target=httpd.serve_forever, daemon=True)
            t.start()
            st.session_state["server_started"] = True
            st.session_state["httpd"] = httpd
        except OSError:
            # Port already bound — server is still running from previous run
            st.session_state["server_started"] = True

# ==========================================
# AGENT STATE TRACKER
# ==========================================
AGENTS_LIST = [
    "Product Manager & Prompt Engineer",
    "Lead UI/UX Web Designer",
    "Senior Front-End Engineer",
    "Senior Back-End Engineer",
    "Release Manager & File System Operator"
]

def detect_agent_from_log(text):
    """Extract agent name from CrewAI verbose output."""
    for agent in AGENTS_LIST:
        if agent.lower() in text.lower():
            return agent
    return None

# ==========================================
# STDOUT → SIDEBAR LOGGER WITH AGENT TRACKING
# ==========================================
class SidebarLogger:
    """Intercepts CrewAI's verbose output and updates sidebar agent status."""
    def __init__(self, container, agent_status_container):
        self.container = container
        self.agent_status_container = agent_status_container
        self._original = sys.stdout
        if "sidebar_logs" not in st.session_state:
            st.session_state.sidebar_logs = []
        if "completed_agents" not in st.session_state:
            st.session_state.completed_agents = set()
        if "active_agent" not in st.session_state:
            st.session_state.active_agent = None

    def write(self, text):
        self._original.write(text)
        stripped = text.strip()
        if stripped:
            st.session_state.sidebar_logs.append(stripped)
            
            # Detect agent changes
            detected_agent = detect_agent_from_log(text)
            if detected_agent and detected_agent != st.session_state.active_agent:
                st.session_state.active_agent = detected_agent
            
            # Mark agent as completed if we see "finished"
            if "finished" in text.lower() and st.session_state.active_agent:
                st.session_state.completed_agents.add(st.session_state.active_agent)
            
            # Update sidebar display
            self._render_agent_status()
            
            # Show logs
            display = "\n".join(st.session_state.sidebar_logs[-15:])
            self.container.markdown(f"```\n{display}\n```", unsafe_allow_html=True)

    def _render_agent_status(self):
        """Render agent status in the sidebar."""
        html = "<div style='margin-top: 12px;'>"
        for agent in AGENTS_LIST:
            status = "✅ Completed" if agent in st.session_state.completed_agents else ("🔄 Working..." if agent == st.session_state.active_agent else "⏳ Queued")
            css_class = "active" if agent == st.session_state.active_agent else ("completed" if agent in st.session_state.completed_agents else "")
            html += f'<div class="agent-item {css_class}"><strong>{agent}</strong><br/><small>{status}</small></div>'
        html += "</div>"
        self.agent_status_container.markdown(html, unsafe_allow_html=True)

    def flush(self):
        self._original.flush()

    def restore(self):
        sys.stdout = self._original


# ==========================================
# SIDEBAR CONFIGURATION
# ==========================================
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key_input = st.text_input(
        "OpenAI API Key",
        type="password",
        placeholder="sk-proj-...",
    )
    selected_model = st.selectbox(
        "Select OpenAI Model",
        ["gpt-4o-mini", "gpt-4o"],
        index=0,
    )
    st.markdown("---")
    st.info("💾 **ChromaDB Ready** — Vector database configured for backend")
    st.markdown("---")
    
    st.header("🤖 Agent Pipeline")
    agent_status_container = st.empty()
    
    # Display agent list at startup
    if not st.session_state.get("sidebar_logs"):
        html = "<div style='margin-top: 12px;'>"
        for agent in AGENTS_LIST:
            html += f'<div class="agent-item"><strong>{agent}</strong><br/><small>⏳ Queued</small></div>'
        html += "</div>"
        agent_status_container.markdown(html, unsafe_allow_html=True)
    
    st.markdown("---")
    st.header("📋 Live Logs")
    sidebar_log_container = st.empty()

# ==========================================
# MAIN INTERFACE
# ==========================================
st.markdown("## 📝 Project Brief")
user_request = st.text_area(
    "Describe what you want to build:",
    value="I need an intelligent knowledge base website where users can upload PDF documents and chat with them using AI. It should look modern and dark-themed.",
    height=120,
    placeholder="E.g., A SaaS dashboard for analytics, an AI chatbot interface, an e-commerce store..."
)

col1, col2, col3 = st.columns([1, 1, 2])
with col1:
    start_button = st.button("🚀 Start Agency Crew", type="primary", use_container_width=True)

if start_button:
    if not api_key_input:
        st.error("⚠️ Please enter your OpenAI API Key in the sidebar.")
        st.stop()

    # Reset sidebar logs on each new run
    st.session_state.sidebar_logs = []
    sidebar_log_container.empty()

    os.environ["OPENAI_API_KEY"] = api_key_input

    try:
        llm = ChatOpenAI(model=selected_model, api_key=api_key_input)
    except Exception as e:
        st.error(f"Failed to initialize LLM: {e}")
        st.stop()

    # ==========================================
    # CUSTOM FILE WRITER TOOL
    # ==========================================
    @tool("File Writer Tool")
    def file_writer(file_path: str, content: str) -> str:
        """Useful to write raw code or text to a specified file path.
        Requires a file_path (string) and the file content (string)."""
        safe_path = os.path.join(WORKSPACE_DIR, file_path)
        os.makedirs(os.path.dirname(safe_path) or '.', exist_ok=True)
        with open(safe_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return f"Success! File saved to {safe_path}"

    # ==========================================
    # AGENTS
    # ==========================================
    requirements_architect = Agent(
        role="Product Manager & Prompt Engineer",
        goal="Transform raw user input into a comprehensive, detailed master project specification.",
        backstory="You are an expert Product Manager. You listen to vague user requests and expand them into detailed specifications that guide designers and developers.",
        llm=llm, allow_delegation=False, verbose=True
    )
    ui_ux_designer = Agent(
        role="Lead UI/UX Web Designer",
        goal="Design modern UI/UX layouts with detailed color palettes and typography specifications.",
        backstory="You are a UI/UX designer. You create beautiful wireframes and layout structures that inspire frontend developers.",
        llm=llm, allow_delegation=False, verbose=True
    )
    frontend_developer = Agent(
        role="Senior Front-End Engineer",
        goal="Build responsive frontend code. MUST output a single index.html file containing HTML, Tailwind CSS via CDN, and JS logic.",
        backstory="You build scalable web apps. To allow for instant live previews, you always bundle your CSS (Tailwind CDN) and JavaScript directly into a comprehensive index.html file.",
        llm=llm, allow_delegation=False, verbose=True
    )
    backend_developer = Agent(
        role="Senior Back-End Engineer",
        goal="Design and implement backend APIs using Python, FastAPI, and ChromaDB.",
        backstory="You are a senior backend engineer. You analyze frontend requirements to build secure backend systems utilizing ChromaDB for vector storage and semantic search.",
        llm=llm, allow_delegation=False, verbose=True
    )
    devops_organizer = Agent(
        role="Release Manager & File System Operator",
        goal="Use the File Writer Tool to save the exact code generated by the engineers to the disk.",
        backstory="You ensure all files are saved correctly so the user can download them and view the raw code.",
        llm=llm, allow_delegation=False, verbose=True, tools=[file_writer]
    )

    # ==========================================
    # TASKS
    # ==========================================
    task_requirements = Task(
        description=f"Analyze the user request: '{user_request}'. Expand it into a Master Project Specification.",
        expected_output="A Markdown document containing the Master Project Specification.",
        agent=requirements_architect
    )
    task_ui_ux = Task(
        description="Using the specification, design the complete UI/UX. List specific hex color codes, font families, and component layouts.",
        expected_output="A comprehensive UI/UX design guide.",
        agent=ui_ux_designer
    )
    task_frontend = Task(
        description="Write the actual frontend code based on the UI/UX design. You MUST write it as a single file incorporating Tailwind CSS (via CDN script tag) and JS. Provide the raw code.",
        expected_output="Raw frontend code meant for an index.html file.",
        agent=frontend_developer
    )
    task_backend = Task(
        description="Write a Python FastAPI backend that includes ChromaDB integration for vector storage/search to serve the frontend.",
        expected_output="Raw Python FastAPI backend code utilizing ChromaDB.",
        agent=backend_developer
    )
    task_devops = Task(
        description="""Review the generated code and use the File Writer Tool to save it EXACTLY as follows:
        1. Save the frontend code as 'index.html' (in the root, NO subfolders).
        2. Save the backend code as 'backend/main.py'.
        3. Save instructions as 'README.md'.""",
        expected_output="A final confirmation message that files were saved.",
        agent=devops_organizer
    )

    web_agency_crew = Crew(
        agents=[requirements_architect, ui_ux_designer, frontend_developer, backend_developer, devops_organizer],
        tasks=[task_requirements, task_ui_ux, task_frontend, task_backend, task_devops],
        process=Process.sequential,
        verbose=True,
    )

    # ==========================================
    # EXECUTION — stdout piped to sidebar with agent tracking
    # ==========================================
    logger = SidebarLogger(sidebar_log_container, agent_status_container)
    sys.stdout = logger   # 🔴 Start capturing

    with st.spinner("🤖 Agency agents are working... (2–4 mins)"):
        try:
            result = web_agency_crew.kickoff()
        except Exception as e:
            sys.stdout = logger.restore() or sys.stdout
            st.error(f"❌ Execution failed: {e}")
            st.stop()
        finally:
            logger.restore()  # ✅ Always restore stdout

    # Start the local preview server now that files exist
    start_preview_server()

    st.success("🎉 Agency Crew has successfully completed the project!")
    st.markdown("---")

    # ==========================================
    # RESULTS — TABS
    # ==========================================
    tab1, tab2, tab3 = st.tabs(["🖥️ Live Website Preview", "📄 Raw Code Preview", "💾 Download Project"])

    index_path    = os.path.join(WORKSPACE_DIR, 'index.html')
    backend_path  = os.path.join(WORKSPACE_DIR, 'backend', 'main.py')
    readme_path   = os.path.join(WORKSPACE_DIR, 'README.md')

    # ---- TAB 1: LOCALHOST IFRAME PREVIEW ----
    with tab1:
        st.subheader("🌐 Live Website Preview")
        if os.path.exists(index_path):
            preview_url = f"http://localhost:{PREVIEW_PORT}/index.html"
            st.markdown(f"<p style='font-size: 13px; color: #94a3b8;'>📡 Serving from: <code>{preview_url}</code></p>", unsafe_allow_html=True)
            
            # Embed iframe with proper styling
            iframe_html = f"""
            <iframe 
                src="{preview_url}" 
                style="width: 100%; height: 700px; border: 1px solid #334155; border-radius: 8px; background: white;"
                allow="autoplay; microphone; camera; accelerometer; gyroscope; magnetometer">
            </iframe>
            """
            components.html(iframe_html, height=750, scrolling=True)
        else:
            st.warning("⚠️ `index.html` was not generated. Check the agent logs in the sidebar for errors.")

    # ---- TAB 2: RAW CODE ----
    with tab2:
        st.subheader("📝 Generated Source Code")
        
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("**Frontend Files:**")
        with col2:
            st.markdown("**Backend & Docs:**")
        
        if os.path.exists(index_path):
            with st.expander("🎨 Frontend (index.html)", expanded=True):
                with open(index_path, 'r', encoding='utf-8') as f:
                    code = f.read()
                    st.code(code, language='html')
                    file_size = len(code) / 1024
                    st.caption(f"📊 {file_size:.1f} KB")

        if os.path.exists(backend_path):
            with st.expander("🔧 Backend (backend/main.py)"):
                with open(backend_path, 'r', encoding='utf-8') as f:
                    code = f.read()
                    st.code(code, language='python')
                    file_size = len(code) / 1024
                    st.caption(f"📊 {file_size:.1f} KB")

        if os.path.exists(readme_path):
            with st.expander("📚 Documentation (README.md)"):
                with open(readme_path, 'r', encoding='utf-8') as f:
                    st.markdown(f.read())

    # ---- TAB 3: DOWNLOAD ----
    with tab3:
        st.subheader("💾 Export Your Project")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("""
            ✅ **All files are ready!**
            
            Your complete project includes:
            - 🎨 **index.html** — Responsive frontend with Tailwind CSS
            - 🔧 **backend/main.py** — FastAPI server + ChromaDB integration
            - 📚 **README.md** — Setup & deployment guide
            """)
        
        with col2:
            st.markdown("")  # spacer
        
        shutil.make_archive(WORKSPACE_DIR, 'zip', WORKSPACE_DIR)
        with open(f"{WORKSPACE_DIR}.zip", "rb") as zip_file:
            st.download_button(
                label="📦  Download ZIP",
                data=zip_file,
                file_name="agency_project.zip",
                mime="application/zip",
                type="primary",
                use_container_width=True
            )
        
        st.markdown("---")
        st.markdown("""
        **Next Steps:**
        1. Extract the ZIP file
        2. Install backend dependencies: `pip install -r requirements.txt`
        3. Run backend: `python backend/main.py`
        4. Open `index.html` in your browser
        """)
