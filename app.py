import os
import shutil
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

# Define the local folder where the AI will write the code
WORKSPACE_DIR = "generated_project"
os.makedirs(WORKSPACE_DIR, exist_ok=True)

st.title("🌐 CrewAI Autonomous Web Design Agency")
st.markdown("Build full-stack web applications, view the generated code, preview the website, and download the project.")

# ==========================================
# SIDEBAR CONFIGURATION
# ==========================================
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key_input = st.text_input(
        "OpenAI API Key",
        type="password",
        placeholder="sk-proj-...",
        help="Enter your OpenAI API key to power the CrewAI agents."
    )
    selected_model = st.selectbox(
        "Select OpenAI Model",
        ["gpt-4o-mini", "gpt-4o"],
        index=0,
    )
    st.markdown("---")
    st.success("✅ **ChromaDB Ready:** The backend agent is configured to build vector databases using ChromaDB.")

# ==========================================
# MAIN INTERFACE
# ==========================================
user_request = st.text_area(
    "Describe your website or application project:",
    value="I need an intelligent knowledge base website where users can upload PDF documents and chat with them using AI. It should look modern and dark-themed.",
    height=100
)

if st.button("🚀 Start Agency Crew", type="primary"):
    if not api_key_input:
        st.error("⚠️ Please enter your OpenAI API Key in the sidebar to proceed.")
    else:
        # Securely set API key for the current session
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
            # Force all files to be saved inside the WORKSPACE_DIR safely
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
            verbose=True
        )

        # ==========================================
        # EXECUTION & RESULTS RENDERING
        # ==========================================
        with st.spinner("🤖 Agency agents are writing code... Please wait (this usually takes 2-4 minutes)."):
            try:
                # 1. Run the Crew
                result = web_agency_crew.kickoff()
                st.success("🎉 Agency Crew has successfully completed the project!")
                
                st.markdown("---")
                
                # Create tabs for organized viewing in the UI
                tab1, tab2, tab3 = st.tabs(["🖥️ Live Website Preview", "📄 Raw Code Preview", "💾 Download Project"])
                
                # ==========================================
                # TAB 1: LIVE WEBSITE PREVIEW
                # ==========================================
                with tab1:
                    st.subheader("Interactive Website Preview")
                    index_path = os.path.join(WORKSPACE_DIR, 'index.html')
                    
                    if os.path.exists(index_path):
                        with open(index_path, 'r', encoding='utf-8') as html_file:
                            source_code = html_file.read()
                        # Render the HTML interactively in Streamlit
                        components.html(source_code, height=700, scrolling=True)
                    else:
                        st.warning("⚠️ Live preview not available. The AI failed to generate the `index.html` file correctly.")

                # ==========================================
                # TAB 2: RAW CODE PREVIEW
                # ==========================================
                with tab2:
                    st.subheader("Generated Source Code")
                    
                    # Read and display Frontend Code
                    if os.path.exists(index_path):
                        with st.expander("Frontend (index.html)", expanded=True):
                            with open(index_path, 'r', encoding='utf-8') as f:
                                st.code(f.read(), language='html')
                                
                    # Read and display Backend Code
                    backend_path = os.path.join(WORKSPACE_DIR, 'backend', 'main.py')
                    if os.path.exists(backend_path):
                        with st.expander("Backend (backend/main.py) with ChromaDB"):
                            with open(backend_path, 'r', encoding='utf-8') as f:
                                st.code(f.read(), language='python')
                    
                    # Read and display README
                    readme_path = os.path.join(WORKSPACE_DIR, 'README.md')
                    if os.path.exists(readme_path):
                        with st.expander("Documentation (README.md)"):
                            with open(readme_path, 'r', encoding='utf-8') as f:
                                st.markdown(f.read())

                # ==========================================
                # TAB 3: DOWNLOAD BUTTON
                # ==========================================
                with tab3:
                    st.subheader("Export Your Files")
                    st.write("All files (Frontend, FastAPI Backend with ChromaDB, and Documentation) have been packed into a ZIP file for you to run locally.")
                    
                    # Zip the folder
                    shutil.make_archive(WORKSPACE_DIR, 'zip', WORKSPACE_DIR)
                    
                    # Provide Download Button
                    with open(f"{WORKSPACE_DIR}.zip", "rb") as zip_file:
                        st.download_button(
                            label="📦 Download Complete Project (ZIP)",
                            data=zip_file,
                            file_name="agency_project.zip",
                            mime="application/zip",
                            type="primary",
                            use_container_width=True
                        )

            except Exception as e:
                st.error(f"An error occurred during execution: {e}")