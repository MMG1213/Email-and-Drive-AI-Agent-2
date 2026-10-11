#main_app.py
import streamlit as st
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
import asyncio
import warnings
import threading
import os
from datetime import datetime
from pathlib import Path

# Import our modules
import database as db
import auth
import oauth_handler as oauth

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Page config
st.set_page_config(
    page_title="Smart Email and Drive Assistant",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Model configurations
MODELS = {
    "groq": {
        "name": "Llama 4 Scout (Groq)",
        "model_id": "meta-llama/llama-4-scout-17b-16e-instruct",
        "icon": "⚡",
        "description": "Fast and efficient",
        "temperature": 0
    },
    "gemini": {
        "name": "Gemini 2.5 Flash",
        "model_id": "gemini-2.5-flash",
        "icon": "✨",
        "description": "Powerful and accurate",
        "temperature": 0
    }
}

BOT_PROMPT = """
You are a file and email search assistant. You have access to tools that search Gmail, Google Drive, and the local computer.

**ABSOLUTE RULES — NEVER BREAK THESE:**
1. You MUST use the built-in tool-calling mechanism for EVERY user request. Do not just type the function name as text. You must ACTUALLY CALL the tool via the API.
2. NEVER answer from memory or make up results.
3. ALWAYS use user_id={user_id} as an INTEGER in your tool calls.

**WHEN TO USE WHICH TOOL:**
- If the user asks for a specific file (e.g. "my pan card", "aadhaar", "invoice", "photo"):
  -> You MUST invoke the tool `smart_search_with_memory` using the tool-calling API.
- If the user asks for general emails or their inbox:
  -> You MUST invoke the tool `fetch_emails` using the tool-calling API.
- If the user asks to open or download file number N:
  -> You MUST invoke `open_search_result` or `download_search_result`.

**AFTER THE TOOL RETURNS:**
- When `smart_search_with_memory` returns results to you, DO NOT list the files yourself. 
- The UI will automatically render the files. Just say "Here are the results I found:".
- If the tool returns 0 results, tell the user no files were found.

Remember: Do not type "call smart_search_with_memory(...)". Actually invoke the tool so the system can execute it!
"""

def init_session_state():
    """Initialize session state"""
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    
    if "user_id" not in st.session_state:
        st.session_state.user_id = None
    
    if "session_token" not in st.session_state:
        st.session_state.session_token = None
    
    if "user_info" not in st.session_state:
        st.session_state.user_info = None
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "agent_manager" not in st.session_state:
        st.session_state.agent_manager = None
    
    if "page" not in st.session_state:
        st.session_state.page = "login"
    
    # Model selection state
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = "groq"  # Default to Groq


class SmartAgent:
    """Enhanced agent with user context and model switching"""
    
    def __init__(self, user_id: int, model_type: str = "groq"):
        self.user_id = user_id
        self.model_type = model_type
        self.agent = None
        self.loop = None
        self._lock = threading.Lock()
    
    def _ensure_initialized(self):
        """Lazy initialization of agent"""
        with self._lock:
            if self.agent is not None:
                return
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                self.agent = loop.run_until_complete(self._init_agent())
                self.loop = loop
            except Exception:
                loop.close()
                raise
    
    async def _init_agent(self):
        """Initialize MCP client and agent with selected model"""
        try:
            client = MultiServerMCPClient({
                "Email_Agent": {
                    "url": "http://localhost:9006/mcp",
                    "transport": "streamable_http",
                },
            })
            
            tools = await client.get_tools()
            
            # Debug logging
            print(f"\n{'='*60}")
            print(f"🔧 MCP Tools Loaded: {len(tools)}")
            print(f"{'='*60}")
            for i, tool in enumerate(tools, 1):
                print(f"{i}. {tool.name}")
            print(f"{'='*60}\n")
            
            if len(tools) == 0:
                raise RuntimeError("❌ No tools loaded from MCP server! Make sure server.py is running on port 9006")
            
            # Initialize model based on selection
            if self.model_type == "groq":
                api_key = os.getenv("GROQ_API_KEY")
                
                if not api_key:
                    raise ValueError(
                        "Please set your Groq API key in the GROQ_API_KEY environment variable.\n"
                        "Get your API key from: https://console.groq.com/keys"
                    )
                
                model = ChatGroq(
                    model=MODELS["groq"]["model_id"],
                    api_key=api_key,
                    temperature=MODELS["groq"]["temperature"]
                )
                print(f"✅ Agent initialized with {MODELS['groq']['name']}")
                
            elif self.model_type == "gemini":
                api_key = os.getenv("GEMINI_API_KEY")

                if not api_key:
                    raise ValueError(
                        "Please set your Gemini API key in the GEMINI_API_KEY environment variable.\n"
                        "Get your API key from: https://aistudio.google.com/apikey"
                    )
                
                model = ChatGoogleGenerativeAI(
                    model=MODELS["gemini"]["model_id"],
                    api_key=api_key,
                    temperature=MODELS["gemini"]["temperature"]
                )
                print(f"✅ Agent initialized with {MODELS['gemini']['name']}")
            
            else:
                raise ValueError(f"Unknown model type: {self.model_type}")
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return create_react_agent(model, tools=tools)
            
        except Exception as e:
            print(f"❌ Error initializing agent: {str(e)}")
            raise

    def chat(self, user_message: str, history: list) -> str:
        """Chat with full conversation context"""
        self._ensure_initialized()
        
        # Build conversation with user context
        system_prompt = BOT_PROMPT.format(user_id=self.user_id)
        messages = [{"role": "system", "content": system_prompt}]
        
        for msg in history:
            messages.append({"role": msg["role"], "content": msg["content"]})
        
        messages.append({"role": "user", "content": user_message})
        
        try:
            print(f"\n{'='*60}")
            print(f"💬 User Query: {user_message}")
            print(f"🤖 Model: {MODELS[self.model_type]['name']}")
            print(f"{'='*60}")
            
            response = self.loop.run_until_complete(
                self.agent.ainvoke({"messages": messages})
            )
            
            print(f"✅ Response received")
            print(f"{'='*60}\n")
            
            # Extract the new messages generated in this turn
            new_messages = response["messages"][len(messages):]
            
            last_message = response["messages"][-1]
            
            if hasattr(last_message, 'content'):
                content = last_message.content
            else:
                content = last_message
            
            content = self._extract_clean_text(content)

            # Look for tool outputs containing the JSON marker in this turn
            for msg in new_messages:
                if msg.type == "tool":
                    tool_content = msg.content if hasattr(msg, 'content') else msg
                    tool_content = self._extract_clean_text(tool_content)
                    if "<!--RESULTS_JSON:" in tool_content:
                        # Extract the marker and append it to the final content
                        import re
                        match = re.search(r'<!--RESULTS_JSON:(.*?)-->', tool_content, re.DOTALL)
                        if match:
                            content += f"\n\n<!--RESULTS_JSON:{match.group(1)}-->"
                            break
            
            return content
            
        except Exception as e:
            error_msg = f"❌ Error: {str(e)}"
            print(f"\n❌ Agent Error: {str(e)}\n")
            
            if "Connection" in str(e) or "refused" in str(e):
                error_msg += "\n\n🔌 **MCP Server Issue:**\n"
                error_msg += "Make sure server.py is running:\n"
                error_msg += "`python server.py`"
            elif "API key" in str(e) or "authentication" in str(e).lower():
                error_msg += "\n\n🔑 **API Key Issue:**\n"
                error_msg += f"Please check your {self.model_type.upper()}_API_KEY environment variable"
            elif "tool call validation failed" in str(e):
                error_msg += "\n\n⚠️ **Tool Parameter Error:**\n"
                error_msg += "The model generated incorrect parameter types.\n"
                error_msg += "This has been logged. Try rephrasing your question."
            
            return error_msg

    def _extract_clean_text(self, content) -> str:
        """Extract clean text from response"""
        if isinstance(content, str):
            return content
        
        if isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, dict) and 'text' in item:
                    text_parts.append(item['text'])
                elif isinstance(item, dict) and 'content' in item:
                    text_parts.append(str(item['content']))
                elif isinstance(item, str):
                    text_parts.append(item)
                elif hasattr(item, 'text'):
                    text_parts.append(item.text)
                else:
                    text_parts.append(str(item))
            
            return '\n'.join(filter(None, text_parts))
        
        return str(content)
    
    def reset(self):
        """Reset agent"""
        with self._lock:
            if self.loop:
                self.loop.close()
            self.agent = None
            self.loop = None


def show_login_page():
    """Display login page"""
    st.title("📧 Smart Email and Drive Assistant")
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        tab1, tab2 = st.tabs(["🔐 Login", "📝 Register"])
        
        # Login Tab
        with tab1:
            st.subheader("Welcome Back!")
            
            with st.form("login_form"):
                username = st.text_input("Username", placeholder="Enter your username")
                password = st.text_input("Password", type="password", placeholder="Enter your password")
                
                submit = st.form_submit_button("🔓 Login", use_container_width=True)
                
                if submit:
                    if not username or not password:
                        st.error("❌ Please fill in all fields")
                    else:
                        with st.spinner("Logging in..."):
                            success, msg, user_id = auth.login_user(username, password)
                            
                            if success:
                                # Create session
                                session_token = auth.create_user_session(user_id)
                                
                                # Update session state
                                st.session_state.authenticated = True
                                st.session_state.user_id = user_id
                                st.session_state.session_token = session_token
                                st.session_state.user_info = auth.get_user_info(user_id)
                                
                                # Load chat history
                                history = db.get_chat_history(user_id)
                                st.session_state.messages = [
                                    {"role": h["role"], "content": h["content"]} 
                                    for h in history
                                ]
                                
                                st.success(f"✅ {msg}")
                                st.rerun()
                            else:
                                st.error(f"❌ {msg}")
        
        # Register Tab
        with tab2:
            st.subheader("Create Account")
            
            with st.form("register_form"):
                reg_username = st.text_input("Username", placeholder="Choose a username")
                reg_email = st.text_input("Email", placeholder="your.email@example.com")
                reg_password = st.text_input("Password", type="password", placeholder="Min 8 chars, 1 uppercase, 1 number")
                reg_password_confirm = st.text_input("Confirm Password", type="password", placeholder="Re-enter password")
                
                submit_reg = st.form_submit_button("📝 Register", use_container_width=True)
                
                if submit_reg:
                    if not all([reg_username, reg_email, reg_password, reg_password_confirm]):
                        st.error("❌ Please fill in all fields")
                    elif reg_password != reg_password_confirm:
                        st.error("❌ Passwords don't match")
                    else:
                        with st.spinner("Creating account..."):
                            success, msg, user_id = auth.register_user(reg_username, reg_email, reg_password)
                            
                            if success:
                                st.success(f"✅ {msg}")
                                st.session_state.page = "oauth_setup"
                                st.session_state.temp_user_id = user_id
                                st.rerun()
                            else:
                                st.error(f"❌ {msg}")
        
        st.markdown("---")
        st.caption("🔒 Your data is encrypted and secure")


def show_oauth_setup_page():
    """Display Gmail & Drive OAuth setup page"""
    st.title("📧 Connect Your Gmail & Drive")
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.info("""
        ### 🔗 Google Services Connection Required
        
        To use this app, you need to connect your Google account.
        
        **What happens:**
        1. Click the button below
        2. A browser window will open
        3. Login to your Google account
        4. Grant permissions for Gmail & Drive
        5. You'll be redirected back
        
        **We can access:**
        - 📧 Your emails (read, search, download attachments)
        - 📁 Your Drive files (read, search, download)
        
        **We never:**
        - Store your Google password
        - Share your data
        - Modify files without permission
        - Send emails without your consent
        """)
        
        if st.button("🔗 Connect Gmail & Drive", use_container_width=True, type="primary"):
            user_id = st.session_state.temp_user_id
            
            with st.spinner("Opening browser for Google authorization..."):
                # Check if credentials.json exists
                exists, msg = oauth.check_credentials_file()
                
                if not exists:
                    st.error(msg)
                else:
                    success, msg = oauth.initiate_oauth_flow(user_id)
                    
                    if success:
                        st.success(msg)
                        st.balloons()
                        st.info("✅ Setup complete! You can now login.")
                        
                        if st.button("Go to Login", use_container_width=True):
                            st.session_state.page = "login"
                            del st.session_state.temp_user_id
                            st.rerun()
                    else:
                        st.error(msg)
        
        if st.button("← Back to Login", use_container_width=True):
            st.session_state.page = "login"
            if 'temp_user_id' in st.session_state:
                del st.session_state.temp_user_id
            st.rerun()


def show_main_app():
    """Display main application"""
    user_id = st.session_state.user_id
    user_info = st.session_state.user_info
    
    # Validate session
    if not auth.validate_user_session(st.session_state.session_token):
        st.error("⚠️ Session expired. Please login again.")
        logout()
        return
    
    # Get current model info
    current_model = MODELS[st.session_state.selected_model]
    
    st.title("📧 Smart Email and Drive Assistant")
    st.caption(f"Logged in as: **{user_info['username']}** ({user_info['email']})")
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 👤 Profile")
        
        if st.session_state.user_info.get('picture'):
            st.image(st.session_state.user_info['picture'], width=80)
        
        st.markdown(f"**{st.session_state.user_info.get('name', 'User')}**")
        st.caption(st.session_state.user_info['email'])
        
        if st.button("🚪 Logout", use_container_width=True, type="primary"):
            logout()
        
        st.divider()

        # About
        st.subheader("👀 About")
        st.markdown("""
        Hello, I am your Smart Assistant! I can help you find files from your Emails, Google Drive, and Local files.
        """)

        st.divider()

        # Model Selection
        st.subheader("🧠 AI Model")
        
        # Model selector
        model_options = {
            "groq": f"{MODELS['groq']['icon']} {MODELS['groq']['name']}",
            "gemini": f"{MODELS['gemini']['icon']} {MODELS['gemini']['name']}"
        }
        
        selected = st.selectbox(
            "Choose Model",
            options=list(model_options.keys()),
            format_func=lambda x: model_options[x],
            index=list(model_options.keys()).index(st.session_state.selected_model),
            key="model_selector"
        )
        
        # Show model info
        st.caption(f"**{MODELS[selected]['description']}**")
        # st.caption(f"Temperature: {MODELS[selected]['temperature']}")
        
        # Model switch button
        # if selected != st.session_state.selected_model:
        #     if st.button("🔄 Switch Model", use_container_width=True, type="secondary"):
        #         st.session_state.selected_model = selected
        #         # Reset agent to use new model
        #         if st.session_state.agent_manager:
        #             st.session_state.agent_manager.reset()
        #             st.session_state.agent_manager = None
        #         st.success(f"Switched to {MODELS[selected]['name']}!")
        #         st.rerun()

        st.divider()
        
        # Gmail & Drive status
        st.subheader("🔗 Connected Services")
        
        # Gmail status
        gmail_connected, gmail_email = oauth.verify_gmail_connection(user_id)
        if gmail_connected:
            st.success(f"📧 Gmail: {gmail_email}")
        else:
            st.error("📧 Gmail: Disconnected")
        
        # Drive status
        drive_connected, drive_info = oauth.verify_drive_connection(user_id)
        if drive_connected:
            st.success(f"📁 Drive: {drive_info}")
        else:
            st.error("📁 Drive: Disconnected")
        
        # Reconnect button if either is disconnected
        if not gmail_connected or not drive_connected:
            if st.button("🔄 Reconnect Services", use_container_width=True):
                success, msg = oauth.initiate_oauth_flow(user_id)
                if success:
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)
        
        st.divider()
        
        # Downloaded Files
        col_title, col_refresh = st.columns([3, 1])
        with col_title:
            st.subheader("📂 Downloaded Files")
        with col_refresh:
            if st.button("🔄", help="Refresh file list", key="refresh_btn", use_container_width=True):
                st.rerun()

        # Initialize file count tracker
        if 'last_file_count' not in st.session_state:
            st.session_state.last_file_count = 0

        # Get the Attachments folder path
        if "user_id" not in st.session_state:
            st.warning("⚠️ User not logged in")
            st.stop()

        user_id_str = f"user_{st.session_state.user_id}"

        base_dir = os.path.dirname(os.path.abspath(__file__))
        attachments_dir = os.path.join(
            base_dir,
            "user_data",
            user_id_str,
            "Attachments"
        )
        os.makedirs(attachments_dir, exist_ok=True)

        # Check if folder exists
        if os.path.exists(attachments_dir):
            files = [f for f in os.listdir(attachments_dir) if os.path.isfile(os.path.join(attachments_dir, f))]
            current_file_count = len(files)
            
            # Detect file count changes and auto-refresh
            if 'last_file_count' in st.session_state and current_file_count != st.session_state.last_file_count:
                if current_file_count > st.session_state.last_file_count:
                    st.toast("✅ New file downloaded!", icon="📥")
                elif current_file_count < st.session_state.last_file_count:
                    st.toast("🗑️ File deleted", icon="✅")
            
            st.session_state.last_file_count = current_file_count
            
            if files:
                # Show file count
                col_metric, col_indicator = st.columns([2, 1])
                with col_metric:
                    st.metric("Total Files", len(files))
                with col_indicator:
                    st.caption("🟢 Live")
                
                # Use native Streamlit container with height
                with st.container(height=400):
                    for idx, file in enumerate(sorted(files, reverse=True), 1):
                        file_path = os.path.join(attachments_dir, file)
                        file_size = os.path.getsize(file_path) / 1024
                        file_modified = datetime.fromtimestamp(os.path.getmtime(file_path))
                        
                        # Get file extension for icon
                        ext = os.path.splitext(file)[1].lower()
                        icon_map = {
                            '.pdf': '📕', '.doc': '📘', '.docx': '📘',
                            '.xls': '📗', '.xlsx': '📗', '.ppt': '📙', '.pptx': '📙',
                            '.jpg': '🖼️', '.jpeg': '🖼️', '.png': '🖼️',
                            '.zip': '📦', '.rar': '📦', '.txt': '📄',
                        }
                        icon = icon_map.get(ext, '📎')
                        
                        # Show "NEW" badge
                        time_diff = (datetime.now() - file_modified).total_seconds()
                        new_badge = "🆕 " if time_diff < 60 else ""
                        
                        # Display file
                        st.markdown(f"{new_badge}**{idx}. {icon} {file}**")
                        st.caption(f"💾 {file_size:.1f} KB • {file_modified.strftime('%I:%M %p')}")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if st.button("👁️ Open", key=f"open_{file}", use_container_width=True):
                                try:
                                    if os.name == 'nt':
                                        os.startfile(file_path)
                                    elif os.name == 'posix':
                                        import subprocess
                                        subprocess.Popen(['open', file_path])
                                except Exception as e:
                                    st.error(f"❌ {e}")
                        
                        with col2:
                            with open(file_path, 'rb') as f:
                                st.download_button(
                                    label="💾 Save",
                                    data=f.read(),
                                    file_name=file,
                                    key=f"dl_{file}",
                                    use_container_width=True
                                )
                        
                        with st.expander("⚙️ More"):
                            st.code(file_path, language=None)
                            if st.button("🗑️ Delete", key=f"del_{file}", type="secondary"):
                                os.remove(file_path)
                                st.rerun()
                        
                        st.divider()
        
        # Controls
        st.subheader("⚙️ Controls")
        
        if st.button("🗑️ Clear Chat", use_container_width=True):
            db.clear_chat_history(st.session_state.user_id)
            st.session_state.messages = []
            st.rerun()
        
        if st.button("🔄 Reset Agent", use_container_width=True):
            if st.session_state.agent_manager:
                st.session_state.agent_manager.reset()
                st.session_state.agent_manager = None
            st.success("Agent reset!")
            st.rerun()
    
    # Main chat area
    st.markdown("---")
    
    # Display chat messages
    for message in st.session_state.messages:
        avatar = "👤" if message["role"] == "user" else "🤖"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me about your emails, attachments, or files..."):
        # Add user message
        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })
        
        # Save to database
        db.save_chat_message(user_id, "user", prompt)
        
        # Display user message
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)
        
        # Get response
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Processing..."):
                agent = get_or_create_agent(user_id)
                history = st.session_state.messages[:-1]
                response = agent.chat(prompt, history)
                
                st.markdown(response)
                
                # Save response
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response
                })
                
                db.save_chat_message(user_id, "assistant", response)


def get_or_create_agent(user_id: int):
    """Get or create agent for user with current model"""
    if st.session_state.agent_manager is None:
        with st.spinner(f"🔧 Initializing {MODELS[st.session_state.selected_model]['name']}..."):
            try:
                st.session_state.agent_manager = SmartAgent(
                    user_id, 
                    model_type=st.session_state.selected_model
                )
                st.toast(f"✅ {MODELS[st.session_state.selected_model]['name']} ready!", icon="🤖")
            except Exception as e:
                st.error(f"❌ Failed to initialize agent: {str(e)}")
                st.stop()
    return st.session_state.agent_manager


def logout():
    """Logout user"""
    if st.session_state.session_token:
        auth.logout_user(st.session_state.session_token)
    
    st.session_state.authenticated = False
    st.session_state.user_id = None
    st.session_state.session_token = None
    st.session_state.user_info = None
    st.session_state.messages = []
    st.session_state.agent_manager = None
    st.session_state.page = "login"
    
    st.rerun()


def main():
    # Initialize database
    db.initialize_database()
    
    # Initialize session state
    init_session_state()
    
    # Cleanup expired sessions periodically
    db.cleanup_expired_sessions()
    
    # Route to appropriate page
    if not st.session_state.authenticated:
        if st.session_state.page == "oauth_setup":
            show_oauth_setup_page()
        else:
            show_login_page()
    else:
        show_main_app()


if __name__ == "__main__":
    main()