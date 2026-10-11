#app.py
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
import asyncio
import threading
import os
import subprocess
from datetime import datetime
from pathlib import Path

# Import our modules
import database as db
import auth
import oauth_handler as oauth

# Page config
st.set_page_config(
    page_title="FindMyFiles",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded"
)

BOT_PROMPT = """
You are an Intelligent Email & Drive Assistant with smart search capabilities.

**IMPORTANT: You MUST pass user_id parameter to ALL tools!**

**Email Tools:**
- fetch_emails(user_id, ...)
- search_emails(user_id, ...)
- download_attachment(user_id, ...)

**Drive Tools:**
- list_drive_files(user_id, max_results, query)
- search_drive_files(user_id, query, max_results)
- download_drive_file(user_id, file_id, filename)
- get_drive_storage_info(user_id)

**File Search:**
- smart_search_with_memory(user_id, query) - Searches local + email + drive
- open_search_result(user_id, file_number)
- download_search_result(user_id, file_number)

**Handling Multiple Results:**
When multiple files are found:
1. Show all results with numbers
2. Ask user: "Which one do you want to view or download? (Say 'file 1', 'file 2', etc.)"
3. When user picks a number, use the appropriate action tool

**Examples:**
User: "show my drive files"
You: list_drive_files(user_id=current_user_id)

User: "search drive for invoice"
You: search_drive_files(user_id=current_user_id, query="invoice")

User: "download that drive file"
You: download_drive_file(user_id=current_user_id, file_id="xyz123")

User: "check my drive storage"
You: get_drive_storage_info(user_id=current_user_id)

Always remember to use the current user's ID!

if the user is asking for a "photo", "image", or "picture", return file formats -> (jpg, png, gif, bmp, tiff).
if the user is asking for a "document", return file formats -> (pdf, docx, txt, odt, rtf).
if the user is asking for a "spreadsheet", return file formats -> (xlsx, csv, ods).

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


class SmartAgent:
    """Enhanced agent with user context"""
    
    def __init__(self, user_id: int):
        self.user_id = user_id
        self.agent = None
        self.loop = None
        self._lock = threading.Lock()
    
    def _ensure_initialized(self):
        """Lazy initialization of agent"""
        with self._lock:
            if self.agent is not None:
                return
            
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            self.agent = self.loop.run_until_complete(self._init_agent())
    
    async def _init_agent(self):
        """Initialize MCP client and agent"""
        client = MultiServerMCPClient({
            "Email_Agent": {
                "url": "http://localhost:9006/mcp",
                "transport": "streamable_http",
            },
        })
        
        tools = await client.get_tools()
        
        # Get API key (you should set this in environment or config)
        api_key = os.getenv("GEMINI_API_KEY", "AIzaSyBAYHru7y4GbDG0QYQwDHM-_jXcqqKoqJM")
        # only available models are gemini-2.5-flash, gemini-2.5-flash-lite, gemini-3-flash.
        model = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            api_key=api_key,
            temperature=0.3
        )
        
        return create_react_agent(model, tools)

    def chat(self, user_message: str, history: list) -> str:
        """Chat with full conversation context"""
        self._ensure_initialized()
        
        # Build conversation with user context
        system_prompt = BOT_PROMPT + f"\n\nCurrent user_id: {self.user_id}"
        messages = [{"role": "system", "content": system_prompt}]
        
        for msg in history:
            messages.append({"role": msg["role"], "content": msg["content"]})
        
        messages.append({"role": "user", "content": user_message})
        
        try:
            response = self.loop.run_until_complete(
                self.agent.ainvoke({"messages": messages})
            )
            
            last_message = response["messages"][-1]
            
            if hasattr(last_message, 'content'):
                content = last_message.content
            else:
                content = last_message
            
            return self._extract_clean_text(content)
            
        except Exception as e:
            error_msg = f"❌ Error: {str(e)}"
            if "Connection" in str(e) or "refused" in str(e):
                error_msg += "\n\n🔌 Make sure MCP server is running: `python server.py`"
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
    st.title("FindMyFiles")
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
    
    st.title("FindMyFiles")
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

        # Features
        st.subheader("👀 About")
        st.markdown("""
        Hello, I am your Smart Assistant! I can help you find files from your Emails, Google Drive and your Local files.
        """)

        st.divider()

        st.subheader("🧠 Model")
        st.markdown("""
        I use the **Gemini 2.5 flash** model for processing your requests.
        """)

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
        # ---------------- USER-SPECIFIC ATTACHMENTS PATH ----------------

        if "user_id" not in st.session_state:
            st.warning("⚠️ User not logged in")
            st.stop()

        user_id = f"user_{st.session_state.user_id}"

        base_dir = os.path.dirname(os.path.abspath(__file__))
        attachments_dir = os.path.join(
            base_dir,
            "user_data",
            user_id,
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
                    # New file added
                    st.toast("✅ New file downloaded!", icon="📥")
                elif current_file_count < st.session_state.last_file_count:
                    # File deleted
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
                with st.container(height=400):  # Set fixed height here
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
            db.clear_chat_history(user_id)
            st.session_state.messages = []
            st.rerun()
        
        if st.button("🔄 Reset Agent", use_container_width=True):
            if st.session_state.agent_manager:
                st.session_state.agent_manager.reset()
                st.session_state.agent_manager = None
            st.success("Agent reset!")
            st.rerun()
        
        # if st.button("🚪 Logout", use_container_width=True, type="primary"):
        #     logout()
    
    # Main chat area
    st.markdown("---")
    
    # Display chat messages
    for message in st.session_state.messages:
        avatar = "🧑" if message["role"] == "user" else "🤖"
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
        with st.chat_message("user", avatar="🧑"):
            st.markdown(prompt)
        
        # Get response
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("🤔 Processing..."):
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
    """Get or create agent for user"""
    if st.session_state.agent_manager is None:
        with st.spinner("🔧 Initializing Email Assistant..."):
            st.session_state.agent_manager = SmartAgent(user_id)
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