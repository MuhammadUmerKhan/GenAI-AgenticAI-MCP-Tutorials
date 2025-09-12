import streamlit as st
from langgraph_backend import chatbot, get_all_threads
from langchain_core.messages import HumanMessage, AIMessage
import uuid

# **************************************** Page Config ******************** **********

st.set_page_config(page_title="AI Chat Assistant", page_icon="💬")

# **************************************** Utility Functions *************************

def generate_thread_id():
    return str(uuid.uuid4())

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state['thread_id'] = thread_id
    add_thread(st.session_state['thread_id'])
    st.session_state['message_history'] = []

def add_thread(thread_id):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)

def load_conversation(thread_id):
    state = chatbot.get_state(config={'configurable': {'thread_id': thread_id}})
    return state.values.get('messages', [])

# **************************************** Session Setup ******************************

if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = get_all_threads()

add_thread(st.session_state['thread_id'])

# **************************************** Custom CSS for Dark Theme *****************

st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Main app background and text */
    .stApp {
        background: linear-gradient(135deg, #0F0F23 0%, #1A1A2E 50%, #16213E 100%);
        color: #E8E8E8;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #1A1A2E;
    }
    ::-webkit-scrollbar-thumb {
        background: #4A5568;
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #718096;
    }

    /* Sidebar styling */
    .stSidebar {
        background: linear-gradient(180deg, #1A1A2E 0%, #16213E 100%);
        border-right: 1px solid #2D3748;
    }
    .stSidebar [data-testid="stSidebarNav"] {
        color: #E8E8E8;
    }
    .stSidebar .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: #FFFFFF;
        border-radius: 12px;
        border: none;
        padding: 12px 20px;
        font-weight: 500;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    .stSidebar .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }

    /* Main content area */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Title styling */
    .stTitle {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 700;
        font-size: 2.5rem;
        margin-bottom: 2rem;
        text-align: center;
    }

    /* Chat container */
    .stChatMessage {
        border-radius: 16px;
        margin: 12px 0;
        padding: 16px 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
        transition: all 0.3s ease;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .stChatMessage:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 25px rgba(0, 0, 0, 0.2);
    }
    
    .stChatMessage.user {
        background: linear-gradient(135deg, #2D3748 0%, #4A5568 100%);
        color: #E8E8E8;
        margin-left: 20%;
        border-left: 4px solid #667eea;
    }
    
    .stChatMessage.assistant {
        background: linear-gradient(135deg, #1A365D 0%, #2C5282 100%);
        color: #FFFFFF;
        margin-right: 20%;
        border-left: 4px solid #4299E1;
    }

    /* Chat input styling */
    .stChatInput {
        background: rgba(26, 26, 46, 0.8);
        border-radius: 16px;
        padding: 8px;
        margin-top: 20px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .stChatInput > div > div > textarea {
        background: rgba(45, 55, 72, 0.8);
        color: #E8E8E8;
        border: 2px solid transparent;
        border-radius: 12px;
        padding: 16px 20px;
        font-size: 16px;
        font-family: 'Inter', sans-serif;
        transition: all 0.3s ease;
        resize: none;
    }
    
    .stChatInput > div > div > textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        outline: none;
    }
    
    .stChatInput > div > div > textarea::placeholder {
        color: #A0AEC0;
        font-style: italic;
    }

    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: #FFFFFF;
        border-radius: 12px;
        border: none;
        padding: 12px 24px;
        font-weight: 600;
        font-family: 'Inter', sans-serif;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    .stButton > button:active {
        transform: translateY(0);
    }
    
    /* Timestamp styling */
    .timestamp {
        font-size: 0.75rem;
        color: #A0AEC0;
        margin-top: 8px;
        font-weight: 400;
        opacity: 0.8;
    }

    /* Sidebar conversation buttons */
    .stSidebar .stButton > button {
        background: rgba(102, 126, 234, 0.1);
        color: #E8E8E8;
        border: 1px solid rgba(102, 126, 234, 0.3);
        border-radius: 8px;
        padding: 10px 16px;
        margin: 4px 0;
        font-size: 0.9rem;
        text-align: left;
        transition: all 0.3s ease;
    }
    
    .stSidebar .stButton > button:hover {
        background: rgba(102, 126, 234, 0.2);
        border-color: rgba(102, 126, 234, 0.5);
        transform: translateX(4px);
    }
    
    /* Loading animation */
    .loading-dots {
        display: inline-block;
    }
    
    .loading-dots::after {
        content: '';
        animation: dots 1.5s steps(4, end) infinite;
    }
    
    @keyframes dots {
        0%, 20% { content: ''; }
        40% { content: '.'; }
        60% { content: '..'; }
        80%, 100% { content: '...'; }
    }
    
    /* Message animations */
    .stChatMessage {
        animation: slideInUp 0.3s ease-out;
    }
    
    @keyframes slideInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Status indicators */
    .status-indicator {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 8px;
    }
    
    .status-online {
        background-color: #48BB78;
        box-shadow: 0 0 6px rgba(72, 187, 120, 0.5);
    }
    
    .status-typing {
        background-color: #ED8936;
        animation: pulse 1.5s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .stChatMessage.user {
            margin-left: 10%;
        }
        .stChatMessage.assistant {
            margin-right: 10%;
        }
        .stTitle {
            font-size: 2rem;
        }
    }
    
    /* Focus states for accessibility */
    .stButton > button:focus,
    .stChatInput > div > div > textarea:focus {
        outline: 2px solid #667eea;
        outline-offset: 2px;
    }
    </style>
""", unsafe_allow_html=True)

# **************************************** Sidebar UI *********************************

# Sidebar header with enhanced styling
st.sidebar.markdown("""
    <div style="text-align: center; padding: 20px 0; border-bottom: 1px solid rgba(255,255,255,0.1); margin-bottom: 20px;">
        <h1 style="color: #667eea; font-size: 1.8rem; margin: 0; font-weight: 700;">🤖 AI Assistant</h1>
        <p style="color: #A0AEC0; margin: 8px 0 0 0; font-size: 0.9rem;">Powered by LangGraph</p>
    </div>
""", unsafe_allow_html=True)

# Status indicator
st.sidebar.markdown("""
    <div style="display: flex; align-items: center; margin-bottom: 20px; padding: 12px; background: rgba(72, 187, 120, 0.1); border-radius: 8px; border: 1px solid rgba(72, 187, 120, 0.3);">
        <span class="status-indicator status-online"></span>
        <span style="color: #E8E8E8; font-size: 0.9rem; font-weight: 500;">Online & Ready</span>
    </div>
""", unsafe_allow_html=True)

# Conversations section
st.sidebar.markdown("### 💬 Conversations")

# New Chat button with enhanced styling
if st.sidebar.button('✨ Start New Chat', key='new_chat', help="Start a fresh conversation"):
    reset_chat()
    st.rerun()

# Conversation history with better formatting
if st.session_state['chat_threads']:
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Recent Chats:**")
    
    for i, thread_id in enumerate(st.session_state['chat_threads']):
        if st.sidebar.button(thread_id, key=f"thread_{thread_id}", help=f"Load conversation {thread_id}"):
            st.session_state['thread_id'] = thread_id
            messages = load_conversation(thread_id)
            temp_messages = []
            for msg in messages:
                role = 'user' if isinstance(msg, HumanMessage) else 'assistant'
                temp_messages.append({
                    'role': role,
                    'content': msg.content,
                })
            st.session_state['message_history'] = temp_messages
            st.rerun()
else:
    st.sidebar.markdown("""
    <div style="text-align: center; padding: 20px; color: #A0AEC0; font-style: italic;">
        No conversations yet.<br>Start chatting to see your history here!
    </div>
    """, unsafe_allow_html=True)

# **************************************** Main UI ************************************

# Enhanced main title
st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h1 class="stTitle">💬 AI Chat Assistant</h1>
        <p style="color: #A0AEC0; font-size: 1.1rem; margin: 0;">Ask me anything! I'm here to help.</p>
    </div>
""", unsafe_allow_html=True)

# Welcome message for new conversations
if not st.session_state['message_history']:
    st.markdown("""
    <div style="text-align: center; padding: 40px 20px; background: rgba(102, 126, 234, 0.1); border-radius: 16px; border: 1px solid rgba(102, 126, 234, 0.3); margin-bottom: 20px;">
        <h3 style="color: #667eea; margin-bottom: 16px;">👋 Welcome!</h3>
        <p style="color: #E8E8E8; margin: 0; font-size: 1.1rem;">I'm your AI assistant powered by LangGraph. I can help you with various tasks, answer questions, and have meaningful conversations.</p>
        <div style="margin-top: 20px; display: flex; justify-content: center; gap: 12px; flex-wrap: wrap;">
            <span style="background: rgba(102, 126, 234, 0.2); padding: 6px 12px; border-radius: 20px; font-size: 0.9rem; color: #E8E8E8;">💡 Ask questions</span>
            <span style="background: rgba(102, 126, 234, 0.2); padding: 6px 12px; border-radius: 20px; font-size: 0.9rem; color: #E8E8E8;">🔍 Get help</span>
            <span style="background: rgba(102, 126, 234, 0.2); padding: 6px 12px; border-radius: 20px; font-size: 0.9rem; color: #E8E8E8;">💬 Chat freely</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Display conversation history with enhanced styling
for message in st.session_state['message_history']:
    with st.chat_message(message['role'], avatar="🧑‍💻" if message['role'] == 'user' else "🤖"):
        st.markdown(message['content'])

# Enhanced chat input with placeholder
user_input = st.chat_input('💬 Type your message here... (Press Enter to send)')

if user_input:
    # Add user message with timestamp
    st.session_state['message_history'].append({
        'role': 'user',
        'content': user_input,
    })
    with st.chat_message('user', avatar="🧑‍💻"):
        st.markdown(user_input)

    CONFIG = {'configurable': {'thread_id': st.session_state['thread_id']}}

    # Show typing indicator and stream response
    with st.chat_message("assistant", avatar="🤖"):
        # Create a placeholder for the typing indicator
        typing_placeholder = st.empty()
        typing_placeholder.markdown("""
            <div style="display: flex; align-items: center; color: #A0AEC0; font-style: italic;">
                <span class="status-indicator status-typing"></span>
                <span class="loading-dots">AI is thinking</span>
            </div>
        """, unsafe_allow_html=True)
        
        def ai_only_stream():
            for message_chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages"
            ):
                if isinstance(message_chunk, AIMessage):
                    yield message_chunk.content

        # Clear the typing indicator and stream the response
        typing_placeholder.empty()
        ai_message = st.write_stream(ai_only_stream())

        # Add the assistant message to history
        st.session_state['message_history'].append({
            'role': 'assistant',
            'content': ai_message,
        })  
    
    # Auto-scroll to bottom (this will be handled by Streamlit's chat interface)
    st.rerun()