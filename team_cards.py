import streamlit as st
import base64

def image_to_base64(path):
    with open(path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode()
        return f"data:image/jpeg;base64,{encoded}"

# Initialize dark mode session state
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False

# Sidebar toggle
st.sidebar.title("Settings")
st.session_state.dark_mode = st.sidebar.checkbox("Enable Dark Mode", value=st.session_state.dark_mode)

# Define colors based on mode
bg_color = "#1e293b" if st.session_state.dark_mode else "#ffffff"
card_shadow = "rgba(0, 0, 0, 0.3)" if st.session_state.dark_mode else "rgba(0, 0, 0, 0.1)"
name_color = "#f1f5f9" if st.session_state.dark_mode else "#0f172a"
desc_color = "#94a3b8" if st.session_state.dark_mode else "#64748b"

def display_team_cards():
    col1, col2, col3 = st.columns(3)

    team_members = [
        {
            "name": "Divyanshi Sharma",
            "role": "Lead Data Scientist",
            "desc": "Machine Learning expert with a passion for financial markets...",
            "image": image_to_base64("/Users/divayanshisharama/Desktop/project/images/team/Divyanshi.jpg"),
            "emoji": "👩‍💻"
        },
        {
            "name": "Ankit Patel",
            "role": "Data Scientist",
            "desc": "Financial expert with deep knowledge of market trends and technical analysis.",
            "image": image_to_base64("/Users/divayanshisharama/Desktop/project/images/team/Screenshot 2025-04-23 at 12.21.03 PM.png"),
            "emoji": "👨‍💻"
        },
        {
            "name": "Piyush Garg",
            "role": "Data Scientist",
            "desc": "Software engineering expert who built the platform's architecture.",
            "image": image_to_base64("/Users/divayanshisharama/Desktop/project/images/team/WhatsApp Image 2025-04-23 at 11.51.41.jpeg"),
            "emoji": "👨‍💻"
        }
    ]

    cols = [col1, col2, col3]

    for i, member in enumerate(team_members):
        with cols[i]:
            html_content = f"""
            <div class="team-card" style="background-color: {bg_color}; 
                                            border-radius: 12px; padding: 20px; text-align: center; 
                                            box-shadow: 0 4px 8px {card_shadow}; 
                                            margin: 10px; animation: fadeInUp 0.8s ease-out forwards {0.2 + i*0.2}s; 
                                            opacity: 0;">
                <img src="{member['image']}" style="width: 150px; height: 150px; border-radius: 50%; object-fit: cover; 
                                                    border: 4px solid #3b82f6; margin-bottom: 15px;">
                <h3 style="font-size: 1.5rem; font-weight: 600; margin-bottom: 5px; color: {name_color};">{member['name']}</h3>
                <p style="font-size: 1.1rem; color: #3b82f6; margin-bottom: 10px; font-weight: 500;">{member['role']}</p>
                <p style="font-size: 0.95rem; color: {desc_color}; line-height: 1.5;">{member['desc']}</p>
                <div style="margin-top: 15px; display: flex; justify-content: center; gap: 10px;">
                    <a href="#" style="font-size: 1.2rem; color: #3b82f6; text-decoration: none;">📱</a>
                    <a href="#" style="font-size: 1.2rem; color: #3b82f6; text-decoration: none;">📧</a>
                    <a href="#" style="font-size: 1.2rem; color: #3b82f6; text-decoration: none;">{member['emoji']}</a>
                </div>
            </div>
            """
            st.markdown(html_content, unsafe_allow_html=True)

# Title
st.title("🚀 Meet the Team")
display_team_cards()
