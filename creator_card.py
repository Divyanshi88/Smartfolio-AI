import streamlit as st
import base64
import os

# Convert image to base64 for embedding
def image_to_base64(path):
    if os.path.exists(path):
        with open(path, "rb") as image_file:
            encoded = base64.b64encode(image_file.read()).decode()
            return f"data:image/jpeg;base64,{encoded}"
    else:
        st.error("Image file not found.")
        return ""

# Dark mode is controlled from main.py
# We'll use the session state that's already set there

def display_creator_card():
    # Load image
    image_path = "/Users/divayanshisharama/Desktop/project/images/team/Divyanshi.jpg"
    encoded_image = image_to_base64(image_path)

    # Custom CSS
    st.markdown("""
    <style>
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .social-icon {
        transition: all 0.3s ease;
    }

    .social-icon:hover {
        transform: translateY(-3px);
    }

    .skill-bar {
        transition: width 1s ease-in-out;
    }
    </style>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        bg_color = "#1e293b" if st.session_state.dark_mode else "white"
        text_color = "#f1f5f9" if st.session_state.dark_mode else "#0f172a"
        desc_color = "#94a3b8" if st.session_state.dark_mode else "#64748b"
        shadow_color = "rgba(0, 0, 0, 0.4)" if st.session_state.dark_mode else "rgba(0, 0, 0, 0.1)"
        skill_bg = "#334155" if st.session_state.dark_mode else "#e2e8f0"
        badge_bg = "rgba(255,255,255,0.1)" if st.session_state.dark_mode else "rgba(59,130,246,0.1)"

        st.markdown(f"""
        <div style="background: {bg_color}; border-radius: 16px; padding: 30px; text-align: center; 
                    box-shadow: 0 10px 25px {shadow_color}; margin: 20px 0; animation: fadeInUp 0.8s ease-out forwards;">
            <img src="{encoded_image}" style="width: 180px; height: 180px; border-radius: 50%; object-fit: cover; 
                                                border: 5px solid #3b82f6; margin-bottom: 15px;">
            <h2 style="font-size: 2rem; font-weight: 700; margin-bottom: 5px; color: {text_color}; 
                       text-shadow: 0 2px 4px rgba(0,0,0,0.1);">Divyanshi Sharma</h2>
            <p style="font-size: 1.2rem; color: #3b82f6; margin-bottom: 15px; font-weight: 600;">
                Creator & Data Scientist</p>
            <div style="width: 50px; height: 3px; background: #3b82f6; margin: 0 auto 20px; border-radius: 3px;"></div>
            <p style="font-size: 1rem; color: {desc_color}; line-height: 1.7; margin-bottom: 25px; 
                     text-align: left; padding: 0 10px;">
                Passionate Data Scientist with expertise in Machine Learning and Financial Markets. 
                I specialize in developing innovative solutions that transform complex data into actionable insights.
                My work combines cutting-edge AI techniques with deep domain knowledge to create powerful, user-friendly applications.
            </p>
            <div style="margin-bottom: 25px; text-align: left;">
                <h4 style="color: {text_color}; font-size: 1.1rem; margin-bottom: 15px; font-weight: 600;">Expertise</h4>
        """, unsafe_allow_html=True)

        # Skills list
        skills = [
            ("Machine Learning", 95),
            ("Financial Analysis", 90),
            ("Data Visualization", 85),
            ("Python", 92)
        ]
        for skill, percent in skills:
            st.markdown(f"""
                <div style="margin-bottom: 12px;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                        <span style="color: {desc_color}; font-size: 0.9rem;">{skill}</span>
                        <span style="color: #3b82f6; font-size: 0.9rem;">{percent}%</span>
                    </div>
                    <div style="background: {skill_bg}; height: 8px; border-radius: 4px; overflow: hidden;">
                        <div class="skill-bar" style="width: {percent}%; height: 100%; background: #3b82f6; border-radius: 4px;"></div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

        # Social & Portfolio Links
        st.markdown(f"""
            </div>
            <div style="margin-top: 25px; display: flex; justify-content: center; gap: 20px;">
                <a href="https://www.linkedin.com/in/divyanshi-sharma-a71a4825a/" target="_blank" class="social-icon" 
                   style="font-size: 1.4rem; color: #3b82f6; text-decoration: none; background: {badge_bg};
                          width: 45px; height: 45px; border-radius: 50%; display: flex; 
                          justify-content: center; align-items: center;">
                    in
                </a>
                <a href="https://github.com/divyanshi88" target="_blank" class="social-icon" 
                   style="font-size: 1.4rem; color: #3b82f6; text-decoration: none; background: {badge_bg};
                          width: 45px; height: 45px; border-radius: 50%; display: flex; 
                          justify-content: center; align-items: center;">
                    GH
                </a>
                <a href="mailto:divyanshi122023@gmail.com" target="_blank" class="social-icon" 
                   style="font-size: 1.4rem; color: #3b82f6; text-decoration: none; background: {badge_bg};
                          width: 45px; height: 45px; border-radius: 50%; display: flex; 
                          justify-content: center; align-items: center;">
                    @
                </a>
                <a href="https://divyanshi88.github.io/" target="_blank" class="social-icon"
                   style="font-size: 1.2rem; font-weight: 600; color: #3b82f6; text-decoration: none;
                          background: {badge_bg}; padding: 10px 16px; border-radius: 12px;">
                    🌐 Portfolio
                </a>
            </div>
        </div>
        """, unsafe_allow_html=True)

# The card will be displayed when called from main.py
# display_creator_card()
