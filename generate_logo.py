from PIL import Image, ImageDraw, ImageFont
import os

# Create a new image with a white background
width, height = 500, 500
background_color = (255, 255, 255)
img = Image.new('RGB', (width, height), background_color)

# Get a drawing context
draw = ImageDraw.Draw(img)

# Draw a blue rectangle as a background for the logo
draw.rectangle([(50, 50), (450, 450)], fill=(59, 130, 246))

# Draw a white portfolio icon (simplified)
# Draw a briefcase shape
draw.rectangle([(150, 150), (350, 350)], fill=(255, 255, 255))
draw.rectangle([(200, 120), (300, 150)], fill=(255, 255, 255))

# Add text
try:
    # Try to use a system font
    font = ImageFont.load_default()
except:
    # Fall back to default font
    font = ImageFont.load_default()

# Add text "SmartFolio AI by Divyanshi Sharma"
draw.text((120, 380), "SmartFolio AI by Divyanshi Sharma", fill=(255, 255, 255), font=font)

# Save the image
logo_path = "images/logo/smartfolio_logo.png"
os.makedirs(os.path.dirname(logo_path), exist_ok=True)
img.save(logo_path)

print(f"Logo created and saved to {logo_path}")