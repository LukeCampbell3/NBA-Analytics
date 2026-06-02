#!/usr/bin/env python3
"""
Realistic NBA Trading Card Generator
Creates Panini/Upper Deck/Topps-style basketball cards
"""

import json
import os
import random
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance, ImageOps
import textwrap
import math
from typing import Tuple, List, Dict, Any, Optional
import colorsys
import random
from datetime import datetime

class RealisticCardGenerator:
    def __init__(self, output_dir: str = "web/images/cards"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Card dimensions (standard trading card size: 2.5" x 3.5" at 300 DPI)
        self.card_width = 750  # 2.5" at 300 DPI
        self.card_height = 1050  # 3.5" at 300 DPI
        self.thumb_width = 300
        self.thumb_height = 420
        
        # Card styles based on real card types
        self.card_styles = {
            "base": {
                "border_color": (40, 40, 40),
                "bg_gradient": [(30, 30, 30), (60, 60, 60)],
                "accent_color": (0, 100, 200),
                "text_color": (255, 255, 255),
                "foil_effect": False
            },
            "premium": {
                "border_color": (200, 180, 50),
                "bg_gradient": [(20, 20, 40), (40, 20, 60)],
                "accent_color": (255, 215, 0),
                "text_color": (255, 255, 255),
                "foil_effect": True
            },
            "rookie": {
                "border_color": (0, 100, 200),
                "bg_gradient": [(20, 40, 80), (40, 60, 120)],
                "accent_color": (0, 150, 255),
                "text_color": (255, 255, 200),
                "foil_effect": True
            },
            "legacy": {
                "border_color": (100, 100, 100),
                "bg_gradient": [(50, 50, 50), (80, 80, 80)],
                "accent_color": (200, 200, 200),
                "text_color": (220, 220, 220),
                "foil_effect": False
            }
        }
        
        # Team color schemes (primary, secondary, accent)
        self.team_colors = {
            "LAL": [(85, 37, 131), (253, 185, 39)],  # Lakers: Purple & Gold
            "GSW": [(29, 66, 138), (255, 199, 44)],    # Warriors: Blue & Yellow
            "BOS": [(0, 122, 51), (139, 111, 78)],      # Celtics: Green & Brown
            "CHI": [(206, 17, 65), (0, 0, 0)],          # Bulls: Red & Black
            "MIA": [(152, 0, 46), (249, 160, 63)],      # Heat: Red & Orange
            "LAC": [(200, 16, 46), (29, 66, 138)],      # Clippers: Red & Blue
            "DEN": [(13, 34, 64), (255, 198, 39)],      # Nuggets: Navy & Gold
            "MIL": [(0, 71, 27), (240, 235, 210)],      # Bucks: Green & Cream
            "PHI": [(0, 107, 182), (237, 23, 76)],      # 76ers: Blue & Red
            "BKN": [(0, 0, 0), (255, 255, 255)],        # Nets: Black & White
            "default": [(30, 30, 30), (200, 200, 200)]   # Default dark/light
        }
        
        # Try to load fonts, fallback to default if not available
        try:
            self.title_font = ImageFont.truetype("arialbd.ttf", 36)
            self.name_font = ImageFont.truetype("arialbd.ttf", 48)
            self.stats_font = ImageFont.truetype("arial.ttf", 20)
            self.small_font = ImageFont.truetype("arial.ttf", 16)
        except:
            # Fallback to default font
            print("Using default font - install arial.ttf for better results")
            self.title_font = ImageFont.load_default()
            self.name_font = ImageFont.load_default()
            self.stats_font = ImageFont.load_default()
            self.small_font = ImageFont.load_default()
    
    def create_gradient(self, width: int, height: int, colors: List[Tuple[int, int, int]], 
                      direction: str = "vertical") -> Image.Image:
        """Create a gradient background"""
        base = Image.new('RGB', (width, height))
        draw = ImageDraw.Draw(base)
        
        if direction == "vertical":
            for y in range(height):
                ratio = y / height
                r = int(colors[0][0] * (1 - ratio) + colors[1][0] * ratio)
                g = int(colors[0][1] * (1 - ratio) + colors[1][1] * ratio)
                b = int(colors[0][2] * (1 - ratio) + colors[1][2] * ratio)
                draw.line([(0, y), (width, y)], fill=(r, g, b))
        else:
            for x in range(width):
                ratio = x / width
                r = int(colors[0][0] * (1 - ratio) + colors[1][0] * ratio)
                g = int(colors[0][1] * (1 - ratio) + colors[1][1] * ratio)
                b = int(colors[0][2] * (1 - ratio) + colors[1][2] * ratio)
                draw.line([(x, 0), (x, height)], fill=(r, g, b))
        
        return base
    
    def add_foil_effect(self, image: Image.Image) -> Image.Image:
        """Add foil/reflective effect to card"""
        # Create a foil overlay
        foil = Image.new('RGBA', image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(foil)
        
        # Add some sparkle effects
        for _ in range(50):
            x = random.randint(0, image.width - 1)
            y = random.randint(0, image.height - 1)
            size = random.randint(1, 3)
            brightness = random.randint(200, 255)
            draw.ellipse([x, y, x + size, y + size], 
                        fill=(brightness, brightness, brightness, 128))
        
        # Add some light streaks
        for _ in range(5):
            x1 = random.randint(0, image.width)
            y1 = random.randint(0, image.height)
            x2 = random.randint(0, image.width)
            y2 = random.randint(0, image.height)
            draw.line([(x1, y1), (x2, y2)], 
                     fill=(255, 255, 255, 100), width=2)
        
        return Image.alpha_composite(image.convert('RGBA'), foil)
    
    def create_player_portrait(self, width: int, height: int, team_colors: Tuple) -> Image.Image:
        """Create a placeholder player portrait with team colors"""
        img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        # Create a silhouette of a basketball player
        # Head
        draw.ellipse([width//2-30, 20, width//2+30, 80], 
                    fill=team_colors[0], outline=team_colors[1], width=3)
        
        # Torso
        draw.rectangle([width//2-20, 80, width//2+20, height-40], 
                      fill=team_colors[0], outline=team_colors[1], width=2)
        
        # Arms
        draw.line([(width//2-20, 100), (width//2-60, height-20)], 
                 fill=team_colors[1], width=8)
        draw.line([(width//2+20, 100), (width//2+60, height-40)], 
                 fill=team_colors[1], width=8)
        
        # Legs
        draw.line([(width//2-10, height-40), (width//2-30, height)], 
                 fill=team_colors[1], width=8)
        draw.line([(width//2+10, height-40), (width//2+30, height)], 
                 fill=team_colors[1], width=8)
        
        # Basketball
        draw.ellipse([width//2-15, height-100, width//2+15, height-70], 
                    fill=(222, 184, 135), outline=team_colors[1], width=2)
        draw.line([(width//2, height-90), (width//2, height-75)], 
                 fill=(139, 69, 19), width=2)
        draw.line([(width//2-8, height-87), (width//2+8, height-87)], 
                 fill=(139, 69, 19), width=2)
        
        return img
    
    def create_stat_bar(self, label: str, value: int, max_value: int, 
                      width: int, height: int, color: Tuple[int, int, int]) -> Image.Image:
        """Create a stat bar visualization"""
        bar = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(bar)
        
        # Draw background
        draw.rounded_rectangle([0, 0, width, height], 
                             fill=(40, 40, 40, 200), radius=5)
        
        # Draw filled portion
        fill_width = int((value / max_value) * (width - 4))
        draw.rounded_rectangle([2, 2, fill_width, height-2], 
                             fill=color, radius=3)
        
        # Add label
        draw.text((width//2, height//2), label, fill=(255, 255, 255), 
                 font=self.small_font, anchor="mm")
        
        return bar
    
    def generate_card(self, player_data: Dict, card_style: str = "base") -> Image.Image:
        """Generate a single trading card"""
        # Get team colors
        team = player_data.get('team', 'LAL')
        team_color = self.team_colors.get(team, self.team_colors['default'])
        
        # Get card style
        style = self.card_styles.get(card_style, self.card_styles['base'])
        
        # Create base card
        card = Image.new('RGB', (self.card_width, self.card_height), 
                        color=style['bg_gradient'][0])
        
        # Add gradient background
        gradient = self.create_gradient(
            self.card_width, self.card_height,
            style['bg_gradient'], 
            "vertical"
        )
        card.paste(gradient, (0, 0))
        
        # Add foil effect if specified
        if style['foil_effect']:
            card = self.add_foil_effect(card)
        
        draw = ImageDraw.Draw(card)
        
        # Draw card border
        border_width = 20
        draw.rounded_rectangle(
            [(border_width, border_width), 
             (self.card_width - border_width, self.card_height - border_width)],
            outline=style['border_color'], 
            width=3,
            radius=15
        )
        
        # Add player portrait
        portrait_size = (self.card_width - 100, 300)
        portrait = self.create_player_portrait(
            portrait_size[0], portrait_size[1], team_color)
        card.paste(portrait, (50, 50), portrait)
        
        # Add player name
        name = player_data.get('name', 'Player Name')
        draw.text((self.card_width // 2, 400), 
                 name.upper(), 
                 font=self.name_font, 
                 fill=style['text_color'],
                 anchor="mm")
        
        # Add team and position
        team_pos = f"{player_data.get('team', 'N/A')} | {player_data.get('position', 'N/A')}"
        draw.text((self.card_width // 2, 450), 
                 team_pos, 
                 font=self.title_font, 
                 fill=style['accent_color'],
                 anchor="mm")
        
        # Add stats
        stats_y = 500
        stats = [
            ("PPG", player_data.get('ppg', 20.5), 35),
            ("RPG", player_data.get('rpg', 8.2), 15),
            ("APG", player_data.get('apg', 6.8), 10),
            ("FG%", player_data.get('fg_pct', 0.48), 0.6),
        ]
        
        for i, (label, value, max_val) in enumerate(stats):
            x = 50 + (i % 2) * 250
            y = stats_y + (i // 2) * 40
            
            # Draw stat bar
            bar_width = 200
            bar = self.create_stat_bar(label, value, max_val, bar_width, 20, 
                                     team_color[0])
            card.paste(bar, (x, y), bar)
        
        # Add card number
        draw.text((self.card_width - 50, self.card_height - 30), 
                 f"#{player_data.get('number', '00')}", 
                 font=self.small_font, 
                 fill=style['text_color'])
        
        # Add card style badge
        draw.ellipse([(30, self.card_height - 60, 80, self.card_height - 10)], 
                    fill=style['accent_color'])
        draw.text((55, self.card_height - 35), 
                 card_style.upper(), 
                 font=self.small_font, 
                 fill=(255, 255, 255),
                 anchor="mm")
        
        return card
    
    def generate_thumbnail(self, card: Image.Image) -> Image.Image:
        """Generate thumbnail version of card"""
        return card.resize((self.thumb_width, self.thumb_height), 
                         Image.Resampling.LANCZOS)
    
    def generate_card_set(self, players_data: List[Dict], output_dir: str):
        """Generate cards for multiple players"""
        os.makedirs(output_dir, exist_ok=True)
        
        for i, player in enumerate(players_data):
            # Cycle through card styles
            style = list(self.card_styles.keys())[i % len(self.card_styles)]
            
            # Generate card
            card = self.generate_card(player, style)
            
            # Save full size
            card_path = os.path.join(output_dir, f"card_{i+1:03d}.png")
            card.save(card_path, 'PNG', quality=95)
            
            # Generate and save thumbnail
            thumb = self.generate_thumbnail(card)
            thumb_path = os.path.join(output_dir, f"thumb_{i+1:03d}.png")
            thumb.save(thumb_path, 'PNG')
            
            print(f"Generated card {i+1}/{len(players_data)}: {player.get('name', 'Unknown')}")

def main():
    """Main function to generate sample cards"""
    # Sample player data
    sample_players = [
        {
            'name': 'LeBron James',
            'team': 'LAL',
            'position': 'SF',
            'ppg': 25.0,
            'rpg': 7.8,
            'apg': 7.8,
            'fg_pct': 0.52,
            'number': 23
        },
        {
            'name': 'Stephen Curry',
            'team': 'GSW',
            'position': 'PG',
            'ppg': 29.4,
            'rpg': 5.5,
            'apg': 6.3,
            'fg_pct': 0.487,
            'number': 30
        },
        {
            'name': 'Giannis Antetokounmpo',
            'team': 'MIL',
            'position': 'PF',
            'ppg': 31.1,
            'rpg': 11.8,
            'apg': 5.7,
            'fg_pct': 0.553,
            'number': 34
        }
    ]
    
    # Create generator
    generator = RealisticCardGenerator()
    
    # Generate cards
    print("Generating realistic basketball cards...")
    generator.generate_card_set(sample_players, "generated_cards")
    print("Cards generated in 'generated_cards' directory")

if __name__ == "__main__":
    main()