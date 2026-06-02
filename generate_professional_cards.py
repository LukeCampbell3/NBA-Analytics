#!/usr/bin/env python3
"""
Professional NBA Trading Card Generator
Creates Panini/Upper Deck/Topps-style basketball cards
"""

import json
import os
import random
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance, ImageOps
import math
import random
from typing import Tuple, List, Dict, Any
import textwrap

class NBACardGenerator:
    def __init__(self, output_dir: str = "generated_cards"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Card dimensions (standard trading card: 2.5" x 3.5" at 300 DPI)
        self.width = 750  # 2.5" at 300 DPI
        self.height = 1050  # 3.5" at 300 DPI
        self.thumb_width = 300
        self.thumb_height = 420
        
        # Team color schemes (primary, secondary, accent)
        self.team_colors = {
            'LAL': ('#552583', '#FDB927', '#000000'),  # Lakers: Purple, Gold, Black
            'GSW': ('#1D428A', '#FFC72C', '#000000'),  # Warriors: Blue, Gold, Black
            'BOS': ('#007A33', '#BA9653', '#000000'),  # Celtics: Green, Gold, Black
            'CHI': ('#CE1141', '#000000', '#FFFFFF'),  # Bulls: Red, Black, White
            'MIA': ('#98002E', '#F9A01B', '#000000'),  # Heat: Red, Orange, Black
            'LAC': ('#C8102E', '#1D428A', '#000000'),  # Clippers: Red, Blue, Black
            'DEN': ('#0E2240', '#FEC524', '#000000'),  # Nuggets: Navy, Gold, Black
            'MIL': ('#00471B', '#EEE1C6', '#000000'),  # Bucks: Green, Cream, Black
            'PHI': ('#006BB6', '#ED174C', '#000000'),  # 76ers: Blue, Red, Black
            'BKN': ('#000000', '#FFFFFF', '#000000'),  # Nets: Black, White, Black
            'default': ('#1E3A8A', '#DC2626', '#000000')  # Default: Blue, Red, Black
        }
        
        # Card styles based on real card types
        self.card_styles = {
            'base': {
                'name': 'Base Card',
                'border_color': (60, 60, 60),
                'bg_gradient': [(40, 40, 40), (80, 80, 80)],
                'accent_color': (0, 100, 200),
                'text_color': (255, 255, 255),
                'has_foil': False,
                'rarity': 'Common'
            },
            'premium': {
                'name': 'Premium Chrome',
                'border_color': (200, 180, 50),
                'bg_gradient': [(20, 20, 40), (40, 20, 60)],
                'accent_color': (255, 215, 0),
                'text_color': (255, 255, 200),
                'has_foil': True,
                'rarity': 'Rare'
            },
            'rookie': {
                'name': 'Rookie Card',
                'border_color': (0, 100, 200),
                'bg_gradient': [(20, 40, 80), (40, 60, 120)],
                'accent_color': (0, 150, 255),
                'text_color': (255, 255, 200),
                'has_foil': True,
                'rarity': 'Rookie'
            },
            'legacy': {
                'name': 'Legacy Edition',
                'border_color': (100, 100, 100),
                'bg_gradient': [(50, 50, 50), (80, 80, 80)],
                'accent_color': (200, 200, 200),
                'text_color': (220, 220, 220),
                'has_foil': False,
                'rarity': 'Vintage'
            }
        }
        
        # Try to load fonts, fallback to default
        try:
            self.title_font = ImageFont.truetype("arialbd.ttf", 36)
            self.name_font = ImageFont.truetype("arialbd.ttf", 48)
            self.stats_font = ImageFont.truetype("arial.ttf", 20)
            self.small_font = ImageFont.truetype("arial.ttf", 16)
        except:
            # Fallback to default font
            self.title_font = ImageFont.load_default()
            self.name_font = ImageFont.load_default()
            self.stats_font = ImageFont.load_default()
            self.small_font = ImageFont.load_default()
    
    def create_gradient(self, width: int, height: int, colors: List[Tuple[int, int, int]]) -> Image.Image:
        """Create a vertical gradient background"""
        base = Image.new('RGB', (width, height))
        draw = ImageDraw.Draw(base)
        
        for y in range(height):
            ratio = y / height
            r = int(colors[0][0] * (1 - ratio) + colors[1][0] * ratio)
            g = int(colors[0][1] * (1 - ratio) + colors[1][1] * ratio)
            b = int(colors[0][2] * (1 - ratio) + colors[1][2] * ratio)
            draw.line([(0, y), (width, y)], fill=(r, g, b))
        
        return base
    
    def add_foil_effect(self, image: Image.Image) -> Image.Image:
        """Add foil/reflective effect to card"""
        foil = Image.new('RGBA', image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(foil)
        
        # Add sparkle effects
        for _ in range(50):
            x = random.randint(0, image.width - 1)
            y = random.randint(0, image.height - 1)
            size = random.randint(1, 3)
            brightness = random.randint(200, 255)
            draw.ellipse([x, y, x + size, y + size], 
                        fill=(brightness, brightness, brightness, 150))
        
        # Add light streaks
        for _ in range(5):
            x1 = random.randint(0, image.width)
            y1 = random.randint(0, image.height)
            x2 = random.randint(0, image.width)
            y2 = random.randint(0, image.height)
            draw.line([(x1, y1), (x2, y2)], 
                     fill=(255, 255, 255, 100), width=2)
        
        return Image.alpha_composite(image.convert('RGBA'), foil)
    
    def create_player_silhouette(self, width: int, height: int, team_colors: Tuple) -> Image.Image:
        """Create a basketball player silhouette"""
        img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        # Draw a simplified basketball player silhouette
        # Head
        draw.ellipse([width//2-15, 20, width//2+15, 50], 
                    fill=team_colors[0], outline=team_colors[1], width=2)
        
        # Torso
        draw.rectangle([width//2-20, 50, width//2+20, 120], 
                      fill=team_colors[0], outline=team_colors[1], width=2)
        
        # Arms
        draw.line([(width//2-20, 70), (width//2-60, 100)], 
                 fill=team_colors[1], width=8)
        draw.line([(width//2+20, 70), (width//2+60, 100)], 
                 fill=team_colors[1], width=8)
        
        # Legs
        draw.line([(width//2-10, 120), (width//2-30, 180)], 
                 fill=team_colors[1], width=8)
        draw.line([(width//2+10, 120), (width//2+30, 180)], 
                 fill=team_colors[1], width=8)
        
        # Basketball
        draw.ellipse([width//2-10, 40, width//2+10, 60], 
                    fill=(222, 184, 135), outline=team_colors[1], width=2)
        
        return img
    
    def create_stat_bar(self, label: str, value: float, max_value: float, 
                      width: int, height: int, color: Tuple[int, int, int]) -> Image.Image:
        """Create a stat bar visualization"""
        bar = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(bar)
        
        # Background
        draw.rounded_rectangle([0, 0, width, height], 
                             fill=(40, 40, 40, 200), radius=5)
        
        # Filled portion
        fill_width = int((value / max_value) * (width - 4))
        draw.rounded_rectangle([2, 2, fill_width, height-2], 
                             fill=color, radius=3)
        
        # Label
        draw.text((width//2, height//2), label, 
                 fill=(255, 255, 255), 
                 font=self.small_font, 
                 anchor="mm")
        
        return bar
    
    def generate_card(self, player_data: Dict, style: str = 'base') -> Image.Image:
        """Generate a trading card for a player"""
        # Get card style
        style_config = self.card_styles.get(style, self.card_styles['base'])
        team = player_data.get('team', 'LAL')
        team_colors = self.team_colors.get(team, self.team_colors['default'])
        
        # Create base image
        card = Image.new('RGB', (self.width, self.height), color='white')
        draw = ImageDraw.Draw(card)
        
        # Create gradient background
        bg = self.create_gradient(self.width, self.height, 
                                style_config['bg_gradient'])
        card.paste(bg)
        
        # Add foil effect if needed
        if style_config['has_foil']:
            card = self.add_foil_effect(card)
        
        # Draw border
        border_width = 15
        draw.rounded_rectangle([border_width, border_width, 
                              self.width-border_width, self.height-border_width], 
                             outline=style_config['border_color'], 
                             width=3, radius=20)
        
        # Add player silhouette
        silhouette = self.create_player_silhouette(200, 300, team_colors)
        card.paste(silhouette, (self.width//2 - 100, 50), silhouette)
        
        # Add player name
        name = player_data.get('name', 'Player Name')
        draw.text((self.width//2, 400), 
                 name.upper(), 
                 fill=style_config['text_color'], 
                 font=self.name_font, 
                 anchor="mm")
        
        # Add team and position
        team_info = f"{player_data.get('team', 'N/A')} | {player_data.get('position', 'N/A')}"
        draw.text((self.width//2, 450), 
                 team_info, 
                 fill=style_config['accent_color'], 
                 font=self.title_font, 
                 anchor="mm")
        
        # Add stats
        stats = [
            ("PPG", player_data.get('ppg', 0), 50),
            ("RPG", player_data.get('rpg', 0), 15),
            ("APG", player_data.get('apg', 0), 10),
            ("FG%", player_data.get('fg_pct', 0.5), 1.0)
        ]
        
        for i, (label, value, max_val) in enumerate(stats):
            y_pos = 500 + (i // 2) * 50
            x_pos = 50 + (i % 2) * 200
            
            bar = self.create_stat_bar(f"{label}: {value}", 
                                     value, max_val, 
                                     180, 30, team_colors[0])
            card.paste(bar, (x_pos, y_pos), bar)
        
        # Add card number and rarity
        draw.text((50, self.height - 50), 
                 f"#{player_data.get('number', '000')}", 
                 fill=style_config['text_color'], 
                 font=self.small_font)
        
        draw.text((self.width - 50, self.height - 50), 
                 style_config['rarity'], 
                 fill=style_config['accent_color'], 
                 font=self.small_font, 
                 anchor="rm")
        
        return card
    
    def generate_thumbnail(self, card: Image.Image) -> Image.Image:
        """Create thumbnail version of card"""
        return card.resize((self.thumb_width, self.thumb_height), 
                         Image.Resampling.LANCZOS)
    
    def save_card(self, card: Image.Image, player_name: str, style: str):
        """Save card to file"""
        filename = f"{player_name.replace(' ', '_')}_{style}.png"
        filepath = self.output_dir / filename
        card.save(filepath, 'PNG', quality=95)
        return filepath

def main():
    """Main function to generate sample cards"""
    generator = NBACardGenerator("nba_trading_cards")
    
    # Sample player data
    players = [
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
    
    print("Generating NBA trading cards...")
    
    for i, player in enumerate(players):
        for style in ['base', 'premium', 'rookie']:
            card = generator.generate_card(player, style)
            thumb = generator.generate_thumbnail(card)
            
            # Save full size
            card_path = generator.save_card(card, player['name'], style)
            print(f"Generated: {card_path}")
    
    print(f"\nAll cards saved to: {generator.output_dir}")

if __name__ == "__main__":
    main()