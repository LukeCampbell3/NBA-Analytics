#!/usr/bin/env python3
"""
NBA Player Card Image Generator
Generates realistic basketball card images using Pillow
"""

import json
import os
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import math
import random
from typing import Dict, List, Tuple, Optional
import colorsys

class CardGenerator:
    def __init__(self, output_dir: str = "web/images", asset_dir: str = "assets"):
        self.output_dir = Path(output_dir)
        self.asset_dir = Path(asset_dir)
        self.card_width = 600
        self.card_height = 840
        self.thumb_width = 300
        self.thumb_height = 420
        
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "full").mkdir(exist_ok=True)
        (self.output_dir / "thumbnails").mkdir(exist_ok=True)
        
        # Team colors (NBA team color schemes)
        self.team_colors = {
            "LAL": ("#552583", "#FDB927"),  # Lakers: Purple & Gold
            "BOS": ("#007A33", "#BA9653"),  # Celtics: Green & Gold
            "GSW": ("#1D428A", "#FFC72C"),  # Warriors: Blue & Gold
            "CHI": ("#CE1141", "#000000"),  # Bulls: Red & Black
            "MIA": ("#98002E", "#F9A01B"),  # Heat: Red & Orange
            "NYK": ("#006BB6", "#F58426"),  # Knicks: Blue & Orange
            "PHI": ("#006BB6", "#ED174C"),  # 76ers: Blue & Red
            "LAC": ("#C8102E", "#1D428A"),  # Clippers: Red & Blue
            "DEN": ("#0E2240", "#FEC524"),  # Nuggets: Blue & Gold
            "MIL": ("#00471B", "#EEE1C6"),  # Bucks: Green & Cream
            "PHX": ("#1D1160", "#E56020"),  # Suns: Purple & Orange
            "DAL": ("#00538C", "#002B5E"),  # Mavericks: Blue & Navy
            "HOU": ("#CE1141", "#C4CED4"),  # Rockets: Red & Silver
            "SAS": ("#C4CED4", "#000000"),  # Spurs: Silver & Black
            "OKC": ("#007AC1", "#EF3B24"),  # Thunder: Blue & Orange
            "POR": ("#E03A3E", "#000000"),  # Blazers: Red & Black
            "UTA": ("#002B5C", "#00471B"),  # Jazz: Navy & Green
            "SAC": ("#5A2D81", "#63727A"),  # Kings: Purple & Gray
            "ORL": ("#0077C0", "#C4CED4"),  # Magic: Blue & Silver
            "WAS": ("#002B5C", "#E31837"),  # Wizards: Navy & Red
            "CLE": ("#860038", "#FDBB30"),  # Cavaliers: Wine & Gold
            "DET": ("#C8102E", "#1D42BA"),  # Pistons: Red & Blue
            "CHA": ("#1D1160", "#00788C"),  # Hornets: Purple & Teal
            "ATL": ("#E03A3E", "#C1D32F"),  # Hawks: Red & Volt
            "MEM": ("#5D76A9", "#12173F"),  # Grizzlies: Blue & Navy
            "NOP": ("#0C2340", "#C8102E"),  # Pelicans: Navy & Red
            "MIN": ("#0C2340", "#236192"),  # Timberwolves: Blue & Blue
            "TOR": ("#CE1141", "#000000"),  # Raptors: Red & Black
            "IND": ("#002D62", "#FDBB30"),  # Pacers: Navy & Gold
            "BRK": ("#000000", "#FFFFFF"),  # Nets: Black & White
        }
        
        # Card families and materials
        self.card_families = {
            "standard": {
                "chrome": {"texture": "metallic", "shine": 0.8},
                "refractor": {"texture": "holographic", "shine": 0.9}
            },
            "fit": {
                "blue-ice": {"color": (13, 71, 161), "transparency": 0.7},
                "zebra-ice": {"color": (33, 33, 33), "pattern": "stripes"},
                "tiger-ice": {"color": (191, 54, 12), "pattern": "tiger"},
                "black-ice": {"color": (0, 0, 0), "transparency": 0.5}
            },
            "value": {
                "auto": {"style": "signature", "texture": "paper"},
                "patch": {"style": "jersey", "texture": "fabric"},
                "manga": {"style": "comic", "texture": "print"},
                "auto-patch": {"style": "combined", "texture": "mixed"}
            }
        }
        
        # Try to load fonts, fall back to default if not available
        try:
            self.title_font = ImageFont.truetype("arial.ttf", 36)
            self.name_font = ImageFont.truetype("arialbd.ttf", 48)
            self.stats_font = ImageFont.truetype("arial.ttf", 24)
            self.small_font = ImageFont.truetype("arial.ttf", 18)
        except:
            print("Warning: Using default font (arial.ttf not found)")
            self.title_font = ImageFont.load_default()
            self.name_font = ImageFont.load_default()
            self.stats_font = ImageFont.load_default()
            self.small_font = ImageFont.load_default()
    
    def hex_to_rgb(self, hex_color: str) -> Tuple[int, int, int]:
        """Convert hex color to RGB tuple"""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    def create_gradient(self, width: int, height: int, color1: Tuple[int, int, int], 
                       color2: Tuple[int, int, int], direction: str = "diagonal") -> Image.Image:
        """Create a gradient background"""
        base = Image.new('RGB', (width, height), color1)
        top = Image.new('RGB', (width, height), color2)
        
        mask = Image.new('L', (width, height), 0)
        draw = ImageDraw.Draw(mask)
        
        if direction == "diagonal":
            for i in range(height):
                alpha = int(255 * (i / height))
                draw.line([(0, i), (width, i)], fill=alpha)
        elif direction == "vertical":
            for i in range(height):
                alpha = int(255 * (i / height))
                draw.rectangle([(0, i), (width, i)], fill=alpha)
        elif direction == "horizontal":
            for i in range(width):
                alpha = int(255 * (i / width))
                draw.rectangle([(i, 0), (i, height)], fill=alpha)
        elif direction == "radial":
            center_x, center_y = width // 2, height // 2
            max_radius = math.sqrt(center_x**2 + center_y**2)
            for y in range(height):
                for x in range(width):
                    distance = math.sqrt((x - center_x)**2 + (y - center_y)**2)
                    alpha = int(255 * (1 - min(distance / max_radius, 1)))
                    mask.putpixel((x, y), alpha)
        
        return Image.composite(base, top, mask)
    
    def add_texture(self, image: Image.Image, texture_type: str = "paper") -> Image.Image:
        """Add texture to image"""
        if texture_type == "paper":
            # Create paper-like texture
            texture = Image.new('RGB', image.size, (255, 255, 255))
            draw = ImageDraw.Draw(texture)
            for _ in range(1000):
                x = random.randint(0, image.width - 1)
                y = random.randint(0, image.height - 1)
                brightness = random.randint(240, 255)
                draw.point((x, y), fill=(brightness, brightness, brightness))
            return Image.blend(image, texture, alpha=0.1)
        
        elif texture_type == "metallic":
            # Add metallic shine
            shine = Image.new('RGB', image.size, (255, 255, 255))
            draw = ImageDraw.Draw(shine)
            for _ in range(500):
                x = random.randint(0, image.width - 1)
                y = random.randint(0, image.height - 1)
                size = random.randint(1, 3)
                draw.ellipse([x, y, x+size, y+size], fill=(255, 255, 255, 128))
            return Image.blend(image, shine, alpha=0.05)
        
        return image
    
    def create_card_background(self, width: int, height: int, primary_color: Tuple[int, int, int],
                              secondary_color: Tuple[int, int, int], family: str, material: str) -> Image.Image:
        """Create card background with team colors and material effects"""
        # Create base gradient
        bg = self.create_gradient(width, height, primary_color, secondary_color, "diagonal")
        
        # Add material effects based on family
        if family == "standard":
            if material == "chrome":
                bg = self.add_texture(bg, "metallic")
                # Add chrome reflection
                reflection = Image.new('RGBA', (width, height), (255, 255, 255, 0))
                draw = ImageDraw.Draw(reflection)
                for i in range(0, width, 20):
                    draw.rectangle([(i, 0), (i+10, height)], fill=(255, 255, 255, 30))
                bg = Image.alpha_composite(bg.convert('RGBA'), reflection)
        
        elif family == "fit":
            # Add ice/glass effect
            ice_overlay = Image.new('RGBA', (width, height), (255, 255, 255, 50))
            draw = ImageDraw.Draw(ice_overlay)
            # Add some ice cracks
            for _ in range(20):
                x1 = random.randint(0, width)
                y1 = random.randint(0, height)
                x2 = x1 + random.randint(-50, 50)
                y2 = y1 + random.randint(-50, 50)
                draw.line([(x1, y1), (x2, y2)], fill=(255, 255, 255, 100), width=2)
            bg = Image.alpha_composite(bg.convert('RGBA'), ice_overlay)
        
        elif family == "value":
            if material == "auto":
                # Add signature-like texture
                bg = self.add_texture(bg, "paper")
            elif material == "patch":
                # Add fabric-like texture
                fabric = Image.new('RGB', (width, height), (200, 200, 200))
                draw = ImageDraw.Draw(fabric)
                for i in range(0, width, 5):
                    for j in range(0, height, 5):
                        if (i + j) % 10 == 0:
                            draw.rectangle([(i, j), (i+2, j+2)], fill=(180, 180, 180))
                bg = Image.blend(bg, fabric, alpha=0.1)
        
        return bg
    
    def draw_player_info(self, draw: ImageDraw.Draw, player: Dict, 
                        primary_color: Tuple[int, int, int], 
                        secondary_color: Tuple[int, int, int],
                        width: int, height: int):
        """Draw player information on card"""
        name = player.get("name", "Unknown Player")
        team = player.get("team", "UNK")
        position = player.get("position", "N/A")
        season = player.get("season", 2025)
        age = player.get("age", 25)
        
        # Draw player name (big and bold)
        name_bbox = draw.textbbox((0, 0), name, font=self.name_font)
        name_width = name_bbox[2] - name_bbox[0]
        name_x = (width - name_width) // 2
        draw.text((name_x, 50), name, font=self.name_font, fill=secondary_color)
        
        # Draw team and position
        team_text = f"{team} • {position}"
        team_bbox = draw.textbbox((0, 0), team_text, font=self.title_font)
        team_width = team_bbox[2] - team_bbox[0]
        team_x = (width - team_width) // 2
        draw.text((team_x, 120), team_text, font=self.title_font, fill=primary_color)
        
        # Draw season and age
        info_text = f"Season: {season} | Age: {age}"
        info_bbox = draw.textbbox((0, 0), info_text, font=self.small_font)
        info_width = info_bbox[2] - info_bbox[0]
        info_x = (width - info_width) // 2
        draw.text((info_x, 170), info_text, font=self.small_font, fill=(200, 200, 200))
        
        # Draw card border
        border_width = 10
        draw.rounded_rectangle(
            [(border_width, border_width), (width - border_width, height - border_width)],
            radius=30,
            outline=secondary_color,
            width=3
        )
        
        # Draw inner border
        inner_border = 20
        draw.rounded_rectangle(
            [(inner_border, inner_border), (width - inner_border, height - inner_border)],
            radius=25,
            outline=primary_color,
            width=2
        )
    
    def draw_stats(self, draw: ImageDraw.Draw, player: Dict, width: int, height: int):
        """Draw player statistics"""
        # Sample stats - in production, these would come from player data
        stats = [
            ("Value Score", player.get("value_metrics", {}).get("player_value_score", 75.5)),
            ("Fit Percentile", random.randint(60, 95)),
            ("Breakout Chance", random.randint(40, 85)),
            ("Trust Score", random.randint(70, 95))
        ]
        
        # Draw stats in a grid
        stat_y = 250
        for i, (label, value) in enumerate(stats):
            col = i % 2
            row = i // 2
            x = 100 + col * 200
            y = stat_y + row * 80
            
            # Draw stat box
            draw.rounded_rectangle(
                [(x-40, y-30), (x+40, y+30)],
                radius=10,
                fill=(40, 40, 40, 180),
                outline=(100, 100, 100),
                width=2
            )
            
            # Draw value
            value_text = f"{value:.1f}" if isinstance(value, float) else str(value)
            value_bbox = draw.textbbox((0, 0), value_text, font=self.title_font)
            value_width = value_bbox[2] - value_bbox[0]
            draw.text((x - value_width//2, y - 20), value_text, font=self.title_font, fill=(255, 255, 255))
            
            # Draw label
            label_bbox = draw.textbbox((0, 0), label, font=self.small_font)
            label_width = label_bbox[2] - label_bbox[0]
            draw.text((x - label_width//2, y + 10), label, font=self.small_font, fill=(180, 180, 180))
    
    def draw_card_elements(self, draw: ImageDraw.Draw, family: str, material: str, 
                          width: int, height: int, primary_color: Tuple[int, int, int]):
        """Draw card decorative elements"""
        # Draw family label
        family_text = f"{family.upper()} SERIES"
        family_bbox = draw.textbbox((0, 0), family_text, font=self.small_font)
        family_x = 30
        family_y = height - 60
        draw.text((family_x, family_y), family_text, font=self.small_font, fill=primary_color)
        
        # Draw material label
        material_text = material.replace("-", " ").upper()
        material_bbox = draw.textbbox((0, 0), material_text, font=self.small_font)
        material_x = width - material_bbox[2] - 30
        draw.text((material_x, family_y), material_text, font=self.small_font, fill=primary_color)
        
        # Draw decorative elements at corners
        corner_size = 40
        # Top-left corner
        draw.arc([(20, 20), (20+corner_size, 20+corner_size)], 180, 270, fill=primary_color, width=3)
        # Top-right corner
        draw.arc([(width-20-corner_size, 20), (width-20, 20+corner_size)], 270, 360, fill=primary_color, width=3)
        # Bottom-left corner
        draw.arc([(20, height-20-corner_size), (20+corner_size, height-20)], 90, 180, fill=primary_color, width=3)
        # Bottom-right corner
        draw.arc([(width-20-corner_size, height-20-corner_size), (width-20, height-20)], 0, 90, fill=primary_color, width=3)
        
        # Add some decorative dots
        for i in range(8):
            angle = (i / 8) * 2 * math.pi
            radius = min(width, height) // 2 - 50
            x = width // 2 + int(radius * math.cos(angle))
            y = height // 2 + int(radius * math.sin(angle))
            draw.ellipse([(x-3, y-3), (x+3, y+3)], fill=primary_color)
    
    def generate_card(self, player: Dict, family: str = "standard", material: str = "chrome") -> Image.Image:
        """Generate a single player card"""
        # Get team colors
        team = player.get("team", "LAL")
        primary_hex, secondary_hex = self.team_colors.get(team, ("#552583", "#FDB927"))
        primary_color = self.hex_to_rgb(primary_hex)
        secondary_color = self.hex_to_rgb(secondary_hex)
        
        # Create background
        bg = self.create_card_background(
            self.card_width, self.card_height,
            primary_color, secondary_color,
            family, material
        )
        
        # Convert to RGBA for drawing
        card = bg.convert('RGBA')
        draw = ImageDraw.Draw(card)
        
        # Draw all elements
        self.draw_player_info(draw, player, primary_color, secondary_color, self.card_width, self.card_height)
        self.draw_stats(draw, player, self.card_width, self.card_height)
        self.draw_card_elements(draw, family, material, self.card_width, self.card_height, primary_color)
        
        return card
    
    def generate_thumbnail(self, card: Image.Image) -> Image.Image:
        """Generate thumbnail from card"""
        return card.resize((self.thumb_width, self.thumb_height), Image.Resampling.LANCZOS)
    
    def save_card(self, card: Image.Image, player: Dict, family: str, material: str, is_thumbnail: bool = False):
        """Save card to file"""
        # Create filename
        name_slug = player["name"].replace(" ", "_")
        team = player.get("team", "UNK")
        season = player.get("season", 2025)
        
        if is_thumbnail:
            filename = f"{name_slug}_{team}_{season}_{family}_{material}_thumb.png"
            filepath = self.output_dir / "thumbnails" / filename
        else:
            filename = f"{name_slug}_{team}_{season}_{family}_{material}_full.png"
            filepath = self.output_dir / "full" / filename
        
        # Save as PNG
        card.save(filepath, "PNG", optimize=True)
        return str(filepath)
    
    def process_player(self, player_data: Dict, family: str, material: str) -> Dict:
        """Process a single player and generate cards"""
        # Extract player info from nested structure
        player_info = player_data.get("player", {})
        name = player_info.get("name", "Unknown Player")
        team = player_info.get("team", "UNK")
        
        print(f"Generating card for {name} ({team}) - {family}/{material}")
        
        # Generate full-size card
        card = self.generate_card(player_info, family, material)
        full_path = self.save_card(card, player_info, family, material, is_thumbnail=False)
        
        # Generate thumbnail
        thumb = self.generate_thumbnail(card)
        thumb_path = self.save_card(thumb, player_info, family, material, is_thumbnail=True)
        
        return {
            "player": player_info,
            "card_assignment": {
                "family": family,
                "material": material,
                "color_scheme": [self.team_colors.get(team, ("#552583", "#FDB927"))[0],
                               self.team_colors.get(team, ("#552583", "#FDB927"))[1]]
            },
            "image_assets": {
                "card_front": f"images/full/{os.path.basename(full_path)}",
                "card_thumb": f"images/thumbnails/{os.path.basename(thumb_path)}"
            }
        }
    
    def generate_cards(self, players_data: List[Dict], max_players: int = 10) -> List[Dict]:
        """Generate cards for multiple players"""
        results = []
        
        # Define card families and materials to test
        test_assignments = [
            ("standard", "chrome"),
            ("standard", "refractor"),
            ("fit", "blue-ice"),
            ("fit", "zebra-ice"),
            ("value", "auto"),
            ("value", "patch")
        ]
        
        # Process players
        for i, player in enumerate(players_data[:max_players]):
            if i >= len(test_assignments):
                break
                
            family, material = test_assignments[i % len(test_assignments)]
            result = self.process_player(player, family, material)
            results.append(result)
        
        return results
    
    def save_manifest(self, results: List[Dict]):
        """Save manifest file"""
        manifest_path = self.output_dir.parent / "card-images.json"
        with open(manifest_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Saved manifest to {manifest_path}")

def main():
    """Main function"""
    print("=== NBA Player Card Image Generator ===\n")
    
    # Load player data
    data_path = Path("web/data/cards.json")
    if not data_path.exists():
        print(f"Error: Data file not found at {data_path}")
        sys.exit(1)
    
    print(f"Loading player data from {data_path}...")
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    players = data if isinstance(data, list) else [data]
    print(f"Loaded {len(players)} players")
    
    # Initialize generator
    generator = CardGenerator()
    
    # Generate cards (test with 6 players to show different card types)
    print("\nGenerating card images...")
    results = generator.generate_cards(players, max_players=6)
    
    # Save manifest
    generator.save_manifest(results)
    
    print(f"\n✅ Generated {len(results)} player cards")
    print(f"📁 Full-size cards: web/images/full/")
    print(f"📁 Thumbnails: web/images/thumbnails/")
    print(f"📄 Manifest: web/card-images.json")
    
    # Show sample of generated files
    print("\n📋 Sample generated files:")
    for result in results[:3]:
        player = result["player"]
        assets = result["image_assets"]
        print(f"  • {player['name']} ({player.get('team', 'UNK')}):")
        print(f"    Full: {assets['card_front']}")
        print(f"    Thumb: {assets['card_thumb']}")

if __name__ == "__main__":
    main()